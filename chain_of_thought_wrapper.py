import re
import torch
import logging
from transformers import PreTrainedModel, AutoTokenizer, GenerationConfig, GenerationMixin
from typing import Optional, List, Tuple, Dict, Union, Any

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MAX_LENGTH = 1024
DEFAULT_REASONING_LIMIT = 10
DEFAULT_CONSISTENCY_ROUNDS = 3
DEFAULT_COMPLEXITY_KEYWORDS = ["explain", "step by step", "plan", "analyze", "reasoning", "logic"]
DEFAULT_FINAL_ANSWER_TAG = "Final_Answer:"

# **Expanded** step‐pattern to catch both "Step 1:" and bare "1."
DEFAULT_STEP_PATTERN = re.compile(
    r"^(?:Step\s*\d+[:.)-]|\d+[:.)-])\s*(.*)", re.IGNORECASE
)

class ChainOfThoughtWrapper:
    """
    A robust, SOTA Chain-of-Thought wrapper for Hugging Face models or custom wrappers.
    ALWAYS uses Chain‑of‑Thought now, with stricter injection and cleaning.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, GenerationMixin, Any],
        tokenizer: AutoTokenizer,
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        reasoning_steps_limit: int = DEFAULT_REASONING_LIMIT,
        self_consistency: bool = False,
        consistency_rounds: int = DEFAULT_CONSISTENCY_ROUNDS,
        complexity_keywords: Optional[List[str]] = None,
        final_answer_tag: str = DEFAULT_FINAL_ANSWER_TAG,
    ):
        """
        model: HF model or wrapper implementing `.generate()`
        tokenizer: corresponding tokenizer
        generation_config: overrides defaults
        device: 'cpu'/'cuda'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reasoning_steps_limit = reasoning_steps_limit
        self.self_consistency = self_consistency
        self.consistency_rounds = max(1, consistency_rounds) if self_consistency else 1
        self.complexity_keywords = complexity_keywords or DEFAULT_COMPLEXITY_KEYWORDS
        self.final_answer_tag = final_answer_tag
        self.final_answer_pattern = re.compile(
            re.escape(final_answer_tag) + r"\s*(.*)", re.IGNORECASE | re.DOTALL
        )

        # Try to locate HF config; fallback to tokenizer if missing
        self._hf_model, self._hf_config = self._find_hf_model_and_config(self.model)
        if self._hf_config is None:
            logger.warning("HF config not found, falling back to tokenizer settings.")
            class PseudoConfig:
                def __init__(self, tok):
                    self.eos_token_id = tok.eos_token_id
                    self.pad_token_id = tok.pad_token_id or tok.eos_token_id
                    self.vocab_size = len(tok)
            self._hf_config = PseudoConfig(self.tokenizer)

        # Setup generation config
        if generation_config:
            self.generation_config = GenerationConfig.from_dict(generation_config.to_dict())
        else:
            self.generation_config = GenerationConfig(
                eos_token_id=self._hf_config.eos_token_id,
                pad_token_id=self._hf_config.pad_token_id,
                max_length=self.max_length,
            )

        # Ensure HF model returns dict outputs
        try:
            setattr(self._hf_config, 'return_dict_in_generate', True)
        except Exception:
            pass

        logger.info("ChainOfThoughtWrapper ready on %s", self.device)

    def _find_hf_model_and_config(self, obj: Any) -> Tuple[Optional[PreTrainedModel], Optional[Any]]:
        """Search for underlying PreTrainedModel and its config."""
        if isinstance(obj, PreTrainedModel) and hasattr(obj, 'config'):
            return obj, obj.config
        for attr in ('model','base_model','transformer'):
            m = getattr(obj, attr, None)
            if isinstance(m, PreTrainedModel) and hasattr(m, 'config'):
                return m, m.config
        return None, getattr(obj, 'config', None)

    def _inject_cot(self, prompt: str) -> str:
        # **More prescriptive CoT template**
        return (
            f"{prompt}\n\n"
            "Let's analyze step by step exactly like this:\n\n"
            "Step 1: \n"
            "Step 2: \n"
            "Step 3: \n\n"
            "Final Answer:\n\n"
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Returns dict with keys: sequences, full_texts, reasoning_steps, final_answers
        ALWAYS uses CoT path.
        """
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # **ALWAYS** do CoT, ignore complexity check
        cot_prompt = self._inject_cot(prompt_text)

        # Merge configs
        cfg = GenerationConfig.from_dict(self.generation_config.to_dict())
        if generation_config:
            cfg.update(**generation_config.to_dict())
        cfg.num_return_sequences = num_return_sequences
        for k,v in kwargs.items(): setattr(cfg, k, v)

        # Encode with injected template
        enc = self.tokenizer(
            cot_prompt, return_tensors='pt', truncation=True,
            max_length=self.max_length - cfg.max_new_tokens
        ).to(self.device)

        out = self.model.generate(
            input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], generation_config=cfg
        )

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        results = [self._parse(text, cot_prompt) for text in decoded]
        seqs = out
        steps = [r[0] for r in results]
        finals = [r[1] for r in results]
        full = [r[2] for r in results]
        return {'sequences': seqs, 'full_texts': full, 'reasoning_steps': steps, 'final_answers': finals}

    def _parse(self, text: str, cot_prompt: str) -> Tuple[List[str], str, str]:
        # Remove the injected prompt
        body = text[len(cot_prompt):].strip() if text.startswith(cot_prompt) else text

        # **Clean out any stray tags or JSON fragments**
        body = re.sub(r"<init>.*?</init>", "", body, flags=re.DOTALL)
        body = re.sub(r"<final_output>.*?</final_output>", "", body, flags=re.DOTALL)
        body = re.sub(r"\{.*?\}", "", body, flags=re.DOTALL)

        lines = [l.strip() for l in body.splitlines() if l.strip()]
        steps = []
        final = ""

        for l in lines:
            m = DEFAULT_STEP_PATTERN.match(l)
            if m:
                steps.append(m.group(1).strip())
            else:
                fa = self.final_answer_pattern.search(l)
                if fa:
                    final = fa.group(1).strip()
                    break

        if not final:
            # assume last non‑step line is the final answer
            final = lines[-1] if lines else ""

        return steps, final, body

    def resize_token_embeddings(self, new_size: int):
        if hasattr(self._hf_model, 'resize_token_embeddings'):
            self._hf_model.resize_token_embeddings(new_size)
            logger.info("Resized embeddings to %d", new_size)
        else:
            logger.error("Cannot resize: no underlying HF model method.")
