# chain_of_thought_wrapper.py

import re
import torch
import logging
from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
    AutoModelForCausalLM # Needed for example usage
)
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from typing import Optional, List, Tuple, Dict, Union, Any
import gc # Import garbage collector for cleanup
import time # Import time for potential timing/logging (unused in final code, but good practice)
from collections import Counter # Needed for example voting

# --- Logging Setup ---
# Configure logging for the module. This helps in debugging and understanding wrapper behavior.
# Set level to DEBUG temporarily to see the detailed logs added below
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Ensure logger doesn't add handlers multiple times if the script is imported repeatedly
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Avoid propagation to the root logger, preventing duplicate messages
    logger.propagate = False

# --- Default Configuration Values ---
# These defaults provide sensible starting points for the wrapper's behavior,
# based on common practices and the audit recommendations.
DEFAULT_MAX_LENGTH = 2048 # Increased default max length to accommodate longer CoT
DEFAULT_REASONING_LIMIT = 15 # A conceptual limit for extracted steps (not strictly enforced by parsing logic)
DEFAULT_CONSISTENCY_ROUNDS = 5 # Default number of chains for self-consistency, increased based on typical research
DEFAULT_COMPLEXITY_KEYWORDS = ["explain", "step by step", "plan", "analyze", "reasoning", "logic"] # Keywords (currently unused as CoT is always on)
DEFAULT_FINAL_ANSWER_TAG = "Final_Answer:" # Explicit tag to signal the final answer

# --- Regex Pattern for Parsing Steps ---
# This pattern is used to identify and extract individual reasoning steps from
# the generated text. It's designed to be flexible, capturing common step formats
# like "Step N:", "N.", etc., case-insensitive for "Step".
# Captures the text *after* the step marker.
DEFAULT_STEP_PATTERN = re.compile(
    r"^(?:Step\s*\d+[:.)-]\s*|\d+[:.)-]\s*)(.*)", re.IGNORECASE
)

# --- Common Artifact Cleanup Regex ---
# Regex patterns to remove common problematic tokens or structures models sometimes emit,
# which are not part of the desired reasoning or answer. Based on audit suggestion.
ARTIFACT_PATTERNS = [
    re.compile(r"<init>.*?</init>", re.DOTALL),       # Example: DeepSeek R1 init tags
    re.compile(r"<final_output>.*?</final_output>", re.DOTALL), # Example: DeepSeek R1 final output tags
    # re.compile(r"\{.*?\}", re.DOTALL), # Removing all {} might be too aggressive, removed based on re-evaluation.
    # Add other specific artifact patterns here as needed for observed model outputs
]

class ChainOfThoughtWrapper:
    """
    A robust Chain-of-Thought (CoT) wrapper for Hugging Face models.

    This wrapper enforces a Chain-of-Thought process by injecting a specific
    template into the prompt. It handles model generation and parses the
    output to extract reasoning steps and a final answer. It is designed
    to generate multiple sequences for potential Self-Consistency voting
    (voting logic is intended for the calling application, e.g., a GUI).

    It incorporates enhancements based on a detailed audit, focusing on
    prompting, decoding, parsing robustness, cross-model compatibility,
    reliability mitigation, and efficiency, while adhering to the "always-on CoT"
    principle.

    Key Features:
    - Forces CoT via a structured, adaptive prompt template.
    - Parses structured reasoning steps and uses robust logic to find the final answer.
    - Supports generating multiple chains for Self-Consistency analysis via GenerationConfig.
    - Handles common cross-model compatibility issues (e.g., pad tokens, device placement).
    - Merges user-provided GenerationConfig with sensible defaults.
    - Includes basic cleanup for common model output artifacts.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, GenerationMixin, Any],
        tokenizer: AutoTokenizer,
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        reasoning_steps_limit: int = DEFAULT_REASONING_LIMIT,
        self_consistency_enabled: bool = False, # Control if multiple chains are generated
        consistency_rounds: int = DEFAULT_CONSISTENCY_ROUNDS,
        complexity_keywords: Optional[List[str]] = None, # Currently unused as CoT is always on
        final_answer_tag: str = DEFAULT_FINAL_ANSWER_TAG,
        # Optional prompt customization for advanced users
        cot_instruction: str = "Let's analyze this problem logically, breaking it down step by step to reach the precise final answer.",
        reasoning_header: str = "Reasoning Process:",
        step_prefix: str = "Step ", # e.g., "Step 1: " - model will ideally continue this
        # Optional reliability controls (simple, prompt-based)
        emphasize_factual: bool = True,
        allow_uncertainty_phrase: Optional[str] = "If information is insufficient or you are unsure, state that clearly.",
        # Optional parsing flexibility
        strip_artifact_patterns: List[re.Pattern] = ARTIFACT_PATTERNS,
    ):
        """
        Initializes the ChainOfThoughtWrapper with enhanced configurations.

        Args:
            model (Union[PreTrainedModel, GenerationMixin, Any]): The language model.
                                                                   Must have a .generate() method.
            tokenizer (AutoTokenizer): The corresponding tokenizer.
            generation_config (Optional[GenerationConfig]): A default generation configuration.
                                                            Values here can be overridden by generate() call.
            device (Optional[str]): The device to load the model onto ('cpu' or 'cuda').
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
            max_length (int): The maximum total length of the input + generated sequence.
                              This should be large enough for the prompt, reasoning, and answer.
            reasoning_steps_limit (int): Conceptual limit for parsed steps. Not strictly enforced by current parsing.
            self_consistency_enabled (bool): If True, enable multi-chain generation for self-consistency.
            consistency_rounds (int): The number of chains to generate if `self_consistency_enabled` is True.
                                      Actual number of sequences is controlled by `num_return_sequences`
                                      in the final `GenerationConfig`.
            complexity_keywords (Optional[List[str]]): List of keywords (unused with always-on CoT).
            final_answer_tag (str): The specific string marker expected before the final answer.
            cot_instruction (str): The core instruction phrase for CoT.
            reasoning_header (str): The header text before the reasoning steps.
            step_prefix (str): The prefix for the first step.
            emphasize_factual (bool): If True, add prompt text emphasizing factual reasoning.
            allow_uncertainty_phrase (Optional[str]): If provided, add a phrase prompting model to state uncertainty.
            strip_artifact_patterns (List[re.Pattern]): List of regex patterns to remove from model output before parsing.
        """
        # --- Device Handling ---
        # Determine and set the device. Log the chosen device.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing ChainOfThoughtWrapper on device: %s", self.device)

        # --- Model and Tokenizer Loading and Configuration ---
        # Move the model to the specified device and set to evaluation mode.
        # Includes error handling for device transfer.
        try:
            self.model = model.to(self.device)
            self.model.eval() # Set model to evaluation mode (disables dropout, etc.)
            logger.info("Model moved to %s and set to eval mode.", self.device)
        except Exception as e:
            logger.error("Failed to move model to device %s: %s", self.device, e)
            raise # Re-raise the exception if device transfer fails

        self.tokenizer = tokenizer

        # Attempt to find the underlying Hugging Face model instance and its config.
        # This helps reliably access attributes like `config.vocab_size`, `resize_token_embeddings`, etc.
        self._hf_model_instance, self._hf_config = self._find_hf_model_and_config(self.model)

        # Handle models/tokenizers without a defined pad_token_id.
        # This is crucial for batch generation (like `num_return_sequences`).
        # If the tokenizer doesn't have a pad_token, try to use the eos_token.
        # If neither exists, add a special token and resize embeddings.
        # The wrapper's `resize_token_embeddings` method is called here if a new token is added.
        if self.tokenizer.pad_token_id is None:
             if self.tokenizer.eos_token_id is not None:
                  self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                  logger.warning("Tokenizer pad_token_id is None, using eos_token_id (%s) as pad_token_id.", self.tokenizer.eos_token_id)
             else:
                 # Fallback: Add a new pad token if neither exists
                 logger.warning("Tokenizer pad_token_id and eos_token_id are both None. Adding a [PAD] token.")
                 self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
                 logger.info("Added new [PAD] token with ID %s.", self.tokenizer.pad_token_id)
                 # Resize model embeddings if we added a new token AND we found a base HF model instance
                 if self._hf_model_instance:
                     self.resize_token_embeddings(len(self.tokenizer)) # Call the instance method
                     logger.info("Resized model embeddings to accommodate new PAD token.")
                 else:
                     logger.warning("Could not resize model embeddings after adding PAD token; underlying HF model instance not found.")
                     logger.warning("Ensure the model can handle a larger vocabulary if batching is used.")

        # --- Configuration Attributes ---
        self.max_length = max_length
        self.reasoning_steps_limit = reasoning_steps_limit
        # The actual number of sequences to generate is controlled by `num_return_sequences` in the final `GenerationConfig`.
        # We store `consistency_rounds` to potentially inform this value.
        self.self_consistency_enabled = self_consistency_enabled
        self.consistency_rounds = max(1, consistency_rounds) if self_consistency_enabled else 1

        # --- Prompt Template Components ---
        self.complexity_keywords = complexity_keywords or list(DEFAULT_COMPLEXITY_KEYWORDS) # Store keywords (currently unused for logic)
        self.final_answer_tag = final_answer_tag
        self._cot_instruction = cot_instruction # Customizable CoT instruction
        self._reasoning_header = reasoning_header # Customizable reasoning header
        self._step_prefix = step_prefix # Customizable step prefix (e.g., "Step ")

        # --- Reliability/Hallucination Mitigation Prompt Components ---
        self._emphasize_factual = emphasize_factual
        self._allow_uncertainty_phrase = allow_uncertainty_phrase

        # --- Parsing Attributes and Compiled Regex ---
        # Compile regex pattern for final answer extraction based on the specified tag.
        # re.escape handles potential special characters in the tag. re.DOTALL matches newline.
        self.final_answer_pattern = re.compile(
            re.escape(final_answer_tag) + r"\s*(.*)", re.IGNORECASE | re.DOTALL
        )
        self._step_pattern = DEFAULT_STEP_PATTERN # Use the default compiled step pattern
        self._artifact_patterns = strip_artifact_patterns # Patterns for cleaning model output

        logger.debug("Final answer pattern compiled: %s", self.final_answer_pattern.pattern)
        logger.debug("Step pattern: %s", self._step_pattern.pattern)

        # --- Base Generation Config Setup ---
        # Create or copy the base GenerationConfig. This config holds the default
        # generation parameters that will be used unless overridden during a generate() call.
        # Use .from_dict(.to_dict()) for a clean copy if a config was provided.
        if generation_config:
            self.base_generation_config = GenerationConfig.from_dict(generation_config.to_dict())
            logger.info("Initialized with provided base GenerationConfig.")
        else:
            # Create a default GenerationConfig if none was provided.
            # Incorporate parameters known to work well for CoT based on audit (temp, top_p, top_k).
            # Ensure pad_token_id and eos_token_id are set from the tokenizer (or the fallback).
            self.base_generation_config = GenerationConfig(
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.max_length, # Max total length
                do_sample=True,             # Always sample for diversity (essential for multi-chain)
                temperature=0.7,            # Balanced randomness
                top_p=0.95,                 # Nucleus sampling
                top_k=50,                   # Top-k sampling cutoff
                num_return_sequences=1,     # Default to 1 sequence (will be overridden by generate call if self-consistency is on)
                # Add a mild repetition penalty, useful for longer CoT
                repetition_penalty=1.1,     # Discourage immediate repetition
                no_repeat_ngram_size=0,     # Default to no n-gram repetition prevention
            )
            logger.info("Initialized with default base GenerationConfig.")

        # Ensure the base config uses the determined pad_token_id
        # This might be redundant if tokenizer already has it, but ensures consistency
        self.base_generation_config.pad_token_id = self.tokenizer.pad_token_id
        logger.debug("Base GenerationConfig pad_token_id set to %s.", self.base_generation_config.pad_token_id)

        # Check if the underlying HF model (if found) supports returning scores, useful for CISC.
        # We set this on the model's config if possible, as `generate` reads from there.
        if self._hf_model_instance and hasattr(self._hf_model_instance.config, 'return_dict_in_generate'):
             try:
                 # Set these attributes directly on the model's config object
                 self._hf_model_instance.config.return_dict_in_generate = True
                 self._hf_model_instance.config.output_scores = True # Also request scores
                 logger.debug("Set underlying HF model config to return dict in generate and output scores.")
             except Exception as e:
                 logger.warning("Failed to set return_dict_in_generate/output_scores on HF model config: %s", e)
        else:
             logger.debug("Underlying HF model instance or config does not support setting return_dict_in_generate/output_scores.")


        logger.info("ChainOfThoughtWrapper initialization complete.")
        logger.debug("Final Base GenerationConfig: %s", self.base_generation_config.to_dict())


    def _find_hf_model_and_config(self, obj: Any) -> Tuple[Optional[PreTrainedModel], Optional[Any]]:
        """
        Recursively searches for an underlying Hugging Face PreTrainedModel
        and its configuration within a potentially wrapped or custom object.
        This helps in accessing standard HF attributes like `config` or
        methods like `resize_token_embeddings`.

        Args:
            obj (Any): The object to inspect (could be the model itself or a wrapper).

        Returns:
            Tuple[Optional[PreTrainedModel], Optional[Any]]: The found HF model instance and its config.
                                                              Returns (None, None) if not found.
        """
        # Add a check to prevent infinite recursion
        if getattr(obj, '_searching_hf_model', False):
             logger.debug("Preventing infinite recursion in _find_hf_model_and_config for object type: %s", type(obj))
             return None, None
        setattr(obj, '_searching_hf_model', True)

        logger.debug("Searching for HF model in object of type: %s", type(obj))
        # If the object is directly a PreTrainedModel and has a config
        if isinstance(obj, PreTrainedModel):
            logger.debug("Found HF PreTrainedModel directly.")
            setattr(obj, '_searching_hf_model', False) # Reset flag
            return obj, getattr(obj, 'config', None) # Return config if it exists

        # Check common attribute names where the base model might be stored
        potential_attrs = ('model', 'base_model', 'transformer', '_original_model', 'module') # Added 'module'
        for attr_name in potential_attrs:
            m = getattr(obj, attr_name, None)
            if m is not None:
                logger.debug("Checking attribute '%s' of type %s", attr_name, type(m))
                # Recursively search within the attribute
                found_model, found_config = self._find_hf_model_and_config(m)
                if found_model or found_config:
                    setattr(obj, '_searching_hf_model', False) # Reset flag before returning
                    return found_model, found_config

        # If no PreTrainedModel found through attributes, check if the object itself has a 'config' attribute
        if hasattr(obj, 'config'):
             logger.debug("Found config attribute on object, but no PreTrainedModel instance.")
             setattr(obj, '_searching_hf_model', False) # Reset flag
             return None, obj.config # Return the config found

        logger.debug("No underlying HF PreTrainedModel instance or config found.")
        setattr(obj, '_searching_hf_model', False) # Reset flag
        return None, None


    def _inject_cot(self, prompt: str) -> str:
        """
        Injects the structured Chain-of-Thought template into the user's prompt.
        This template guides the model's response format.
        Incorporates reliability prompts based on settings.

        Args:
            prompt (str): The original user prompt.

        Returns:
            str: The prompt with the CoT template appended.
        """
        # Start with the cleaned original prompt
        injected_prompt = f"{prompt.strip()}\n\n"

        # Add the core CoT instruction phrase
        injected_prompt += self._cot_instruction + "\n"

        # Add reliability-focused instructions if enabled
        if self._emphasize_factual:
             injected_prompt += "Think through the problem step-by-step using only factual information and logical deduction. Do not assume any facts that are not given.\n"
        if self._allow_uncertainty_phrase:
             injected_prompt += self._allow_uncertainty_phrase + "\n"

        # Add the structured template for reasoning steps and final answer tag
        injected_prompt += f"\n{self._reasoning_header}\n\n"
        injected_prompt += f"{self._step_prefix}1: " # Explicitly start the first step to guide format consistency

        logger.debug("Injected CoT template. Full prompt starts with: %s...", injected_prompt[:200].replace('\n', '\\n'))
        return injected_prompt


    @torch.no_grad() # Disable gradient calculation during generation for efficiency
    def generate(
        self,
        input_text: str,
        generation_config: Optional[GenerationConfig] = None, # Optional override config for this call
        num_return_sequences: Optional[int] = None, # Explicitly request N sequences
    ) -> Dict[str, Any]:
        """
        Generates text using the wrapped model with Chain-of-Thought injection.
        Handles tokenization, prompt injection, generation, and parsing.
        Efficiently generates multiple sequences using `num_return_sequences`.

        Args:
            input_text (str): The user's input text/question.
            generation_config (Optional[GenerationConfig]): Additional generation parameters
                                                            to override the base config for this call.
            num_return_sequences (Optional[int]): Number of independent sequences (chains) to generate.
                                                  If None, uses the value from the merged generation config
                                                  (defaulting to 1 or `consistency_rounds` if enabled).

        Returns:
            Dict[str, Any]: A dictionary containing the generation results:
                            - 'sequences': The raw generated token IDs (list of tensors).
                            - 'full_texts': List of raw, cleaned text outputs (after stripping prompt/artifacts) for each chain.
                            - 'reasoning_steps': List of lists of extracted reasoning steps for each chain.
                            - 'final_answers': List of extracted final answer strings for each chain.
                            - 'generation_scores': Scores if requested and available (for CISC externally).
        """
        logger.info("Received generate call with input text starting: '%s...'", input_text[:100])

        # 1) Inject the CoT prompt into the original input text
        cot_prompt_text = self._inject_cot(input_text)

        # 2) Tokenize the full CoT prompt
        # Ensure padding is handled correctly. Use return_tensors="pt" for PyTorch tensors.
        # truncation=True ensures the input fits within max_length.
        # max_length applies to the input sequence here.
        try:
            encoded_input = self.tokenizer(
                cot_prompt_text,
                return_tensors="pt",
                padding="longest", # Pad to the longest sequence in the batch (only 1 here, but good practice)
                truncation=True,
                max_length=self.max_length, # Truncate if the prompt itself is too long
            ).to(self.device)
            logger.debug("Input text tokenized. Input IDs shape: %s, on device: %s", encoded_input['input_ids'].shape, encoded_input['input_ids'].device)
        except Exception as e:
            logger.error("Failed to tokenize input text: %s", e)
            # Attempt cleanup before raising
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            raise # Re-raise tokenization error


        # 3) Build the final GenerationConfig for this specific call
        # Start with the base config, then merge any provided overrides.
        # Use .from_dict(.to_dict()) for safe merging.
        cfg = GenerationConfig.from_dict(self.base_generation_config.to_dict())

        if generation_config is not None:
             logger.debug("Merging provided generation_config overrides...")
             cfg.update(**generation_config.to_dict())
             logger.debug("Merged user-provided GenerationConfig.")

        # Explicitly set num_return_sequences for this call based on the argument.
        # This overrides any num_return_sequences set in the base config or the provided override config.
        if num_return_sequences is not None:
            cfg.num_return_sequences = num_return_sequences
            logger.debug("Using num_return_sequences from function argument: %s", cfg.num_return_sequences)
        elif self.self_consistency_enabled:
             # Fallback: If num_return_sequences argument is None, use consistency_rounds if self_consistency is enabled
             cfg.num_return_sequences = self.consistency_rounds
             logger.debug("num_return_sequences argument is None, using consistency_rounds (%s) because self_consistency is enabled.", cfg.num_return_sequences)
        else:
             # Fallback: If num_return_sequences argument is None and self_consistency is disabled, default to 1
             cfg.num_return_sequences = 1
             logger.debug("num_return_sequences argument is None and self_consistency disabled, defaulting to 1.")


        # Ensure max_length in the config respects the wrapper's max_length setting
        # max_length in generate() config is the *total* length (input + new tokens)
        # max_new_tokens is the number of *new* tokens generated
        # Prefer max_new_tokens if set, otherwise calculate from max_length
        input_length = encoded_input['input_ids'].shape[1]
        if cfg.max_new_tokens is None:
             # If max_new_tokens is NOT set, ensure the total length does not exceed the wrapper's max_length
             if cfg.max_length is not None:
                 # Only adjust cfg.max_length if it's set in the base/override config
                 cfg.max_length = min(self.max_length, cfg.max_length)
             else:
                 # If neither max_new_tokens nor max_length were set in base/override, use wrapper's max_length
                 cfg.max_length = self.max_length
             logger.debug("max_new_tokens not set in config. Using total max_length: %s (Input length: %s)", cfg.max_length, input_length)
        else:
            # If max_new_tokens IS set, the total length will be input_length + max_new_tokens
            # We should check if this effective total length exceeds the wrapper's overall max_length
            effective_total_length = input_length + cfg.max_new_tokens
            if effective_total_length > self.max_length:
                 logger.warning("Effective total length (input %d + new %d = %d) exceeds wrapper max_length (%d). Adjusting max_new_tokens.",
                                input_length, cfg.max_new_tokens, effective_total_length, self.max_length)
                 # Adjust max_new_tokens down to respect the wrapper's limit
                 cfg.max_new_tokens = max(0, self.max_length - input_length)
                 logger.warning("Adjusted max_new_tokens to %d.", cfg.max_new_tokens)

        # Ensure pad_token_id and eos_token_id are correctly set in the final config
        # Use tokenizer's IDs as the source of truth
        cfg.pad_token_id = self.tokenizer.pad_token_id
        cfg.eos_token_id = self.tokenizer.eos_token_id

        logger.debug("Final GenerationConfig for this call after resolving overrides and num_return_sequences: %s", cfg.to_dict())

        # --- Debugging: Inspect inputs immediately before generation ---
        # ADDED LOGGING HERE TO DIAGNOSE CUDA ERROR
        logger.debug("-" * 30 + " Inputs to model.generate " + "-" * 30)
        logger.debug("  Input Text Snippet: '%s...'", input_text[:100])
        logger.debug("  CoT Prompt Text Snippet: '%s...'", cot_prompt_text[:200].replace('\n', '\\n'))
        logger.debug("  Input IDs shape: %s, dtype: %s, device: %s", encoded_input["input_ids"].shape, encoded_input["input_ids"].dtype, encoded_input["input_ids"].device)
        if encoded_input.get("attention_mask", None) is not None:
            logger.debug("  Attention Mask shape: %s, dtype: %s, device: %s", encoded_input["attention_mask"].shape, encoded_input["attention_mask"].dtype, encoded_input["attention_mask"].device)
            # Log a snippet of the attention mask for inspection (only first batch item, first 20 tokens)
            if encoded_input["attention_mask"].numel() > 0:
                 logger.debug("  Attention Mask snippet (first 20): %s", encoded_input["attention_mask"][0, :20].tolist())
            # Check if mask seems valid (contains only 0s and 1s) - might not catch all CUDA errors but helps debug
            if not torch.all((encoded_input["attention_mask"] == 0) | (encoded_input["attention_mask"] == 1)):
                 logger.error("!!! Attention mask contains values other than 0 or 1 !!!")
        else:
            logger.warning("!!! No attention mask provided to model.generate !!!")
        logger.debug("  GenerationConfig.pad_token_id: %s", cfg.pad_token_id)
        logger.debug("  GenerationConfig.eos_token_id: %s", cfg.eos_token_id)
        logger.debug("  GenerationConfig.num_return_sequences: %s", cfg.num_return_sequences)
        logger.debug("-" * 30 + " End Inputs to model.generate " + "-" * 30)
        # --- End Debugging ---


        # 4) Generate text using the model's generate method
        # Pass input_ids and attention_mask. Pass the *final* GenerationConfig object.
        try:
            generation_output = self.model.generate(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input.get("attention_mask", None),
                generation_config=cfg, # Pass the fully configured GenerationConfig
                # Request scores if supported by the model/config for potential CISC implementation externally
                return_dict_in_generate=True, # Request dict output
                output_scores=True,          # Request scores
            )
            generated_sequences = generation_output.sequences
            # If scores were requested and returned, they are available in generation_output.scores
            # These can be used by the caller for CISC voting.
            generation_scores = generation_output.scores if hasattr(generation_output, 'scores') else None
            logger.info("Generation complete. Generated %d sequence(s).", len(generated_sequences))
            if generation_scores:
                 logger.debug("Generation scores available (%d scores tensors).", len(generation_scores))

        except Exception as e:
            logger.error("Model generation failed: %s", e)
            # Log the exception details
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback

             # Attempt cleanup even on failure - this *might* also trigger the CUDA error again,
             # but it's the correct place to *try* to clean up GPU memory associated with the model.
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.debug("Attempted torch.cuda.empty_cache() after generation failure.")
                except Exception as cache_e:
                    logger.error("Error during cuda empty_cache after generation failure: %s", cache_e)
            gc.collect()
            logger.debug("Attempted gc.collect() after generation failure.")

            raise # Re-raise generation error


        # 5) Decode and Parse the generated sequences
        # Ensure generated_sequences is a list or tensor before decoding
        if not isinstance(generated_sequences, (list, torch.Tensor)) or len(generated_sequences) == 0:
             logger.warning("No sequences generated. Returning empty results.")
             return {
                "sequences": [],
                "full_texts": [],
                "reasoning_steps": [],
                "final_answers": [],
                "generation_scores": None,
             }

        decoded_outputs = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        logger.debug("Batch decoding complete.")
        parsed_results = [self._parse(text, cot_prompt_text) for text in decoded_outputs]
        logger.debug("Parsing complete for %d sequences.", len(parsed_results))


        # Unpack the parsed results
        all_steps = [result[0] for result in parsed_results]
        all_final_answers = [result[1] for result in parsed_results]
        full_generated_bodies = [result[2] for result in parsed_results]

        # 6) Construct and return the results dictionary
        # The actual self-consistency voting logic is handled by the caller,
        # but the wrapper provides the necessary outputs (multiple chains and parsed answers).
        return {
            "sequences": generated_sequences, # Raw sequences (token IDs)
            "full_texts": full_generated_bodies, # Cleaned generated text bodies
            "reasoning_steps": all_steps, # Parsed reasoning steps for each chain
            "final_answers": all_final_answers, # Parsed final answer for each chain
            "generation_scores": generation_scores, # Scores if requested and available (for CISC)
        }


    def _parse(self, text: str, cot_prompt: str) -> Tuple[List[str], str, str]:
        """
        Parses the generated text to extract reasoning steps and the final answer.
        This is a robust parsing function that handles different formats,
        artifacts, and provides fallback logic for finding the answer.

        Args:
            text (str): The raw text output from the model for a single chain.
            cot_prompt (str): The exact prompt text that was injected (used to remove it from the output).

        Returns:
            Tuple[List[str], str, str]: A tuple containing:
                                        - A list of extracted reasoning step strings.
                                        - The extracted final answer string.
                                        - The full body of the generated text (after removing the prompt and artifacts).
        """
        logger.debug("Starting parsing for a single generated text chunk...")

        # 1) Remove the exact injected prompt from the beginning of the text.
        # This isolates the model's generated continuation.
        body = text
        if text.startswith(cot_prompt):
            body = text[len(cot_prompt):] # Remove the prefix
            logger.debug("Removed exact CoT prompt (%d characters) from beginning.", len(cot_prompt))
        else:
            logger.warning("Generated text does not start with the injected CoT prompt. Attempting to parse entire text after initial whitespace strip.")
            body = text.lstrip() # Just strip leading whitespace if template wasn't followed

        # 2) Apply artifact cleanup patterns
        logger.debug("Applying artifact cleanup patterns...")
        original_body_len = len(body)
        cleaned_body = body # Start with body after prompt removal
        for pattern in self._artifact_patterns:
            cleaned_body = pattern.sub("", cleaned_body)
        if len(cleaned_body) < original_body_len:
            logger.debug("Artifact cleanup removed %d characters.", original_body_len - len(cleaned_body))
        else:
            logger.debug("No artifacts found matching patterns.")

        # Ensure body is stripped after cleanup
        cleaned_body = cleaned_body.strip()
        body_lines = [l.strip() for l in cleaned_body.splitlines() if l.strip()] # Split into non-empty, stripped lines

        steps = [] # List to store extracted steps
        final_answer = "" # Variable to store the final answer
        found_final_answer_tagged = False # Flag to track if the specific tag was found

        # 3) Extract Steps and Final Answer (Primary Method: Tagged Answer)
        # Iterate through lines and apply regex patterns.
        # Prioritize finding the explicit final answer tag.
        logger.debug("Attempting to extract steps and final answer using explicit tag '%s'...", self.final_answer_tag)
        for i, line in enumerate(body_lines):
            # Check for the explicit final answer tag pattern first
            final_answer_match = self.final_answer_pattern.search(line)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                logger.debug("Extracted final answer using explicit tag: '%s'", final_answer[:100])
                found_final_answer_tagged = True
                # Once the tagged answer is found, we can stop processing lines for it
                # We still iterate through ALL lines below to capture all steps BEFORE the tag.
                # No break here because we need to collect steps that might appear after the tag was first encountered on a line.
                # E.g., "Step 1: ... Final_Answer: X Step 2: ..." (unlikely but possible)
                # The logic below ensures we capture steps *before* the final answer.


        # Now, iterate through lines AGAIN to collect steps.
        # This second pass ensures we collect steps even if the answer tag was found early.
        # We stop collecting steps once we encounter the line that *contained* the final answer tag,
        # or if we apply a step limit.
        logger.debug("Collecting reasoning steps...")
        for i, line in enumerate(body_lines):
            # Stop collecting steps if we found the final answer tag on this line or a previous one
            # And if we've reached or passed the line where the tag was found (if it was found)
            # This requires knowing the index of the line where the tag was found.
            # A simpler approach: just collect all lines matching step pattern UP TO the first line
            # where the final answer tag was found.
            final_answer_line_index = -1
            for idx, l in enumerate(body_lines):
                 if self.final_answer_pattern.search(l):
                      final_answer_line_index = idx
                      break # Found the first occurrence of the tag

            if final_answer_line_index != -1 and i >= final_answer_line_index:
                 logger.debug("Stopped collecting steps at line index %d because final answer tag was found on line %d.", i, final_answer_line_index)
                 break # Stop collecting steps once we reach the line with the answer tag

            # Check for reasoning step pattern
            step_match = self._step_pattern.match(line)
            if step_match:
                step_text = step_match.group(1).strip()
                if step_text: # Only add non-empty steps
                    steps.append(step_text)
                    # logger.debug("Extracted step: '%s'", steps[-1][:50]) # Too verbose usually
                # Stop adding steps if we've reached a defined limit
                if len(steps) >= self.reasoning_steps_limit:
                   logger.debug("Reached reasoning steps limit (%d). Stopping step extraction.", self.reasoning_steps_limit)
                   break # Stop collecting steps if limit is reached


        # 4) Fallback for Final Answer (If Tag Still Not Found)
        # If the explicit final answer tag was not found after both passes, apply fallback heuristics.
        if not found_final_answer_tagged:
            logger.debug("Explicit final answer tag not found. Applying fallback heuristics.")

            # Fallback: Assume the last non-step line is the answer.
            # Iterate backwards through the processed lines to find the last line that doesn't look like a step.
            # Using the 'body_lines' list after cleanup and stripping.
            last_non_step_line = ""
            for line in reversed(body_lines): # Iterate backwards through non-empty, stripped lines
                if line and not self._step_pattern.match(line):
                    last_non_step_line = line.strip()
                    logger.debug("Fallback: Identified last non-step line: '%s'", last_non_step_line[:100])
                    break # Found the last non-step line, stop searching backwards

            if last_non_step_line:
                 # Check if the last non-step line *contains* the final answer tag,
                 # even if it didn't *start* with it or wasn't the line where the tag was first found.
                 fa_match_fallback = self.final_answer_pattern.search(last_non_step_line)
                 if fa_match_fallback:
                      final_answer = fa_match_fallback.group(1).strip()
                      logger.debug("Fallback found tagged answer in last non-step line: '%s'", final_answer[:100])
                 else:
                    # If no tag in the last non-step line, just use the line itself as the answer
                    final_answer = last_non_step_line
                    logger.debug("Fallback using last non-step line as answer: '%s'", final_answer[:100])
            else:
                 # If no non-empty or non-step lines were found, the final answer is empty
                 final_answer = ""
                 logger.debug("Fallback: No non-empty or non-step lines found in body. Final answer is empty.")

        # 5) Basic Post-Parsing Cleanup on Final Answer
        # Remove any trailing punctuation from the final answer, unless it's part of specific symbols (like !?)
        # This helps normalize answers for voting.
        if final_answer:
            # Remove common trailing characters like periods, commas, etc.
            final_answer = re.sub(r'[.,;:]+$', '', final_answer).strip()
            # Remove common leading "Answer: " or similar preambles if they weren't removed by tag matching
            # This needs to be case-insensitive
            final_answer = re.sub(r'^\s*(?:Answer|Result|Output|Final Answer)\s*[:\-]?\s*', '', final_answer, flags=re.IGNORECASE).strip()
            logger.debug("Applied basic post-parsing cleanup to final answer: '%s'", final_answer[:100])

        # Final check: Ensure steps list doesn't contain the final answer line or text
        # This is a belt-and-suspenders approach as the logic above should prevent it,
        # but safeguards against edge cases where the tag wasn't found but the line
        # looked like a step *and* contained the answer.
        if final_answer and steps:
             # Remove any step that exactly matches the final answer after stripping
             steps = [step for step in steps if step.strip() != final_answer.strip()]
             # Also check if the final answer is contained *within* a step (less likely but possible)
             steps = [step for step in steps if final_answer.strip() not in step.strip()]


        logger.info("Parsing complete. Steps found: %d, Final Answer: '%s'", len(steps), final_answer[:100])

        # Return the extracted steps, the final answer, and the cleaned generated body text
        return steps, final_answer, cleaned_body # Return steps, final answer, and the cleaned body text


    def resize_token_embeddings(self, new_size: int):
        """
        Resizes the model's token embeddings to match a new vocabulary size,
        useful after adding new tokens (like a custom PAD token) to the tokenizer.
        This operation is crucial if the tokenizer size changes and the model
        is used for generation or training.

        Only works if the underlying model object is a PreTrainedModel
        or has a `resize_token_embeddings` method.

        Args:
            new_size (int): The new size of the vocabulary/embedding layer.
                            Should typically be `len(self.tokenizer)`.
        """
        # Use the stored HF model instance found during initialization
        hf_model_instance = self._hf_model_instance

        if hf_model_instance and hasattr(hf_model_instance, 'resize_token_embeddings'):
            try:
                old_size = hf_model_instance.get_input_embeddings().weight.size(0)
                if new_size != old_size:
                    logger.info("Attempting to resize model token embeddings from %d to %d.", old_size, new_size)
                    # Ensure the model is on the correct device before resizing
                    hf_model_instance.to(self.device)
                    hf_model_instance.resize_token_embeddings(new_size)
                    logger.info("Successfully resized token embeddings.")
                    # Update model config's vocab size if available
                    if hasattr(hf_model_instance, 'config') and hasattr(hf_model_instance.config, 'vocab_size'):
                         hf_model_instance.config.vocab_size = new_size
                         logger.debug("Updated underlying model config vocab_size to %d.", new_size)
                    # Attempt garbage collection after a potentially memory-intensive operation
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.info("Embedding size is already %d, no resizing needed.", new_size)
            except Exception as e:
                 logger.error("Failed to resize token embeddings: %s", e)
                 # Attempt cleanup even on failure
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 gc.collect()
                 # Note: Not re-raising here by default, as a failure might not be critical
                 # depending on the user's intended use (e.g., if they don't use the new tokens for generation).
                 # Could be re-raised if this is deemed a critical error.
        else:
            logger.warning("Cannot resize token embeddings: The underlying model object does not have a 'resize_token_embeddings' method or HF model instance not found.")


# Example Usage (Illustrative)
if __name__ == "__main__":
    print("--- ChainOfThoughtWrapper Example Usage ---")
    print("This block demonstrates loading a small HF model and using the wrapper.")
    print("Setting logging level to DEBUG to see detailed wrapper logs.")
    logger.setLevel(logging.DEBUG) # Set logger to DEBUG for example

    # You would replace this with your actual model loading logic
    try:
        # Use a tiny, fast model for a quick test
        # NOTE: distilgpt2 might still hit CUDA errors with num_return_sequences > 1
        # if there are underlying driver/CUDA/PyTorch compatibility issues or
        # subtle model-specific padding bugs in HF transformers for this architecture.
        # If this example still fails, try a different simple causal model like 'gpt2' or a small LLaMA variant.
        model_id = "distilbert/distilgpt2" # A slightly larger but still fast GPT-2 variant
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Attempting to load model {model_id} on {device}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Ensure pad token is set for generation robustness (common requirement for GPT-like models)
        # Handle this *before* loading the model if possible, or ensure embeddings are resized.
        if tokenizer.pad_token_id is None:
             if tokenizer.eos_token_id is not None:
                  tokenizer.pad_token_id = tokenizer.eos_token_id
                  logger.warning("Tokenizer pad_token_id is None, using eos_token_id (%s) as pad_token_id.", tokenizer.eos_token_id)
             else:
                 # Add a pad token if neither eos nor pad exists.
                 # This *must* be done before loading the model or resizing embeddings.
                 logger.warning("Tokenizer pad_token_id and eos_token_id are both None. Adding a [PAD] token.")
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
                 logger.info("Added new [PAD] token with ID %s.", tokenizer.pad_token_id)
                 # Note: Resizing embeddings will be handled by the wrapper during initialization
                 # if a compatible HF model instance is found.


        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_id)


        # Instantiate the wrapper
        # Simulate parameters that would come from a GUI or config
        # This GenerationConfig will override some defaults in the wrapper's base config for this call.
        simulated_base_gen_config = GenerationConfig(
            max_new_tokens=128,      # Limit generated tokens
            temperature=0.85,        # Slightly higher temp for diversity in multiple chains
            do_sample=True,          # Crucial for sampling-based generation
            # num_return_sequences is intentionally NOT set here; it's set by the wrapper based on generate() argument
            pad_token_id=tokenizer.pad_token_id, # Pass pad_token_id explicitly
            eos_token_id=tokenizer.eos_token_id, # Pass eos_token_id explicitly
            # Add other parameters based on tuning recommendations if desired
            repetition_penalty=1.1 # Apply repetition penalty
        )

        # Instantiate the wrapper, enabling self-consistency flags in init
        # These flags inform the wrapper's default behavior if generate() args are None
        cot_wrapper = ChainOfThoughtWrapper(
            model=model,
            tokenizer=tokenizer,
            generation_config=simulated_base_gen_config, # Pass overrides here if desired as base
            device=device,
            self_consistency_enabled=True, # Simulate SC enabled
            consistency_rounds=5, # Simulate consistency rounds setting
            final_answer_tag="Final Answer:", # Use a slightly different tag for demo
            # Keep factual emphasis on for demo
            emphasize_factual=True,
            allow_uncertainty_phrase="If you cannot determine a definitive answer, state that.",
        )

        # Prepare input prompt
        # Use a prompt that encourages steps and a clear answer
        prompt_text = "If a train travels at 60 mph for 2.5 hours, how far does it travel? Calculate step-by-step."
        logger.info(f"Generating reasoning for prompt: '{prompt_text}'")

        # Generate outputs
        # We explicitly pass num_return_sequences to the generate call (e.g., from GUI slider)
        num_chains_to_generate = 3 # Simulate GUI setting num_chains slider to 3
        logger.info(f"Calling wrapper.generate() requesting {num_chains_to_generate} chains.")

        start_time = time.time()
        outputs = cot_wrapper.generate(
            input_text=prompt_text,
            # No explicit generation_config override here; uses the base config initialized in the wrapper
            # but you *could* pass overrides like: generation_config=GenerationConfig(temperature=1.0)
            num_return_sequences=num_chains_to_generate, # Pass the desired number of sequences here
        )
        end_time = time.time()
        logger.info(f"Generation of {len(outputs.get('sequences', []))} sequences took {end_time - start_time:.2f} seconds.")


        # --- Process Results (including simulated Self-Consistency voting logic) ---
        print("\n" + "="*50)
        print("--- Generation Results ---")
        print("="*50)

        full_texts = outputs.get('full_texts', [])
        reasoning_steps = outputs.get('reasoning_steps', [])
        final_answers_raw = outputs.get('final_answers', []) # Raw answers from wrapper

        if not full_texts:
             print("No chains were generated or parsed.")
        else:
            for i, (full_text, steps, final_answer_raw) in enumerate(zip(full_texts, reasoning_steps, final_answers_raw)):
                print(f"\n--- Chain {i+1} ---")
                print("Full Text (Cleaned):")
                print(full_text)
                print("\nReasoning Steps Parsed:")
                if steps:
                    # Ensure steps is a list before iterating
                    steps = steps if isinstance(steps, list) else []
                    for j, step in enumerate(steps):
                         # Ensure step is a string before printing
                         if isinstance(step, str) and step.strip():
                            print(f"  Step {j+1}: {step.strip()}")
                         elif not isinstance(step, str):
                              print(f"  [Step {j+1} has invalid format]")
                    if not steps: # If steps list was empty after checks
                        print("  [No steps parsed]")
                else: # If steps was None or not a list initially
                    print("  [No steps parsed]")
                print("\nFinal Answer Parsed (Raw):")
                # Ensure raw answer is a string before printing
                display_raw_answer = final_answer_raw if isinstance(final_answer_raw, str) and final_answer_raw.strip() else "[No final answer parsed]"
                print(f"  '{display_raw_answer}'")


        # --- Simulate Self-Consistency Voting (as would be done in GUI) ---
        print("\n" + "="*50)
        print("--- Simple Self-Consistency Voting Simulation ---")
        print("="*50)

        if final_answers_raw:
            # Perform the actual voting using the helper functions
            consensus_answer, answer_distribution_dict = perform_self_consistency_voting(final_answers_raw)
            answer_distribution = Counter(answer_distribution_dict) # Convert to Counter for display

            print(f"Raw Answers Submitted for Voting: {final_answers_raw}")
            print(f"Normalized Answers for Voting: {list(answer_distribution_dict.keys())}") # Show unique normalized answers
            print(f"Answer Counts: {dict(answer_distribution)}")

            if consensus_answer:
                print(f"\nConsensus Answer: '{consensus_answer}'")
                # Get count of the winning normalized answer
                winner_count = answer_distribution.get(normalize_answer(consensus_answer), 0)
                print(f"(Voted by {winner_count} chain(s) out of {len(final_answers_raw)})")

                # Optional: Check for ties (more sophisticated tie-breaking would go here in a real voter)
                if len(answer_distribution) > 1 and answer_distribution.most_common(2)[0][1] == answer_distribution.most_common(2)[1][1]:
                     print("Note: There is a tie for the most common normalized answer.")

            else:
                print("No valid final answers found for voting.")
        else:
            print("No final answers were parsed from any chain for voting.")


    except Exception as e:
        logger.error("An error occurred during the example usage: %s", e)
        import traceback
        traceback.print_exc() # Print detailed traceback for the example failure

    print("\n--- Example Usage End ---")
    # Attempt final cleanup
    if torch.cuda.is_available():
         try:
             torch.cuda.empty_cache()
             print("GPU memory cache cleared.")
         except Exception as cleanup_e:
              print(f"Error during final cuda empty_cache: {cleanup_e}")
    gc.collect()
    print("Garbage collected.")