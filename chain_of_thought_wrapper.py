# chain_of_thought_wrapper.py

import re
import torch
import logging
from transformers import PreTrainedModel, AutoTokenizer, GenerationConfig, GenerationMixin
from transformers.utils import is_accelerate_available, is_bitsandbytes_available
from typing import Optional, List, Tuple, Dict, Union, Any
import gc # Import garbage collector for cleanup
import time

# --- Logging Setup ---
# Configure logging for the module
logging.basicConfig(level=logging.INFO) # Default logging level
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if imported multiple times
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent logs from going to root logger multiple times


# --- Default Configuration Values ---
# These defaults provide sensible starting points for the wrapper's behavior.
DEFAULT_MAX_LENGTH = 1024          # Default maximum length of the generated output sequence.
DEFAULT_REASONING_LIMIT = 10       # Limit on the number of steps to extract during parsing (currently unused in parse logic, but good to keep as a concept).
DEFAULT_CONSISTENCY_ROUNDS = 3     # Default number of chains to generate for self-consistency (used in __init__, passed via GUI num_chains).
DEFAULT_COMPLEXITY_KEYWORDS = ["explain", "step by step", "plan", "analyze", "reasoning", "logic"] # Keywords to potentially trigger CoT (currently unused, CoT is always on).
DEFAULT_FINAL_ANSWER_TAG = "Final_Answer:" # The specific tag expected before the final answer.

# --- Regex Pattern for Parsing Steps ---
# This pattern is used to identify and extract individual reasoning steps from
# the generated text. It's designed to be flexible, capturing:
# - "Step N:"
# - "Step N."
# - "Step N-"
# - "N:"
# - "N."
# - "N-"
# Where N is one or more digits, case-insensitive for "Step".
DEFAULT_STEP_PATTERN = re.compile(
    r"^(?:Step\s*\d+[:.)-]|\d+[:.)-])\s*(.*)", re.IGNORECASE
)


class ChainOfThoughtWrapper:
    """
    A robust Chain-of-Thought (CoT) wrapper for Hugging Face models.

    This wrapper enforces a Chain-of-Thought process by injecting a specific
    template into the prompt. It handles model generation and parses the
    output to extract reasoning steps and a final answer. It is designed
    to generate multiple sequences for potential Self-Consistency voting
    (voting logic is expected to be handled by the calling application,
    like the Streamlit GUI).

    Key Features:
    - Forces CoT via prompt injection.
    - Parses structured reasoning steps and final answer from output.
    - Supports generating multiple chains for Self-Consistency analysis.
    - Compatible with Hugging Face PreTrainedModels or objects implementing `.generate()`.
    - Handles device placement and merges GenerationConfig.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, GenerationMixin, Any],
        tokenizer: AutoTokenizer,
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        reasoning_steps_limit: int = DEFAULT_REASONING_LIMIT, # Parameter included as per provided code
        self_consistency: bool = False, # Parameter included as per provided code (__init__ attribute)
        consistency_rounds: int = DEFAULT_CONSISTENCY_ROUNDS, # Parameter included as per provided code (__init__ attribute)
        complexity_keywords: Optional[List[str]] = None, # Parameter included as per provided code
        final_answer_tag: str = DEFAULT_FINAL_ANSWER_TAG,
        # self_consistency_enabled: bool = False # Removed this based on user's 'keep as is' and gui interaction
    ):
        """
        Initializes the ChainOfThoughtWrapper.

        Args:
            model (Union[PreTrainedModel, GenerationMixin, Any]): The language model.
                                                                   Must have a `.generate()` method.
            tokenizer (AutoTokenizer): The corresponding tokenizer.
            generation_config (Optional[GenerationConfig]): A default generation configuration.
                                                            Values here can be overridden by `generate()` call.
            device (Optional[str]): The device to load the model onto ('cpu' or 'cuda').
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
            max_length (int): The maximum total length of the input + generated sequence.
            reasoning_steps_limit (int): Conceptual limit for parsed steps (currently not enforced in _parse).
            self_consistency (bool): Flag indicating if self-consistency is intended (Informs `consistency_rounds` attribute).
            consistency_rounds (int): The number of chains to generate if self-consistency is active (Informs `consistency_rounds` attribute).
                                      The actual number generated is controlled by `num_return_sequences` in `generate()` or `generation_config`.
            complexity_keywords (Optional[List[str]]): List of keywords to potentially trigger CoT (currently unused).
            final_answer_tag (str): The specific string marker expected before the final answer.
        """
        # Determine and set the device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing wrapper on device: %s", self.device)

        # Move the model to the specified device
        try:
            self.model = model.to(self.device)
            self.model.eval() # Set model to evaluation mode for consistent behavior
            logger.info("Model moved to %s and set to eval mode.", self.device)
        except Exception as e:
            logger.error("Failed to move model to device %s: %s", self.device, e)
            raise # Re-raise the exception after logging

        self.tokenizer = tokenizer

        # Set core parameters
        self.max_length = max_length
        self.reasoning_steps_limit = reasoning_steps_limit
        self.self_consistency = self_consistency # Attribute stored, actual generation count controlled elsewhere
        self.consistency_rounds = max(1, consistency_rounds) if self_consistency else 1 # Attribute stored
        self.complexity_keywords = complexity_keywords or list(DEFAULT_COMPLEXITY_KEYWORDS) # Ensure it's a mutable list
        self.final_answer_tag = final_answer_tag
        # Compile regex pattern for final answer extraction
        self.final_answer_pattern = re.compile(
            re.escape(final_answer_tag) + r"\s*(.*)", re.IGNORECASE | re.DOTALL
        )
        logger.debug("Final answer pattern compiled: %s", self.final_answer_pattern.pattern)
        logger.debug("Step pattern: %s", DEFAULT_STEP_PATTERN.pattern)

        # Attempt to find the underlying Hugging Face model and its config
        # This is useful for accessing standard attributes like eos_token_id, etc.
        self._hf_model, self._hf_config = self._find_hf_model_and_config(self.model)

        # Fallback to tokenizer settings if HF config isn't found
        if self._hf_config is None:
            logger.warning("Underlying HF model config not found. Relying on tokenizer for eos/pad tokens and vocab size.")
            # Create a pseudo-config with essential tokenizer info
            class PseudoConfig:
                def __init__(self, tok):
                    self.eos_token_id = tok.eos_token_id
                    # Use eos_token_id as pad_token_id if pad_token_id is None (common for GPT-like models)
                    self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                    # Fallback if both are None (less common but possible)
                    if self.pad_token_id is None:
                        logger.warning("Tokenizer pad_token_id and eos_token_id are both None. Generation might be unstable without padding.")
                        # Assign a arbitrary value or handle externally if this happens in practice
                        # For now, keep it None, generation might fail or behave unexpectedly
                        pass # Keep pad_token_id as None

                    self.vocab_size = len(tok) # Vocabulary size from tokenizer

                def __getattr__(self, name):
                     # Allow accessing other attributes, returning None if not found
                     # This prevents errors if generation_config tries to read something unexpected
                    logger.debug("Accessing undefined attribute '%s' on PseudoConfig. Returning None.", name)
                    return None

            self._hf_config = PseudoConfig(self.tokenizer)
            logger.debug("Created PseudoConfig: eos_token_id=%s, pad_token_id=%s, vocab_size=%s",
                         self._hf_config.eos_token_id, self._hf_config.pad_token_id, self._hf_config.vocab_size)
        else:
            logger.info("Found underlying HF model config.")
            logger.debug("HF Config: eos_token_id=%s, pad_token_id=%s, vocab_size=%s",
                         getattr(self._hf_config, 'eos_token_id', None),
                         getattr(self._hf_config, 'pad_token_id', None),
                         getattr(self._hf_config, 'vocab_size', None))


        # --- Setup Generation Config ---
        # Start with a base config, either provided or a default one
        if generation_config:
            # Use from_dict and to_dict for safe merging/copying of GenerationConfig
            self.generation_config = GenerationConfig.from_dict(generation_config.to_dict())
            logger.info("Initialized with provided GenerationConfig.")
        else:
            # Create a default GenerationConfig using info from HF config or tokenizer fallback
            self.generation_config = GenerationConfig(
                eos_token_id=self._hf_config.eos_token_id,
                pad_token_id=self._hf_config.pad_token_id,
                max_length=self.max_length, # Set max_length from wrapper param
                # Add other common defaults if not provided
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1, # Default to 1 sequence
                no_repeat_ngram_size=0, # Default to no ngram repetition prevention
            )
            logger.info("Initialized with default GenerationConfig.")

        # Ensure the underlying HF model (if found) is set to return dict outputs from generate
        # This is necessary for accessing scores, hidden states etc. if needed, and for consistency.
        # Use a check as some custom models might not have this attribute on their config.
        if hasattr(self._hf_config, 'return_dict_in_generate'):
            try:
                setattr(self._hf_config, 'return_dict_in_generate', True)
                logger.debug("Set _hf_config.return_dict_in_generate = True.")
            except Exception as e:
                logger.warning("Failed to set return_dict_in_generate on _hf_config: %s", e)
        else:
             logger.debug("_hf_config does not have return_dict_in_generate attribute.")


        logger.info("ChainOfThoughtWrapper initialization complete on device: %s", self.device)
        logger.debug("Initial GenerationConfig: %s", self.generation_config.to_dict())


    def _find_hf_model_and_config(self, obj: Any) -> Tuple[Optional[PreTrainedModel], Optional[Any]]:
        """
        Recursively searches for an underlying Hugging Face PreTrainedModel
        and its configuration within a potentially wrapped object.

        Args:
            obj (Any): The object to inspect (could be the model itself or a wrapper).

        Returns:
            Tuple[Optional[PreTrainedModel], Optional[Any]]: The found HF model instance and its config.
                                                              Returns (None, None) if not found.
        """
        logger.debug("Searching for HF model in object of type: %s", type(obj))
        # If the object is directly a PreTrainedModel and has a config
        if isinstance(obj, PreTrainedModel) and hasattr(obj, 'config'):
            logger.debug("Found HF PreTrainedModel directly.")
            return obj, obj.config

        # Check common attribute names where the base model might be stored
        potential_attrs = ('model', 'base_model', 'transformer', 'hf_model')
        for attr_name in potential_attrs:
            m = getattr(obj, attr_name, None)
            if m is not None:
                logger.debug("Checking attribute '%s' of type %s", attr_name, type(m))
                # Recursively search within the attribute
                found_model, found_config = self._find_hf_model_and_config(m)
                if found_model or found_config:
                    return found_model, found_config

        # If no PreTrainedModel found, check if the object itself has a 'config' attribute
        if hasattr(obj, 'config'):
            logger.debug("Found config attribute on object, but no PreTrainedModel.")
            return None, obj.config

        logger.debug("No HF PreTrainedModel or config found.")
        return None, None


    def _inject_cot(self, prompt: str) -> str:
        """
        Injects the prescriptive Chain-of-Thought template into the user's prompt.

        This method defines the expected format the model should follow for reasoning.

        Args:
            prompt (str): The original user prompt.

        Returns:
            str: The prompt with the CoT template appended.
        """
        # The template strongly guides the model to produce step-by-step reasoning
        # followed by a specific tag for the final answer.
        cot_prompt = (
            f"{prompt.strip()}\n\n" # Use strip() to clean user prompt
            "Let's analyze this problem logically, breaking it down step by step to reach the precise final answer.\n\n" # Enhanced instruction
            "Reasoning Process:\n\n" # Clearer heading for steps
            "Step 1: " # Start the first step explicitly
            # More steps are not needed here, the model learns to continue the pattern
        )
        logger.debug("Injected CoT template. Full prompt starts with: %s...", cot_prompt[:100].replace('\n', '\\n'))
        return cot_prompt


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        num_return_sequences: int = 1, # This argument controls how many sequences are generated
        **kwargs: Any # Allows passing arbitrary generation parameters
    ) -> Dict[str, Any]:
        """
        Generates text using the wrapped model, enforcing Chain-of-Thought.

        This method prepares the input by injecting the CoT template, calls the
        underlying model's generate method, and then parses the raw outputs
        to extract structured reasoning steps and final answers.

        Args:
            input_ids (torch.LongTensor): Tokenized input prompt (batch size 1 expected).
                                          Shape [1, sequence_length].
            attention_mask (Optional[torch.LongTensor]): Attention mask for the input.
                                                        Shape [1, sequence_length].
            generation_config (Optional[GenerationConfig]): Specific generation config
                                                           for this call. Overrides defaults.
            num_return_sequences (int): The number of independent sequences to generate.
                                        This is crucial for Self-Consistency.
                                        Comes from the GUI's 'num_chains'.
            **kwargs (Any): Additional keyword arguments passed to the model's `generate` method.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - 'sequences' (torch.LongTensor): The raw generated token sequences.
                            - 'full_texts' (List[str]): The complete decoded text for each sequence.
                            - 'reasoning_steps' (List[List[str]]): List of parsed reasoning steps for each sequence.
                            - 'final_answers' (List[str]): List of parsed final answers for each sequence.
                            - 'consensus_answer' (Optional[str]): The consensus answer if self-consistency is active and possible (Handled by calling code).
        """
        # Ensure input is on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Decode the original prompt text for CoT injection
        # Assume batch size is 1 for the input prompt tensor [1, sequence_length]
        if input_ids.size(0) != 1:
             logger.warning("Batch size > 1 detected for input_ids (%d). CoT injection assumes batch size 1. Using the first item.", input_ids.size(0))
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # --- Inject CoT Template ---
        # This is the core step that forces the model into a reasoning mode.
        cot_prompt = self._inject_cot(prompt_text)
        logger.debug("Injected CoT prompt. Encoding...")

        # --- Prepare Generation Configuration ---
        # Merge the wrapper's default config with the call-specific config and kwargs.
        # The num_return_sequences from the function argument takes precedence here.
        cfg = GenerationConfig.from_dict(self.generation_config.to_dict()) # Start with wrapper's default
        if generation_config:
             cfg.update(**generation_config.to_dict()) # Update with call-specific config
             logger.debug("Updated GenerationConfig with call-specific config.")

        # Explicitly set num_return_sequences from the function argument
        cfg.num_return_sequences = num_return_sequences
        logger.info("Generating %d sequence(s).", cfg.num_return_sequences)

        # Update with any remaining keyword arguments passed to generate()
        for k, v in kwargs.items():
             if hasattr(cfg, k):
                  setattr(cfg, k, v)
                  logger.debug("Updating GenerationConfig kwarg: %s=%s", k, v)
             else:
                 # Allow passing arbitrary kwargs to model.generate if the underlying method supports them
                 # These won't be part of the GenerationConfig object itself unless it's a supported param.
                 # However, the model's generate method might accept extra args.
                 # Log a warning if it's not a standard GenerationConfig parameter.
                 if k not in GenerationConfig().__dict__: # Check if it's NOT a standard param
                     logger.debug("Passing non-standard kwarg '%s' to model.generate.", k)
                 # We pass all kwargs to model.generate below anyway.

        logger.debug("Final GenerationConfig for call: %s", cfg.to_dict())

        # --- Encode the CoT Prompt ---
        # Max length for input should be total max_length minus max_new_tokens
        # to leave space for the generation.
        # Ensure padding and truncation are handled.
        try:
            enc = self.tokenizer(
                cot_prompt,
                return_tensors='pt',
                padding='longest', # Pad to the longest sequence in the batch (always 1 here)
                truncation=True,   # Crucially, truncate if the prompt is too long
                max_length=self.max_length - cfg.max_new_tokens # Leave room for generation
            ).to(self.device)
            logger.debug("Encoded CoT prompt. Input shape: %s", enc['input_ids'].shape)

        except Exception as e:
            logger.error("Failed to encode CoT prompt: %s", e)
            raise # Re-raise the exception after logging


        # --- Generate Text ---
        # Call the underlying model's generate method with the prepared input and config.
        # torch.no_grad() context is already applied to the whole method.
        try:
            logger.info("Calling model.generate()...")
            start_time = time.time() # Measure generation time
            out = self.model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                generation_config=cfg,
                **kwargs # Pass through any extra kwargs
            )
            elapsed_time = time.time() - start_time
            logger.info("model.generate() finished in %.2f seconds.", elapsed_time)
            logger.debug("Raw output shape: %s", out.shape)


        except Exception as e:
            logger.error("Model generation failed: %s", e)
            # Attempt to clean up GPU memory in case of OOM or other errors
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()
            gc.collect() # Trigger Python garbage collection
            raise # Re-raise the exception after logging

        # --- Decode and Parse Outputs ---
        # Decode the generated token sequences back into text.
        logger.debug("Decoding and parsing outputs...")
        decoded_outputs = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # Process each decoded output to extract steps and final answer
        parsed_results = [self._parse(text, cot_prompt) for text in decoded_outputs]

        # Separate the parsed components into lists
        all_steps = [r[0] for r in parsed_results]
        all_finals = [r[1] for r in parsed_results]
        all_full_texts = [r[2] for r in parsed_results] # The 'body' after removing template

        logger.info("Generated and parsed %d sequence(s).", len(decoded_outputs))

        # --- Return Results ---
        # The calling code (e.g., the GUI) is responsible for implementing
        # Self-Consistency voting based on the list of 'final_answers' provided here.
        return {
            'sequences': out, # Return raw sequences in case they are needed
            'full_texts': all_full_texts, # Text body after template removal
            'reasoning_steps': all_steps,
            'final_answers': all_finals,
            # 'consensus_answer' is not computed here, it's done externally.
            # Keeping the structure consistent with GUI expectation.
            'consensus_answer': None # Placeholder, computed externally
        }


    def _parse(self, text: str, cot_prompt: str) -> Tuple[List[str], str, str]:
        """
        Parses the generated text to extract reasoning steps and the final answer.

        Applies regex patterns to find lines matching the step format and the
        final answer tag. Includes cleanup for stray model artifacts.

        Args:
            text (str): The raw text output from the model for a single chain.
            cot_prompt (str): The exact prompt text that was injected (used to remove it from the output).

        Returns:
            Tuple[List[str], str, str]: A tuple containing:
                                        - A list of extracted reasoning step strings.
                                        - The extracted final answer string.
                                        - The full body of the generated text (after removing the prompt).
        """
        logger.debug("Parsing generated text...")

        # Remove the exact injected prompt from the beginning of the text.
        # This isolates the model's generated continuation.
        body = text
        if text.startswith(cot_prompt):
            body = text[len(cot_prompt):].strip()
            logger.debug("Removed CoT prompt (%d characters) from beginning.", len(cot_prompt))
        else:
            logger.warning("Generated text does not start with the injected CoT prompt. Parsing entire text.")
            body = text.strip() # Just strip whitespace if template wasn't followed

        # --- Cleanup stray model artifacts ---
        # Remove common problematic tags or partial JSON structures that models sometimes emit.
        # This makes the raw output cleaner before step/answer extraction.
        logger.debug("Cleaning stray artifacts...")
        body = re.sub(r"<init>.*?</init>", "", body, flags=re.DOTALL)
        body = re.sub(r"<final_output>.*?</final_output>", "", body, flags=re.DOTALL)
        # Note: Removing all {} might be aggressive if model uses them naturally.
        # Keeping it as per provided code, but be aware this could remove desired output.
        # Consider making this optional or more specific if needed.
        body = re.sub(r"\{.*?\}", "", body, flags=re.DOTALL)
        logger.debug("Artifact cleanup complete.")


        lines = [l.strip() for l in body.splitlines() if l.strip()] # Split into non-empty, stripped lines
        steps = [] # List to store extracted steps
        final_answer = "" # Variable to store the final answer

        # --- Extract Steps and Final Answer ---
        # Iterate through lines and apply regex patterns.
        found_final_answer_line = False
        for i, line in enumerate(lines):
            # Check for reasoning step pattern
            step_match = DEFAULT_STEP_PATTERN.match(line)
            if step_match:
                # If a step is found, add the captured group (the text after the number/tag)
                steps.append(step_match.group(1).strip())
                logger.debug("Extracted step %d: '%s'", len(steps), steps[-1][:50])
                # Stop adding steps if we've reached a defined limit (though limit isn't currently enforced after parsing)
                # if len(steps) >= self.reasoning_steps_limit:
                #    logger.debug("Reached reasoning steps limit (%d). Stopping step extraction.", self.reasoning_steps_limit)
                #    # Continue iterating to potentially find the final answer after the limit
                #    # break # DO NOT break if we still need to find the final answer tag after the limit


            else:
                # If it's not a step, check for the final answer tag
                final_answer_match = self.final_answer_pattern.search(line)
                if final_answer_match:
                    # If the final answer tag is found, extract the text following it
                    final_answer = final_answer_match.group(1).strip()
                    logger.debug("Extracted final answer tagged: '%s'", final_answer[:50])
                    found_final_answer_line = True
                    # Once the final answer tag is found, we can stop processing lines for *this specific pattern*
                    # However, the provided code breaks the loop entirely here.
                    # Keeping the break to match the original logic.
                    break # Stop processing lines after finding the tagged answer

        # --- Fallback for Final Answer ---
        # If the specific final answer tag was not found, assume the last non-step line
        # is the intended final answer. This is a heuristic fallback.
        if not found_final_answer_line:
            logger.debug("Final answer tag not found. Applying fallback heuristic.")
            # Find the last line that is not a step
            last_non_step_line = ""
            for line in reversed(lines): # Iterate backwards
                if line.strip() and not DEFAULT_STEP_PATTERN.match(line):
                    last_non_step_line = line.strip()
                    logger.debug("Fallback: Last non-step line found: '%s'", last_non_step_line[:50])
                    break # Found the last non-step line

            if last_non_step_line:
                 # Check if the last non-step line *contains* the final answer tag,
                 # even if it didn't *start* with it or was the last line processed.
                 # This handles cases where the tag might be mid-line or in a different format.
                 fa_match_fallback = self.final_answer_pattern.search(last_non_step_line)
                 if fa_match_fallback:
                      final_answer = fa_match_fallback.group(1).strip()
                      logger.debug("Fallback found tagged answer in last non-step line: '%s'", final_answer[:50])
                 else:
                    # If no tag in the last non-step line, just use the line itself
                    final_answer = last_non_step_line
                    logger.debug("Fallback using last non-step line as answer: '%s'", final_answer[:50])
            else:
                 # If no non-empty lines were found, the final answer is empty
                 final_answer = ""
                 logger.debug("No lines found in body. Final answer is empty.")


        logger.debug("Parsing complete. Steps found: %d, Final Answer: '%s'", len(steps), final_answer[:50])

        return steps, final_answer, body # Return steps, final answer, and the cleaned body text


    def resize_token_embeddings(self, new_size: int):
        """
        Resizes the model's token embeddings, useful after adding new tokens
        to the tokenizer (like a custom PAD token).

        Only works if the underlying model object has a `resize_token_embeddings` method.

        Args:
            new_size (int): The new size of the vocabulary/embedding layer.
                            Should match the size of the tokenizer's vocabulary.
        """
        # Find the actual HF model if wrapped
        hf_model_instance, _ = self._find_hf_model_and_config(self.model)

        if hasattr(hf_model_instance, 'resize_token_embeddings'):
            try:
                old_size = hf_model_instance.get_input_embeddings().weight.size(0)
                if new_size != old_size:
                    hf_model_instance.resize_token_embeddings(new_size)
                    logger.info("Resized model token embeddings from %d to %d.", old_size, new_size)
                    # Update model config's vocab size if available
                    if hasattr(hf_model_instance, 'config') and hasattr(hf_model_instance.config, 'vocab_size'):
                        hf_model_instance.config.vocab_size = new_size
                        logger.debug("Updated model config vocab_size to %d.", new_size)
                else:
                    logger.info("Embedding size is already %d, no resizing needed.", new_size)
            except Exception as e:
                 logger.error("Failed to resize token embeddings: %s", e)
                 # Attempt cleanup
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 gc.collect()
        else:
            logger.error("Cannot resize token embeddings: The underlying model object does not have a 'resize_token_embeddings' method.")


# Example Usage (Illustrative - requires a real HF model and tokenizer)
if __name__ == "__main__":
    print("--- ChainOfThoughtWrapper Example Usage ---")
    print("This block requires a Hugging Face model to run.")
    print("Loading a small dummy model for demonstration...")

    # You would replace this with your actual model loading logic
    try:
        # Use a tiny, fast model for a quick test
        model_id = "hf-internal-testing/tiny-random-gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Attempting to load model {model_id} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Ensure pad token is set for generation (common requirement)
        if tokenizer.pad_token_id is None:
             if tokenizer.eos_token_id is not None:
                  tokenizer.pad_token_id = tokenizer.eos_token_id
             else:
                 # Add a pad token if neither eos nor pad exists
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 model.resize_token_embeddings(len(tokenizer)) # Resize embeddings after adding token
                 tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
                 logger.warning("Added and set [PAD] token, resized embeddings.")

        # Instantiate the wrapper
        # Simulate parameters that would come from the GUI
        simulated_gen_config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            num_return_sequences=2, # Simulate asking for 2 chains
            pad_token_id=tokenizer.pad_token_id, # Pass pad_token_id explicitly
            eos_token_id=tokenizer.eos_token_id, # Pass eos_token_id explicitly
        )

        cot_wrapper = ChainOfThoughtWrapper(
            model=model,
            tokenizer=tokenizer,
            generation_config=simulated_gen_config,
            device=device,
            self_consistency=True, # Simulate SC enabled
            consistency_rounds=2, # Simulate consistency rounds setting
        )

        # Prepare input prompt
        prompt_text = "What is 2 + 2? Think step-by-step."
        input_enc = tokenizer(prompt_text, return_tensors='pt').to(device)

        logger.info(f"Generating reasoning for prompt: '{prompt_text}'")

        # Generate outputs
        # The num_return_sequences from simulated_gen_config will be used here
        outputs = cot_wrapper.generate(
            input_ids=input_enc['input_ids'],
            attention_mask=input_enc['attention_mask']
        )

        # Process results (including simulated Self-Consistency voting logic)
        print("\n--- Generation Results ---")
        for i, (full_text, steps, final_answer) in enumerate(zip(outputs['full_texts'], outputs['reasoning_steps'], outputs['final_answers'])):
            print(f"\n--- Chain {i+1} ---")
            print("Full Text:")
            print(full_text)
            print("\nReasoning Steps:")
            if steps:
                for j, step in enumerate(steps):
                    print(f"  Step {j+1}: {step}")
            else:
                print("  [No steps parsed]")
            print("\nFinal Answer:")
            print(f"  {final_answer or '[No final answer parsed]'}")

        # --- Simulate Self-Consistency Voting (as would be done in GUI) ---
        print("\n--- Self-Consistency Voting ---")
        final_answers = [ans for ans in outputs['final_answers'] if ans.strip()] # Filter empty answers
        if final_answers:
            answer_counts = Counter(final_answers)
            most_common_answer, count = answer_counts.most_common(1)[0]
            print(f"Raw Answers Submitted for Voting: {final_answers}")
            print(f"Answer Counts: {dict(answer_counts)}")
            print(f"Consensus Answer: '{most_common_answer}' (Voted by {count} chain(s))")
        else:
            print("No valid final answers found for voting.")


    except Exception as e:
        logger.error("Example usage failed: %s", e)
        import traceback
        traceback.print_exc() # Print detailed traceback for the example failure

    print("\n--- Example Usage End ---")