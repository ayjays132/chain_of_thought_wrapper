
from __future__ import annotations
# chain_of_thought_wrapper.py

import os
import re
import torch
import logging
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    GenerationConfig,
    # Keep AutoModelForCausalLM for example usage block, but not used in main wrapper logic
    # We rely on AutoModel now
    # AutoModelForCausalLM, # Removed as AutoModel is more general
    # ADDED: AutoProcessor and AutoModel for multimodal handling
    AutoProcessor,
    AutoModel,
    # Import specific model classes if AutoModel isn't sufficient for a specific type
    # from transformers import LlamaForCausalLM # Example
    # from transformers import LlavaForConditionalGeneration # Example multimodal model class
)
from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Union, Any
import gc  # Import garbage collector for cleanup
from PIL import Image # Needed for handling image data
import io # Needed for handling image bytes

if TYPE_CHECKING:
    from transformers import AutoTokenizer


# ─── NEW: memory imports ─────────────────────────────────────────
# Assuming these custom classes are provided and handle text-based data
# Ensure these files (Enhanced_MemoryEngine.py, etc.) are in the same directory
try:
    from Enhanced_MemoryEngine import MemoryEngine       # 📝🧠💾✨🔍
    from NeuroMemoryProcessor import NeuroMemoryProcessor # 📝⚙️🧬🔄
    from AGIEnhancer import AGIEnhancer                   # ✍️❤️‍🩹🧠
    from FullAGI_ExpansionModule import NeoSentientCore   # 🤖💭✨
    # ADDED: Import the new Self Assessment module
    from SimulatedSelfAssessment import SimulatedSelfAssessment # 📈📊🧠

    AGI_IMPORTS_SUCCESS = True
    logger = logging.getLogger(__name__) # Re-get logger after potential basicConfig in imported modules
    logger.info("AGI helper modules imported successfully.")
except ImportError as e:
    AGI_IMPORTS_SUCCESS = False
    logger = logging.getLogger(__name__) # Re-get logger
    logger.error(f"Failed to import AGI helper modules. AGI features will be disabled: {e}")
    # Define dummy classes/objects or handle None checks later if imports fail
    class MemoryEngine: # Dummy class to prevent NameError
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None # Mock methods
    class NeuroMemoryProcessor: # Dummy class
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    class AGIEnhancer: # Dummy class
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    class NeoSentientCore: # Dummy class
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    # ADDED: Dummy class for Self Assessment if import fails
    class SimulatedSelfAssessment: # Dummy class
         def __init__(self, *args, **kwargs): pass
         def __getattr__(self, name): return lambda *args, **kwargs: {"state_summary": "Simulated self-assessment module not available."} # Mock method returning default summary


# --- Logging Setup for Wrapper ---
# Configure logging only when no handlers exist on the root logger or when the
# COT_DEBUG environment variable is set. This prevents noisy output during tests
DEBUG_MODE = bool(os.getenv("COT_DEBUG"))
if DEBUG_MODE or not logging.root.handlers:
    level = logging.DEBUG if DEBUG_MODE else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
if not logger.handlers and (DEBUG_MODE or not logging.root.handlers):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
elif logger.handlers:
    logger.setLevel(logger.handlers[0].level)
else:
    logger.addHandler(logging.NullHandler())


# --- Default Configuration Values ---
# These defaults provide sensible starting points for the wrapper's behavior,
# based on common practices and the audit recommendations.
DEFAULT_MAX_LENGTH = 2048 # Increased default max length to accommodate longer CoT
DEFAULT_REASONING_LIMIT = 15 # A conceptual limit for extracted steps (not strictly enforced by parsing logic)
DEFAULT_CONSISTENCY_ROUNDS = 5 # Default number of chains for self-consistency, increased based on typical research
# DEFAULT_COMPLEXITY_KEYWORDS = ["explain", "step by step", "plan", "analyze", "reasoning", "logic"] # Keywords (currently unused as CoT is always on)
DEFAULT_FINAL_ANSWER_TAG = "Final Answer:" # Explicit tag to signal the final answer, reverted to a more common default


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


# --- Self-Consistency Voting (Defined here, but used by the GUI) ---
# Keep the normalize_answer function here as it's a utility
def normalize_answer(answer: str) -> str:
    """
    Normalizes a string answer for robust comparison during voting.
    - Converts to lowercase.
    - Strips leading/trailing whitespace.
    - Removes common punctuation and articles.
    - Handles simple cases of number words (e.g., "two" -> "2").
    - Removes extra internal whitespace.
    """
    if not isinstance(answer, str):
        return "" # Handle non-string inputs

    normalized = answer.lower().strip()

    # Remove common trailing characters like periods, commas, etc.
    normalized = re.sub(r'[.,!?;:]+$', '', normalized).strip()

    # Remove common leading preambles (case-insensitive)
    normalized = re.sub(r'^\s*(?:the answer is|result|output)\s*[:\-]?\s*', '', normalized, flags=re.IGNORECASE).strip()

    # Remove common articles (a, an, the) only if they appear at the start of the answer
    normalized = re.sub(r'^\s*(a|an|the)\s+', '', normalized, flags=re.IGNORECASE).strip()

    # Basic number word to digit conversion for common cases (can be expanded)
    num_word_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
        'million': '1000000', 'billion': '1000000000'
    }
    # Simple word replacement - might fail on "twenty-two" or "one hundred".
    # More robust parsing is complex.
    words = normalized.split()
    normalized_words = [num_word_map.get(word, word) for word in words]
    normalized = " ".join(normalized_words)


    # Remove extra whitespace within the string (replace multiple spaces with single)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Remove trailing spaces from the very end again just in case
    normalized = normalized.strip()


    return normalized


def validate_device_selection(selected_device: str) -> str:
    """Validate CUDA device selection and fall back to CPU when invalid."""
    if isinstance(selected_device, str) and selected_device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            return "cpu"

        # Default to device index 0 when none is specified
        index = 0
        if ":" in selected_device:
            try:
                index = int(selected_device.split(":")[-1])
            except ValueError:
                logger.warning(
                    f"Invalid CUDA device specification '{selected_device}'. Falling back to CPU."
                )
                return "cpu"

        if index < 0 or index >= torch.cuda.device_count():
            logger.warning(
                f"Selected CUDA device index {index} is out of range (Max index: {torch.cuda.device_count() - 1}). Falling back to CPU."
            )
            return "cpu"

        # Return the canonical device string
        return f"cuda:{index}"

    return selected_device

# NOTE: This voting function is for the EXAMPLE USAGE BLOCK only and is NOT
# directly used by the ChainOfThoughtWrapper.generate method.
# It's included here for completeness if the user wanted to test the wrapper
# standalone, but the GUI implements its own voting logic using normalize_answer.
# Removed this function as it's explicitly not used by the wrapper itself and the GUI has its own.
# def perform_self_consistency_voting(...)


# --- ChainOfThoughtWrapper Class (Multimodal Enabled) ---
class ChainOfThoughtWrapper:
    """
    ChainOfThoughtWrapper: Orchestrates model generation with CoT prompting
    and interacts with AGI helper modules.

    Supports multimodal input (image + text) for compatible models
    loaded with Hugging Face's AutoModel and AutoProcessor.
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, GenerationMixin, AutoModel, Any], # Accept AutoModel
        processor: Union[AutoTokenizer, AutoProcessor, Any], # Accept AutoProcessor (can be AutoTokenizer)
        device: Union[str, torch.device],
        # cot_template is less critical now as multimodal models often use specific chat templates
        # We'll keep a basic CoT prompt component but rely on processor for formatting
        cot_instruction: str = "Analyze this step by step to find the answer.",
        reasoning_header: str = "Reasoning:",
        step_prefix: str = "Step", # e.g., "Step 1: " - model will ideally continue this
        final_answer_tag: str = DEFAULT_FINAL_ANSWER_TAG, # Explicit tag to signal the final answer
        max_length: int = DEFAULT_MAX_LENGTH # Max length for tokenization (input + output)
    ):
        """
        Initializes the ChainOfThoughtWrapper.

        Args:
            model (Union[PreTrainedModel, GenerationMixin, AutoModel, Any]): The loaded Hugging Face model.
            processor (Union[AutoTokenizer, AutoProcessor, Any]): The loaded Hugging Face processor
                                                                  (tokenizer or multimodal processor).
            device (Union[str, torch.device]): The device the model is on.
            cot_instruction (str): The core instruction phrase for CoT.
            reasoning_header (str): The header text before the reasoning steps.
            step_prefix (str): The prefix for the first step.
            final_answer_tag (str): The specific string marker expected before the final answer.
            max_length (int): The maximum combined length of input prompt and generated tokens.
        """
        logger.debug("ChainOfThoughtWrapper __init__ started.")
        self.model = model
        self.processor = processor  # Store the processor (can be AutoProcessor or AutoTokenizer)

        # Validate the requested device string when provided
        if isinstance(device, str):
            self.device = validate_device_selection(device)
        else:
            self.device = device
        self.cot_instruction = cot_instruction
        self.reasoning_header = reasoning_header
        self.step_prefix = step_prefix
        self.final_answer_tag = final_answer_tag
        self.max_length = max_length
        # If the loaded model defines a maximum positional embedding length,
        # ensure the wrapper does not allow inputs longer than that limit.
        if hasattr(self.model, "config"):
            model_max_len = getattr(self.model.config, "max_position_embeddings", None)
            if model_max_len is None:
                model_max_len = getattr(self.model.config, "n_positions", None)
            if model_max_len is not None and self.max_length > model_max_len:
                logger.warning(
                    "Specified max_length %s exceeds model's max_position_embeddings (%s). Reducing to model limit.",
                    self.max_length,
                    model_max_len,
                )
                self.max_length = model_max_len
        self._artifact_patterns = ARTIFACT_PATTERNS # Use default artifact patterns
        self.reasoning_steps_limit = DEFAULT_REASONING_LIMIT # Use default limit for parsing

        # Determine if the loaded processor has an image processor component -> Multimodal capability flag
        # This is how we check if the loaded model/processor pair is multimodal capable for input
        self.multimodal_capable = hasattr(self.processor, 'image_processor') and self.processor.image_processor is not None
        logger.info(f"Wrapper initialized on {self.device}. Multimodal capability detected: {self.multimodal_capable}")

        # Ensure we have a tokenizer, whether the processor is multimodal or text-only
        # If processor IS the tokenizer, getattr will return the processor itself.
        # CORRECTED: Use getattr to get the tokenizer from the processor
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)

        if self.tokenizer is None:
             logger.error("Processor does not contain a tokenizer.")
             # Depending on model, this might be fatal. Proceed, but expect errors during tokenization/decoding.

        # Handle models/tokenizers without a defined pad_token_id for batch generation
        # Only attempt this if a tokenizer was found
        if self.tokenizer and self.tokenizer.pad_token_id is None:
             if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                  self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                  logger.warning("Tokenizer pad_token_id is None, using eos_token_id (%s) as pad_token_id for batching.", self.tokenizer.eos_token_id)
             else:
                 # Fallback: Add a new pad token if neither exists
                 logger.warning("Tokenizer pad_token_id and eos_token_id are both None. Attempting to add a [PAD] token.")
                 try:
                     # Check if the token already exists before adding
                     if hasattr(self.tokenizer, 'vocab') and '[PAD]' not in self.tokenizer.vocab:
                         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                         # Note: Resizing embeddings should ideally happen on the model *after* adding the token.
                         # The GUI's loading function attempts this, but log if it's needed and might not happen here.
                         logger.warning("Added new [PAD] token to tokenizer. Model embeddings may need resizing.")
                     elif not hasattr(self.tokenizer, 'vocab'):
                          logger.warning("Tokenizer does not have a vocabulary attribute. Cannot check for or add [PAD] token.")
                     else:
                         logger.info("[PAD] token already exists in tokenizer vocabulary.")

                     # After potentially adding the token, set pad_token_id if it's still None
                     if self.tokenizer.pad_token_id is None and hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                          self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
                          logger.info("Set pad_token_id to ID of [PAD] token (%s).", self.tokenizer.pad_token_id)
                     elif self.tokenizer.pad_token_id is None:
                          logger.warning("Cannot set pad_token_id as convert_tokens_to_ids method is missing.")


                 except Exception as e:
                     logger.error(f"Failed to add [PAD] token or set pad_token_id: {e}")
                     self.tokenizer.pad_token_id = None # Ensure it stays None if adding fails
                     logger.warning("Failed to set pad_token_id. Batch generation might fail.")
        elif self.tokenizer:
             logger.debug("Tokenizer has pad_token_id: %s", self.tokenizer.pad_token_id)
        else:
             logger.warning("No tokenizer available to check or set pad_token_id.")


        # Compile regex pattern for final answer extraction based on the specified tag.
        # re.escape handles potential special characters in the tag. re.DOTALL matches newline.
        self.final_answer_pattern = re.compile(
            re.escape(final_answer_tag) + r"\s*(.*)", re.IGNORECASE | re.DOTALL
        )
        self._step_pattern = DEFAULT_STEP_PATTERN # Use the default compiled step pattern

        logger.debug("Final answer pattern compiled: %s", self.final_answer_pattern.pattern)
        logger.debug("Step pattern: %s", self._step_pattern.pattern)


        # --- Initialize AGI Helper Modules ---
        # Instantiate your AGI components here, only if imports were successful
        self.memory_engine = None
        self.neuro_processor = None
        self.agi_enhancer = None
        self.neo_sentient_core = None
        # ADDED: Initialize the Self Assessment module
        self.self_assessment_module = None # Initialize the attribute

        if AGI_IMPORTS_SUCCESS:
            try:
                self.memory_engine = MemoryEngine()
                logger.info("MemoryEngine initialized.")
            except Exception as e:
                 self.memory_engine = None
                 logger.error(f"Failed to initialize MemoryEngine: {e}")

            try:
                self.neuro_processor = NeuroMemoryProcessor()
                logger.info("NeuroMemoryProcessor initialized.")
            except Exception as e:
                 self.neuro_processor = None
                 logger.error(f"Failed to initialize NeuroMemoryProcessor: {e}")

            try:
                self.agi_enhancer = AGIEnhancer()
                logger.info("AGIEnhancer initialized.")
            except Exception as e:
                 self.agi_enhancer = None
                 logger.error(f"Failed to initialize AGIEnhancer: {e}")

            try:
                self.neo_sentient_core = NeoSentientCore(name="NeoAGI")
                logger.info("NeoSentientCore initialized.")
            except Exception as e:
                 self.neo_sentient_core = None
                 logger.error(f"Failed to initialize NeoSentientCore: {e}")

            # ADDED: Initialize the Self Assessment module instance
            try:
                 self.self_assessment_module = SimulatedSelfAssessment()
                 logger.info("SimulatedSelfAssessment initialized.")
            except Exception as e:
                 self.self_assessment_module = None
                 logger.error(f"Failed to initialize SimulatedSelfAssessment: {e}")

        else:
             logger.warning("AGI helper modules were not imported, AGI features will not be available.")


        logger.debug("ChainOfThoughtWrapper __init__ finished.")


    def _build_agi_preamble(
        self,
        input_text: str,
        image_data: Optional[List[bytes]] = None,
    ) -> Tuple[str, str]:
        """Gather context from AGI helper modules for the generation prompt."""
        agi_pre_prompt_elements: List[str] = []
        self_assessment_summary_text: Optional[str] = None

        if AGI_IMPORTS_SUCCESS and self.neo_sentient_core:
            perception_detail = (
                f"User input: '{input_text[:200]}{'...' if len(input_text) > 200 else ''}'"
            )
            if image_data:
                perception_detail += f" (with {len(image_data)} image(s))"
            try:
                self.neo_sentient_core.perceive(perception_detail)
                logger.debug("NeoSentientCore perceived input.")
            except Exception as e:
                logger.warning(f"NeoSentientCore perceive failed: {e}")

            try:
                current_goal = self.neo_sentient_core.decide_goal()
                if current_goal and isinstance(current_goal, str):
                    agi_pre_prompt_elements.append(f"Intention: {current_goal.strip()}")
            except Exception as e:
                logger.warning(f"NeoSentientCore decide_goal failed: {e}")

            try:
                inner_monologue = self.neo_sentient_core.inner_voice()
                if inner_monologue and isinstance(inner_monologue, str):
                    agi_pre_prompt_elements.append(f"InnerVoice: {inner_monologue.strip()}")
            except Exception as e:
                logger.warning(f"NeoSentientCore inner_voice failed: {e}")

            try:
                qualia_token = self.neo_sentient_core.generate_qualia_token("curiosity")
                if qualia_token and isinstance(qualia_token, str):
                    agi_pre_prompt_elements.insert(0, qualia_token.strip())
            except Exception as e:
                logger.warning(f"NeoSentientCore generate_qualia_token failed: {e}")

        if AGI_IMPORTS_SUCCESS and self.agi_enhancer:
            enhancer_experience_detail = (
                f"User input: '{input_text[:200]}{'...' if len(input_text) > 200 else ''}'"
            )
            if image_data:
                enhancer_experience_detail += f" (with {len(image_data)} image(s))"
            try:
                self.agi_enhancer.log_experience(enhancer_experience_detail)
                logger.debug("AGIEnhancer logged experience.")
            except Exception as e:
                logger.warning(f"AGIEnhancer log_experience failed: {e}")

        if (
            AGI_IMPORTS_SUCCESS
            and self.self_assessment_module
            and self.memory_engine
            and self.neuro_processor
            and self.neo_sentient_core
        ):
            try:
                recent_reflections_snapshot = self.memory_engine.recall(
                    include_long_term=True, include_working=True, limit=5
                )
                top_biases_snapshot = self.neuro_processor.recall_biases(top_k=10)
                synaptic_weights_snapshot = self.neuro_processor.recall_weights(top_k=10)
                neo_state_snapshot = self.neo_sentient_core.get_state()
                current_emotions_snapshot = neo_state_snapshot.get("emotions", {})
                intent_pool_snapshot = neo_state_snapshot.get("intent_pool", [])
                qri_snapshot_data = None

                assessment_result = self.self_assessment_module.perform_assessment(
                    recent_reflections=recent_reflections_snapshot,
                    top_biases=top_biases_snapshot,
                    synaptic_weights_snapshot=synaptic_weights_snapshot,
                    current_emotions=current_emotions_snapshot,
                    intent_pool=intent_pool_snapshot,
                    trace_summary=self.memory_engine.get_trace()[-10:]
                    if self.memory_engine and len(self.memory_engine.get_trace()) > 0
                    else [],
                    qri_snapshot=qri_snapshot_data,
                )
                self_assessment_summary_text = assessment_result.get("state_summary", None)
                logger.debug(
                    "Performed simulated self-assessment and retrieved summary for prompt."
                )
            except Exception as e:
                logger.error(f"Failed to perform simulated self-assessment: {e}")
                self_assessment_summary_text = (
                    "\n--- Simulated Self-Assessment Error ---\n"
                    "Internal assessment module encountered an issue and cannot provide a state summary.\n---\n"
                )

        agi_pre_prompt = "\n".join(agi_pre_prompt_elements) + "\n\n" if agi_pre_prompt_elements else ""
        self_assessment_prompt_part = (
            self_assessment_summary_text + "\n\n" if self_assessment_summary_text else ""
        )

        return agi_pre_prompt, self_assessment_prompt_part


    @torch.no_grad() # Ensure no gradients are calculated during inference
    def generate(
        self,
        input_text: str,
        image_data: Optional[List[bytes]] = None, # Accept list of image bytes
        multimodal_model: bool = False,
        generation_params: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[Optional[List[Dict[str, str]]], Optional[str], Optional[str]]:
        """
        Generates a Chain-of-Thought response from the language model, optionally
        handling multimodal input (text + image). Integrates AGI helper modules
        (MemoryEngine, NeuroProcessor, AGIEnhancer, NeoSentientCore, SelfAssessment)
        and includes conversation history in the prompt.

        Args:
            prompt (str): The user's input prompt (text part).
            image (Optional[Image.Image]): The input image, if any.
            multimodal_model (bool): True if the loaded model is multimodal.
            generation_params (Optional[Dict[str, Any]]): Dictionary of generation parameters
            chat_history (Optional[List[Dict[str, str]]]): A list of dictionaries
                          representing previous turns of the conversation. Each dict
                          is expected to have keys 'role' ('user' or 'assistant')
                          and 'content' (the message text).

        Returns:
            Tuple[Optional[List[Dict[str, str]]], Optional[str], Optional[str]]:
            A tuple containing:
            1. List of dictionaries representing the parsed CoT steps (or None).
            2. The extracted final answer string (or None).
            3. The raw body text of the model's response (or None).
        """
        logger.debug("Wrapper generate method called.")
        # Added check for model generation compatibility at the start of generate
        if self.model is None or self.processor is None or self.tokenizer is None or \
           not (hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate', None)) or isinstance(self.model, GenerationMixin)):
            logger.error("Model, Processor, Tokenizer not loaded or loaded model is not generation compatible.")
            # Return an empty result dict to indicate failure, GUI will handle displaying error
            return {"full_texts": [], "reasoning_steps": [], "final_answers": [], "generated_images": [], "generation_scores": None}


        # Safely get generation parameters
        params = generation_params if generation_params is not None else {}
        effective_num_return_sequences = params.get("num_return_sequences", 1)
        # Use default values if not provided in params


        logger.info(f"Generating {effective_num_return_sequences} sequence(s) with params: {params}")
        if image_data:
             logger.info(f"Received {len(image_data)} image(s). Wrapper multimodal capable: {self.multimodal_capable}")


        # --- Collect AGI Context (Pre-Generation) ---
        agi_pre_prompt, self_assessment_prompt_part = self._build_agi_preamble(
            input_text, image_data
        )


        # Construct the core CoT prompt string for the text part of the input
        # Include instructions, reasoning header, and step prefix to guide the model
        cot_instruction_text = (
             f"{self.cot_instruction}\n\n"
             # Optional: Add an instruction to the model about using the assessment summary
             "Based on the provided 'Simulated Internal State Assessment', incorporate insights about your perceived internal state, coherence, and well-being into your response and reasoning process.\n\n"
        )


        cot_prompt_core_text = (
            cot_instruction_text +
            f"{self.reasoning_header}\n\n"
            f"{self.step_prefix} 1: " # Explicitly start the first step
        )

        # Combine AGI pre-prompt, Self-Assessment summary, and the core CoT text prompt
        history_prompt_part = ""
        if chat_history:
            logger.debug(f"Including {len(chat_history)} turns in conversation history prompt part.")
            formatted_history_lines = []
            for turn in chat_history:
                role = turn.get('role', 'unknown').capitalize()

                raw_content = turn.get('content', '')
                if isinstance(raw_content, str):
                    content = raw_content.strip()
                else:
                    content = str(raw_content).strip()

                if role and content:
                    formatted_history_lines.append(f"{role}: {content}")
            # Join history lines with a separator, add a final separator
            history_prompt_part = "\n".join(formatted_history_lines) + "\n\n---\n\n" if formatted_history_lines else ""
            logger.debug(f"Formatted history prompt part:\n{history_prompt_part[:500]}...") # Log snippet


        # Combine history, AGI pre-prompt, Self-Assessment summary, and the core CoT text prompt
        # ADDED: Prepend history_prompt_part
        full_text_prompt = history_prompt_part + agi_pre_prompt + self_assessment_prompt_part + cot_prompt_core_text


        # --- Prepare Multimodal Input ---
        input_tensors = {} # Dictionary to hold input tensors

        try:
            # Use the processor to handle both text and image inputs
            # This is the core change for multimodal input processing
            # Multimodal models often require a specific format for messages (e.g., interleaved text/image)
            # We'll create a simple message structure for the processor: [image(s)], text prompt
            messages = []
            if image_data and self.multimodal_capable:
                 for img_bytes in image_data:
                      try:
                           img = Image.open(io.BytesIO(img_bytes))
                           messages.append({"type": "image", "content": img}) # Use PIL Image object
                      except Exception as e:
                           logger.warning(f"Could not open image from bytes for processing: {e}. Skipping this image.")
                           # Decide if you want to continue without the image or raise an error
                           # For robustness, we'll just skip this image and log a warning

            # Append the text part of the prompt as a text message
            # It's often beneficial to include the user's original text input as part of the prompt
            # for the model to explicitly reference.
            # Let's use a simple structure: User Query + [Image(s)] + CoT Guiding text

            # Revised message structure for processor:
            processor_messages = []
            # Add user's original input text first
            if input_text and input_text.strip():
                 processor_messages.append({"type": "text", "content": f"User Input: {input_text.strip()}"})

            # Add image messages *after* the initial text input if images are available and wrapper is multimodal
            if image_data and self.multimodal_capable and messages: # Check if images were successfully loaded into `messages` list
                 processor_messages.extend(messages)
                 logger.debug(f"Prepared {len(messages)} image messages for processor.")
            elif image_data and not self.multimodal_capable:
                 logger.warning("Image data provided but wrapper/model is text-only. Images will be ignored by the processor.")

            # Add the core CoT guiding text (AGI + template) as the final text message
            # This guides the *output* format regardless of input modality
            if full_text_prompt.strip():
                 processor_messages.append({"type": "text", "content": full_text_prompt.strip()})
            elif not processor_messages: # If no text input, no images, and no CoT prompt text, add a default
                 logger.warning("No text or image content in messages. Adding a default text message.")
                 processor_messages.append({"type": "text", "content": "Please provide input."})
                 # Note: An empty prompt might cause issues for some models. This is a safeguard.


            # Log the structured messages for debugging
            logger.debug(f"Messages prepared for processor: {processor_messages}")


            # Use the processor to handle input, adapting based on chat template availability
            tokenizer_for_template = getattr(self.processor, 'tokenizer', None) # Access tokenizer via processor
            has_chat_template = tokenizer_for_template and hasattr(tokenizer_for_template, 'apply_chat_template') and tokenizer_for_template.chat_template

            if hasattr(self.processor, '__call__') and has_chat_template:
                 # Scenario 1: Processor is callable AND has a chat template
                 logger.debug("Processor is callable and has a chat template. Using processor's chat template to format messages.")
                 # apply_chat_template returns a string, so we then tokenize this string
                 # Use add_generation_prompt=True to ensure the template is completed for the model to generate
                 chat_prompt_text = tokenizer_for_template.apply_chat_template(processor_messages, tokenize=False, add_generation_prompt=True)
                 logger.debug(f"Chat template applied. Resulting text prompt: {chat_prompt_text[:200]}...")

                 # Now tokenize the formatted text prompt
                 inputs = self.tokenizer( # Use the stored tokenizer from __init__
                      chat_prompt_text,
                      return_tensors="pt",
                      padding="longest",
                      truncation=True,
                      max_length=self.max_length,
                 ).to(self.device)

                 # Need to also process images separately if using chat template, as apply_chat_template is text-only
                 if image_data and self.multimodal_capable and messages: # Check if images were successfully loaded into `messages` list
                      image_processor_component = getattr(self.processor, 'image_processor', None)
                      if image_processor_component:
                           try:
                                # Extract PIL Images from the 'messages' list
                                pil_images = [msg["content"] for msg in messages if msg["type"] == "image" and isinstance(msg["content"], Image.Image)]
                                if pil_images:
                                     image_inputs = image_processor_component(
                                          pil_images, # Process list of images
                                          return_tensors="pt"
                                     ).to(self.device)
                                     # Merge image inputs (pixel_values) with text inputs (input_ids, attention_mask)
                                     inputs.update(image_inputs)
                                     logger.debug(f"Image inputs processed separately and merged for chat template case. Keys now: {inputs.keys()}")
                                else:
                                     logger.warning("No valid PIL images found in messages despite image_data for chat template case. Skipping image processing.")

                           except Exception as image_process_e:
                                logger.error(f"Failed to process image inputs separately for chat template case: {image_process_e}. Generation might fail.")
                                # Continue with text inputs only, but log error
                      else:
                           logger.warning("Processor's image_processor component is missing despite multimodal capability flag for chat template case. Cannot process images.")


            elif hasattr(self.processor, '__call__'):
                 # Scenario 2: Processor is callable but NO chat template.
                 # Attempt to pass concatenated text and separate image inputs to processor.__call__
                 logger.debug("Processor is callable but no chat template. Concatenating text messages and processing images separately.")

                 # Concatenate text content from all text messages
                 concatenated_text_input = "\n".join([msg["content"] for msg in processor_messages if msg["type"] == "text"])

                 if not concatenated_text_input.strip() and any(msg["type"] == "image" for msg in processor_messages):
                      # Handle case where there's only image input but no text input.
                      # Some multimodal models might still need a minimal text input like "".
                      logger.warning("No text content in messages, but images are present. Passing empty string as text input.")
                      concatenated_text_input = ""
                 elif not concatenated_text_input.strip():
                      # Handle case with no text and no images
                      logger.warning("No text or image content in messages. Passing empty string as text input.")
                      concatenated_text_input = ""

                 # Duplicate the concatenated text string for batching
                 text_input_for_processor = [concatenated_text_input] * effective_num_return_sequences
                 logger.debug(f"Concatenated text input for processor: '{concatenated_text_input[:200]}...' (duplicated {effective_num_return_sequences} times for batching)")

                 # Process images separately if images are present
                 image_inputs = {} # Initialize empty image inputs
                 if image_data and self.multimodal_capable and messages: # Check if images were successfully loaded into `messages` list
                      image_processor_component = getattr(self.processor, 'image_processor', None)
                      if image_processor_component:
                           try:
                                # Extract PIL Images from the 'messages' list
                                pil_images = [msg["content"] for msg in messages if msg["type"] == "image" and isinstance(msg["content"], Image.Image)]
                                if pil_images:
                                     # Process images once and add them.
                                     # Note: For batching num_return_sequences > 1, the model's generate method
                                     # is usually expected to handle the batching dimension for image inputs
                                     # if the image processor outputs batched tensors. If this causes errors,
                                     # model-specific handling might be needed here.
                                     image_inputs = image_processor_component(
                                          pil_images, # Process list of images
                                          return_tensors="pt"
                                     ).to(self.device)
                                     logger.debug(f"Image inputs processed separately for callable processor without chat template. Keys now: {image_inputs.keys()}")

                                else:
                                     logger.warning("No valid PIL images found in messages despite image_data for callable processor without chat template. Skipping image processing.")

                           except Exception as image_process_e:
                                logger.error(f"Failed to process image inputs separately for callable processor without chat template: {image_process_e}. Generation might fail.")
                                # Continue with text inputs only, but log error
                      else:
                           logger.warning("Processor's image_processor component is missing despite multimodal capability flag for callable processor without chat template. Cannot process images.")


                 # Pass the concatenated text (as a list for batching) and image inputs (if any)
                 # to the processor's __call__ method.
                 # Assuming the processor.__call__ signature handles this pattern.
                 inputs = self.processor(
                     text=text_input_for_processor, # Pass list of strings for batching
                     **image_inputs, # Unpack image inputs (e.g., pixel_values)
                     return_tensors="pt",
                     padding="longest",
                     truncation=True,
                     max_length=self.max_length,
                 ).to(self.device)
                 logger.debug("Input processed using processor.__call__ with concatenated text and separate image inputs.")


            elif hasattr(self.processor, 'tokenizer'): # Fallback for text-only models loaded with AutoTokenizer
                 # Scenario 3: Processor is NOT callable, but HAS a tokenizer (text-only model)
                 logger.debug("Processor is text-only (using tokenizer). Processing text input only.")
                 # Use the stored tokenizer from __init__ to process only the combined text prompt
                 # Combine user input and CoT guiding text for text-only models
                 # Let's use a simple format: User Input + CoT Template Text
                 combined_text_for_tokenizer = f"User Input: {input_text.strip()}\n\n{full_text_prompt.strip()}"

                 inputs = self.tokenizer(
                     combined_text_for_tokenizer,
                     return_tensors="pt",
                     padding="longest",
                     truncation=True,
                     max_length=self.max_length,
                 ).to(self.device)
                 logger.debug("Input processed using tokenizer directly.")


            else:
                 # Safeguard: Should not happen if tokenizer check passes, but as a safeguard
                 raise TypeError("Loaded processor is neither callable nor contains a tokenizer attribute.")

            # ... (rest of input preparation block) ...
            # Prepare the input tensors dictionary for the model's generate method
            input_tensors = inputs # 'inputs' is already a dictionary or object acting like one

            # Log the keys present in the input_tensors for debugging
            logger.debug("Input tensors prepared for model.generate. Keys: %s", list(input_tensors.keys()))
            if 'input_ids' in input_tensors:
                 logger.debug("Input IDs shape: %s, dtype: %s, on device: %s", input_tensors['input_ids'].shape, input_tensors['input_ids'].dtype, input_tensors['input_ids'].device)
            if 'pixel_values' in input_tensors:
                 logger.debug("Pixel values shape: %s, dtype: %s, on device: %s", input_tensors['pixel_values'].shape, input_tensors['pixel_values'].dtype, input_tensors['pixel_values'].device)


        except Exception as e:
            logger.error("Failed to prepare input tensors (tokenization/image processing): %s", e)
            # Attempt cleanup before raising
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as cleanup_e:
                    logger.warning(f"Error during cuda empty_cache: {cleanup_e}")
            gc.collect()
            # Do not re-raise here, return empty lists and let the GUI handle the error
            return {"full_texts": [], "reasoning_steps": [], "final_answers": [], "generated_images": [], "generation_scores": None}


        # --- Generate Response ---
        generated_outputs = None
        try:
            # Build the final GenerationConfig for this specific call
            # Start with a default, then update with provided params
            # Ensure pad_token_id and eos_token_id are set from the tokenizer
            cfg = GenerationConfig() # Start with an empty config
            if self.tokenizer:
                 # Safely get pad_token_id and eos_token_id, defaulting to None if not found
                 cfg.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                 cfg.eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            else:
                 logger.warning("Tokenizer not available, GenerationConfig may lack pad/eos tokens.")

            # Update config with parameters from the GUI/caller
            if params:
                # Remove 'self_consistency_enabled' and 'requested_chains' as they are not GenerationConfig parameters
                params_for_gen_config = {k: v for k, v in params.items() if k not in ['self_consistency_enabled', 'requested_chains', 'pad_token_id', 'eos_token_id']}
                cfg.update(**params_for_gen_config)
                logger.debug("Merged generation_params into GenerationConfig.")


            # Ensure required parameters for batch generation are set
            cfg.num_return_sequences = effective_num_return_sequences
            if cfg.num_return_sequences > 1 and not cfg.do_sample:
                 logger.warning("num_return_sequences > 1 but do_sample is False. Generated sequences will be identical.")
            if cfg.do_sample and cfg.temperature == 0:
                 logger.warning("do_sample is True but temperature is 0. Generation will be deterministic.")


            # Ensure max_length or max_new_tokens is handled correctly
            # Use max_new_tokens from params if available, otherwise calculate from max_length
            # Safely get input_length, defaulting to 0 if input_ids is missing or empty
            input_ids_tensor = input_tensors.get('input_ids', torch.tensor([[]]))
            input_length = input_ids_tensor.shape[-1] if input_ids_tensor.numel() > 0 else 0

            # Prioritize max_new_tokens from input params if provided, otherwise use max_length
            if 'max_new_tokens' in params:
                 cfg.max_new_tokens = params['max_new_tokens']
                 effective_total_length = input_length + cfg.max_new_tokens if input_length + cfg.max_new_tokens > 0 else 0
                 if effective_total_length > self.max_length:
                     logger.warning(
                         "Effective total length (%d) exceeds wrapper max_length (%d). Adjusting max_new_tokens.",
                         effective_total_length,
                         self.max_length,
                     )
                     cfg.max_new_tokens = max(0, self.max_length - input_length)
                     effective_total_length = input_length + cfg.max_new_tokens
                     logger.warning("Adjusted max_new_tokens to %d.", cfg.max_new_tokens)
                 if cfg.max_length is None or effective_total_length > cfg.max_length:
                     cfg.max_length = effective_total_length if effective_total_length > 0 else None
                 logger.debug(
                     "Using max_new_tokens from params: %s. Calculated total max_length: %s",
                     cfg.max_new_tokens,
                     cfg.max_length,
                 )

            elif cfg.max_new_tokens is None:
                 # If max_new_tokens is NOT set in params or default cfg, ensure the total length
                 # does not exceed the wrapper's max_length limit. Use wrapper's default max_length.
                 cfg.max_length = min(self.max_length, cfg.max_length if cfg.max_length is not None else self.max_length)
                 # If max_length is set this way, max_new_tokens should effectively be the difference
                 cfg.max_new_tokens = max(0, cfg.max_length - input_length) # Ensure it's not negative
                 logger.debug("max_new_tokens not set in params or default cfg. Using wrapper max_length: %s. Calculated max_new_tokens: %s", cfg.max_length, cfg.max_new_tokens)
            else:
                 # If max_new_tokens was set in default cfg but not params, validate against wrapper's max_length
                 effective_total_length = input_length + cfg.max_new_tokens
                 if effective_total_length > self.max_length:
                     logger.warning("Effective total length (%d) exceeds wrapper max_length (%d). Adjusting max_new_tokens.", effective_total_length, self.max_length)
                     cfg.max_new_tokens = max(0, self.max_length - input_length)
                     cfg.max_length = input_length + cfg.max_new_tokens if input_length + cfg.max_new_tokens > 0 else None
                     logger.warning("Adjusted max_new_tokens to %d.", cfg.max_new_tokens)
                 else:
                     # If max_new_tokens was set and is within limits, ensure cfg.max_length is also set correctly
                     cfg.max_length = input_length + cfg.max_new_tokens if input_length + cfg.max_new_tokens > 0 else None
                     logger.debug("Using max_new_tokens from default cfg: %s. Calculated total max_length: %s", cfg.max_new_tokens, cfg.max_length)


            # Ensure max_length is not None unless input_length + max_new_tokens is 0 or less
            if cfg.max_length is None and (input_length + (cfg.max_new_tokens if cfg.max_new_tokens is not None else 0)) > 0:
                 calculated_max_length = input_length + (cfg.max_new_tokens if cfg.max_new_tokens is not None else 0)
                 if calculated_max_length > 0:
                     cfg.max_length = calculated_max_length
                 else:
                      cfg.max_length = None # If calculation somehow results in <= 0


            # Final check: if max_new_tokens became 0 or less, maybe generation isn't possible?
            if cfg.max_new_tokens is not None and cfg.max_new_tokens <= 0:
                 logger.warning("Calculated max_new_tokens is 0 or less. Generation might return only prompt.")
                 # Set max_new_tokens to a small value like 1 to attempt at least one new token if possible
                 if input_length < self.max_length and self.max_length > 0:
                      cfg.max_new_tokens = 1
                      # Re-calculate max_length to reflect the adjusted max_new_tokens
                      cfg.max_length = input_length + cfg.max_new_tokens
                      logger.warning("Setting max_new_tokens to 1 to attempt minimal generation.")
                 else:
                      # If input already max_length or max_length is 0, cannot generate new tokens
                      cfg.max_new_tokens = 0 # Explicitly 0
                      logger.warning("Input length is already at max_length or max_length is zero. Cannot generate new tokens (max_new_tokens = 0).")


            logger.debug("Final GenerationConfig for this call after resolving params: %s", cfg.to_dict())


            # --- Call model.generate ---
            # Pass the prepared input tensors (which may include pixel_values) and generation config
            # The model's generate method will handle the multimodal input if supported
            generated_outputs = self.model.generate(
                **input_tensors, # Unpack the input tensors (input_ids, attention_mask, pixel_values, etc.)
                generation_config=cfg, # Pass the fully configured GenerationConfig
                return_dict_in_generate=True, # Ensure we get a dictionary output
                output_scores=True # Request scores if needed (though not used in parsing currently)
            )
            logger.info(f"Model generation complete. Generated {len(generated_outputs.sequences)} sequences.")

            # If scores were requested and returned, they are available in generation_output.scores
            generation_scores = generated_outputs.scores if hasattr(generated_outputs, 'scores') else None
            if generation_scores is not None: # Check explicitly for None
                 logger.debug("Generation scores available (%d scores tensors).", len(generation_scores))


        except Exception as e:
            logger.error("Failed during model generation: %s", e)
            # Attempt cleanup before raising
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as cleanup_e:
                    logger.warning(f"Error during cuda empty_cache: {cleanup_e}")
            gc.collect()
            # Do not re-raise here, return empty lists and let the GUI handle the error
            return {"full_texts": [], "reasoning_steps": [], "final_answers": [], "generated_images": [], "generation_scores": None}


        # --- Process Generated Outputs ---
        full_texts: List[str] = []
        reasoning_steps: List[List[str]] = [] # List of lists, one list of steps per sequence
        final_answers: List[Optional[str]] = [] # List of final answers per sequence
        # Placeholder for future generated images (multimodal output)
        generated_images_list: List[Any] = [] # Will store image data if generated


        if generated_outputs and hasattr(generated_outputs, 'sequences'):
             # Decode the generated token sequences
             # Need the tokenizer from the processor
             if self.tokenizer is None:
                  logger.error("Tokenizer is missing. Cannot decode generated sequences.")
                  # Return empty lists but don't stop processing
             else:
                 # Get the length of the input prompt's token IDs for prompt removal
                 # Safely get input_length, defaulting to 0 if input_ids is missing or empty
                 input_ids_tensor = input_tensors.get('input_ids', torch.tensor([[]]))
                 input_length = input_ids_tensor.shape[-1] if input_ids_tensor.numel() > 0 else 0
                 logger.debug(f"Input token length determined for prompt removal during decoding: {input_length}")


                 for i, sequence in enumerate(generated_outputs.sequences):
                     # Decode the entire generated sequence back to text
                     # Need to handle potential prompt remnants in the output for causal models.
                     # A common approach is to find the start of the generation (length of input_ids)
                     # and decode only from that point onwards.

                     # Ensure sequence is a tensor before slicing and decoding
                     if isinstance(sequence, torch.Tensor):
                          # Decode only the newly generated tokens (after the input prompt)
                          # Use max(0, input_length) to handle cases where input_length might be negative or zero
                          # Ensure the slice is valid (sequence might be shorter than input_length in error cases)
                          start_index = max(0, input_length)
                          # Use skip_special_tokens=True to remove EOS, BOS, PAD tokens from output text
                          decoded_text = self.tokenizer.decode(sequence[start_index:], skip_special_tokens=True)
                          logger.debug(f"Decoded new tokens for sequence {i} (input length {input_length}, decoded from index {start_index}): {decoded_text[:200]}...")
                     else:
                          # If sequence is not a tensor, decode the whole thing and log a warning
                          logger.warning(f"Generated sequence {i} is not a tensor (type: {type(sequence)}). Decoding full sequence and hoping parsing handles it.")
                          # Decode the full sequence, including potential prompt if it's not handled correctly upstream
                          decoded_text = self.tokenizer.decode(sequence, skip_special_tokens=True)
                          logger.debug(f"Decoded full sequence {i}: {decoded_text[:200]}...")


                     # In a multimodal generation scenario, the output might *also* contain image tokens
                     # or encoded image data. Extracting those would require model-specific parsing.
                     # For now, we assume text output, potentially with text-encoded image info that parsing might ignore.
                     # Placeholder for future image extraction:
                     # extracted_image_data = self._extract_image_data_from_text(decoded_text) # Conceptual

                     # Parse the decoded text for CoT steps and final answer
                     # Pass the original user text and the constructed CoT prompt text for parsing reference
                     steps, answer, full_output_text_cleaned = self._parse(
                         decoded_text, # The raw decoded output (just the new tokens part)
                         input_text, # Original user text input (for potential robust prompt removal in parse)
                         full_text_prompt # The constructed CoT prompt text (AGI + template) (for potential robust prompt removal in parse)
                     )

                     full_texts.append(full_output_text_cleaned) # Append the cleaned output body
                     reasoning_steps.append(steps)
                     final_answers.append(answer)
                     # Append placeholder or extracted image data
                     # generated_images_list.append(extracted_image_data if extracted_image_data is not None else None)


        else:
             logger.warning("Model generation did not return sequences in expected format or returned no sequences.")
             # Return empty lists


        # --- AGI Helper Module Interaction (Post-Generation) ---
        # Use NeoSentientCore and AGIEnhancer to process the generated output
        # Process the output of the first generated chain as the main experience, if any were generated.
        if AGI_IMPORTS_SUCCESS and full_texts:
             # Use the first chain's full output text for AGI processing
             main_output_text = full_texts[0]

             if self.memory_engine:
                 try:
                     # Observe the generated output (text)
                     # Pass text content. Image observation would need adapting MemoryEngine
                     self.memory_engine.observe(main_output_text)
                     logger.debug("MemoryEngine observed generated output (text).")
                 except Exception as e:
                     logger.warning(f"MemoryEngine observe failed: {e}")

                 try:
                     # Save reasoning chains (example: save steps from the first chain)
                     if reasoning_steps and reasoning_steps[0]:
                          # Ensure steps list contains strings before saving
                          valid_steps = [step for step in reasoning_steps[0] if isinstance(step, str) and step.strip()]
                          if valid_steps:
                               self.memory_engine.save_reasoning_chain(1, valid_steps) # Save steps from the first chain
                               logger.debug("MemoryEngine saved reasoning chain (from first chain).")
                          else:
                               logger.debug("MemoryEngine skipping saving empty or invalid reasoning chain.")
                 except Exception as e:
                      logger.warning(f"MemoryEngine save_reasoning_chain failed: {e}")

                 # Consider reflecting periodically - this logic should be managed externally or less frequently
                 # logger.debug("MemoryEngine reflection not called here.")


             if self.neuro_processor:
                 try:
                     # Record the generation experience (text)
                     generation_experience_detail = f"Generated response (first chain): {main_output_text[:200]}{'...' if len(main_output_text) > 200 else ''}"
                     # Pass text content. Image experience would need adapting NeuroMemoryProcessor
                     self.neuro_processor.record_experience("generation", generation_experience_detail)
                     logger.debug("NeuroMemoryProcessor recorded generation experience (text).")
                 except Exception as e:
                      logger.warning(f"NeuroMemoryProcessor record_experience failed: {e}")

                 # Update biases based on the output (example: process the text)
                 # Consider moving to scheduled task
                 # try:
                 #      self.neuro_processor._evolve_cognitive_bias(main_output_text) # Direct call for simplicity
                 #      logger.debug("NeuroProcessor evolved biases based on output.")
                 # except Exception as e:
                 #      logger.warning(f"NeuroProcessor _evolve_cognitive_bias failed: {e}")


             if self.agi_enhancer:
                 try:
                     # Log the generation experience (text)
                     enhancer_experience_detail = f"Generated response (first chain): {main_output_text[:200]}{'...' if len(main_output_text) > 200 else ''}"
                     # Pass text content. Image logging would need adapting AGIEnhancer
                     self.agi_enhancer.log_experience(enhancer_experience_detail)
                     logger.debug("AGIEnhancer logged experience.")
                 except Exception as e:
                      logger.warning(f"AGIEnhancer log_experience failed: {e}")

                 # Engage in reflection periodically - this logic should be managed externally or less frequently
                 # logger.debug("AGIEnhancer reflection not called here post-gen.")

             # NeoSentientCore post-generation actions (perception of its own output is handled above)
             if self.neo_sentient_core:
                 try:
                     # Simulate the core processing the generated output (text)
                     # Assuming NeoSentientCore has a process_output method that accepts text
                     if hasattr(self.neo_sentient_core, 'process_output'):
                          self.neo_sentient_core.process_output(main_output_text)
                          logger.debug("NeoSentientCore processed generated output (text).")
                     else:
                          logger.warning("NeoSentientCore does not have a 'process_output' method. Skipping output processing.")

                 except Exception as e:
                     logger.warning(f"NeoSentientCore process_output failed: {e}")



        # Attempt cleanup after generation attempt (success or failure)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared after generation attempt.")
            except Exception as cleanup_e:
                 logger.warning(f"Error during cuda empty_cache after generation attempt: {cleanup_e}")
                 pass # Suppress this warning unless in debug mode
        gc.collect()
        logger.debug("Garbage collection performed after generation attempt.")


        # Return the collected results
        return {
            "full_texts": full_texts,
            "reasoning_steps": reasoning_steps,
            "final_answers": final_answers,
            "generation_scores": generation_scores, # Include scores (will be None if not requested/available)
            # In a future multimodal version, generated_images might be included here
            "generated_images": generated_images_list # Return the list (might be empty)
        }


    def _parse(self, text: str, user_input: str, cot_prompt_text: str) -> Tuple[List[str], Optional[str], str]:
        """
        Parses one chain’s generated text into steps + final answer.
        Handles artifact cleaning. Attempts to handle potential prompt remnants.
        Returns: (steps_list, final_answer_string_or_None, cleaned_body_text)
        """
        logger.debug("_parse method called.")
        # Ensure input is a string
        if not isinstance(text, str):
            logger.warning(f"Attempted to parse non-string output: {type(text)}. Returning empty.")
            return [], None, str(text) # Return empty lists/None and the stringified input

        body = text.strip() # Start with the raw decoded text and strip leading/trailing whitespace

        # 1) Clean up artifacts using compiled patterns
        for pattern in self._artifact_patterns:
            body = pattern.sub("", body)
        body = body.strip()
        logger.debug(f"Text body after artifact cleanup: {body[:200]}...")

        # 2) Split into non‐empty lines for parsing
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        logger.debug(f"Split into {len(lines)} non-empty lines.")

        # 3) Extract tagged answer if present
        steps: List[str] = []
        final_answer: Optional[str] = None # Use Optional[str]
        tagged = False
        answer_line_index = -1 # Track line index of the answer tag

        # Search for the final answer tag *anywhere* in the lines
        # Use the compiled pattern
        for i, line in enumerate(lines):
            m = self.final_answer_pattern.search(line)
            if m:
                final_answer = m.group(1).strip()
                tagged = True
                answer_line_index = i # Store the index
                logger.debug(f"Found final answer tag on line {i}: '{final_answer[:100]}...'")
                break # Stop searching once the tag is found

        # 4) Collect steps from the beginning up to the line containing the answer tag (if tagged)
        # If not tagged, collect steps from all lines that match the step pattern.
        step_lines = []
        if tagged and answer_line_index != -1:
             # Collect steps from lines *before* the answer line index
             step_lines = lines[:answer_line_index]
             logger.debug(f"Collecting steps from lines before answer tag (up to line {answer_line_index}).")
        else:
             # If not tagged, consider all lines for steps
             step_lines = lines
             logger.debug("Final answer tag not found. Collecting steps from all lines matching step pattern.")


        # Extract steps using the step pattern from the identified step lines
        for line in step_lines:
            m = self._step_pattern.match(line)
            if m:
                steps.append(m.group(1).strip())
                # Apply conceptual limit *during* collection if needed, though parsing is usually fast.
                if self.reasoning_steps_limit > 0 and len(steps) >= self.reasoning_steps_limit:
                    logger.debug("Reached reasoning steps limit (%d). Stopping step collection.", self.reasoning_steps_limit)
                    break # Stop collecting steps if limit is reached

        logger.debug(f"Extracted {len(steps)} reasoning steps.")

        # 5) Fallback for final answer if no tagged answer was found
        # If no tagged answer was found AND no final_answer was extracted (e.g., tag was empty),
        # try to find the last non-step line as the answer.
        if not tagged and (final_answer is None or not final_answer.strip()): # Only attempt if no valid tagged answer found
             logger.debug("Attempting fallback for final answer...")
             # Iterate backwards from the end
             # Start from the last line, or just before the answer tag line if tag was found but empty
             start_index_for_fallback = answer_line_index if tagged and answer_line_index != -1 else len(lines) - 1
             for i in range(start_index_for_fallback, -1, -1):
                  line = lines[i]
                  # Check if the line is *not* a step line AND is not empty
                  if line.strip() and not self._step_pattern.match(line):
                       # Attempt to remove common answer prefixes from the fallback line
                       fallback_answer_attempt = re.sub(
                           r"^\s*(?:Answer|Result|Output|Final Answer)\s*[:\-]?\s*",
                           "",
                           line, # Use the original line for prefix removal attempt
                           flags=re.IGNORECASE
                       ).strip()
                       # If after removing prefixes, the line is not empty, use it as the fallback answer
                       if fallback_answer_attempt:
                            final_answer = fallback_answer_attempt
                            logger.debug("Fallback answer found: '%s'", final_answer[:100])
                            break # Found the fallback answer
                       # If removing prefixes resulted in an empty string, maybe the original line is the answer?
                       elif line.strip():
                            final_answer = line.strip()
                            logger.debug("Using last non-empty, non-step line as fallback answer: '%s'", final_answer[:100])
                            break # Found the fallback answer

        logger.debug(f"Final Answer (after fallback): '{final_answer[:100] if final_answer is not None else 'None'}'")

        # 6) Final cleanup on the extracted answer
        # Remove trailing punctuation that might be part of the model's generation habit
        if final_answer is not None:
            final_answer = re.sub(r'[.,;:]+$', '', final_answer).strip()
            logger.debug(f"Final Answer (after cleanup): '{final_answer[:100] if final_answer is not None else 'None'}'")


        logger.debug("Parsing complete. %d steps, Final Answer: '%s'", len(steps), final_answer[:100] if final_answer is not None else 'None')
        # Return steps list, final answer string (or None), and cleaned body text
        return steps, final_answer, body # Return the cleaned body text


    # Add placeholder for potential image data extraction from text output
    # This method would be highly model-specific
    # Multimodal output is not currently supported by this wrapper's parsing/extraction
    # def _extract_image_data_from_text(self, text: str) -> Optional[Any]:
    #      """
    #      Conceptual: Extracts encoded image data or image tokens from text output.
    #      Requires model-specific parsing logic.
    #      Returns image data or None.
    #      """
    #      logger.debug("Attempting to extract image data from text output (not implemented).")
    #      return None