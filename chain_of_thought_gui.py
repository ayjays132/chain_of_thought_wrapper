#!/usr/bin/env python3
"""
NeuroReasoner Chain-of-Thought GUI (Dark Theme Enhanced)
-------------------------------------------------------------
A premium Streamlit app for step-by-step reasoning
across any Hugging Face model (causal or seq2seq).
Featuring a dark theme, model-type detection, self-consistency
sampling & voting, robust handling, and GPU telemetry.
"""
import os
import time
import re # Needed for answer normalization
import streamlit as st
import torch
import pynvml # For GPU telemetry
import numpy as np # Imported, but currently unused in core logic
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    PretrainedConfig
)
from collections import Counter # For self-consistency voting
import gc # Import garbage collector
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Import the Enhanced ChainOfThoughtWrapper ---
# Assuming chain_of_thought_wrapper.py is in the same directory.
# The wrapper is expected to:
# 1. Accept model, tokenizer, GenerationConfig (base), device, etc. in __init__.
# 2. Have a .generate() method that takes input_text (str), GenerationConfig (overrides),
#    and crucially, num_return_sequences (int) to generate multiple chains efficiently.
# 3. Return a dictionary with keys 'full_texts', 'reasoning_steps', 'final_answers' (lists).
try:
    from chain_of_thought_wrapper import ChainOfThoughtWrapper
except ImportError:
    st.error("Error: chain_of_thought_wrapper.py not found. Please ensure the enhanced wrapper script is in the same directory.")
    st.stop() # Halt execution if the wrapper is not found

# --- Logging Setup for GUI ---
# Use Streamlit's built-in logging or configure a separate logger
# For this example, we'll keep it simple and rely mostly on st.status and st.exception
# if needed, a more detailed logger could be configured here.

# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 NeuroReasoner CoT GUI",
    page_icon="🧠",
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded", # Sidebar open by default
    menu_items={
        'Get Help': 'https://github.com/ayjays132/NeuroReasoner', # Example repo link
        'Report a bug': "https://github.com/ayjays132/NeuroReasoner/issues", # Example repo issues link
        'About': """
        **NeuroReasoner Chain-of-Thought GUI**
        An open-source interface powered by Hugging Face models and the enhanced NeuroReasoner wrapper.
        Explore step-by-step reasoning with various language models.
        \n\n**Features:** Dark Theme, GPU Telemetry, Model Caching, Self-Consistency Voting,
        Robust Generation Parameters, Support for Causal and Seq2Seq models.
        """
    }
)

# --- Dark Theme CSS ---
# Comprehensive CSS for a professional dark theme inspired by VS Code.
st.markdown("""
<style>
    /* Overall Page Background & Text (Dark Theme) */
    /* Target the main container and the root app div */
    .stApp {
        background-color: #1E1E1E; /* Dark grey background */
        color: #D4D4D4; /* Light grey text */
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: #2D2D2D; /* Slightly lighter dark grey for sidebar */
        padding: 2rem 1rem;
        border-right: 1px solid #3E3E3E; /* Subtle border */
        color: #D4D4D4; /* Ensure text in sidebar is light */
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
          color: #569CD6 !important; /* Visual Studio Code blue for sidebar headers */
    }
    .stSidebar label {
        color: #D4D4D4 !important; /* Ensure sidebar labels are visible */
    }


    /* Main Content Area */
    /* No specific background needed here, .stApp covers it */

    /* Titles and Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #569CD6; /* VS Code blue headings */
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    h1 { font-size: 2.5rem; color: #4EC9B0; } /* Teal for main title */
    h2 { font-size: 2rem; border-bottom: 2px solid #569CD6; padding-bottom: 0.5rem; margin-bottom: 1rem;}


    /* Buttons */
    .stButton>button {
        background-color: #1E4D2B; /* Dark green */
        color: #4EC9B0; /* Teal text */
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        transition: background-color 0.2s ease, transform 0.1s ease;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        margin-top: 1.65rem; /* Add top margin to align with text area */
    }
    .stButton>button:hover {
        background-color: #27633A; /* Lighter green on hover */
        transform: translateY(-1px);
    }
    .stButton>button:active {
        background-color: #1A3C23; /* Darker green on click */
        transform: translateY(0);
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.4);
    }


    /* Text areas and inputs */
    /* Target specific classes used by Streamlit for input/text areas */
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input {
        border: 1px solid #3E3E3E; /* Dark border */
        border-radius: 0.4rem;
        padding: 0.75rem;
        font-size: 1rem;
        background-color: #252526; /* VS Code background */
        color: #D4D4D4; /* Light text */
        box-shadow: inset 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
     div[data-baseweb="textarea"] label,
     div[data-baseweb="input"] label,
     .stSlider label, .stSelectbox label, .stCheckbox label {
        font-weight: bold;
        color: #9CDCFE !important; /* Light blue labels */
        margin-bottom: 0.5rem;
        display: block;
    }
     /* Streamlit status box styling - Target the main container and its contents */
    .st-emotion-cache-vj1l9j { /* This class might change with Streamlit versions */
        background-color: #2D2D2D; /* Match sidebar background */
        border: 1px solid #3E3E3E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
     .st-emotion-cache-vj1l9j .stMarkdown p,
     .st-emotion-cache-vj1l9j .stAlert { /* Style text and alerts inside status */
         color: #D4D4D4 !important;
         background-color: transparent !important; /* Don't want alert backgrounds inside status */
         border: none !important; /* No borders for alerts inside status */
         padding: 0.5rem 0 !important; /* Adjust padding */
     }


    /* Info/Success/Error/Warning boxes */
    .stAlert {
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        padding: 1rem;
        font-size: 1rem;
        border-left: 5px solid transparent; /* Base style */
        color: #D4D4D4; /* Default text color for alerts */
    }
    .stAlert.stAlert-info { border-left-color: #569CD6; background-color: #2A3E52; } /* Dark blue info */
    .stAlert.stAlert-success { border-left-color: #4EC9B0; background-color: #28403A; } /* Dark teal success */
    .stAlert.stAlert-warning { border-left-color: #DCDCAA; background-color: #454032; } /* Dark yellow warning */
    .stAlert.stAlert-error { border-left-color: #F44747; background-color: #4A3030; } /* Dark red error */


    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #3E3E3E; /* Dark grey header */
        color: #D4D4D4; /* Light grey text */
        border-radius: 0.5rem;
        padding: 0.75rem 1.2rem;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #4E4E4E; /* Slightly lighter on hover */
    }
    .streamlit-expanderContent {
        background-color: #252526; /* VS Code background */
        border: 1px solid #3E3E3E;
        border-top: none;
        border-bottom-left-radius: 0.5rem;
        border-bottom-right-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 0;
        color: #D4D4D4;
    }
    .streamlit-expanderContent .stMarkdown p {
         color: #D4D4D4 !important; /* Ensure text inside expanders is light */
    }


    /* Labels for the output text areas */
    .output-label {
        font-weight: bold !important;
        color: #9CDCFE !important; /* Light blue */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        display: block;
        font-size: 1.1rem;
    }

    /* Custom class for output text areas to differentiate from input */
    /* Need to target the specific Streamlit internal class for the text area */
    .output-text-area div[data-baseweb="textarea"] textarea {
        background-color: #1E1E1E; /* Even darker background for outputs */
        border: 1px solid #3E3E3E;
        border-radius: 0.4rem;
        padding: 0.75rem;
        font-size: 1rem;
        color: #D4D4D4;
    }

    /* Telemetry box styling */
    .telemetry-box {
        background-color: #2D2D2D; /* Match sidebar */
        border: 1px solid #3E3E3E;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #D4D4D4;
        text-align: center;
    }

    /* Self-Consistency Consensus Styling */
    .consensus-answer {
        background-color: #28403A; /* Dark green */
        color: #7AC7A3; /* Light green text */
        border: 1px solid #3A5048;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .consensus-answer strong {
        color: #4EC9B0 !important; /* Teal for "Consensus Answer" label */
    }
     .consensus-answer p {
         color: #D4D4D4 !important; /* Ensure the answer text is light */
         margin: 0 !important; /* Remove default paragraph margins */
         padding: 0 !important; /* Remove default paragraph padding */
     }


</style>
""", unsafe_allow_html=True)


# --- GPU Telemetry Setup ---
# Initialize NVML for GPU monitoring if available
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    # Get the number of devices to pick the first one (index 0)
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
    if GPU_COUNT == 0:
        GPU_AVAILABLE = False
        st.warning("NVML initialized but no NVIDIA GPUs found.")
except Exception:
    GPU_AVAILABLE = False
    # st.info("NVIDIA Management Library (pynvml) not found or failed to initialize. GPU telemetry disabled.")

# Use st.empty to hold the telemetry status text, defined *outside* cached functions
telemetry_placeholder = st.empty()

def update_telemetry():
    """Updates the telemetry display in the dedicated placeholder."""
    telemetry_text = "📊 System Status: [Initializing...]"
    if not GPU_AVAILABLE or not torch.cuda.is_available():
        telemetry_text = "📊 System Status: [No GPU Available]"
    else:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Use the first GPU
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = memory.used // 1024**2
            mem_total_mb = memory.total // 1024**2
            telemetry_text = f"📊 System Status: GPU 0: {utilization.gpu}% | Mem {mem_used_mb}/{mem_total_mb} MB"
        except Exception:
             # If NVML fails after initialization, report error
             telemetry_text = "📊 System Status: [Telemetry Error]"

    # Use markdown with a custom class for styling the container
    telemetry_placeholder.markdown(f'<div class="telemetry-box">{telemetry_text}</div>', unsafe_allow_html=True)

# Initial telemetry update when the script starts
update_telemetry()


# --- Caching Model Loading (Core Logic Only) ---
# Use st.cache_resource for heavy objects like models and tokenizers.
# This function MUST NOT call Streamlit elements that affect the layout
# or state outside of its own scope (except for the final return value).
@st.cache_resource(show_spinner=False) # Spinner handled manually in safe_load_model_with_status
def _load_model_and_tokenizer_cached(model_name: str, device: str, forced_model_type: str = None):
    """
    Loads the model and tokenizer. This function is cached by Streamlit.
    It should perform resource-intensive loading only.
    """
    # Use low_cpu_mem_usage=True to reduce RAM usage during loading
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True)
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
    detected_type = "Seq2Seq" if is_encoder_decoder else "Causal"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure padding token is set for generation robustness, especially for batching (num_return_sequences)
    # This mirrors the logic in the wrapper's __init__ but is good to do here too
    # before the model is potentially loaded with a different vocab size.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
             tokenizer.pad_token = tokenizer.eos_token
             # Set the ID explicitly as well
             tokenizer.pad_token_id = tokenizer.eos_token_id
             # logger.warning(f"Tokenizer pad_token is None, using eos_token '{tokenizer.eos_token}' as pad_token.")
        else:
            # Fallback: Add a new pad token if neither eos nor pad exists.
            # The wrapper's init will handle resizing embeddings if possible.
            # logger.warning("Tokenizer has no pad_token and no eos_token. Adding a new [PAD] token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Need to get the ID for the new token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            # logger.info(f"Added new [PAD] token with ID {tokenizer.pad_token_id}.")


    # Determine the model class based on detection or forced selection
    actual_model_type = forced_model_type if forced_model_type != "Auto" else detected_type
    model = None # Initialize model to None

    try:
        if actual_model_type == "Seq2Seq":
             model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        elif actual_model_type == "Causal":
             model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        else:
             raise ValueError(f"Unsupported model type selected: {actual_model_type}. Please select 'Auto', 'Causal', or 'Seq2Seq'.")

        # Move model to device and set to eval mode
        model.to(device)
        model.eval() # Crucial for consistent inference behavior and disabling dropout etc.

        # Ensure return_dict_in_generate is True for structured outputs (needed by wrapper for scores/sequences)
        if not getattr(model.config, 'return_dict_in_generate', False):
             model.config.return_dict_in_generate = True
        # Request output_scores by default for potential future CISC use in the GUI voter
        if not getattr(model.config, 'output_scores', False):
             model.config.output_scores = True

        # The wrapper's __init__ will perform its own pad token handling and embedding resizing check.
        # We ensure the tokenizer passed to it has a pad_token_id.

    except Exception as e:
        # Clean up resources if model loading failed
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        raise e # Re-raise the exception for the caller to handle status updates

    return model, tokenizer, actual_model_type

# --- Wrapper function to handle status reporting for cached loading ---
def safe_load_model_with_status(model_name: str, device: str, forced_model_type: str = None):
    """
    Calls the cached loading function (_load_model_and_tokenizer_cached)
    and handles Streamlit status updates and error reporting.
    """
    # Use st.status here, defined outside the cached function, for live updates
    status_text = f"🌐 Loading model '{model_name}' on device '{device}'..."
    with st.status(status_text, expanded=True) as status_box:
        status_box.write("Checking system status...")
        update_telemetry() # Update the separate telemetry box

        try:
            status_box.write("Loading configuration and tokenizer...")
            # Call the actual cached loading function
            model, tokenizer, actual_model_type = _load_model_and_tokenizer_cached(
                model_name=model_name,
                device=device,
                forced_model_type=forced_model_type
            )

            # Report padding token status after loading
            if tokenizer and tokenizer.pad_token_id is None:
                 status_box.warning(f"Tokenizer has no pad_token_id. Batch generation (Self-Consistency) might be unstable.")
            elif tokenizer:
                 status_box.write(f"Tokenizer pad_token_id set to {tokenizer.pad_token_id}.")


            status_box.success(f"✅ Model '{model_name}' ({actual_model_type}) loaded successfully on '{device}'.")
            update_telemetry() # Final telemetry update after success
            return model, tokenizer, actual_model_type

        except Exception as e:
            status_box.error(f"❌ Model loading failed.")
            update_telemetry() # Final telemetry update after error
            st.exception(e) # Display the full exception traceback within the status box
            # No need for manual cleanup here, as the exception in the cached function
            # should have triggered cleanup within that function, and Streamlit's
            # cache resource management handles state on failure.
            return None, None, None # Return None on failure


# --- Self-Consistency Voting Logic ---
def normalize_answer(answer: str) -> str:
    """
    Normalizes a string answer for robust comparison during voting.
    - Converts to lowercase.
    - Strips leading/trailing whitespace.
    - Removes common punctuation.
    - Can be extended with more sophisticated normalization (e.g., number words to digits).
    """
    if not isinstance(answer, str):
        return "" # Handle non-string inputs

    # Simple normalization: lowercase, strip whitespace, remove common punctuation
    normalized = answer.lower().strip()
    # Remove common trailing characters like periods, commas, etc.
    normalized = re.sub(r'[.,!?;:]+$', '', normalized).strip()
    # Remove common leading "Answer: " or similar preambles (case-insensitive)
    normalized = re.sub(r'^\s*(?:the answer is|result|output)\s*[:\-]?\s*', '', normalized, flags=re.IGNORECASE).strip()
    # Add more normalization rules if needed (e.g., handling "forty two" vs "42")

    return normalized

def perform_self_consistency_voting(final_answers: List[str]) -> Tuple[Optional[str], Dict[str, int]]:
    """
    Performs simple majority voting on a list of final answers.
    Filters out empty answers and normalizes them before voting.

    Args:
        final_answers (List[str]): A list of raw final answer strings from the wrapper.

    Returns:
        Tuple[Optional[str], Dict[str, int]]: A tuple containing:
            - The winning (most common) normalized answer, or None if no valid answers.
            - A dictionary mapping normalized answers to their vote counts.
    """
    if not final_answers:
        return None, {}

    # 1. Filter out empty or non-string answers
    valid_answers = [ans for ans in final_answers if isinstance(ans, str) and ans.strip()]

    if not valid_answers:
        return None, {}

    # 2. Normalize answers
    normalized_answers = [normalize_answer(ans) for ans in valid_answers]
    # Filter out answers that became empty after normalization
    normalized_answers = [ans for ans in normalized_answers if ans.strip()]

    if not normalized_answers:
         return None, {}


    # 3. Perform majority voting
    answer_counts = Counter(normalized_answers)

    # 4. Determine the consensus answer
    # most_common(1) returns a list like [('answer', count)]
    most_common_item = answer_counts.most_common(1)

    if most_common_item:
        consensus_answer = most_common_item[0][0]
        return consensus_answer, dict(answer_counts)
    else:
        # This case should ideally not happen if normalized_answers is not empty
        return None, dict(answer_counts)


# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Core Settings")
    st.markdown("Configure the foundational aspects of the NeuroReasoner.")

    with st.expander("🧠 Model Configuration", expanded=True):
        model_name = st.text_input(
            "Hugging Face Model ID or Path",
            "ayjays132/NeuroReasoner-1-NR-1", # Default model
            help="Enter the model ID from huggingface.co or a local path. Changing this requires reloading."
        )

        # --- Dynamic Model Type Detection ---
        # Attempt to load config to detect type without caching (lightweight check)
        # This provides immediate feedback on the likely model type.
        detected_type_display = "Unknown (Enter Model ID)"
        initial_config = None
        try:
            if model_name and model_name.strip(): # Only attempt if input is not empty
                with st.spinner("Detecting model type..."): # Small spinner for detection
                    # Use from_pretrained without loading weights (low_cpu_mem_usage=True helps)
                    initial_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True)
                    is_encoder_decoder_initial = getattr(initial_config, "is_encoder_decoder", False)
                    detected_type_display = "Seq2Seq" if is_encoder_decoder_initial else "Causal"
                    # Store the actual detected type for use in the selectbox default index
                    actual_detected_type_for_index = "Seq2Seq" if is_encoder_decoder_initial else "Causal"
            else:
                 detected_type_display = "Unknown (Enter Model ID)"
                 actual_detected_type_for_index = "Auto" # Default to Auto index if no model name

        except Exception:
            detected_type_display = "Unknown (Config Load Error)" # Indicate config load itself failed
            actual_detected_type_for_index = "Auto" # Default to Auto index if config load fails


        # Options match the strings used in the loading function
        model_type_options = ["Auto", "Causal", "Seq2Seq"]
        # Set the default index based on the detected type, falling back to "Auto"
        try:
             default_model_type_index = model_type_options.index(actual_detected_type_for_index) if actual_detected_type_for_index in model_type_options else 0 # Default to Auto (index 0)
        except ValueError:
             default_model_type_index = 0 # Should not happen if logic is correct, but safety fallback


        forced_model_type = st.selectbox(
            "Architecture Type",
            model_type_options,
            index=default_model_type_index, # Use the detected type as the default selection
            help=f"Detected: {detected_type_display}. 'Auto' uses the detected type. Select manually if detection is incorrect or overridden."
        )

        # --- Device Selection ---
        # List available devices, prioritizing CUDA if available
        available_devices = ["cpu"]
        if torch.cuda.is_available():
            available_devices.insert(0, "cuda") # Put cuda first if available

        device = st.selectbox(
            "Device",
            available_devices,
            index=(0 if "cuda" in available_devices else 0), # Default to cuda if available, else cpu
            help="Select the hardware device for computation (GPU recommended)."
        )

        st.markdown("""
            <small>💡 Changing model settings requires reloading the model.</small>
        """, unsafe_allow_html=True)


    st.markdown("---") # Visual separator

    st.header("✨ Generation Parameters")
    st.markdown("Define how the AI generates reasoning steps and answers.")

    with st.expander("Basic Parameters", expanded=True):
        # Number of Reasoning Chains slider
        num_chains = st.slider(
            "Number of Reasoning Chains",
            min_value=1,
            max_value=15, # Allow generating up to 15 chains
            value=5,      # Default to 5 chains for a good balance
            help="How many independent reasoning chains to generate for analyzing the problem. More chains can improve Self-Consistency but take longer and use more memory."
        )

        # Self-Consistency checkbox
        self_consistency_enabled_gui = st.checkbox(
            "Enable Self-Consistency Voting",
            value=True, # Default to enabled
            help="When enabled, the system generates multiple chains and identifies the most common final answer as the consensus via majority voting. Requires 'Number of Reasoning Chains' > 1."
        )

        # Conditional warning if Self-Consistency is on but num_chains is 1
        if self_consistency_enabled_gui and num_chains <= 1:
             st.warning("Self-Consistency voting is most effective with 2 or more chains.")


    # Advanced Parameters
    with st.expander("🧪 Advanced Sampling Parameters"):
        # Using standard parameter names from Hugging Face GenerationConfig
        max_new_tokens = st.slider(
            "Max New Tokens per Chain",
            50, 2048, 768, # Min, Max, Default
            help="Maximum number of new tokens to generate for *each* individual reasoning chain. Adjust based on expected reasoning complexity and answer length."
        )
        temperature = st.slider(
            "Temperature",
            0.0, 2.0, 0.8, # Min, Max, Default
            step=0.05, # Allow finer control
            help="Controls the randomness of sampling. 0.0 is deterministic (greedy). Higher values increase diversity in reasoning paths."
        )
        top_k = st.slider(
            "Top-k",
            0, 200, 50, # Min, Max, Default (increased max k)
            help="Filter to consider only the top_k most likely tokens at each step (0 disables). Used with sampling."
        )
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            0.0, 1.0, 0.95, # Min, Max, Default
            step=0.01, # Allow finer control
            help="Filter to consider tokens with cumulative probability below top_p (0.0 disables). Used with sampling."
        )
        repetition_penalty = st.slider(
            "Repetition Penalty",
            1.0, 2.0, 1.1, # Min, Max, Default
            step=0.05, # Allow finer control
            help="Penalizes repeated tokens or sequences. Higher values reduce repetition in the output."
        )
        no_repeat_ngram_size = st.slider(
             "No-repeat Ngram Size",
             0, 10, 0, # Min, Max, Default (changed default to 0, often less needed with repetition penalty)
             help="Avoids repeating sequences of N tokens. Set to 0 to disable. Can help prevent loops in reasoning."
        )
        do_sample = st.checkbox(
            "Enable Sampling",
            value=True, # Default to enabled
            help="If checked, uses probabilistic sampling (controlled by Temperature, Top-k, Top-p). If unchecked, uses greedy decoding (deterministic)."
        )
        if not do_sample:
             st.info("Sampling disabled. Temperature, Top-k, and Top-p will be ignored.")


    st.markdown("---") # Visual separator

    # Update the persistent telemetry box in the sidebar footer area
    update_telemetry()


# --- Main Content Layout ---
st.title("🧠 NeuroReasoner: Chain-of-Thought Explorer")
st.markdown("Unpack complex problems with step-by-step AI reasoning.")

# Container for input and primary controls
input_container = st.container()

with input_container:
    # Use columns for prompt input and action button
    prompt_col, button_col = st.columns([3, 1])

    with prompt_col:
        prompt = st.text_area(
            "📝 Enter your query or problem:",
            height=180, # Increased height for better input experience
            placeholder="Example: If a train travels at 60 mph and a car at 40 mph, starting at the same time from cities 300 miles apart, how long until they meet? Think step-by-step.",
            key="user_prompt" # Unique key for the widget
        )

    with button_col:
        # Add some vertical space to align the button nicely with the text area
        st.markdown("<div style='height: 3.25rem;'></div>", unsafe_allow_html=True) # Adjusted height
        run_button = st.button("✨ Generate Reasoning", use_container_width=True, key="generate_button") # Unique key

# Container for status updates and results
results_container = st.container()


# --- Generation Logic Trigger ---
if run_button:
    if not prompt or not prompt.strip():
        results_container.warning("Please enter a prompt to begin generation.")
        # No need to st.stop() here, warning is sufficient
    else:
        # --- Prepare for Generation ---
        # Load model and tokenizer (handles caching internally via safe_load_model_with_status)
        # This happens only when the button is clicked and parameters might have changed
        model, tokenizer, loaded_model_type = safe_load_model_with_status(model_name, device, forced_model_type)

        if model is None or tokenizer is None:
            # Error was already shown by safe_load_model_with_status
            results_container.error("Model or tokenizer failed to load. Please check settings and traceback above.")
            # No need to st.stop() here, error message is displayed

        else: # Model and tokenizer loaded successfully
            # --- Configure GenerationConfig ---
            # Build the GenerationConfig object from sidebar parameters
            # This config will be passed to the wrapper's __init__ as the base config
            # and to the wrapper's generate() call as overrides.
            # The wrapper's generate method will ultimately use these settings.
            try:
                # Base config for the wrapper's __init__ - defines default behavior
                base_gen_config = GenerationConfig(
                    max_new_tokens=max_new_tokens, # Use sidebar value
                    temperature=temperature,       # Use sidebar value
                    top_k=top_k,                   # Use sidebar value
                    top_p=top_p,                   # Use sidebar value
                    do_sample=do_sample,           # Use sidebar value
                    repetition_penalty=repetition_penalty, # Use sidebar value
                    no_repeat_ngram_size=no_repeat_ngram_size, # Use sidebar value
                    eos_token_id=tokenizer.eos_token_id, # Always pass eos_token_id from tokenizer
                    pad_token_id=tokenizer.pad_token_id, # Always pass pad_token_id from tokenizer
                    # Other parameters like num_beams, etc., could be added here if exposed in sidebar
                )


            except Exception as e:
                results_container.error(f"❌ Failed to create GenerationConfig from parameters: {e}")
                st.exception(e)
                st.stop() # Stop if config creation fails

            # --- Instantiate Wrapper ---
            # Use a status box for ongoing generation process
            with results_container:
                st.markdown("## ⏳ Generation Progress") # Use a clear header for the status section
                generation_status = st.status("Initializing Chain-of-Thought wrapper...", expanded=True)
                update_telemetry() # Update telemetry while status is active

                try:
                    # Pass the loaded model, tokenizer, device, and relevant settings.
                    # The wrapper uses the generation_config passed here as its base defaults.
                    # Self-consistency settings from GUI are passed to wrapper's init attributes.
                    cot_wrapper = ChainOfThoughtWrapper(
                        model=model,
                        tokenizer=tokenizer,
                        # Pass the configured generation params as the base config for the wrapper
                        generation_config=base_gen_config,
                        device=device,
                        # Pass GUI self-consistency settings
                        # The wrapper uses self_consistency_enabled to decide if it *should* generate >1 chains
                        # when num_return_sequences is not explicitly passed to generate().
                        # We *are* explicitly passing num_return_sequences to generate(),
                        # so these init flags primarily inform internal wrapper behavior/logging.
                        self_consistency_enabled=self_consistency_enabled_gui,
                        consistency_rounds=num_chains # Inform the wrapper about intended rounds
                        # Pass other wrapper-specific init args if needed (e.g., custom tags)
                        # final_answer_tag="Final Answer:" # Example if different from default
                    )
                    generation_status.write("Wrapper initialized.")
                    update_telemetry()

                except Exception as e:
                    generation_status.error(f"❌ Failed to initialize CoT wrapper: {e}")
                    st.exception(e)
                    # Clean up resources in case of failure
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()
                    st.stop() # Stop if wrapper initialization fails

                # --- Generate ---
                # Call the wrapper's generate method.
                # Pass the original input text.
                # Explicitly pass num_return_sequences to request the desired number of chains.
                generation_status.update(label=f"⏳ Generating {num_chains} reasoning chain(s)...", state="running")
                start_time = time.time()

                try:
                    # The wrapper's generate takes input_text and optional config/num_return_sequences overrides.
                    # We rely on the wrapper's internal config logic and pass the desired number of sequences.
                    outputs = cot_wrapper.generate(
                        input_text=prompt,
                        # Optional: Pass a GenerationConfig override for this specific call if needed, e.g.:
                        # generation_config=GenerationConfig(temperature=temperature + 0.1),
                        # Pass the requested number of chains directly to the generate method:
                        num_return_sequences=num_chains,
                    )
                    # Expected `outputs` dict structure: {'full_texts': [...], 'reasoning_steps': [...], 'final_answers': [...], 'generation_scores': [...]}

                except Exception as e:
                    generation_status.error(f"❌ Generation failed: {e}")
                    st.exception(e) # Display the full exception traceback
                    # Clean up resources after potential OOM or other errors
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect() # Python garbage collection
                    # No need to st.stop() here, error message is displayed
                    outputs = None # Ensure outputs is None if generation fails

                elapsed_time = time.time() - start_time

                # Update final status based on success or failure
                if outputs is not None:
                    generation_status.update(label=f"✨ Generation complete in {elapsed_time:.2f}s", state="complete")
                else:
                     generation_status.update(label=f"❌ Generation failed after {elapsed_time:.2f}s", state="error")

                update_telemetry() # Final telemetry update after generation attempt


            # --- Process and Display Results ---
            if outputs is not None: # Only display if generation was attempted and returned results
                with results_container:
                    st.markdown("## 📚 Reasoning Output")

                    # Display Self-Consistency Consensus first if enabled and results are available
                    # Implement the voting logic here in the GUI using the wrapper's output
                    final_answers_list = outputs.get('final_answers', [])

                    if self_consistency_enabled_gui and final_answers_list:
                        # Perform the actual voting
                        consensus_answer, answer_distribution_dict = perform_self_consistency_voting(final_answers_list)
                        # Convert dict to Counter for sorting by count in display
                        answer_distribution = Counter(answer_distribution_dict)


                        st.markdown('<div class="consensus-answer">', unsafe_allow_html=True)
                        st.write("💡 **Consensus Answer (Self-Consistency):**")
                        if consensus_answer:
                             st.write(f'<p>{consensus_answer}</p>', unsafe_allow_html=True)
                             # Optional: Add a note about the confidence/number of votes for the winner
                             # We need the count of the winning answer from the distribution
                             winner_count = answer_distribution.get(normalize_answer(consensus_answer), 0) # Get count for the winning normalized answer
                             st.write(f"*(Based on {winner_count} {'vote' if winner_count == 1 else 'votes'} out of {len(final_answers_list)} chains)*")
                        else:
                             st.write("<p>[Could not determine consensus - no valid answers found]</p>", unsafe_allow_html=True)
                             st.write(f"*(Examined {len(final_answers_list)} chains)*")

                        st.markdown('</div>', unsafe_allow_html=True)

                        # Display answer distribution if there's more than one unique answer after normalization
                        if len(answer_distribution) > 1:
                            st.markdown("###### Answer Distribution:")
                            # Display sorted distribution by vote count
                            for ans, count in answer_distribution.most_common():
                                # Display the normalized answer and its count
                                st.write(f"- '{ans}' ({count} {'vote' if count == 1 else 'votes'})")
                        elif len(answer_distribution) == 1 and consensus_answer:
                             st.info(f"All {len(final_answers_list)} valid chains agreed on the normalized answer: '{consensus_answer}'.")
                        else:
                             st.warning(f"No valid answers ({len(answer_distribution)} unique normalized answers) were found to determine distribution from {len(final_answers_list)} chains.")


                        st.markdown("---") # Separator after consensus section


                    # Display individual chains
                    full_texts = outputs.get('full_texts', [])
                    reasoning_steps_list = outputs.get('reasoning_steps', [])
                    final_answers_list_raw = outputs.get('final_answers', []) # Keep raw answers for display

                    if not full_texts:
                        st.warning("No reasoning chains were generated or parsed successfully.")
                    else:
                        st.markdown(f"### Individual Chains ({len(full_texts)} generated)")
                        # Iterate and display each chain in an expander
                        # Ensure lists are iterable and have consistent length, padding with placeholders if necessary
                        max_len_outputs = len(full_texts)
                        # Ensure reasoning_steps_list and final_answers_list_raw match the length of full_texts
                        reasoning_steps_list = (reasoning_steps_list if isinstance(reasoning_steps_list, list) else []) + [[]] * (max_len_outputs - len(reasoning_steps_list))
                        final_answers_list_raw = (final_answers_list_raw if isinstance(final_answers_list_raw, list) else []) + ["[N/A - Parsing Failed]"] * (max_len_outputs - len(final_answers_list_raw))


                        for idx, (text, steps, ans_raw) in enumerate(zip(full_texts, reasoning_steps_list, final_answers_list_raw), 1):
                            # Use try-except for displaying each chain just in case of unexpected data format
                            try:
                                # Expander for each chain, starting collapsed
                                with st.expander(f"Chain {idx}", expanded=False):
                                    # Display full generated text
                                    st.markdown('<div class="output-label">Full Generated Text (Cleaned):</div>', unsafe_allow_html=True)
                                    st.text_area(f"chain_text_area_{idx}", text if isinstance(text, str) else "[Invalid Text Data]", height=250, label_visibility="collapsed", help="The complete generated output for this chain after cleaning artifacts.", key=f"chain_text_{idx}") # Added key

                                    # Display parsed reasoning steps
                                    st.markdown('<div class="output-label">Reasoning Steps Parsed:</div>', unsafe_allow_html=True)
                                    if steps and isinstance(steps, list) and len(steps) > 0:
                                        # Display steps as a list with strong emphasis on step number
                                        for i, step in enumerate(steps, 1):
                                            if isinstance(step, str) and step.strip():
                                                 st.markdown(f"**Step {i}:** {step.strip()}")
                                            elif not isinstance(step, str):
                                                 st.warning(f"Step {i} has invalid format in Chain {idx}.")
                                    elif isinstance(steps, list) and len(steps) == 0:
                                         st.info("No specific steps were extracted for this chain.")
                                    else:
                                         st.warning(f"Reasoning steps data is invalid or missing for Chain {idx}.")


                                    # Display parsed final answer (raw)
                                    st.markdown('<div class="output-label">Final Answer Parsed:</div>', unsafe_allow_html=True)
                                    display_answer = ans_raw if isinstance(ans_raw, str) and ans_raw.strip() else "[No answer extracted]"
                                    st.write(f"**{display_answer}**")

                                    # Optional: Add a separator between chain sections within the expander if desired
                                    # st.markdown("---")

                            except Exception as chain_e:
                                st.error(f"Error displaying content for Chain {idx}: {chain_e}")
                                st.exception(chain_e)

                        # Final separator after all chains
                        st.markdown("---")
                        st.info(f"Displayed details for {len(full_texts)} generated chains.")

            # Clean up GPU memory after generation and display are complete
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()
            gc.collect() # Python garbage collection
            # st.write("GPU memory cache cleared and garbage collected.") # Optional status message