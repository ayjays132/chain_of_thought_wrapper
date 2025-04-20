#!/usr/bin/env python3
"""
NeuroReasoner Chain-of-Thought GUI (Dark Theme Enhanced)
-------------------------------------------------------------
A premium Streamlit app for step-by-step reasoning
across any Hugging Face model (causal or seq2seq).
Featuring a dark theme, model-type detection, self-consistency
sampling, and robust handling.
"""
import os
import time
import streamlit as st
import torch
import pynvml # For GPU telemetry
import numpy as np
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

# Assuming chain_of_thought_wrapper.py is in the same directory
# and is designed to work with standard Hugging Face models and GenerationConfig.
# Make sure the wrapper correctly handles num_return_sequences for CoT and SC,
# and returns the expected dictionary structure:
# {'full_texts': [...], 'reasoning_steps': [...], 'final_answers': [...], 'consensus_answer': '...'}
try:
    from chain_of_thought_wrapper import ChainOfThoughtWrapper
except ImportError:
    st.error("Error: chain_of_thought_wrapper.py not found. Please ensure it's in the same directory.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 NeuroReasoner CoT GUI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your_repo_link_here', # Replace or remove
        'Report a bug': "https://github.com/your_repo_link_here/issues", # Replace or remove
        'About': """
        **NeuroReasoner Chain-of-Thought GUI**
        An open-source interface powered by Hugging Face models and the NeuroReasoner wrapper.
        Explore step-by-step reasoning with various language models.
        """
    }
)

# --- Dark Theme CSS ---
st.markdown("""
<style>
    /* Overall Page Background & Text (Dark Theme) */
    body {
        background-color: #1E1E1E; /* Dark grey background */
        color: #D4D4D4; /* Light grey text */
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    }
    .stApp {
        background-color: #1E1E1E;
        color: #D4D4D4;
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: #2D2D2D; /* Slightly lighter dark grey for sidebar */
        padding: 2rem 1rem;
        border-right: 1px solid #3E3E3E; /* Subtle border */
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
         color: #569CD6; /* Visual Studio Code blue for sidebar headers */
    }
    .stSidebar label {
        color: #D4D4D4 !important; /* Ensure sidebar labels are visible */
    }


    /* Main Content Area */
    .stContainer {
        padding: 2rem;
    }

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
    .stTextArea textarea, .stTextInput input {
        border: 1px solid #3E3E3E; /* Dark border */
        border-radius: 0.4rem;
        padding: 0.75rem;
        font-size: 1rem;
        background-color: #252526; /* VS Code background */
        color: #D4D4D4; /* Light text */
        box-shadow: inset 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
     .stTextArea label, .stTextInput label {
        font-weight: bold;
        color: #9CDCFE !important; /* Light blue labels */
        margin-bottom: 0.5rem;
        display: block;
    }
    /* Streamlit status box styling */
    .st-emotion-cache-vj1l9j { /* Target the status box content div */
        background-color: #2D2D2D; /* Match sidebar background */
        border: 1px solid #3E3E3E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
     .st-emotion-cache-vj1l9j .stMarkdown p { /* Style text inside status */
        color: #D4D4D4 !important;
     }
     /* Status box icons/text (might need to target specific internal classes) */
     .st-emotion-cache-vj1l9j .stAlert {
        background-color: transparent !important; /* Don't want alert backgrounds inside status */
     }


    /* Info/Success/Error/Warning boxes */
    .stAlert {
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        padding: 1rem;
        font-size: 1rem;
        border-left: 5px solid transparent; /* Base style */
    }
    .stAlert.stAlert-info { border-left-color: #569CD6; background-color: #2A3E52; color: #9CDCFE; } /* Dark blue info */
    .stAlert.stAlert-success { border-left-color: #4EC9B0; background-color: #28403A; color: #7AC7A3; } /* Dark teal success */
    .stAlert.stAlert-warning { border-left-color: #DCDCAA; background-color: #454032; color: #FFDAA6; } /* Dark yellow warning */
    .stAlert.stAlert-error { border-left-color: #F44747; background-color: #4A3030; color: #F48787; } /* Dark red error */


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
    .output-text-area textarea {
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
        color: #4EC9B0; /* Teal for "Consensus Answer" label */
    }
    .consensus-answer div {
        color: #D4D4D4; /* Ensure the answer text is light */
    }


</style>
""", unsafe_allow_html=True)


# --- GPU Telemetry Setup ---
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# Use st.empty to hold the telemetry status text, defined *outside* cached functions
telemetry_placeholder = st.empty()

def update_telemetry():
    """Updates the telemetry display in the dedicated placeholder."""
    telemetry_text = "[Checking System Status...]"
    if not GPU_AVAILABLE or not torch.cuda.is_available():
        telemetry_text = "📊 System Status: [No GPU Available]"
    else:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem_used_mb = m.used // 1024**2
            mem_total_mb = m.total // 1024**2
            telemetry_text = f"📊 System Status: GPU {u.gpu}% | Mem {mem_used_mb}/{mem_total_mb} MB"
        except Exception:
             telemetry_text = "📊 System Status: [Telemetry Error]"

    # Use markdown with a custom class for styling
    telemetry_placeholder.markdown(f'<div class="telemetry-box">{telemetry_text}</div>', unsafe_allow_html=True)


# Initial telemetry update when the script starts
update_telemetry()


# --- Caching Model Loading (Core Logic Only) ---
# Use st.cache_resource for heavy objects like models and tokenizers.
# This function MUST NOT call Streamlit elements that affect the layout
# or state outside of its own scope.
@st.cache_resource(show_spinner=False) # Spinner handled manually
def _load_model_and_tokenizer_cached(model_name: str, device: str, forced_model_type: str = None):
    """
    Loads the model and tokenizer. This function is cached and should
    contain minimal Streamlit calls to avoid caching issues.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True)
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
    detected_type = "Seq2Seq" if is_encoder_decoder else "Causal"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure padding token is set for generation robustness
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
             tokenizer.pad_token = tokenizer.eos_token
        else:
             # Fallback - adding tokens might require resizing model embeddings
             # which is complex and model-dependent. This is a basic attempt.
             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             tokenizer.pad_token = '[PAD]' # Set the attribute
             # Attempt to get the new pad token ID - may not work for all tokenizers
             try:
                 tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
             except Exception:
                 tokenizer.pad_token_id = None # Indicate failure to get ID

    # Determine the model class based on detection or forced selection
    actual_model_type = forced_model_type if forced_model_type != "Auto" else detected_type

    if actual_model_type == "Seq2Seq":
         model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config, trust_remote_code=True)
    elif actual_model_type == "Causal":
         model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
    else:
         raise ValueError(f"Unsupported model type selected: {actual_model_type}. Please select 'Auto', 'Causal', or 'Seq2Seq'.")

    model.to(device)
    model.eval() # Crucial for consistent inference behavior and disabling dropout etc.

    # Ensure return_dict_in_generate is True for structured outputs
    if not getattr(model.config, 'return_dict_in_generate', False):
         model.config.return_dict_in_generate = True

    return model, tokenizer, actual_model_type

# --- Wrapper function to handle status reporting for cached loading ---
def safe_load_model_with_status(model_name: str, device: str, forced_model_type: str = None):
    """
    Calls the cached loading function and handles Streamlit status updates.
    """
    status_text = f"🌐 Loading model '{model_name}' on device '{device}'..."
    # Use st.status here, defined outside the cached function
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

            # Report padding token status if available
            if tokenizer and tokenizer.pad_token_id is None:
                 status_box.warning(f"Tokenizer has no pad_token_id. Generation might fail for models requiring padding (e.g., batching).")
            elif tokenizer:
                 status_box.write(f"Tokenizer pad_token_id set to {tokenizer.pad_token_id}.")


            status_box.success(f"✅ Model '{model_name}' ({actual_model_type}) loaded successfully on '{device}'.")
            update_telemetry() # Final telemetry update after success
            return model, tokenizer, actual_model_type

        except Exception as e:
            status_box.error(f"❌ Model loading failed.")
            update_telemetry() # Final telemetry update after error
            st.exception(e) # Display the full exception traceback
            # Clean up resources in case of failure before returning None
            # These are manual attempts; cache handles cleanup on its own state changes
            # but explicit cleanup is good practice on error paths.
            try:
                if 'model' in locals() and model is not None: del model
            except NameError: pass
            try:
                if 'tokenizer' in locals() and tokenizer is not None: del tokenizer
            except NameError: pass
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            return None, None, None # Return None on failure


# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Core Settings")
    st.markdown("Configure the foundational aspects of the NeuroReasoner.")

    with st.expander("🧠 Model Configuration", expanded=True):
        model_name = st.text_input(
            "Hugging Face Model ID or Path",
            "ayjays132/NeuroReasoner-1-NR-1",
            help="Enter the model ID from huggingface.co or a local path."
        )

        # --- Dynamic Model Type Detection ---
        detected_type = "Unknown (Enter Model ID)"
        # Options match the strings used in the loading function
        model_type_options = ["Auto", "Causal", "Seq2Seq"]
        default_model_type_index = model_type_options.index("Auto")

        # Attempt to load config to detect type without caching (lightweight check)
        try:
            if model_name and model_name.strip(): # Only attempt if input is not empty
                initial_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True)
                is_encoder_decoder_initial = getattr(initial_config, "is_encoder_decoder", False)
                detected_type = "Seq2Seq" if is_encoder_decoder_initial else "Causal"
            else:
                 detected_type = "Unknown (Enter Model ID)"
        except Exception:
            detected_type = "Unknown (Config Load Error)" # Indicate config load itself failed

        forced_model_type = st.selectbox(
            "Architecture Type",
            model_type_options,
            index=default_model_type_index,
            help=f"Detected: {detected_type}. 'Auto' uses the detected type. Select manually if detection is incorrect or overridden."
        )

        # --- Device Selection ---
        available_devices = ["cpu"]
        if torch.cuda.is_available():
             available_devices.insert(0, "cuda") # Put cuda first if available

        device = st.selectbox(
            "Device",
            available_devices,
            help="Select the hardware device for computation (GPU recommended)."
        )

        st.markdown("""
             <small>💡 Changing model settings requires reloading the model.</small>
        """, unsafe_allow_html=True)


    st.markdown("---") # Visual separator

    st.header("✨ Generation Parameters")
    st.markdown("Define how the AI generates reasoning steps and answers.")

    with st.expander("Basic Parameters", expanded=True):
        # Finalized 'Number of Reasoning Chains' parameter
        num_chains = st.slider(
            "Number of Reasoning Chains",
            min_value=1,
            max_value=15, # Kept the higher max for more robustness
            value=5,      # Kept the default of 5
            help="How many independent reasoning chains to generate for analyzing the problem. More chains can improve Self-Consistency but take longer."
        )

        # Finalized 'No-repeat Ngram Size' parameter
        no_repeat_ngram_size = st.slider( # Using the standard name for GenerationConfig
            "No-repeat Ngram Size",
            min_value=0,
            max_value=10,
            value=3,
            help="Avoids generating repeating sequences of N tokens. Set to 0 to disable."
        )

        # Self-Consistency checkbox remains
        self_consistency = st.checkbox(
            "Enable Self-Consistency Voting",
            value=True,
            help="When enabled, the system generates multiple chains and identifies the most common final answer as the consensus. Requires 'Number of Reasoning Chains' > 1."
        )

        # Conditional warning if Self-Consistency is on but num_chains is 1
        if self_consistency and num_chains <= 1:
             st.warning("Self-Consistency is most effective with 2 or more chains.") # Slightly rephrased warning


    # Advanced Parameters
    with st.expander("🧪 Advanced Sampling Parameters"):
        max_new_tokens = st.slider(
            "Max Tokens per Chain",
            50, 2048, 768,
            help="Maximum number of new tokens to generate for *each* individual reasoning chain. Adjust based on complexity expected."
        )
        temperature = st.slider(
            "Temperature",
            0.0, 2.0, 0.8,
            help="Controls the randomness of sampling. 0.0 is deterministic (greedy). Higher values increase diversity."
        )
        top_k = st.slider(
            "Top-k",
            0, 100, 50,
            help="Filter to consider only the top_k most likely tokens at each step (0 disables). Used with sampling."
        )
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            0.0, 1.0, 0.95,
            help="Filter to consider tokens with cumulative probability below top_p (0.0 disables). Used with sampling."
        )
        do_sample = st.checkbox(
            "Enable Sampling",
            value=True,
            help="If checked, uses probabilistic sampling (controlled by Temperature, Top-k, Top-p). If unchecked, uses greedy decoding."
        )
        if not do_sample:
             st.info("Sampling disabled. Temperature, Top-k, and Top-p will be ignored.")

        no_repeat_ngram_size = st.slider(
            "No-repeat Ngram Size",
            0, 10, 3,
            help="Avoids repeating sequences of N tokens. Set to 0 to disable."
        )
        # Optional: Add a seed for reproducibility if desired
        # generation_seed = st.number_input("Generation Seed (Optional)", value=-1, help="Set a positive integer for reproducible generation.")


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
            height=150,
            placeholder="Example: If a train travels at 60 mph and a car at 40 mph, starting at the same time from cities 300 miles apart, how long until they meet? Think step-by-step.",
            key="user_prompt" # Added key for stability
        )

    with button_col:
        # Add some vertical space to align the button nicely
        st.markdown("<div style='height: 3.5rem;'></div>", unsafe_allow_html=True)
        run_button = st.button("✨ Generate Reasoning", use_container_width=True, key="generate_button") # Added key

# Container for status updates and results
results_container = st.container()


# --- Generation Logic Trigger ---
if run_button:
    if not prompt or not prompt.strip():
        results_container.warning("Please enter a prompt to begin generation.")
        st.stop() # Stop execution until prompt is entered

    # --- Prepare for Generation ---
    # Load model and tokenizer (handles caching internally with st.cache_resource
    # via safe_load_model_with_status which also reports status)
    # This happens only when the button is clicked and parameters might have changed
    model, tokenizer, loaded_model_type = safe_load_model_with_status(model_name, device, forced_model_type)

    if model is None or tokenizer is None:
        # Error was already shown by safe_load_model_with_status
        st.error("Model or tokenizer failed to load. Please check settings and traceback above.")
        st.stop() # Stop if loading failed


    # --- Configure Generation ---
    # Use a status box for ongoing generation process
    with results_container:
        st.markdown("---") # Separator before results
        generation_status = st.status("Preparing generation config...", expanded=True)
        update_telemetry() # Update telemetry while status is active


    try:
        # Build GenerationConfig based on sidebar parameters
        # num_return_sequences should match num_chains for the wrapper to process them
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_chains, # <--- Corrected: Use num_chains directly
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )
        generation_status.write(f"Generation parameters set: {gen_cfg.to_dict()}")
        update_telemetry()

        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_chains,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    except Exception as e:
        generation_status.error(f"❌ Failed to create GenerationConfig: {e}")
        st.exception(e)
        st.stop()

    # --- Instantiate Wrapper ---
    try:
        generation_status.write("Initializing Chain-of-Thought wrapper...")
        # Pass the configured generation config to the wrapper
        # The wrapper should internally use num_return_sequences from gen_cfg
        cot_wrapper = ChainOfThoughtWrapper(
            model=model,
            tokenizer=tokenizer,
            generation_config=cfg,
            device=device,
            self_consistency=self_consistency,
            consistency_rounds=(num_chains if self_consistency else 1)
        )
        generation_status.write("Wrapper initialized.")
        update_telemetry()

    except Exception as e:
        generation_status.error(f"❌ Failed to initialize CoT wrapper: {e}")
        st.exception(e)
        st.stop()

    # --- Tokenize Input ---
    try:
        generation_status.write("Tokenizing input prompt...")
        # Use model_max_length or a reasonable cap for input length
        max_input_length = tokenizer.model_max_length
        if max_input_length is None or max_input_length > 4096: # Cap input length if tokenizer reports None or very large
             max_input_length = 4096
             if tokenizer.model_max_length is None:
                  generation_status.warning(f"Tokenizer has no model_max_length, capping input to {max_input_length}.")


        enc = tokenizer(
            prompt,
            return_tensors='pt',
            padding='longest', # Pad to the longest sequence in the batch (batch size is 1 here)
            truncation=True,
            max_length=max_input_length, # Use a proper max length for the input
        ).to(device)
        generation_status.write(f"Input token length: {enc['input_ids'].shape[1]}")
        update_telemetry()

    except Exception as e:
        generation_status.error(f"❌ Tokenization failed: {e}")
        st.exception(e)
        st.stop()

    # --- Generate ---
    generation_status.update(label=f"⏳ Generating {num_chains} reasoning chains...", state="running")
    start_time = time.time()

    try:
        # Call the wrapper's generate method
        # It should handle the loop for multiple chains and self-consistency internally
        outputs = cot_wrapper.generate(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            # Pass any other necessary arguments to your wrapper's generate method
        )
        # Expected `outputs` dict structure: {'full_texts': [...], 'reasoning_steps': [...], 'final_answers': [...], 'consensus_answer': '...'}
        # The wrapper should handle extracting steps/answers if needed.

    except Exception as e:
        generation_status.error(f"❌ Generation failed: {e}")
        st.exception(e)
        # Clean up resources after potential OOM or other errors
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect() # Python garbage collection
        st.stop()

    elapsed_time = time.time() - start_time
    generation_status.update(label=f"✨ Generation complete in {elapsed_time:.2f}s", state="complete")
    update_telemetry() # Final telemetry update after successful generation

    # --- Display Results ---
    with results_container:
        st.markdown("## 📚 Reasoning Output")

        # Display Self-Consistency Consensus first if enabled and results are available
        if self_consistency and outputs and 'consensus_answer' in outputs and outputs.get('final_answers'):
            consensus = outputs.get('consensus_answer')
            answers = outputs.get('final_answers', [])

            st.markdown('<div class="consensus-answer">', unsafe_allow_html=True)
            st.write("💡 **Consensus Answer (Self-Consistency):**")
            st.write(consensus if consensus else "[Could not determine consensus]")
            st.markdown('</div>', unsafe_allow_html=True)

            if answers and len(answers) > 1: # Only show distribution if more than one answer was found
                st.markdown("###### Answer Distribution:")
                answer_counts = Counter(answers)
                # Display sorted distribution
                for ans, count in answer_counts.most_common():
                     st.write(f"- '{ans}' ({count} {'vote' if count == 1 else 'votes'})")
            st.markdown("---") # Separator


        # Display individual chains
        full_texts = outputs.get('full_texts', [])
        reasoning_steps = outputs.get('reasoning_steps', [])
        final_answers = outputs.get('final_answers', [])

        if not full_texts:
            st.warning("No reasoning chains were generated.")
        else:
            st.markdown(f"### Individual Chains ({len(full_texts)} generated)")
            # Iterate and display each chain in an expander
            # Ensure lists are iterable, even if empty
            full_texts = full_texts if isinstance(full_texts, list) else []
            reasoning_steps = reasoning_steps if isinstance(reasoning_steps, list) else []
            final_answers = final_answers if isinstance(final_answers, list) else []

            # Pad lists to the same length in case the wrapper returned inconsistent outputs
            max_len_outputs = max(len(full_texts), len(reasoning_steps), len(final_answers))
            full_texts.extend(["[N/A - Generation Failed for this chain]"] * (max_len_outputs - len(full_texts)))
            reasoning_steps.extend([[]] * (max_len_outputs - len(reasoning_steps)))
            final_answers.extend(["[N/A]"] * (max_len_outputs - len(final_answers)))


            for idx, (text, steps, ans) in enumerate(zip(full_texts, reasoning_steps, final_answers), 1):
                # Use try-except just in case a single chain output is malformed
                try:
                    # Expander for each chain, starting collapsed
                    with st.expander(f"Chain {idx}", expanded=False):
                        # Use custom class for styling the label
                        st.markdown('<div class="output-label">Full Generated Text:</div>', unsafe_allow_html=True)
                        # Use custom class for styling the text area background
                        st.text_area(f"chain_text_area_{idx}", text, height=250, label_visibility="collapsed", help="The complete generated output for this chain.")

                        if steps and isinstance(steps, list):
                            st.markdown('<div class="output-label">Reasoning Steps:</div>', unsafe_allow_html=True)
                            # Display steps as a list
                            if steps:
                                for i, step in enumerate(steps, 1):
                                     if isinstance(step, str) and step.strip():
                                          st.write(f"**Step {i}:** {step.strip()}")
                                     elif not isinstance(step, str):
                                          st.warning(f"Step {i} has invalid format.")
                            else:
                                 st.info("No specific steps were extracted for this chain.")


                        st.markdown('<div class="output-label">Final Answer:</div>', unsafe_allow_html=True)
                        st.write(f"**{ans if ans else '[No answer extracted]'}**")

                        # Optional: Add a separator between chain sections
                        st.markdown("---", help="End of Chain details.")

                except Exception as chain_e:
                    st.error(f"Error displaying Chain {idx}: {chain_e}")
                    st.exception(chain_e)


        st.markdown("---") # Final separator
        st.info("Generation process concluded. Review the chains above.")

    # Clean up GPU memory after generation is complete and results are displayed
    if torch.cuda.is_available():
         torch.cuda.empty_cache()
    gc.collect() # Python garbage collection