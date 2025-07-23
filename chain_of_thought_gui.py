#!/usr/bin/env python3
"""
NeuroReasoner Chain-of-Thought GUI (Final Version)
-----------------------------------------------
A highly refined and premium Streamlit app for exploring
step-by-step reasoning across compatible Hugging Face models,
with enhanced model loading logic, functional multimodal input support,
fixed Streamlit state issues, and preserving all previous features.

Featuring an advanced, futuristic, layered dark theme, dynamic model-type detection,
enhanced self-consistency sampling & voting display, robust handling with improved feedback,
detailed Chain-of-Thought output presentation including code rendering,
integrated GPU telemetry, and accessibility improvements.

This version prioritizes loading specific model types (CausalLM, Seq2SeqLM)
for better generation compatibility, falls back to AutoModel, adds a check
for generation capability after loading, includes a minor fix for telemetry,
resolves the Streamlit expander nesting issue by restructuring output display,
ensures full output display, and incorporates extensive futuristic styling.
"""
import time
import re  # Needed for answer normalization
import streamlit as st
import torch
import pynvml  # For GPU telemetry
import logging  # Import logging

from transformers import (
    AutoConfig,
    # We will explicitly try CausalLM and Seq2SeqLM first
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    # Keep AutoModel for fallback
    AutoModel,
    AutoProcessor,  # Needed for multimodal models that combine tokenizer and image processor
    # Import Vision-to-Sequence models if AutoModel isn't sufficient for a specific type
    # from transformers import AutoModelForVision2Seq # Can be used for specific types if AutoModel fails
    # Import GenerationMixin to check for generate capability
    GenerationMixin,
)
from collections import Counter  # For self-consistency voting
import gc  # Import garbage collector
from typing import Dict, List, Optional, Tuple
# No need for markdown library, st.markdown is used.
from PIL import Image # Needed for handling uploaded images
import io # Needed for handling image bytes


# Configure logging for the GUI
logger = logging.getLogger(__name__)
# Ensure logger doesn't add handlers multiple times if the script is imported repeatedly
if not logger.root.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent logs from going to root logger if root also has handlers
logger.setLevel(logging.INFO) # Set default level to INFO


# --- Import the Enhanced ChainOfThoughtWrapper ---
# Assuming chain_of_thought_wrapper.py is in the same directory.
# The wrapper now needs to accept the processor (tokenizer + image processor).
try:
    # Ensure the wrapper is imported correctly
    from chain_of_thought_wrapper import ChainOfThoughtWrapper, validate_device_selection
    logger.info("Successfully imported ChainOfThoughtWrapper.")
except ImportError:
    logger.error("🚨 Fatal Error: `chain_of_thought_wrapper.py` not found. Please ensure the enhanced wrapper script is in the same directory.")
    st.error("🚨 Fatal Error: The `chain_of_thought_wrapper.py` script was not found. Please ensure it's in the same directory as the GUI script.")
    st.stop() # Halt execution if the wrapper is not found


# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 NeuroReasoner CoT (Multimodal)", # Updated title
    page_icon="🧠",
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded", # Sidebar open by default
    menu_items={
        'Get Help': 'https://github.com/ayjays132/NeuroReasoner', # Example repo link - UPDATE THIS TO YOUR ACTUAL REPO
        'Report a bug': "https://github.com/ayjays132/NeuroReasoner/issues", # Example repo issues link - UPDATE THIS
        'About': """
        **NeuroReasoner Chain-of-Thought GUI (Final Version)**
        Version 2.4 (Futuristic Styling & Output Fix)

        An open-source interface for exploring Chain-of-Thought reasoning with Hugging Face models,
        now with enhanced model loading logic, functional multimodal (image + text) input support,
        fixed Streamlit state issues, and preserving all previous features.

        \n\n**Key Features:**
        - **Futuristic, Layered, Magical Dark Theme**
        - Real-time GPU Telemetry
        - Efficient Model Caching (including processor)
        - Enhanced Model Loading Strategy (Prioritizes CausalLM/Seq2SeqLM)
        - Self-Consistency Voting
        - Detailed CoT Output Visualization (Markdown/Code support, **Full Output Display Fixed**)
        - Robust Generation Controls
        - Accessibility improvements
        - Functional Multimodal (Image + Text) Input (Requires compatible model)
        - Integrated AGI Helper Module Interactions (Conceptual)
        - Corrected Streamlit Session State Handling
        - **Fixed Expander Nesting Issue**

        **Note on Multimodal Models:** While this version supports multimodal input, you must load
        a model from Hugging Face that is designed for multimodal tasks (e.g., LLaVA, InstructBLIP, SmolVLM)
        and is compatible with Hugging Face's `AutoProcessor` and model classes designed for generation.
        Loading a text-only model will disable the image processing logic for that session. Loading
        a model that does not have a language modeling head (like the base `GPT2Model` instead of
        `GPT2LMHeadModel`) will result in a generation error. This version attempts to load
        generation-compatible classes more robustly.

        Developed by [Your Name/Org Here - Optional]
        Built with Streamlit, Hugging Face Transformers, and pynvml.
        """
    }
)

# --- Theme Selection ---
with st.sidebar:
    theme_choice = st.selectbox(
        "🎨 Theme",
        ["Dark", "Light", "Premium", "Alien"],
        index=0,
        key="theme_select",
    )
    auto_scroll = st.checkbox("Auto-scroll to latest", value=True, key="auto_scroll")

LIGHT_THEME_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    color: #111 !important;
}
.stApp {
    background: linear-gradient(160deg, #FFFFFF 0%, #F0F0F0 50%, #FFFFFF 100%) !important;
    color: #111 !important;
}
.stSidebar {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F5F5 50%, #EAEAEA 100%) !important;
    color: #111 !important;
    border-right: 1px solid #CCCCCC !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #0A84FF !important;
}
.stButton>button {
    background: linear-gradient(180deg, #0A84FF 0%, #0066CC 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #0A84FF !important;
}
.stButton>button:hover {
    background: linear-gradient(180deg, #4DA3FF 0%, #0A84FF 100%) !important;
}
</style>
"""

PREMIUM_THEME_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    color: #F0F0F0 !important;
}
.stApp {
    background: radial-gradient(circle at 20% 20%, #1b1b3a 0%, #0b0b20 100%) !important;
    color: #F0F0F0 !important;
}
.stSidebar {
    background: linear-gradient(180deg, #151525 0%, #202040 100%) !important;
    border-right: 1px solid #444 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #9E84FF !important;
}
.stButton>button {
    background: linear-gradient(180deg, #9E84FF 0%, #6A5ACD 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #6A5ACD !important;
}
.stButton>button:hover {
    background: linear-gradient(180deg, #B39BFF 0%, #9E84FF 100%) !important;
}
</style>
"""

ALIEN_THEME_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    color: #d0fffc !important;
}
.stApp {
    background: radial-gradient(circle at 30% 30%, #002b36 0%, #001014 100%) !important;
    color: #d0fffc !important;
}
.stSidebar {
    background: linear-gradient(180deg, #00181e 0%, #002b36 100%) !important;
    border-right: 1px solid #005f6b !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #2be4d4 !important;
}
.stButton>button {
    background: linear-gradient(180deg, #2be4d4 0%, #008b94 100%) !important;
    color: #001014 !important;
    border: 1px solid #008b94 !important;
}
.stButton>button:hover {
    background: linear-gradient(180deg, #5ffbf1 0%, #2be4d4 100%) !important;
}
</style>
"""

# --- Extensive Futuristic and Layered CSS Styling ---
st.markdown("""
<style>
    /* Basic Reset and Body/App Styling */
    html, body, [data-testid="stAppViewContainer"] {
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
        color: #EDF0F5 !important; /* Refined off-white text */
        line-height: 1.62 !important;
        letter-spacing: 0.018rem !important;
    }

    .stApp {
        background: linear-gradient(160deg, #050508 0%, #0A0A10 50%, #0D0D15 100%); /* More complex gradient */
        color: #EDF0F5; /* Refined off-white text with slight blue tint */
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
        padding: 0;
        margin: 0;
        letter-spacing: 0.018rem; /* Enhanced letter spacing */
        line-height: 1.62;
    }
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%230E1E3A' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        pointer-events: none;
    }

    /* Main Content Area */
    .stApp > header:first-child,
    .stApp > div:has(div[data-testid="stSidebarContent"]) + div,
    .stApp > div:first-child > div:has(div[data-testid="stVerticalBlock"]) > div:first-child {
        padding: 2rem 3rem 3.5rem 3rem !important; /* Expanded padding */
        position: relative !important;
        z-index: 1 !important; /* Above background pattern */
        backdrop-filter: blur(3px) !important; /* Subtle blur for depth */
    }

    /* Hide Streamlit Header */
    .stAppHeader {
        display: none !important;
    }

    /* Futuristic Sidebar */
    .stSidebar {
        background: linear-gradient(180deg, #0D0D15 0%, #101020 50%, #131325 100%) !important; /* Darker, more intense gradient */
        padding: 2.5rem 1.75rem !important;
        border-right: 1px solid #1F1F33 !important; /* Stronger border */
        position: relative !important;
        color: #E9EDF0 !important;
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.8) !important; /* Deeper shadow */
        z-index: 10 !important; /* Ensure sidebar is on top */
    }
    /* Glowing edge for sidebar */
    .stSidebar::before {
        content: "";
        position: absolute;
        top: 0;
        bottom: 0;
        right: 0;
        width: 2px; /* Wider glow edge */
        background: linear-gradient(to bottom,
            transparent,
            rgba(0, 180, 255, 0.4) 20%, /* Brighter blue glow */
            rgba(0, 180, 255, 0.7) 50%,
            rgba(0, 180, 255, 0.4) 80%,
            transparent) !important;
        filter: blur(1px); /* Soften the glow */
    }
    /* Subtle radial highlight in sidebar */
    .stSidebar::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at top left, rgba(0, 180, 255, 0.04), transparent 70%) !important;
        pointer-events: none;
    }
    /* Sidebar Headers */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
        color: #00BFFF !important; /* Brighter, more vibrant blue */
        margin-top: 0.75rem !important;
        margin-bottom: 1.25rem !important;
        position: relative !important;
        padding-bottom: 0.75rem !important;
        text-shadow: 0px 0px 5px rgba(0, 180, 255, 0.3) !important; /* More prominent text glow */
    }
    /* Sidebar Header Underlines */
    .stSidebar h1::after, .stSidebar h2::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: 0;
        width: 60%; /* Wider underline */
        height: 3px; /* Thicker underline */
        background: linear-gradient(90deg, #00BFFF, transparent) !important;
        box-shadow: 0px 0px 6px rgba(0, 180, 255, 0.5) !important; /* Stronger glowing underline */
    }
     .stSidebar h3::after, .stSidebar h4::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: 0;
        width: 40%; /* Wider underline */
        height: 2px; /* Thicker underline */
        background: linear-gradient(90deg, #00BFFF, transparent) !important;
        box-shadow: 0px 0px 4px rgba(0, 180, 255, 0.3) !important; /* Stronger glowing underline */
    }
    /* Sidebar Labels */
    .stSidebar label {
        color: #8BE9FD !important; /* Cyan-like color */
        font-weight: normal !important;
        margin-bottom: 0.75rem !important;
        display: block !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02rem !important;
    }
     .stSidebar .stSelectbox > label,
    .stSidebar .stSlider > label,
    .stSidebar .stCheckbox > label,
    .stSidebar .stTextInput > label,
    .stSidebar .stFileUploader > label {
        font-weight: bold !important;
        color: #50FA7B !important; /* Green */
        text-shadow: 0px 0px 3px rgba(80, 250, 123, 0.4) !important; /* Enhanced glow */
        position: relative !important;
        padding-left: 0.75rem !important;
        margin-top: 1.5rem !important; /* Space above controls */
        margin-bottom: 0.5rem !important;
    }
    /* Vertical accent bar for sidebar labels */
    .stSidebar .stSelectbox > label::before,
    .stSidebar .stSlider > label::before,
    .stSidebar .stCheckbox > label::before,
    .stSidebar .stTextInput > label::before,
     .stSidebar .stFileUploader > label::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.1rem;
        bottom: 0.1rem;
        width: 4px; /* Thicker bar */
        border-radius: 2px;
        background: linear-gradient(to bottom, #50FA7B, rgba(80, 250, 123, 0.4)) !important; /* Gradient vertical bar */
        box-shadow: 0px 0px 4px rgba(80, 250, 123, 0.4) !important; /* Glowing accent */
    }


    /* Futuristic Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #8BE9FD !important; /* Cyan-like color */
        margin-top: 2rem !important;
        margin-bottom: 1.25rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: none !important;
        letter-spacing: -0.015rem !important;
        position: relative !important;
        text-shadow: 0px 0px 3px rgba(139, 233, 253, 0.2) !important; /* Subtle text glow */
    }
    h1 {
        font-size: 3rem !important; /* Larger main title */
        color: #FF79C6 !important; /* Pink */
        padding-bottom: 1rem !important;
        margin-bottom: 2.5rem !important;
        text-shadow: 0px 1px 6px rgba(255, 121, 198, 0.3) !important; /* Enhanced text glow */
    }
    /* Main Title Underline/Accent */
    h1::before {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 4px; /* Thicker line */
        background: linear-gradient(to right, #FF79C6, #BD93F9, transparent 95%) !important; /* Pink to purple gradient */
        border-radius: 3px !important;
        box-shadow: 0px 0px 8px rgba(255, 121, 198, 0.5) !important; /* Glowing border */
    }
    h1::after {
        content: "";
        position: absolute;
        bottom: -10px; /* Further below */
        left: 0;
        width: 50%; /* Wider secondary line */
        height: 2px; /* Thicker secondary line */
        background: linear-gradient(to right, rgba(255, 121, 198, 0.6), transparent) !important;
        border-radius: 1px !important;
    }
    h2 {
        font-size: 2.4rem !important; /* Larger secondary header */
        background: linear-gradient(90deg, #50FA7B, #8BE9FD) !important; /* Green to Cyan gradient */
        -webkit-background-clip: text !important;
        background-clip: text !important;
        color: transparent !important;
        margin-top: 3.5rem !important; /* More space above */
        border-bottom: 1px solid #3A3A4C !important; /* Darker separator */
        padding-bottom: 1rem !important; /* More padding */
        margin-bottom: 2.25rem !important; /* More space below */
    }
    /* H2 Underline/Accent */
    h2::before {
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 160px; /* Wider accent */
        height: 3px; /* Thicker accent */
        background: linear-gradient(90deg, #50FA7B, rgba(80, 250, 123, 0.3)) !important;
        border-radius: 2px !important;
        box-shadow: 0px 0px 5px rgba(80, 250, 123, 0.4) !important; /* Glowing accent */
    }
     h2::after {
        content: "";
        position: absolute;
        bottom: -5px; /* Further below */
        left: 0;
        width: 80px; /* Wider secondary */
        height: 1px;
        background: linear-gradient(90deg, rgba(80, 250, 123, 0.5), transparent) !important;
    }
    h3 {
        font-size: 2rem !important; /* Larger h3 */
        color: #BD93F9 !important; /* Purple */
        margin-top: 2.5rem !important;
        margin-bottom: 1.75rem !important;
        position: relative !important;
        padding-left: 1.5rem !important; /* More padding */
        text-shadow: 0px 0px 3px rgba(189, 147, 249, 0.3) !important; /* Enhanced glow */
    }
    /* H3 Vertical Bar Accent */
    h3::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.1rem;
        bottom: 0.1rem;
        width: 5px; /* Thicker bar */
        background: linear-gradient(to bottom, #BD93F9, rgba(189, 147, 249, 0.4)) !important; /* Gradient vertical bar */
        border-radius: 3px !important;
        box-shadow: 0px 0px 5px rgba(189, 147, 249, 0.4) !important; /* Glowing accent */
    }
     h4 {
        font-size: 1.6rem !important; /* Larger h4 */
        background: linear-gradient(90deg, #FFB86C, #FF79C6) !important; /* Orange to Pink gradient */
        -webkit-background-clip: text !important;
        background-clip: text !important;
        color: transparent !important;
        margin-top: 2rem !important;
        margin-bottom: 1.25rem !important;
        position: relative !important;
        padding-left: 1rem !important; /* More padding */
    }
     /* H4 Vertical Bar Accent */
     h4::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.2rem;
        bottom: 0.2rem;
        width: 4px; /* Thicker bar */
        background: linear-gradient(to bottom, #FFB86C, rgba(255, 184, 108, 0.4)) !important; /* Gradient vertical bar */
        border-radius: 2px !important;
     }
     h5 {
        font-size: 1.4rem !important;
        color: #D4D4E8 !important;
        margin-top: 1.8rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 1px dashed #4F4F68 !important; /* Stronger dashed border */
        padding-bottom: 0.5rem !important;
        position: relative !important;
        padding-left: 0.5rem !important;
     }
     h5::before {
         content: "◊"; /* Diamond accent */
         position: absolute;
         left: -0.5rem;
         top: 0.5rem;
         color: #BD93F9 !important; /* Purple accent */
         text-shadow: 0px 0px 3px rgba(189, 147, 249, 0.3) !important;
         font-size: 1em;
     }
     h6 {
        font-size: 1.2rem !important;
        color: #B0B0C0 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
        position: relative !important;
        padding-left: 0.3rem !important;
     }
      h6::before {
         content: "▪"; /* Square accent */
         position: absolute;
         left: -0.3rem;
         top: 0.4rem;
         color: #FFB86C !important; /* Orange accent */
         text-shadow: 0px 0px 2px rgba(255, 184, 108, 0.3) !important;
         font-size: 0.8em;
      }


    /* Futuristic Button */
    .stButton>button {
        background: linear-gradient(180deg, #50FA7B 0%, #48E772 50%, #40D067 100%) !important; /* Vibrant green gradient */
        color: #0A0A12 !important; /* Very dark text */
        border: 1px solid #3ACD5A !important;
        border-radius: 0.6rem !important; /* Slightly larger border radius */
        padding: 1rem 2.5rem !important; /* More padding */
        font-size: 1.1rem !important; /* Slightly larger font */
        font-weight: 700 !important; /* Bolder */
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important; /* Smoother easing */
        box-shadow:
            0px 3px 0px #2F994A, /* Deeper bottom shadow */
            0px 5px 15px rgba(0, 0, 0, 0.6), /* Stronger outer shadow */
            inset 0px 1px 0px rgba(255, 255, 255, 0.2) !important; /* Enhanced inner highlight */
        margin-top: 2.5rem !important;
        letter-spacing: 0.8px !important; /* More spacing */
        position: relative !important;
        overflow: hidden !important;
        text-shadow: 0px 1px 3px rgba(0, 0, 0, 0.3) !important; /* Deeper text shadow */
    }
    /* Button Top Highlight */
    .stButton>button::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(255, 255, 255, 0.15), transparent) !important;
        opacity: 0.8;
    }
    /* Button Hover State */
    .stButton>button:hover {
        background: linear-gradient(180deg, #69FF94 0%, #50FA7B 50%, #48E772 100%) !important; /* Brighter gradient */
        border-color: #50FA7B !important;
        transform: translateY(-3px) scale(1.02) !important; /* More pronounced lift */
        box-shadow:
            0px 5px 0px #3ACD5A, /* Match hover lift */
            0px 8px 18px rgba(0, 0, 0, 0.7), /* Stronger hover shadow */
            inset 0px 1px 0px rgba(255, 255, 255, 0.3) !important; /* Brighter inner highlight */
    }
    /* Button Active State */
    .stButton>button:active {
        background: linear-gradient(180deg, #3A8E51 0%, #40D067 50%, #40D067 100%) !important;
        border-color: #40D067 !important;
        transform: translateY(1px) scale(0.98) !important;
        box-shadow:
            0px 1px 0px #2F994A,
            0px 3px 8px rgba(0, 0, 0, 0.6),
            inset 0px 1px 4px rgba(0, 0, 0, 0.4) !important; /* Deeper inner shadow */
    }
     /* Button Shine Effect */
     .stButton>button::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        opacity: 0;
        transform: rotate(30deg);
        background: linear-gradient(
            to right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.4) 50%, /* Brighter shine */
            rgba(255, 255, 255, 0) 100%
        );
        transition: opacity 0.8s ease, transform 0.8s ease !important; /* Slower, smoother transition */
    }
    .stButton>button:hover::after {
        opacity: 1;
        transform: rotate(30deg) translate(60%, 60%) !important; /* More pronounced shine movement */
    }


    /* Futuristic Text Areas and Inputs */
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input,
    .stTextInput>div>div>input,
    .stSelectbox>div>div[data-baseweb="select"]>div,
    .stFileUploader>div>div {
        border: 1px solid #3F3F55 !important; /* Stronger border */
        border-radius: 0.6rem !important; /* Slightly larger radius */
        padding: 1.1rem 1.35rem !important; /* More padding */
        font-size: 1.05rem !important; /* Slightly larger font */
        background-color: #1A1A28 !important; /* Dark bluish background */
        color: #E9EDF0 !important;
        box-shadow:
            inset 1px 1px 10px rgba(0, 0, 0, 0.4), /* Deeper inner shadow */
            0px 1px 2px rgba(255, 255, 255, 0.05),
            0px 0px 0px 2px rgba(63, 63, 85, 0.6) !important; /* Wider outer glow */
        width: 100% !important;
        line-height: 1.6 !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        position: relative !important;
    }
     .stSelectbox>div>div[data-baseweb="select"]>div {
         padding: 0.8rem 1.35rem !important; /* Adjust padding for select */
     }
     .stFileUploader>div>div {
          padding: 1rem !important; /* Adjust padding for file uploader drop area */
          border: 1px dashed #556270 !important; /* Dashed border for uploader */
          background-color: #1A1A28 !important;
          box-shadow: inset 1px 1px 10px rgba(0, 0, 0, 0.4), 0px 2px 8px rgba(0, 0, 0, 0.3);
     }
     .stFileUploader>div>div>button {
          font-size: 0.95rem !important;
          padding: 0.6rem 1rem !important;
          margin-top: 1rem !important;
          background: linear-gradient(180deg, #BD93F9 0%, #B08FE5 100%) !important; /* Purple gradient */
          color: #0A0A12 !important;
          border: 1px solid #A989D2 !important;
          border-radius: 0.4rem !important;
          box-shadow: 0px 2px 0px #7A639B, 0px 4px 8px rgba(0, 0, 0, 0.4);
     }
      .stFileUploader>div>div>button:hover {
          background: linear-gradient(180deg, #D7AEFB 0%, #BD93F9 100%) !important;
          border-color: #BD93F9 !important;
          transform: translateY(-1px);
          box-shadow: 0px 3px 0px #7A639B, 0px 5px 10px rgba(0, 0, 0, 0.5);
     }
      .stFileUploader>div>div>button:active {
          transform: translateY(1px);
          box-shadow: 0px 1px 0px #7A639B, 0px 2px 5px rgba(0, 0, 0, 0.4);
     }


    /* Input/Select Top Edge Highlight */
    div[data-baseweb="textarea"]::after,
    div[data-baseweb="input"]::after,
    .stTextInput>div>div::after,
    .stSelectbox>div>div[data-baseweb="select"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px; /* Thicker highlight */
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent) !important;
        pointer-events: none;
    }
    /* Futuristic focus state with glow */
    div[data-baseweb="textarea"] textarea:focus,
    div[data-baseweb="input"] input:focus,
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div[data-baseweb="select"]:focus-within { /* Use :focus-within for selectbox */
        border-color: #00BFFF !important; /* Brighter blue glow color */
        background-color: #1F1F30 !important; /* Lighter when focused */
        box-shadow:
            inset 1px 1px 6px rgba(0, 0, 0, 0.3), /* Deeper inner shadow */
            0 0 0 3px rgba(0, 180, 255, 0.35), /* Wider, more intense outer glow */
            0 0 12px rgba(0, 180, 255, 0.2) !important; /* Ambient glow */
        outline: none !important;
    }

    /* Futuristic Status Box */
    [data-testid="stStatus"] {
        background: linear-gradient(135deg, #1A1A28, #28283D) !important; /* Richer gradient */
        border: 1px solid #40405A !important; /* Stronger border */
        border-radius: 0.7rem !important; /* Larger radius */
        padding: 1.8rem !important; /* More padding */
        margin-bottom: 2.5rem !important; /* More space below */
        box-shadow:
            3px 4px 20px rgba(0, 0, 0, 0.45), /* Stronger shadow */
            0px 1px 0px rgba(255, 255, 255, 0.08) inset,
            0px 0px 0px 2px rgba(64, 64, 90, 0.8) !important; /* Wider outer glow */
        position: relative !important;
        overflow: hidden !important;
    }
    /* Status Box Top Accent */
    [data-testid="stStatus"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px; /* Thicker accent */
        background: linear-gradient(90deg, #00BFFF, #FF79C6) !important; /* Blue to Pink gradient */
        border-top-left-radius: 0.7rem !important;
        border-top-right-radius: 0.7rem !important;
        box-shadow: 0px 0px 10px rgba(0, 180, 255, 0.6) !important; /* Stronger glow */
    }
    /* Status Box Internal Lighting */
    [data-testid="stStatus"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            radial-gradient(circle at top right, rgba(255, 121, 198, 0.06), transparent 70%), /* Pink highlight */
            linear-gradient(135deg, rgba(255, 255, 255, 0.04), transparent) !important; /* Glass effect */
        pointer-events: none;
    }
    [data-testid="stStatus"] .stMarkdown p,
    [data-testid="stStatus"] .stAlert {
        color: #D4D4E8 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0.3rem 0 !important; /* More padding */
        position: relative !important;
        z-index: 1 !important;
    }
    [data-testid="stStatus"] .stProgress {
        margin-top: 1.5rem !important; /* More space */
        margin-bottom: 1rem !important;
        position: relative !important;
    }
    [data-testid="stStatus"] .stProgress > div > div {
        background: linear-gradient(90deg, #00BFFF, #50FA7B) !important; /* Blue to Green gradient */
        box-shadow: 0px 0px 8px rgba(0, 180, 255, 0.5) !important; /* Glowing progress bar */
        height: 8px !important; /* Taller */
        border-radius: 4px !important;
    }
     [data-testid="stStatus"] .stProgress::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(20, 20, 30, 0.4) !important; /* Darker track */
        border-radius: 4px !important;
        z-index: -1;
     }

    /* Futuristic Alert Boxes */
    .stAlert {
        border-radius: 0.8rem !important; /* Larger radius */
        margin-bottom: 2.5rem !important; /* More space */
        padding: 1.6rem 1.8rem !important; /* More padding */
        font-size: 1.05rem !important;
        border-left: 6px solid transparent !important; /* Thicker border */
        color: #E9EDF0 !important;
        box-shadow:
            3px 4px 20px rgba(0, 0, 0, 0.5),
            0px 0px 0px 1px rgba(64, 64, 90, 0.6) !important; /* Subtle outer glow */
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(5px) !important; /* More blur for glass */
        background-color: rgba(30, 30, 45, 0.6) !important; /* Semi-transparent background */
    }
    /* Alert Glass Effect */
    .stAlert::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.08), transparent), /* Stronger highlight */
            radial-gradient(circle at top left, rgba(255, 255, 255, 0.05), transparent 70%) !important;
        pointer-events: none;
    }
    /* Alert Top Accent Line */
    .stAlert::after {
        content: "";
        position: absolute;
        top: 0;
        left: 6px; /* Account for border-left */
        width: calc(100% - 6px);
        height: 2px; /* Thicker line */
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.15), transparent) !important;
        pointer-events: none;
    }
    .stAlert.stAlert-info {
        border-left-color: #00BFFF !important; /* Brighter blue */
         background-color: rgba(15, 30, 45, 0.6) !important; /* Match border color */
    }
    .stAlert.stAlert-success {
        border-left-color: #50FA7B !important; /* Green */
        background-color: rgba(15, 40, 30, 0.6) !important; /* Match border color */
    }
    .stAlert.stAlert-warning {
        border-left-color: #FFB86C !important; /* Orange */
        background-color: rgba(40, 35, 15, 0.6) !important; /* Match border color */
    }
    .stAlert.stAlert-error {
        border-left-color: #F44747 !important; /* Red */
        background-color: rgba(45, 20, 25, 0.6) !important; /* Match border color */
    }


    /* Futuristic Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #1A1A28, #28283D) !important; /* Richer gradient */
        color: #E0E0EA !important;
        border-radius: 0.7rem !important; /* Larger radius */
        padding: 1.4rem 1.8rem !important; /* More padding */
        margin-top: 2rem !important; /* More space above */
        margin-bottom: 0.8rem !important; /* More space below */
        font-weight: 700 !important; /* Bolder */
        font-size: 1.2rem !important; /* Larger font */
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        box-shadow:
            0px 3px 10px rgba(0, 0, 0, 0.4),
            0px 0px 0px 2px rgba(64, 64, 90, 0.7) !important; /* Wider outer glow */
        position: relative !important;
        border: 1px solid #3F3F55 !important; /* Stronger border */
        overflow: hidden !important;
        backdrop-filter: blur(4px) !important; /* Glass effect */
    }
     /* Expander Header Glass Effect */
    .streamlit-expanderHeader::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.05), transparent),
            radial-gradient(circle at top right, rgba(189, 147, 249, 0.03), transparent 70%) !important; /* Purple highlight */
        pointer-events: none;
        opacity: 0.8;
        transition: opacity 0.3s ease !important;
    }
    /* Expander Header Hover State */
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #28283D, #3A3A4C) !important;
        color: #F0F0F5 !important;
        box-shadow:
            0px 4px 12px rgba(0, 0, 0, 0.5),
            0px 0px 0px 2px rgba(189, 147, 249, 0.3) !important; /* Purple glow on hover */
        transform: translateY(-2px) !important;
    }
    .streamlit-expanderHeader:hover::before {
        opacity: 1 !important;
    }
    /* Expander Header Arrow */
    .streamlit-expanderHeader::after {
        content: "";
        position: absolute;
        right: 1.8rem; /* More space */
        top: 50%;
        transform: translateY(-50%);
        width: 14px; /* Larger arrow */
        height: 14px; /* Larger arrow */
        border-right: 3px solid #BD93F9 !important; /* Thicker, purple arrow */
        border-bottom: 3px solid #BD93F9 !important; /* Thicker, purple arrow */
        transform-origin: 75% 75%;
        transition: transform 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55) !important; /* Bouncier animation */
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3) !important; /* Stronger shadow on arrow */
    }
    .streamlit-expanderHeader[aria-expanded="true"]::after {
        transform: translateY(-50%) rotate(45deg) scale(1.15) !important; /* Larger scale */
    }
    .streamlit-expanderHeader[aria-expanded="false"]::after {
        transform: translateY(-50%) rotate(-45deg) scale(1.15) !important;
    }

    /* Futuristic Expander Content */
    .streamlit-expanderContent {
        background: linear-gradient(135deg, #1A1A28, #232335) !important; /* Subtle gradient */
        border: 1px solid #3F3F55 !important;
        border-top: none !important;
        border-bottom-left-radius: 0.7rem !important;
        border-bottom-right-radius: 0.7rem !important;
        padding: 2.5rem !important; /* More padding */
        margin-top: 0 !important;
        color: #E9EDF0 !important;
        box-shadow:
            inset 0 3px 12px rgba(0, 0, 0, 0.3), /* Deeper inner shadow */
            0px 4px 12px rgba(0, 0, 0, 0.3) !important; /* Outer shadow */
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(4px) !important; /* Match header blur */
    }
    /* Expander Content Top Separator */
    .streamlit-expanderContent::before {
        content: "";
        position: absolute;
        top: 0;
        left: 5%;
        width: 90%;
        height: 2px; /* Thicker separator */
        background: linear-gradient(90deg, transparent, rgba(189, 147, 249, 0.4), transparent) !important; /* Purple gradient separator */
        box-shadow: 0px 0px 6px rgba(189, 147, 249, 0.3) !important; /* Glowing separator */
    }
    /* Expander Content Internal Lighting */
    .streamlit-expanderContent::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            radial-gradient(circle at bottom right, rgba(80, 250, 123, 0.04), transparent 70%), /* Green highlight */
            linear-gradient(135deg, rgba(255, 255, 255, 0.03), transparent) !important; /* Glass effect */
        pointer-events: none;
    }
     /* Text within expander content */
    .streamlit-expanderContent .stMarkdown p,
    .streamlit-expanderContent .stMarkdown li {
        color: #E9EDF0 !important;
        margin-bottom: 1rem !important; /* More space */
        line-height: 1.8 !important; /* Increased line height */
        letter-spacing: 0.01rem !important;
    }
    /* Futuristic code blocks inside expanders */
    .streamlit-expanderContent .stMarkdown pre code {
        background: linear-gradient(135deg, #0D0D15, #101020) !important; /* Darker gradient */
        border: 1px solid #3F3F55 !important;
        border-radius: 0.7rem !important; /* Larger radius */
        padding: 1.8rem !important; /* More padding */
        margin: 1.2rem 0 !important; /* More margin */
        overflow-x: auto !important;
        color: #BD93F9 !important; /* Purple code text */
        font-family: 'Fira Code', Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace !important; /* Use Fira Code if available */
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
        display: block !important;
        word-wrap: normal !important;
        white-space: pre-wrap !important;
        box-shadow:
            inset 0 0 20px rgba(0, 0, 0, 0.4) !important, /* Deeper inner shadow */
            0px 4px 15px rgba(0, 0, 0, 0.3) !important; /* Enhanced outer shadow */
        position: relative !important;
        backdrop-filter: blur(2px); /* Slight blur for code block */
    }
     .streamlit-expanderContent .stMarkdown pre code::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.02), transparent) !important,
            radial-gradient(circle at top right, rgba(189, 147, 249, 0.03), transparent 70%) !important; /* Purple highlight */
        pointer-events: none;
        border-radius: 0.7rem !important;
     }
    /* Futuristic inline code inside expanders */
    .streamlit-expanderContent .stMarkdown code {
        background: linear-gradient(135deg, #28283D, #3F3F55) !important; /* Gradient background */
        border-radius: 0.5rem !important; /* Larger radius */
        padding: 0.3em 0.6em !important; /* More padding */
        color: #FFB86C !important; /* Orange inline code */
        font-family: 'Fira Code', Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace !important;
        font-size: 0.95rem !important;
        border: 1px solid rgba(255, 184, 108, 0.3) !important; /* Subtle orange border */
        box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.3) !important; /* Subtle shadow */
    }

    /* Ultra Premium Telemetry Box */
    .telemetry-box {
        background: linear-gradient(135deg, #1F1F30, #2A2A40) !important; /* Richer gradient */
        border: 1px solid #4F4F68 !important; /* Stronger border */
        border-radius: 0.8rem !important; /* Larger radius */
        padding: 1.5rem !important; /* More padding */
        margin-top: 2.5rem !important; /* More space */
        font-size: 0.9rem !important;
        color: #B0B0C0 !important;
        text-align: center !important;
        box-shadow:
            3px 4px 20px rgba(0, 0, 0, 0.45),
            0px 0px 0px 2px rgba(79, 79, 104, 0.8) !important; /* Wider outer glow */
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(3px) !important; /* Glass effect */
    }
    /* Telemetry Box Glass Effect */
    .telemetry-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.04), transparent),
            radial-gradient(circle at bottom right, rgba(139, 233, 253, 0.03), transparent 70%) !important; /* Cyan highlight */
        pointer-events: none;
    }
    /* Telemetry Box Top Accent Line */
    .telemetry-box::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px; /* Thicker line */
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent) !important;
        pointer-events: none;
    }
    .telemetry-box strong {
        color: #F8F8F2 !important; /* Dracular foreground */
        font-weight: 700 !important; /* Bolder */
        letter-spacing: 0.03rem !important; /* More spacing */
        text-shadow: 0px 0px 2px rgba(248, 248, 242, 0.3) !important; /* Stronger text glow */
    }

    /* Ultra Premium Self-Consistency Consensus Styling */
    .consensus-answer { /* Use hyphenated class name */
        background: linear-gradient(135deg, #153323, #1A482C, #206038) !important; /* Richer green gradient */
        color: #A3E6CE !important;
        border: 1px solid #3E8E56 !important; /* Stronger border */
        border-radius: 1rem !important; /* Very large radius */
        padding: 2rem !important; /* More padding */
        margin-top: 3rem !important; /* More space */
        margin-bottom: 2.5rem !important;
        font-size: 1.4rem !important; /* Larger font */
        font-weight: 700 !important; /* Bolder */
        box-shadow:
            0 6px 25px rgba(0, 0, 0, 0.6), /* Stronger shadow */
            0 1px 0 rgba(255, 255, 255, 0.08) inset,
            0px 0px 0px 3px rgba(62, 142, 86, 0.8) !important; /* Wide green outer glow */
        display: flex !important;
        align-items: center !important;
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(6px) !important; /* More blur for glass */
        background-color: rgba(21, 51, 35, 0.7) !important; /* Semi-transparent green */
    }
    /* Consensus Answer Left Accent */
    .consensus-answer::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 8px; /* Wider accent */
        height: 100%;
        background: linear-gradient(to bottom, #69FF94, #3E8E56) !important; /* Brighter green gradient */
        box-shadow: 0px 0px 10px rgba(105, 255, 148, 0.6) !important; /* Stronger glowing accent */
        border-top-left-radius: 1rem !important;
        border-bottom-left-radius: 1rem !important;
    }
    /* Consensus Answer Glass Effect */
    .consensus-answer::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.1), transparent), /* Stronger highlight */
            radial-gradient(circle at bottom right, rgba(105, 255, 148, 0.05), transparent 70%) !important;
        pointer-events: none;
    }
    .consensus-answer strong {
        color: #8BE9FD !important; /* Cyan label */
        margin-right: 1.5rem !important; /* More space */
        flex-shrink: 0 !important;
        position: relative !important;
        padding-right: 1.5rem !important;
        text-shadow: 0px 0px 4px rgba(139, 233, 253, 0.4) !important; /* Stronger text glow */
    }
     /* Consensus Answer Label Separator */
    .consensus-answer strong::after {
        content: "";
        position: absolute;
        right: 0;
        top: 15%; /* Adjust position */
        height: 70%; /* Adjust height */
        width: 2px; /* Thicker separator */
        background: linear-gradient(to bottom, transparent, rgba(139, 233, 253, 0.8), transparent) !important;
        box-shadow: 0px 0px 4px rgba(139, 233, 253, 0.5) !important;
    }
    .consensus-answer p {
        color: #F8F8F2 !important; /* Dracular foreground */
        margin: 0 !important;
        padding: 0 !important;
        flex-grow: 1 !important;
        word-break: break-word !important;
        line-height: 1.8 !important;
        letter-spacing: 0.01rem !important;
    }


    /* Ultra Premium Vote Counts Styling */
    .vote-counts {
        background: linear-gradient(135deg, #1F1F30, #2A2A40) !important; /* Richer gradient */
        border: 1px solid #4F4F68 !important;
        border-radius: 0.8rem !important; /* Larger radius */
        padding: 1.8rem !important; /* More padding */
        margin-top: 2.5rem !important; /* More space */
        margin-bottom: 2.5rem !important;
        font-size: 1rem !important;
        box-shadow:
            3px 4px 20px rgba(0, 0, 0, 0.45),
            0px 0px 0px 2px rgba(79, 79, 104, 0.7) !important; /* Wider outer glow */
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(5px) !important; /* Glass effect */
         background-color: rgba(30, 30, 45, 0.6) !important; /* Semi-transparent */
    }
    /* Vote Counts Top Accent */
    .vote-counts::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px; /* Thicker accent */
        background: linear-gradient(90deg, #8BE9FD, transparent) !important; /* Cyan gradient */
        box-shadow: 0px 0px 8px rgba(139, 233, 253, 0.5) !important; /* Glowing top border */
    }
     /* Vote Counts Glass Effect */
    .vote-counts::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.04), transparent),
            radial-gradient(circle at top right, rgba(139, 233, 253, 0.03), transparent 70%) !important; /* Cyan highlight */
        pointer-events: none;
    }
    .vote-counts strong {
        color: #8BE9FD !important; /* Cyan label */
        display: block !important;
        margin-bottom: 1.2rem !important; /* More space */
        font-weight: 700 !important; /* Bolder */
        letter-spacing: 0.02rem !important;
        position: relative !important;
        padding-bottom: 0.6rem !important;
        text-shadow: 0px 0px 3px rgba(139, 233, 253, 0.3) !important; /* Text glow */
    }
     /* Vote Counts Label Underline */
    .vote-counts strong::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 70px; /* Wider underline */
        height: 2px; /* Thicker underline */
        background: linear-gradient(90deg, rgba(139, 233, 253, 0.7), transparent) !important;
    }
    .vote-counts ul {
        padding-left: 2.5rem !important; /* More padding */
        margin-bottom: 1rem !important;
        list-style-type: none !important;
        position: relative !important;
    }
    .vote-counts li {
        margin-bottom: 0.8rem !important; /* More space */
        color: #D4D4E8 !important;
        position: relative !important;
        padding-left: 0.6rem !important; /* More padding */
        line-height: 1.7 !important;
    }
    /* Vote Counts Custom Bullets */
    .vote-counts li::before {
        content: "•"; /* Custom bullet */
        position: absolute;
        left: -1.4rem; /* Adjust position */
        color: #BD93F9 !important; /* Purple bullets */
        text-shadow: 0px 0px 3px rgba(189, 147, 249, 0.4) !important; /* Glowing bullets */
        font-size: 1em;
    }


    /* Ultra Premium Chat Message Styling */
    [data-testid="chatContainer"] [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, #1A1A28, #232335) !important; /* Richer gradient */
        border-radius: 1rem !important; /* Larger radius */
        padding: 1.8rem 2.2rem !important; /* More padding */
        margin-bottom: 2rem !important; /* More space */
        border: 1px solid #3F3F55 !important; /* Stronger border */
        box-shadow:
            0 6px 22px rgba(0, 0, 0, 0.5), /* Stronger shadow */
            0px 0px 0px 2px rgba(63, 63, 85, 0.8) !important; /* Wider outer glow */
        position: relative !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        overflow: hidden !important;
        backdrop-filter: blur(5px) !important; /* Glass effect */
        background-color: rgba(26, 26, 40, 0.6) !important; /* Semi-transparent */
    }
    /* Chat Message Glass Effect */
    [data-testid="chatContainer"] [data-testid="stChatMessage"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.05), transparent), /* Stronger highlight */
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.02), transparent 70%) !important;
        pointer-events: none;
        opacity: 0.9; /* More visible glass */
        transition: opacity 0.3s ease !important;
    }
    /* Chat Message Hover State */
    [data-testid="chatContainer"] [data-testid="stChatMessage"]:hover {
        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.6),
            0px 0px 0px 2px rgba(80, 250, 123, 0.4) !important; /* Green outer glow on hover */
        transform: translateY(-3px) !important; /* More pronounced lift */
    }
    [data-testid="chatContainer"] [data-testid="stChatMessage"]:hover::before {
        opacity: 1 !important; /* Full glass effect on hover */
    }
    /* User vs Assistant message differentiation with enhanced styling */
    [data-testid="chatContainer"] [data-testid="stChatMessage"][data-testid="user-message"] {
        border-left: 5px solid #BD93F9 !important; /* Thicker purple accent */
    }
     /* User Message Left Accent */
    [data-testid="chatContainer"] [data-testid="stChatMessage"][data-testid="user-message"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: -5px; /* Match border width */
        height: 100%;
        width: 5px; /* Match border width */
        background: linear-gradient(to bottom, #BD93F9, rgba(189, 147, 249, 0.4)) !important; /* Gradient border */
        box-shadow: 0px 0px 10px rgba(189, 147, 249, 0.6) !important; /* Glowing border */
        border-top-left-radius: 1rem !important;
        border-bottom-left-radius: 1rem !important;
        pointer-events: none;
    }
    [data-testid="chatContainer"] [data-testid="stChatMessage"]:not([data-testid="user-message"]) {
        border-left: 5px solid #69FF94 !important; /* Thicker green accent */
    }
     /* Assistant Message Left Accent */
    [data-testid="chatContainer"] [data-testid="stChatMessage"]:not([data-testid="user-message"])::after {
        content: "";
        position: absolute;
        top: 0;
        left: -5px; /* Match border width */
        height: 100%;
        width: 5px; /* Match border width */
        background: linear-gradient(to bottom, #69FF94, rgba(105, 255, 148, 0.4)) !important; /* Gradient border */
        box-shadow: 0px 0px 10px rgba(105, 255, 148, 0.6) !important; /* Glowing border */
        border-top-left-radius: 1rem !important;
        border-bottom-left-radius: 1rem !important;
        pointer-events: none;
    }
    /* Enhanced content within the chat message */
    [data-testid="chatContainer"] [data-testid="stChatMessage"] .stMarkdown p {
        color: #F8F8F2 !important; /* Dracular foreground */
        margin-bottom: 1rem !important;
        line-height: 1.8 !important;
        letter-spacing: 0.01rem !important;
    }
    /* Futuristic code blocks inside chat messages */
    [data-testid="chatContainer"] [data-testid="stChatMessage"] .stMarkdown pre code {
        background: linear-gradient(135deg, #0D0D15, #101020) !important; /* Darker gradient */
        border: 1px solid #3F3F55 !important;
        border-radius: 0.7rem !important;
        padding: 1.8rem !important;
        margin: 1.2rem 0 !important;
        overflow-x: auto !important;
        color: #BD93F9 !important; /* Purple code text */
        font-family: 'Fira Code', Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace !important;
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
        display: block !important;
        word-wrap: normal !important;
        white-space: pre-wrap !important;
        box-shadow:
            inset 0 0 20px rgba(0, 0, 0, 0.4) !important,
            0px 4px 15px rgba(0, 0, 0, 0.3) !important;
        position: relative !important;
        backdrop-filter: blur(2px); /* Slight blur */
    }
     [data-testid="chatContainer"] [data-testid="stChatMessage"] .stMarkdown pre code::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.02), transparent) !important,
            radial-gradient(circle at top right, rgba(189, 147, 249, 0.03), transparent 70%) !important; /* Purple highlight */
        pointer-events: none;
        border-radius: 0.7rem !important;
     }
    /* Futuristic inline code inside chat messages */
    [data-testid="chatContainer"] [data-testid="stChatMessage"] code {
        background: linear-gradient(135deg, #28283D, #3F3F55) !important; /* Gradient background */
        border-radius: 0.5rem !important;
        padding: 0.35em 0.7em !important; /* More padding */
        color: #FFB86C !important; /* Orange inline code */
        font-family: 'Fira Code', Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace !important;
        font-size: 0.95rem !important;
        border: 1px solid rgba(255, 184, 108, 0.3) !important; /* Subtle orange border */
        box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.3) !important;
    }

    /* Ultra Premium File Uploader Styling */
    [data-testid="stFileUploader"] label {
        color: #FFB86C !important; /* Orange label */
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        display: block !important;
        position: relative !important;
        letter-spacing: 0.01rem !important;
        text-shadow: 0px 0px 2px rgba(255, 184, 108, 0.3) !important; /* Subtle text glow */
    }
    /* File Uploader Label Underline */
    [data-testid="stFileUploader"] label::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: -4px;
        width: 50px; /* Wider underline */
        height: 2px;
        background: linear-gradient(90deg, #FFB86C, transparent) !important; /* Orange gradient */
        border-radius: 1px !important;
        box-shadow: 0px 0px 5px rgba(255, 184, 108, 0.4) !important; /* Glowing underline */
    }
    [data-testid="stFileUploader"] div[data-baseweb="file-uploader"] {
        background: linear-gradient(135deg, #1A1A28, #232335) !important; /* Gradient background */
        border: 2px dashed #556270 !important; /* Thicker dashed border */
        border-radius: 0.8rem !important; /* Larger radius */
        padding: 2.5rem !important; /* More padding */
        text-align: center !important;
        color: #B0B0C0 !important;
        margin-bottom: 2rem !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow:
            inset 0 3px 12px rgba(0, 0, 0, 0.3),
            0px 3px 10px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(4px) !important; /* Glass effect */
    }
     /* File Uploader Drop Area Glow */
    [data-testid="stFileUploader"] div[data-baseweb="file-uploader"]::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background:
            radial-gradient(circle, rgba(189, 147, 249, 0.2) 0%, transparent 70%), /* Purple radial glow */
            linear-gradient(135deg, rgba(255, 255, 255, 0.04), transparent) !important;
        opacity: 0;
        transition: opacity 0.6s ease, transform 0.6s ease !important;
        transform: scale(0.9);
        transform-origin: center;
    }
    [data-testid="stFileUploader"] div[data-baseweb="file-uploader"]:hover {
        border-color: #BD93F9 !important; /* Purple border on hover */
        box-shadow:
            inset 0 3px 12px rgba(0, 0, 0, 0.4),
            0 0 0 2px rgba(189, 147, 249, 0.4), /* Purple outer glow */
            0px 4px 15px rgba(0, 0, 0, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stFileUploader"] div[data-baseweb="file-uploader"]:hover::before {
        opacity: 1 !important;
        transform: scale(1) !important;
    }
    /* Futuristic uploaded file chip styling */
    [data-testid="stFileUploader"] div[data-testid="FileThumb"] {
        background: linear-gradient(135deg, #28283D, #3F3F55) !important; /* Gradient background */
        border: 1px solid #556270 !important;
        border-radius: 0.7rem !important;
        padding: 1.2rem !important; /* More padding */
        margin-top: 1.5rem !important;
        color: #E9EDF0 !important;
        box-shadow:
            3px 4px 15px rgba(0, 0, 0, 0.35),
            0px 0px 0px 1px rgba(79, 79, 104, 0.7) !important;
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(3px); /* Glass effect */
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
     /* File Chip Glass Effect */
     [data-testid="stFileUploader"] div[data-testid="FileThumb"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), transparent) !important;
        pointer-events: none;
     }
    [data-testid="stFileUploader"] div[data-testid="FileThumb"] p {
        color: #F8F8F2 !important; /* Dracular foreground */
        font-size: 0.95rem !important;
        position: relative !important;
    }
    [data-testid="stFileUploader"] div[data-testid="FileThumb"]:hover {
        box-shadow:
            3px 4px 18px rgba(0, 0, 0, 0.45),
            0px 0px 0px 1px rgba(80, 250, 123, 0.3) !important; /* Green outer glow */
        transform: translateY(-2px) !important;
        border-color: #69FF94 !important; /* Brighter green */
    }
    /* Futuristic image display in chat */
    .chat-image {
        max-width: 100% !important;
        border-radius: 0.8rem !important;
        margin: 1.5rem 0 !important; /* More margin */
        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.6),
            0px 0px 0px 2px rgba(63, 63, 85, 0.8) !important; /* Premium outer glow */
        border: 1px solid #3F3F55 !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
     /* Chat Image Top Accent */
    .chat-image::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px; /* Thicker line */
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.15), transparent) !important;
        border-top-left-radius: 0.8rem !important;
        border-top-right-radius: 0.8rem !important;
        pointer-events: none;
    }
    .chat-image:hover {
        transform: scale(1.02) !important;
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.7),
            0px 0px 0px 2px rgba(105, 255, 148, 0.4) !important; /* Green outer glow on hover */
        border-color: #69FF94 !important; /* Brighter green */
    }
    /* Style for the thumbnail images in chat */
     [data-testid="chatContainer"] [data-testid="stChatMessage"] img {
         border-radius: 0.6rem !important;
         border: 1px solid #3F3F55 !important;
         box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3) !important;
         transition: all 0.2s ease !important;
     }
     [data-testid="chatContainer"] [data-testid="stChatMessage"] img:hover {
          transform: scale(1.05);
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4) !important;
          border-color: #BD93F9 !important;
     }
     /* Remove default Streamlit spacing for columns inside chat message */
    [data-testid="chatContainer"] [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] {
          gap: 1rem !important;
     }

    /* Copy button for code blocks */
    .copy-button {
        position: absolute !important;
        top: 0.4rem !important;
        right: 0.4rem !important;
        background: #3F3F55 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 0.3rem !important;
        padding: 0.2rem 0.6rem !important;
        font-size: 0.8rem !important;
        cursor: pointer !important;
        opacity: 0.8 !important;
    }
    .copy-button:hover {
        opacity: 1 !important;
    }


</style>
""", unsafe_allow_html=True)

if theme_choice == "Light":
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
elif theme_choice == "Premium":
    st.markdown(PREMIUM_THEME_CSS, unsafe_allow_html=True)
elif theme_choice == "Alien":
    st.markdown(ALIEN_THEME_CSS, unsafe_allow_html=True)

# --- Code Copy Script ---
st.markdown(
    """
    <script>
    function addCopyButtons(){
        document.querySelectorAll('pre').forEach(function(pre){
            const parent = pre.parentElement;
            if(parent.classList.contains('code-with-copy')) return;
            parent.classList.add('code-with-copy');
            const btn = document.createElement('button');
            btn.textContent = 'Copy';
            btn.className = 'copy-button';
            btn.addEventListener('click', function(){
                navigator.clipboard.writeText(pre.innerText);
                btn.textContent = 'Copied!';
                setTimeout(function(){btn.textContent='Copy';}, 2000);
            });
            parent.style.position = 'relative';
            parent.appendChild(btn);
        });
    }
    document.addEventListener('DOMContentLoaded', addCopyButtons);
    new MutationObserver(addCopyButtons).observe(document.body,{subtree:true,childList:true});
    </script>
    """,
    unsafe_allow_html=True,
)


# --- GPU Telemetry Setup ---
# Initialize NVML for GPU monitoring if available
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    # Get the number of devices to pick the first one (index 0)
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
    if GPU_COUNT == 0:
        GPU_AVAILABLE = False
        logger.info("No NVML-compatible GPU devices found.")
    else:
         # Check if the selected device is a CUDA device and if telemetry works for it
         selected_device = validate_device_selection(
             st.session_state.get("device_select", "cpu")
         )
         if selected_device.startswith("cuda"):
              try:
                   # Attempt to get handle for the selected cuda device index
                   device_index = int(selected_device.split(":")[-1]) if ":" in selected_device else 0
                   # Check if the device index is valid
                   if device_index >= GPU_COUNT:
                       GPU_AVAILABLE = False
                       logger.warning(f"Selected CUDA device index {device_index} is out of range (Max index: {GPU_COUNT - 1}). Telemetry disabled for this selection.")
                   else:
                        pynvml.nvmlDeviceGetHandleByIndex(device_index) # Just check if handle is accessible
                        logger.info(f"NVML initialized and ready for device {device_index}. Total GPU(s) found: {GPU_COUNT}") # Log total count
              except Exception as e:
                   GPU_AVAILABLE = False # Telemetry failed for the selected CUDA device
                   logger.error(f"Error accessing NVML handle for device {selected_device}: {e}. Telemetry disabled.")
         else:
             GPU_AVAILABLE = False # Telemetry only for CUDA devices
             logger.info(f"Selected device '{selected_device}' is not a CUDA device. Telemetry disabled.")

except Exception as e:
    GPU_AVAILABLE = False
    logger.error(f"pynvml initialization failed: {e}. GPU Telemetry disabled.")


# Use st.empty to hold the telemetry status text, defined *outside* cached functions
# Place it in the sidebar footer area using markdown
telemetry_placeholder = st.sidebar.empty() # Place in sidebar


def update_telemetry():
    """Updates the telemetry display in the dedicated placeholder."""
    telemetry_text = "📊 System Status: [Initializing...]"
    selected_device = validate_device_selection(
        st.session_state.get("device_select", "cpu")
    )

    if not GPU_AVAILABLE or not selected_device.startswith("cuda"):
        telemetry_text = "📊 System Status: [No Compatible GPU Available or Selected for Telemetry]"
    else:
        try:
            # Use the index from the selected device if it's specific (e.g. cuda:1)
            device_index = int(selected_device.split(":")[-1]) if ":" in selected_device else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # Fix for telemetry decode error: handle potential bytes or string
            gpu_name_bytes = pynvml.nvmlDeviceGetName(handle)
            try:
                 gpu_name = gpu_name_bytes.decode('utf-8')
            except (UnicodeDecodeError, AttributeError):
                 # If decoding fails or it's not bytes, assume it's already a string or handle as is
                 gpu_name = str(gpu_name_bytes) # Ensure it's a string


            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = memory.used // 1024**2
            mem_total_mb = memory.total // 1024**2
            telemetry_text = f"📊 <strong>GPU {device_index} ({gpu_name}):</strong> {utilization.gpu}% Util | {mem_used_mb}/{mem_total_mb} MB VRAM"
        except pynvml.NVMLError as e:
             telemetry_text = f"📊 System Status: [Telemetry Error: {e}]"
             logger.error(f"pynvml error during telemetry update: {e}")
        except Exception as e:
             telemetry_text = "📊 System Status: [Telemetry Error]"
             logger.error(f"Unexpected error during telemetry update: {e}")


    # Use markdown with a custom class for styling the container
    telemetry_placeholder.markdown(f'<div class="telemetry-box">{telemetry_text}</div>', unsafe_allow_html=True)

# Initial telemetry update when the script starts
update_telemetry()


# --- Caching Model Loading (Now loads AutoModel and AutoProcessor for multimodal) ---
@st.cache_resource(show_spinner=False)
def _load_model_and_processor_cached(model_name: str, device: str):
    """
    Loads the model and its associated processor (tokenizer + image processor if multimodal)
    using a prioritized approach (CausalLM, Seq2SeqLM, then AutoModel).
    This function is cached by Streamlit.
    """
    logger.info(f"Attempting to load model '{model_name}' and processor on '{device}'...")

    model = None
    processor = None # Use a general processor variable
    model_config = None
    model_loaded_successfully = False

    status_box = st.empty() # Placeholder for status in cached function context

    try:
        # Load configuration first to inspect model type (optional but helpful)
        status_box.write("Fetching model configuration...")
        try:
             model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
             logger.info(f"Model config loaded. Architecture: {model_config.architectures}")
             status_box.write("Configuration loaded. Loading processor...")
        except Exception as config_e:
             logger.warning(f"Could not load model config: {config_e}. Proceeding without config.")
             status_box.warning(f"Could not load model config: {config_e}. Proceeding without config.")
             model_config = None # Ensure it's None if loading failed


        # Use AutoProcessor to handle multimodal processors (which can include tokenizers)
        # Setting trust_remote_code=True is often necessary for custom model types
        status_box.write("Loading processor...")
        try: # Wrap processor loading in try/except
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"Processor loaded successfully. Processor type: {type(processor)}")
            status_box.write("Processor loaded. Loading model weights...")
        except Exception as proc_e:
            logger.error(f"Failed to load processor: {proc_e}")
            status_box.error(f"Failed to load processor for '{model_name}'. Details: {proc_e}")
            # Clean up resources
            if model_config is not None: del model_config
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as cleanup_e:
                    logger.warning(f"Error during cuda empty_cache: {cleanup_e}")
            gc.collect()
            return None, None, False # Return failure


        # Determine appropriate dtype based on device and model config
        torch_dtype = torch.float32 # Default to float32
        if torch.cuda.is_available() and device.startswith("cuda"):
            # Check if bfloat16 is generally supported (requires Ampere or newer GPU)
            # And if the model config indicates bfloat16 is preferred or supported
            # This is a heuristic; explicit model docs are best.
            try:
                cuda_dev = torch.device(device)
                if cuda_dev.index is None:
                    cuda_dev = torch.device("cuda", torch.cuda.current_device())

                if cuda_dev.index >= torch.cuda.device_count():
                    raise IndexError(f"CUDA device index {cuda_dev.index} out of range")

                gpu_major_version = torch.cuda.get_device_properties(cuda_dev).major
                if gpu_major_version >= 8: # Ampere or newer
                    # Check model config for preferred dtype if available
                    if model_config is not None and hasattr(model_config, 'torch_dtype') and model_config.torch_dtype == torch.bfloat16:
                         torch_dtype = torch.bfloat16
                         logger.info("GPU supports bfloat16 and model config specifies bfloat16. Loading with bfloat16.")
                    elif model_config is not None and model_config.architectures and any(arch.lower() in ["llama", "mistral", "gemma", "qwen", "smol", "idefics", "paligemma"] for arch in model_config.architectures):
                         # Common architectures that often support bfloat16, adding Idefics and Paligemma
                         torch_dtype = torch.bfloat16
                         logger.info("GPU supports bfloat16 and model architecture suggests bfloat16 compatibility. Loading with bfloat16.")
                    else:
                         logger.info("GPU supports bfloat16 but model config/architecture doesn't explicitly suggest it. Loading with float32.")
            except Exception as dtype_e:
                 logger.error(f"Failed to inspect CUDA device '{device}': {dtype_e}. Falling back to CPU.")
                 status_box.error(f"Invalid CUDA device '{device}'. Falling back to CPU.")
                 device = "cpu"
                 torch_dtype = torch.float32
        else:
             logger.info("Not on CUDA device. Loading with float32.")


        # --- Prioritized Model Loading ---
        # Try loading generation-specific classes first, then fall back to general AutoModel
        loading_attempts = [
            (AutoModelForCausalLM, "CausalLM"),
            (AutoModelForSeq2SeqLM, "Seq2SeqLM"),
            (AutoModel, "General AutoModel")
        ]

        for auto_model_class, class_name in loading_attempts:
            status_box.write(f"Attempting to load model as {class_name}...")
            try:
                model = auto_model_class.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype
                )
                model_loaded_successfully = True
                logger.info(f"Model loaded successfully as {class_name}.")
                # If loaded with general AutoModel, explicitly check for generation capability
                if auto_model_class == AutoModel:
                     if not hasattr(model, 'generate') or not callable(getattr(model, 'generate', None)):
                          logger.error(f"Model loaded with {class_name} does not have a compatible .generate() method.")
                          status_box.error(f"Loaded model ({type(model).__name__}) does not have a compatible .generate() method required for text generation. Please try a different model.")
                          model = None # Treat as failed if not generation compatible
                          model_loaded_successfully = False # Ensure flag is False
                     else:
                          logger.info(f"Model loaded with {class_name} has a compatible .generate() method.")

                break # Exit loop if loading was successful (and generation compatible if using AutoModel)

            except Exception as e:
                logger.warning(f"Failed to load as {class_name}: {e}")
                model = None # Ensure model is None if this specific attempt failed


        # If all loading methods failed:
        if not model_loaded_successfully:
            logger.error(f"All model loading attempts failed for '{model_name}'.")
            status_box.error(f"Failed to load model '{model_name}' using any method. Please check the model ID/path and its compatibility.")
            # Clean up resources if loading failed
            if model is not None: del model
            if processor is not None: del processor
            if model_config is not None: del model_config
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as cleanup_e:
                    logger.warning(f"Error during cuda empty_cache: {cleanup_e}")
            gc.collect()
            return None, processor, False # Return None for model and False for multimodal on failure


        # If loading was successful, move model to device and set to eval mode
        model.to(device)
        model.eval() # Crucial for consistent inference behavior
        logger.info(f"Model moved to {device} and set to eval mode. Model type: {type(model)}")


        # Attempt to resize model embeddings if tokenizer added a new pad token during loading
        # This is important for models where pad_token_id was None initially.
        # Access tokenizer via the loaded processor
        tokenizer = getattr(processor, 'tokenizer', None)
        if tokenizer and tokenizer.pad_token_id is not None:
             # Check if the tokenizer's vocabulary size is larger than the model's embedding size
             # This can happen if special tokens (like [PAD]) were added.
             try:
                 if hasattr(model, 'resize_token_embeddings') and hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None and len(tokenizer) > model.get_input_embeddings().weight.size(0):
                      logger.info(f"Tokenizer size ({len(tokenizer)}) is larger than model embedding size ({model.get_input_embeddings().weight.size(0)}). Resizing model embeddings...")
                      model.resize_token_embeddings(len(tokenizer))
                      logger.info("Model embeddings resized successfully.")
                 elif hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
                      logger.debug("Tokenizer size matches or is smaller than model embedding size. No resize needed.")
                 else:
                      logger.warning("Model does not have get_input_embeddings method or embeddings are None. Cannot check or resize embeddings.")
             except Exception as resize_e:
                  logger.warning(f"Failed to check or resize model embeddings: {resize_e}")
                  status_box.warning(f"Failed to check or resize model embeddings: {resize_e}")
        elif tokenizer is None:
             logger.warning("Processor does not have a tokenizer. Skipping embedding resize check.")


        # Ensure necessary config settings for structured generation output if applicable
        # These might vary for multimodal models, check model documentation
        # Generally good practice for models used with generate()
        if hasattr(model, 'config'):
             model.config.return_dict_in_generate = True
             # output_scores might not be supported by all models, skip if config doesn't have it
             if hasattr(model.config, 'output_scores'):
                  model.config.output_scores = True


        # Check if the loaded processor has an image processor component -> Multimodal capability flag
        # This flag determines if the GUI's image upload should be enabled
        has_image_processor = hasattr(processor, 'image_processor') and processor.image_processor is not None
        model_architecture_name = type(model).__name__ if model else "Unknown"
        processor_architecture_name = type(processor).__name__ if processor else "Unknown"
        logger.info(f"Loaded Model Class: {model_architecture_name}")
        logger.info(f"Loaded Processor Class: {processor_architecture_name}")
        logger.info(f"Image Processor available (Multimodal Capable for Input): {has_image_processor}")

        status_box.empty() # Clear the temporary status box

        # Return the loaded model, processor, and multimodal capability flag
        return model, processor, has_image_processor


    except Exception as e:
        logger.error(f"❌ Model or Processor loading failed for '{model_name}'. Details: {e}")
        status_box = st.empty() # Use a final status box to show error
        status_box.error(f"❌ Model or Processor loading failed for '{model_name}'. Details: {e}")
        # Clean up resources if loading failed
        if model is not None: del model
        if processor is not None: del processor
        if model_config is not None: del model_config
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_e:
                logger.warning(f"Error during cuda empty_cache: {cleanup_e}")
        gc.collect()
        # No re-raise here, return None and the calling function handles the error state

        return None, None, False


# --- Wrapper function to handle status reporting for cached loading ---
def safe_load_model_with_status(model_name: str, device: str):
    """
    Calls the cached loading function and handles Streamlit status updates
    and error reporting. Now loads processor and checks multimodal capability.
    """
    # Use st.status here, defined outside the cached function, for live updates
    status_text = f"🌐 Loading model and processor for '{model_name}'..."
    # Use a main status container for the loading process that persists
    main_loading_status = st.status(status_text, expanded=True)

    with main_loading_status:
        main_loading_status.write(f"Initiating model and processor load on device '{device}'...")
        update_telemetry() # Update the separate telemetry box

        try:
            # Call the actual cached loading function which also writes its own status
            # This function returns None, None, False if loading fails, after logging the error
            model, processor, has_image_processor = _load_model_and_processor_cached(
                model_name=model_name,
                device=device
            )

            # After cached function returns, continue status here
            if model and processor:
                 # Check tokenizer padding/eos tokens for generation compatibility
                 tokenizer = getattr(processor, 'tokenizer', None)
                 if tokenizer:
                      if tokenizer.pad_token_id is None:
                           main_loading_status.warning(f"Tokenizer has no pad_token_id. Batch generation (Self-Consistency) might be unstable.")
                      else:
                           main_loading_status.write(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}.")

                      if tokenizer.eos_token_id is not None:
                           main_loading_status.write(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}.")
                      else:
                            main_loading_status.warning(f"Tokenizer has no eos_token_id.")
                 else:
                      main_loading_status.warning("Processor does not have a tokenizer attribute. Cannot check pad/eos token IDs.")


                 # Report on multimodal capability based on the loaded processor
                 if has_image_processor:
                      st.session_state.multimodal_capable_model_loaded = True
                      main_loading_status.info("🖼️ Loaded model supports image inputs (multimodal).")
                 else:
                      st.session_state.multimodal_capable_model_loaded = False
                      main_loading_status.warning("ℹ️ Loaded model appears text-only (no image processor found). Multimodal features will be disabled.")

                 update_telemetry() # Final telemetry update after success

                 # Use update() method for the final status within the context manager
                 main_loading_status.update(label=f"✅ Model Loaded: {model_name}", state="complete")


                 # Return model, processor, and the multimodal capability flag
                 return model, processor, has_image_processor

            else:
                 # If _load_model_and_processor_cached returned None, an error occurred and was logged/displayed by it
                 # The status was already updated to error inside the cached function or right after it returned None.
                 st.session_state.multimodal_capable_model_loaded = False # Ensure flag is False on failure
                 update_telemetry() # Final telemetry update after error
                 # No need to duplicate error message here, it was handled by _load_model_and_processor_cached


                 return None, None, False # Return False for multimodal capability on failure

        except Exception as e:
             # Catch any unexpected errors during the status wrapping
             logger.error(f"An unexpected error occurred during model loading status wrapping: {e}")
             st.session_state.multimodal_capable_model_loaded = False # Ensure flag is False on failure
             main_loading_status.error(f"❌ An unexpected error occurred during loading status: {e}")
             update_telemetry() # Final telemetry update after error
             st.exception(e) # Display the exception
             main_loading_status.update(label="🔴 Model Loading Failed", state="complete") # Mark status as complete

             return None, None, False


# --- Self-Consistency Voting Logic ---
# Keep the same normalize_answer and perform_self_consistency_voting functions as provided previously.
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

def perform_self_consistency_voting(final_answers: List[str]) -> Tuple[Optional[str], Dict[str, int], List[str]]:
    """
    Performs simple majority voting on a list of final answers after normalization.

    Args:
        final_answers (List[str]): A list of raw final answer strings from the wrapper.

    Returns:
        Tuple[Optional[str], Dict[str, int], List[str]]: A tuple containing:
            - The winning (most common) normalized answer, or None if no valid answers.
            - A dictionary mapping normalized answers to their vote counts.
            - A list of the raw answers that matched the consensus normalized answer.
    """
    if not final_answers:
        return None, {}, []

    # 1. Filter out empty or non-string answers and normalize
    # Ensure we convert to string safely before normalization
    normalized_answer_pairs = [(normalize_answer(str(ans)), str(ans)) for ans in final_answers if ans is not None and str(ans).strip()]

    # 2. Count occurrences of each normalized answer
    normalized_answers_only = [pair[0] for pair in normalized_answer_pairs]
    if not normalized_answers_only: # No valid answers after normalization
         return None, {}, []

    answer_distribution = Counter(normalized_answers_only)

    # 3. Determine the consensus answer (most common normalized answer)
    # most_common(1) returns a list like [('normalized_answer', count)]
    # Check if there's at least one item in the counter
    if not answer_distribution:
         return None, {}, []

    # Get the most common normalized answer string
    # Safely access the first item in most_common result
    most_common_list = answer_distribution.most_common(1)
    if not most_common_list:
        return None, dict(answer_distribution), [] # Should not happen if answer_distribution is not empty
    consensus_normalized_answer = most_common_list[0][0]

    # 4. Collect the original raw answers that map to the consensus normalized answer
    # Ensure we collect the *raw* answers from the original list that map to the winning normalized answer
    consensus_raw_answers = [
        original_ans for original_ans in final_answers
        if original_ans is not None and normalize_answer(str(original_ans)) == consensus_normalized_answer
    ]


    return consensus_normalized_answer, dict(answer_distribution), consensus_raw_answers


# --- Streamlit App Layout and Logic ---

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add an initial welcome message if history is empty
    st.session_state.chat_history.append({
        "role": "assistant",
        # Updated content for a more premium and immersive welcome message using "AI companion"
        "content": "✨ Greetings. I am your AI companion, ready to explore the frontiers of thought with you. Present your question or problem below, and sculpt the analytical process using the parameters in the sidebar.",
        "type": "text",
        "id": 0 # Assign ID
    })


# Initialize model and tokenizer/processor in session state
if "model" not in st.session_state:
    st.session_state.model = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "cot_wrapper" not in st.session_state:
    st.session_state.cot_wrapper = None
# New state variable to track if the loaded model is multimodal capable
if "multimodal_capable_model_loaded" not in st.session_state:
    st.session_state.multimodal_capable_model_loaded = False


st.title("🧠 NeuroReasoner CoT (Multimodal)")

# Sidebar for Controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Model Selection
    st.subheader("🌐 Model Configuration")
    # Use a multimodal model as the default example, or load the previously selected one
    # Changed default to a common GPT-2 model for initial testing compatibility
    default_model = "gpt2" # Example Text-Only Model
    # You can change this to a multimodal model like "HuggingFaceTB/SmolVLM-Instruct"
    # if you are sure the model is compatible with Auto classes for generation.
    model_name_input = st.text_input(
        "Hugging Face Model ID or Local Path",
        value=st.session_state.get("model_name_select", default_model), # Default to text-only
        key="model_name_select" # Persistent key
    )

    # Device Selection
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            available_devices.append(f"cuda:{i}")
    # Select the first available device by default, or the previously selected one
    default_device_index = 0
    if "device_select" in st.session_state and st.session_state.device_select in available_devices:
         default_device_index = available_devices.index(st.session_state.device_select)
    elif len(available_devices) > 0:
         default_device_index = 0


    device_select = st.selectbox(
        "Device",
        available_devices,
        index=default_device_index,
        key="device_select", # Persistent key
        help="Select the device (CPU or GPU) to load the model onto. Larger or multimodal models often require GPU."
    )
    # Validate device choice against current CUDA availability
    validated_device = validate_device_selection(device_select)
    if validated_device != device_select:
        st.session_state.device_select = validated_device
        device_select = validated_device

    # Load Model Button
    if st.button("Load Model"):
        logger.info("Load Model button clicked. Initiating model loading process.")
        # Clear cached resources and session state model/wrapper before loading new one
        # Clearing st.cache_resource clears ALL cached resources
        st.cache_resource.clear()
        logger.info("Streamlit resource cache cleared.")

        # Explicitly clear model/wrapper state in session state
        st.session_state.model = None
        st.session_state.processor = None
        st.session_state.cot_wrapper = None
        st.session_state.multimodal_capable_model_loaded = False # Reset capability flag

        # Attempt to load the model and processor with status updates
        model, processor, has_image_processor = safe_load_model_with_status(model_name_input, device_select)

        # Store the loaded model, processor, and capability flag in session state
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.multimodal_capable_model_loaded = has_image_processor

        # If loading was successful (model and processor are not None), create the wrapper
        # Also check if the loaded model object is actually generation compatible before creating the wrapper
        if model is not None and processor is not None and (hasattr(model, 'generate') and callable(getattr(model, 'generate', None)) or isinstance(model, GenerationMixin)):
             try:
                  # Initialize the ChainOfThoughtWrapper with the loaded model and processor
                  st.session_state.cot_wrapper = ChainOfThoughtWrapper(
                       model=model,
                       processor=processor, # Pass the processor
                       device=device_select,
                       # The wrapper will determine text-only vs multimodal based on processor
                       # You can override default template/settings here if needed, e.g.:
                       # cot_instruction="Solve this problem by thinking step-by-step.",
                       # final_answer_tag="Final Answer:"
                  )
                  st.sidebar.success("🚀 CoT Wrapper initialized!")
                  logger.info("CoT Wrapper initialized successfully in GUI.")
             except Exception as e:
                  st.session_state.cot_wrapper = None
                  st.sidebar.error(f"❌ Failed to initialize CoT Wrapper: {e}")
                  logger.error(f"Failed to initialize CoT Wrapper in GUI: {e}")
                  st.exception(e) # Display exception in sidebar
        elif model is not None:
             # Model loaded, but not generation compatible according to the check in safe_load_model_with_status
             # The error message was already shown in the status box within safe_load_model_with_status
             logger.warning("Model loaded but not generation compatible, skipping wrapper initialization.")
             st.sidebar.warning("Wrapper initialization skipped: Loaded model is not compatible with text generation.")
        else:
             logger.warning("Model or Processor loading failed, skipping wrapper initialization.")
             st.sidebar.warning("Wrapper initialization skipped due to model loading failure.")


    # Generation Parameters
    st.subheader("✨ Generation Parameters")
    # Add parameters for self-consistency and number of chains
    # CORRECTED: Remove assignment back to st.session_state
    st.checkbox(
        "Enable Self-Consistency Voting",
        value=st.session_state.get("self_consistency_checkbox", True), # Use value for initial state
        key="self_consistency_checkbox", # Streamlit manages state at this key
        help="Generate multiple reasoning chains and vote for the most common answer."
    )
    # Access the value later using st.session_state.self_consistency_checkbox

    # CORRECTED: Remove assignment back to st.session_state
    st.slider(
        "Number of Chains (N)",
        min_value=1,
        max_value=10,
        step=1,
        key="num_chains", # Streamlit manages state at this key
        help="The number of independent reasoning chains to generate for self-consistency."
             "Requires a model that supports batch generation and a tokenizer with a pad_token_id."
    )
    # Access the value later using st.session_state.num_chains

    # Display effective number of chains (1 if self-consistency is off)
    effective_num_chains_display = st.session_state.get("num_chains", 1) if st.session_state.get("self_consistency_checkbox", False) else 1
    st.info(f"Effective Chains Generated: {effective_num_chains_display}")


    # Add other generation parameters you want to expose
    # CORRECTED: Remove assignment back to st.session_state
    st.slider(
        "Max New Tokens",
        min_value=50,
        max_value=2048, # Increased max tokens for more extensive reasoning
        value=st.session_state.get("max_new_tokens", 512), # Use value for initial state
        step=10,
        key="max_new_tokens", # Streamlit manages state at this key
        help="Maximum number of tokens to generate in each chain."
    )
    # Access the value later using st.session_state.max_new_tokens

    # CORRECTED: Remove assignment back to st.session_state
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get("temperature", 0.7), # Use value for initial state
        step=0.05,
        key="temperature", # Streamlit manages state at this key
        help="Controls the randomness of generation. Lower = more deterministic, Higher = more creative."
             "Set to 0 for greedy decoding (if do_sample is False)."
    )
    # Access the value later using st.session_state.temperature

    # CORRECTED: Remove assignment back to st.session_state
    st.slider(
        "Top K",
        min_value=0,
        max_value=100, # Increased max for more diverse options
        value=st.session_state.get("top_k", 50), # Use value for initial state
        step=1,
        key="top_k", # Streamlit manages state at this key
        help="Limits the sampling pool to the top K most likely tokens."
             "Set to 0 to disable Top-K sampling (when do_sample is True)."
             "Recommended: Use either Top-K or Top-P, not both (e.g., Top-P 1.0 with Top-K > 0, or Top-K 0 with Top-P < 1.0)."
    )
    # Access the value later using st.session_state.top_k

    # CORRECTED: Remove assignment back to st.session_state
    st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("top_p", 1.0), # Use value for initial state
        step=0.01,
        key="top_p", # Streamlit manages state at this key
        help="Limits the sampling pool to the smallest set of tokens whose cumulative probability exceeds Top P."
             "Set to 1.0 to disable Top-P sampling (when do_sample is True)."
             "Recommended: Use either Top-K or Top-P, not both (e.g., Top-P 1.0 with Top-K > 0, or Top-K 0 with Top-P < 1.0)."
    )
    # Access the value later using st.session_state.top_p

    # CORRECTED: Remove assignment back to st.session_state
    st.checkbox(
        "Do Sample",
        value=st.session_state.get("do_sample", True), # Use value for initial state
        key="do_sample", # Streamlit manages state at this key
        help="If True, use sampling (with Temperature, Top K, Top P). Otherwise, use greedy decoding (Temperature, Top K, Top P are ignored)."
    )
    # Access the value later using st.session_state.do_sample

    # CORRECTED: Remove assignment back to st.session_state
    st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=st.session_state.get("repetition_penalty", 1.1), # Use value from slider
        step=0.01,
        key="repetition_penalty", # Streamlit manages state at this key
        help="Penalizes repeated tokens based on the number of repetitions. Value > 1.0 encourages diversity."
    )
    # Access the value later using st.session_state.repetition_penalty

    # CORRECTED: Remove assignment back to st.session_state
    st.slider(
        "No Repeat Ngram Size",
        min_value=0,
        max_value=10, # Increased max size
        value=st.session_state.get("no_repeat_ngram_size", 0), # Use value for initial state
        step=1,
        key="no_repeat_ngram_size", # Streamlit manages state at this key
        help="All ngrams of this size appearing in the generation are penalized so that they do not appear more than once in the generation."
             "Set to 0 to disable."
    )
    # Access the value later using st.session_state.no_repeat_ngram_size


    # Memory Management
    st.subheader("💾 Memory Management")
    if st.button("Clear Chat History"):
        logger.info("Clear Chat History button clicked.")
        st.session_state.chat_history = []
        # Re-add the initial welcome message after clearing
        st.session_state.chat_history.append({
            "role": "assistant",
            # Updated content for a more premium and immersive welcome message using "AI companion"
            "content": "✨ Greetings. I am your AI companion, ready to explore the frontiers of thought with you. Present your question or problem below, and sculpt the analytical process using the parameters in the sidebar.",
            "type": "text",
            "id": 0 # Assign ID
        })
        logger.info("Chat history cleared and welcome message added.")
        st.rerun() # Rerun to clear the display

    def _chat_history_to_text(history):
        """Convert chat history to a plain text transcript."""
        lines = []
        for msg in history:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            msg_type = msg.get("type", "text")
            if isinstance(content, dict):
                if role == "user":
                    user_text = content.get("text", "")
                    lines.append(f"User: {user_text}")
                    if content.get("image_data"):
                        lines.append(f"[Attached {len(content.get('image_data'))} image(s)]")
                elif msg_type == "cot_response":
                    lines.append(f"Assistant: {content.get('main_output_display', '')}")
                else:
                    lines.append(f"{role}: {str(content)}")
            else:
                lines.append(f"{role}: {str(content)}")
        return "\n".join(lines)

    def _last_cot_to_markdown(history):
        """Extract the last chain-of-thought response as markdown."""
        for msg in reversed(history):
            if msg.get("type") == "cot_response" and isinstance(msg.get("content"), dict):
                content = msg["content"]
                md_lines = [f"**{content.get('main_output_label', 'Output')}:**", "", str(content.get('main_output_display', ''))]
                details = content.get("details", {})
                chains = details.get("generated_results", [])
                if chains:
                    md_lines.append("\n### Generated Chains\n")
                    for i, chain in enumerate(chains, 1):
                        md_lines.append(f"#### Chain {i}")
                        full_text = chain.get("full_text", "")
                        md_lines.append(f"```\n{full_text}\n```")
                        final_ans = chain.get("final_answer")
                        if final_ans:
                            md_lines.append(f"**Final Answer:** {final_ans}")
                consensus = details.get("consensus_answer")
                if consensus:
                    md_lines.append(f"\n**Consensus Answer:** {consensus}")
                return "\n".join(md_lines)
        return None

    if st.session_state.chat_history:
        history_text = _chat_history_to_text(st.session_state.chat_history)
        st.download_button(
            label="📥 Download Chat History",
            data=history_text,
            file_name="chat_history.txt",
            mime="text/plain",
        )

        last_cot_md = _last_cot_to_markdown(st.session_state.chat_history)
        if last_cot_md:
            st.download_button(
                label="📥 Download Last Reasoning",
                data=last_cot_md,
                file_name="last_reasoning.md",
                mime="text/markdown",
            )

    if st.button("Reset Session"):
        logger.info("Reset Session button clicked.")
        st.session_state.clear()
        st.rerun()


    # Telemetry Footer (Automatically updated)
    # The telemetry_placeholder was defined before the sidebar
    # Its content is updated by the update_telemetry function


# --- Main Chat Interface ---
# Dynamic message about multimodal capability based on the loaded model
if st.session_state.multimodal_capable_model_loaded:
    st.info("🖼️ Multimodal model loaded. You can now upload images alongside your text input.")
else:
     st.warning("ℹ️ Text-only model loaded or no model loaded or loaded model is not multimodal compatible. Multimodal input (images) will not be processed for generation. Load a multimodal model (like `HuggingFaceTB/SmolVLM-Instruct`) compatible with Hugging Face Auto classes for generation to enable this.")


# Display chat history
# Iterate through history and render messages in original order for chronological display
for message in st.session_state.chat_history:
    role = message["role"]
    # Check if 'content' is a dictionary (for user messages with images/structured data) or string
    content_data = message["content"]
    message_type = message.get("type", "text") # Default to text type
    message_id = message.get("id", None) # Get message ID

    # Use a container for each message to apply consistent styling and layout
    # The CSS handles the main chat message styling based on role
    with st.chat_message(role):
        if role == "user":
            # User message can have text and images
            # Ensure user_text is a string regardless of content_data type:
            user_text = content_data.get("text", "") if isinstance(content_data, dict) else str(content_data)
            image_data_list = content_data.get("image_data", []) if isinstance(content_data, dict) else []

            # Display user text
            if user_text:
                 # ENSURE user_text IS STRING FOR MARKDOWN:
                 st.markdown(str(user_text), unsafe_allow_html=True)

            # Display uploaded images if they exist for this user turn
            if image_data_list:
                 if user_text:
                     st.write("---") # Separator only if there was text
                 # ENSURE THE MESSAGE ITSELF IS STRING:
                 st.write(f"Uploaded Image{'s' if len(image_data_list) > 1 else ''}:")
                 # Display images in a row if multiple, or centered if single
                 # Use a container for single image to better control centering/width within the chat bubble
                 cols = st.columns(len(image_data_list)) if len(image_data_list) > 1 else [st.container()]
                 for i, img_bytes in enumerate(image_data_list):
                      try:
                           img = Image.open(io.BytesIO(img_bytes))
                           # Use the column or container to display
                           with (cols[i] if len(image_data_list) > 1 else cols[0]):
                                # Use a custom class for the image for potential specific styling
                                # Captions are handled correctly as strings by st.image
                                st.image(img, caption=f"Image {i+1}", use_column_width=False, width=150) # Display smaller thumbnail
                      except Exception as e:
                           logger.error(f"Could not display uploaded image {i+1} for message ID {message_id}: {e}")
                           with (cols[i] if len(image_data_list) > 1 else cols[0]):
                                # ENSURE WARNING MESSAGE IS STRING:
                                st.warning(f"Could not display image {i+1}")


        elif role == "assistant":
            # Assistant messages are typically strings or structured dicts from CoT response
            if message_type == "text":
                # Standard text response (e.g., welcome message)
                # ENSURE CONTENT IS STRING FOR MARKDOWN:
                st.markdown(str(content_data), unsafe_allow_html=True)
            elif message_type == "error_message":
                # Error message response - explicitly marked
                # ENSURE CONTENT IS STRING FOR ERROR MESSAGE:
                st.error(str(content_data))
            elif message_type == "cot_response":
                # Structured CoT response data - content_data should be a dict here
                if not isinstance(content_data, dict):
                    logger.error(f"Assistant message ID {message_id} content is not in expected dictionary format for cot_response.")
                    st.error("Error: Assistant CoT response content is not in expected dictionary format.")
                    # Display as raw text as a fallback, ENSURING IT'S STRING:
                    st.markdown(str(content_data), unsafe_allow_html=True)
                    continue # Skip structured rendering

                # --- START Refactored Assistant CoT Response Rendering (Fixed Expander Nesting & Full Output as Main) ---

                # Access data from the structured dictionary
                # Use the new keys for the main output display
                # SAFELY GET AND ENSURE STRING FOR main_output_display:
                main_output_display = str(content_data.get("main_output_display", "Error: Could not retrieve response."))
                # SAFELY GET AND ENSURE STRING FOR main_output_label:
                main_output_label = str(content_data.get("main_output_label", "Output"))

                # Access data for the details section
                details = content_data.get("details", {}) # Safely get the details dictionary

                # Get detailed data from the 'details' dictionary (used in the expander)
                generated_results = details.get("generated_results", [])
                consensus_answer = details.get("consensus_answer") # Will be handled as string below
                vote_counts = details.get("vote_counts") # Will be handled as dictionary/items below
                consensus_raw_answers = details.get("consensus_raw_answers", []) # Will be handled as list of strings below
                self_consistency_was_enabled = details.get("self_consistency_enabled", False)
                requested_chains = details.get("requested_chains", len(generated_results))

                # Get model/param info from the top level of content_data for the param section (used in the expander)
                generation_params = content_data.get("generation_params", {}) # Will be handled as dictionary/items below
                # Build model_info string, SAFELY GETTING AND ENSURING STRINGS:
                model_name = str(content_data.get('model_name', 'N/A'))
                device_name = str(content_data.get('device', 'N/A'))
                model_type_name = str(content_data.get('model_type', 'Unknown'))
                is_multimodal = content_data.get('is_multimodal_compatible', False)
                model_info = f"Model: {model_name} | Device: {device_name} | Loaded Class: {model_type_name} | Capability: {'Multimodal' if is_multimodal else 'Text-Only'}"


                # 1. Display the main output (Full Output of Chain 1) directly in the chat bubble
                # Use markdown for the label, SAFELY ENSURING STRING:
                st.markdown(f'**{str(main_output_label)}:**', unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency

                # Use st.code to ensure the full output is displayed correctly, preserving formatting
                # ENSURE INPUT TO st.code IS STRING:
                st.code(str(main_output_display), language='markdown')


                # 2. Create a single main expander for all reasoning and generation details
                # Show this expander if there are generated results OR if SC was enabled with voting data
                should_show_details_expander = len(generated_results) > 0 or (self_consistency_was_enabled and requested_chains > 1 and (consensus_answer is not None or vote_counts is not None))

                if should_show_details_expander:
                    # Use a clear title for the main details expander
                    with st.expander("View Reasoning Details"):

                        # --- Content INSIDE the Main Expander (No Nested Expanders) ---

                        # a) Consensus and Vote Counts Section (only if SC was enabled and >1 chain requested)
                        if self_consistency_was_enabled and requested_chains > 1:
                            st.markdown("#### Consensus Details", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency

                            # Display Consensus Answer if available (useful redundancy in detail view)
                            # CHECK IF NOT NONE AND ENSURE STRING BEFORE STRIPPING AND DISPLAYING:
                            if consensus_answer is not None and str(consensus_answer).strip():
                                 # Use custom CSS class for premium consensus display within the expander
                                 # ENSURE consensus_answer IS A STRING FOR DISPLAY:
                                 st.markdown(f'<div class="consensus-answer"><strong>Consensus Answer:</strong> <p>{str(consensus_answer)}</p></div>', unsafe_allow_html=True)
                            else:
                                 st.info("No consensus answer found among the generated chains (all unique or invalid).")

                            # Display Self-Consistency Vote Counts
                            if vote_counts and isinstance(vote_counts, dict): # Explicitly check if vote_counts is a dictionary
                                # Sort by vote count descending
                                sorted_votes = sorted(vote_counts.items(), key=lambda item: item[1], reverse=True)
                                # Use custom CSS class for premium vote counts display within the expander
                                # ITERATE THROUGH ITEMS AND ENSURE BOTH KEY (ans) AND VALUE (count) ARE STRINGS FOR F-STRING:
                                vote_list_items = "".join([f"<li><code>{str(ans)[:150]}{'...' if len(str(ans)) > 150 else ''}</code>: {str(count)} vote(s)</li>" for ans, count in sorted_votes])
                                st.markdown(f'<div class="vote-counts"><strong>Self-Consistency Vote Counts ({len(generated_results)} Chains Analyzed):</strong><ul style="list-style-type: disc;">{vote_list_items}</ul></div>', unsafe_allow_html=True)
                            # Only show this info if SC was on and >1 chain requested but no votes recorded (and vote_counts wasn't a valid dict)
                            elif self_consistency_was_enabled and requested_chains > 1 and len(generated_results) > 0:
                                 st.info("Self-Consistency voting skipped: Fewer than 2 valid answers extracted from generated chains for voting.")

                            # Add a separator after Consensus/Votes if that section was shown
                            # Only add if the consensus details section was actually displayed
                            if (consensus_answer is not None and str(consensus_answer).strip()) or (vote_counts and isinstance(vote_counts, dict)):
                                st.markdown("---")


                        # b) Individual Reasoning Chains Section
                        st.markdown("#### Individual Reasoning Chains", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency

                        if not generated_results:
                             st.warning("No reasoning chains were generated successfully by the wrapper.")
                        elif not isinstance(generated_results, list): # Add check if generated_results is unexpectedly not a list
                            st.error("Error: generated_results content is not in expected list format.")
                            st.markdown(str(generated_results), unsafe_allow_html=True) # Display raw if not list
                        else:
                             # Iterate through EACH generated chain and display its details sequentially within the main expander
                             for i, chain_data in enumerate(generated_results):
                                 # Ensure chain_data is a dictionary before processing its contents
                                 if not isinstance(chain_data, dict):
                                     st.error(f"Error: Content for Chain {i+1} is not in expected dictionary format.")
                                     st.markdown(str(chain_data), unsafe_allow_html=True) # Display raw if not dict
                                     continue # Skip this chain if format is wrong

                                 chain_number = i + 1

                                 # Display Chain Number and Header (without an expander around each)
                                 st.markdown(f"##### Chain {chain_number}", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency


                                 # Safely get chain answer for display
                                 chain_answer = chain_data.get("final_answer", None)
                                 # Ensure the full extracted answer is used for display here and is a string
                                 # CHECK IF NOT NONE AND ENSURE STRING BEFORE STRIPPING:
                                 chain_answer_display = str(chain_answer) if chain_answer is not None and str(chain_answer).strip() else "N/A (Extraction Failed)"


                                 # Display Parsed Reasoning Steps
                                 reasoning_steps = chain_data.get("reasoning_steps", [])
                                 # Ensure reasoning_steps is a list and its items are strings before joining/displaying
                                 # CHECK IF LIST AND ENSURE EACH STEP IS A STRING BEFORE STRIPPING:
                                 if reasoning_steps and isinstance(reasoning_steps, list) and any(str(step).strip() for step in reasoning_steps):
                                     st.markdown("###### Reasoning Steps", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency
                                     # Use markdown to render list items properly, ensuring steps are strings and stripping whitespace
                                     # ENSURE EACH STEP IS STRING AND STRIP:
                                     steps_markdown = "\n".join([f"- {str(step).strip()}" for step in reasoning_steps if str(step).strip()])
                                     st.markdown(steps_markdown, unsafe_allow_html=True)
                                 else:
                                     st.info("No reasoning steps parsed for this chain.")


                                 # Display Parsed Final Answer (Chain Only) - Ensure full answer is shown and is a string
                                 st.markdown("###### Final Answer (Chain Only)", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency
                                 # Display the full extracted final answer without truncation
                                 # ENSURE THE DISPLAYED ANSWER IS STRING:
                                 st.markdown(f'**{str(chain_answer_display)}**', unsafe_allow_html=True) # Bold the answer


                                 # Display Full Generated Output (Chain Only) - Ensure full output is shown and is a string
                                 st.markdown("###### Full Generated Output (Chain Only)", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency
                                 # SAFELY GET AND ENSURE STRING:
                                 full_text = str(chain_data.get("full_text", "N/A (No output returned)"))
                                 # Use st.code for displaying raw text/code blocks clearly, handles long text
                                 # ENSURE INPUT TO st.code IS STRING:
                                 st.code(str(full_text), language='markdown') # Use markdown language hint


                                 # In a future multimodal output version, generated images for this chain might be displayed here too
                                 # if "generated_image_data" in chain_data and chain_data["generated_image_data"]:
                                 #     st.markdown("###### Generated Image (Chain Only)") # Added unsafe_allow_html=True for consistency
                                 #     try:
                                 #         img = Image.open(io.BytesIO(chain_data["generated_image_data"]))
                                 #         st.image(img, caption=f"Generated Image for Chain {chain_number}", use_column_width=True)
                                 #     except Exception as e:
                                 #         st.warning(f"Could not display generated image for Chain {chain_number}: {e}")

                                 # Add a separator BETWEEN chains, but not after the last one
                                 if i < len(generated_results) - 1:
                                     st.markdown("---", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency


                        # Add a separator before Generation Parameters if chains were shown OR consensus details were shown
                        # Only add if either the chains section or the consensus section was displayed
                        if generated_results or (self_consistency_was_enabled and requested_chains > 1 and (consensus_answer is not None or (vote_counts and isinstance(vote_counts, dict)))):
                             st.markdown("---", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency


                        # c) Generation Parameters Section (still inside the main details expander, NOT a nested expander)
                        st.markdown("#### Generation Parameters Used", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency

                        st.markdown("###### Model Info", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency
                        # ENSURE model_info IS STRING FOR DISPLAY:
                        st.info(str(model_info))

                        st.markdown("###### Parameters", unsafe_allow_html=True) # Added unsafe_allow_html=True for consistency
                        if generation_params and isinstance(generation_params, dict): # Explicitly check if generation_params is a dictionary
                             # Display parameters as a list, excluding redundant/internal ones
                             param_keys_to_exclude_from_display = ['self_consistency_enabled', 'pad_token_id', 'eos_token_id', 'requested_chains'] # Exclude from this display

                             filtered_gen_params = {k: v for k, v in generation_params.items() if k not in param_keys_to_exclude_from_display}

                             if filtered_gen_params:
                                  # Sort parameters alphabetically by key for consistent display
                                  sorted_display_params = sorted(filtered_gen_params.items())
                                  # Iterate through items and ENSURE BOTH KEY AND VALUE ARE STRINGS FOR F-STRING:
                                  params_markdown = "\n".join([f"- **{str(key)}:** {str(value)}" for key, value in sorted_display_params])
                                  st.markdown(params_markdown, unsafe_allow_html=True)
                             else:
                                  st.info("No specific generation parameters recorded for this turn.")
                        else:
                             st.info("No generation parameters recorded for this turn.")

                        # --- End of Content INSIDE the Main Expander ---

                    # Add a little space after the main expander if it was shown
                    st.markdown("<br>", unsafe_allow_html=True)

                # --- END Refactored Assistant CoT Response Rendering (Fixed Expander Nesting & Full Output as Main) ---

                # In a future multimodal output version, generated images that are NOT tied to specific chains
                # but might be a collective output could be displayed here (OUTSIDE the main details expander).

# Auto-scroll to the bottom of the chat if enabled
if st.session_state.get("auto_scroll"):
    st.markdown(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        unsafe_allow_html=True,
    )

# --- User input field and file uploader at the bottom ---
# Use a form to group input and file uploader so pressing enter submits both
with st.form("chat_form", clear_on_submit=True):
    # Use columns to place the text area and file uploader side-by-side
    col1, col2 = st.columns([4, 1]) # Adjust column ratios as needed

    with col1:
        user_input = st.text_area(
            "Your Message",
            key="user_input_text_area", # Persistent key
            label_visibility="collapsed", # Hide the label
            placeholder="Enter your question or problem here...",
            height=100 # Adjust height
        )

    with col2:
         # Multimodal Input (Functional if capable model is loaded)
         # Add a label above the uploader using markdown for consistent styling
         st.markdown("###### Upload Image")
         # Only allow file upload if a multimodal capable model is loaded
         if st.session_state.multimodal_capable_model_loaded:
              uploaded_files = st.file_uploader(
                  "Upload Image(s)",
                  type=["png", "jpg", "jpeg", "gif", "webp"], # Allowed image types
                  accept_multiple_files=True,
                  label_visibility="collapsed", # Hide the default label for compactness
                  # Limit 200MB per file handled by Streamlit automatically
              )
              # Store uploaded file bytes temporarily if files are uploaded
              uploaded_image_data = []
              if uploaded_files:
                   for uploaded_file in uploaded_files:
                        # Read file as bytes
                        try:
                            file_bytes = uploaded_file.getvalue()
                            uploaded_image_data.append(file_bytes)
                        except Exception as e:
                            logger.error(f"Error reading uploaded file {uploaded_file.name}: {e}")
                            st.warning(f"Could not read file: {uploaded_file.name}")
                            # Optionally clear the uploader here if a read fails, or just skip the file


         else:
              # Display the placeholder message and disable the uploader
              st.info("Upload disabled (Text-only or incompatible model loaded)")
              uploaded_files = None # Ensure uploaded_files is None
              uploaded_image_data = [] # Ensure uploaded_image_data is empty list


         # Send button - placed below the file uploader
         # This button is part of the form and submits when clicked or Enter is pressed in the text area
         # Add a markdown label above the button for consistent spacing/alignment
         st.markdown("###### Send")
         send_button = st.form_submit_button("Send")


# Define thinking_message_placeholder outside the form so it can be cleared later
# Place it before the main processing logic block that is triggered by the button press
thinking_message_placeholder = st.empty()


# --- Main generation logic triggered by send_button ---
# When the send button is clicked and there is user input OR uploaded files
if send_button and (user_input or uploaded_image_data):
    # Check if a model and wrapper are loaded AND the model is generation compatible
    is_model_generation_compatible = (
        st.session_state.model is not None and
        (hasattr(st.session_state.model, 'generate') and callable(getattr(st.session_state.model, 'generate', None)) or isinstance(st.session_state.model, GenerationMixin))
    )

    if st.session_state.model is None or st.session_state.processor is None or st.session_state.cot_wrapper is None or not is_model_generation_compatible:
        logger.warning("Attempted generation without loaded model/wrapper or with incompatible model.")
        if st.session_state.model is not None and not is_model_generation_compatible:
             st.warning("Loaded model is not compatible with text generation. Please load a different model.")
             assistant_msg_content = "🔴 Loaded model is not compatible with text generation. Please load a different model."
        else:
             st.warning("Please load a model first.")
             assistant_msg_content = "Please load a model first to start the conversation."


        # Add an assistant message indicating model is not loaded or incompatible
        # Use type "error_message" or "text" for simple text responses
        st.session_state.chat_history.append({
             "role": "assistant",
             "content": assistant_msg_content,
             "type": "error_message", # Use error_message type for warnings/errors
             "id": len(st.session_state.chat_history) # Assign new ID
        })
        st.rerun() # Rerun to display the warning

    else:
        # Add user message (text and images) to history
        # Store the user input as a dictionary to include image data
        user_message_content = {
            "text": user_input,
            "image_data": uploaded_image_data # Store list of image bytes
        }
        # Only add the user message and proceed with generation if there is text OR image data
        if user_input.strip() or uploaded_image_data:
             st.session_state.chat_history.append({"role": "user", "content": user_message_content, "type": "user_input", "id": len(st.session_state.chat_history)})
             logger.info(f"User input received (text: {len(user_input.strip()) > 0}, images: {len(uploaded_image_data)}). Proceeding with generation.")
        else:
             # If empty input and no image, do nothing and don't rerun
             logger.info("Empty user input and no image uploaded. Skipping generation.")
             # Clear the thinking placeholder just in case it was triggered somehow
             thinking_message_placeholder.empty()
             st.stop() # Stop script execution for this run if no input


        # Display a "Thinking..." message immediately below the user's prompt
        # Use the placeholder defined before this block
        thinking_message_placeholder.info("💭 NeuroReasoner is thinking...")


        # Prepare parameters for the wrapper
        # Access the parameter values directly from st.session_state using their keys
        self_consistency_enabled = st.session_state.get("self_consistency_checkbox", False)
        num_chains_requested = st.session_state.get("num_chains", 1)
        effective_num_chains = num_chains_requested if self_consistency_enabled else 1


        # Ensure required generation parameters are passed
        current_gen_params = {
            "max_new_tokens": st.session_state.get("max_new_tokens", 512),
            "temperature": st.session_state.get("temperature", 0.7),
            "top_k": st.session_state.get("top_k", 50),
            "top_p": st.session_state.get("top_p", 1.0),
            "do_sample": st.session_state.get("do_sample", True),
            "repetition_penalty": st.session_state.get("repetition_penalty", 1.1), # Use value from slider
            "no_repeat_ngram_size": st.session_state.get("no_repeat_ngram_size", 0), # Use value from slider
            "num_return_sequences": effective_num_chains, # Pass this to the wrapper

            # Include pad/eos token IDs and SC status for storage with response history
            # Safely access tokenizer attributes via processor
            "pad_token_id": getattr(getattr(st.session_state.processor, 'tokenizer', None), 'pad_token_id', None),
            "eos_token_id": getattr(getattr(st.session_state.processor, 'tokenizer', None), 'eos_token_id', None),
            "self_consistency_enabled": self_consistency_enabled, # Pass SC status boolean
            "requested_chains": num_chains_requested # Pass number of chains requested by slider
            # Add other parameters if needed
        }
        model_name_current = st.session_state.get("model_name_select", "N/A")
        device_current = st.session_state.get("device_select", "N/A")
        # Get actual model type and multimodal capability from session state after loading
        # Use the model's actual class name and the determined capability flag
        actual_model_class_name = type(st.session_state.model).__name__ if st.session_state.model else "Unknown"
        is_multimodal_compatible_loaded = st.session_state.multimodal_capable_model_loaded


        generation_status_text = f"✨ Generating {effective_num_chains} reasoning chain(s)..."
        with st.status(generation_status_text, expanded=True) as status_box:
             start_time = time.time()
             status_box.write("Starting generation process...")
             # If multimodal, indicate image processing is happening
             if uploaded_image_data and is_multimodal_compatible_loaded:
                  status_box.write(f"Processing {len(uploaded_image_data)} uploaded image(s)...")

             update_telemetry() # Update telemetry at start of generation

             outputs = None # Initialize outputs to None
             try:
                 # Call the wrapper's generate method
                 outputs = st.session_state.cot_wrapper.generate(
                     input_text=user_input,
                     image_data=uploaded_image_data,
                     generation_params=current_gen_params,
                     chat_history=st.session_state.chat_history
                 )
                 end_time = time.time()
                 duration = outputs.get("generation_duration", end_time - start_time) \
                            if isinstance(outputs, dict) else (end_time - start_time)
                 status_box.write(f"Generation complete in {duration:.2f} seconds.")
                 update_telemetry()  # Update telemetry after generation

                 # Process the outputs from the wrapper
                 generated_results_list = []
                 all_final_answers = []

                 # Ensure outputs is a dictionary and contains the expected keys with lists
                 # Check for the required keys and that their values are lists and are not None
                 if outputs and isinstance(outputs, dict) and \
                    isinstance(outputs.get("full_texts"), list) and \
                    isinstance(outputs.get("reasoning_steps"), list) and \
                    isinstance(outputs.get("final_answers"), list):

                      num_returned = len(outputs.get("full_texts", []))
                      logger.info(f"Wrapper returned {num_returned} sequences.")


                      if num_returned == 0:
                           status_box.warning("Wrapper returned no generated sequences.")
                      elif num_returned != effective_num_chains:
                           status_box.warning(f"Wrapper returned {num_returned} sequences, expected {effective_num_chains}. Displaying returned sequences.")

                      # Iterate through the number of sequences actually returned by the wrapper
                      for i in range(num_returned):
                           # Safely get elements for EACH sequence (chain)
                           # Use get with default None or "N/A" and ensure they are treated as strings later if needed for display
                           full_text = outputs.get("full_texts", [])[i] if i < len(outputs.get("full_texts", [])) else "N/A (No output)"
                           # Ensure reasoning_steps is a list for consistent storage, default to empty list if missing
                           reasoning_steps_data = outputs.get("reasoning_steps", [])[i] if i < len(outputs.get("reasoning_steps", [])) else [] # Default to empty list
                           # If reasoning_steps_data is not a list, store it as a single item list or string
                           reasoning_steps_list = reasoning_steps_data if isinstance(reasoning_steps_data, list) else [str(reasoning_steps_data)]

                           final_answer = outputs.get("final_answers", [])[i] if i < len(outputs.get("final_answers", [])) else None # Use None for missing/failed answer


                           # Append a dictionary for EACH chain to the list
                           chain_result = {
                                "full_text": full_text,
                                "reasoning_steps": reasoning_steps_list, # Store as list
                                "final_answer": final_answer
                                # In a future multimodal version, generated_image_data might be included here
                                # "generated_image_data": outputs.get("generated_images", [])[i] if i < len(outputs.get("generated_images", [])) else None # Placeholder
                           }
                           generated_results_list.append(chain_result)
                           # Collect all final answers (including None) for voting
                           all_final_answers.append(final_answer)


                 else:
                      logger.error("Wrapper returned unexpected output structure or empty lists.")
                      status_box.error("❌ Wrapper did not return expected output structure or empty lists.")
                      # generated_results_list and all_final_answers remain as initialized (empty or partially built)


                 # Perform Self-Consistency Voting if enabled and multiple chains were generated
                 consensus_answer = None
                 vote_counts = None
                 consensus_raw_answers = [] # Store raw answers matching consensus
                 # Only vote if self-consistency is enabled AND there's more than one chain returned AND there is at least one valid answer among them
                 if self_consistency_enabled and len(generated_results_list) > 1 and any(ans is not None and str(ans).strip() for ans in all_final_answers):
                     status_box.write("Performing Self-Consistency voting...")
                     # Pass the collected final answers from all chains (including None/empty for filtering in voting function)
                     consensus_answer, vote_counts, consensus_raw_answers = perform_self_consistency_voting(all_final_answers)
                     if consensus_answer is not None:
                         status_box.success(f"Self-Consistency consensus reached: '{consensus_answer}'")
                     else:
                          # This case should be covered by the any() check above, but as a safeguard
                          status_box.warning("Self-Consistency voting found no common answers or no valid answers among the generated chains.")

                 elif self_consistency_enabled and len(generated_results_list) > 0 and num_chains_requested > 1:
                      # SC is on, >1 chains requested, but <=1 valid chain was returned
                      # Check if any chains were returned at all before showing this
                      if len(generated_results_list) > 0:
                           status_box.info("Self-Consistency voting skipped: Fewer than 2 valid answers extracted from generated chains for voting.")
                      else:
                           # No chains generated at all
                           status_box.warning("Self-Consistency voting skipped: No chains were generated.")


                 # --- BEGIN Refactored Response Data Structure for Improved UX ---
                 # Determine the main answer to display directly in the chat bubble
                 main_output_to_show = None
                 main_output_label = "Output" # Default label for the main display

                 # Prioritize showing the full text of the first chain
                 if generated_results_list and len(generated_results_list) > 0:
                     # Safely get the full text from the first chain
                     first_chain_full_text = generated_results_list[0].get("full_text", "N/A (No full output returned for Chain 1)")
                     main_output_to_show = str(first_chain_full_text) # Ensure it's a string
                     main_output_label = "Full Output (Chain 1)"

                     # Add a note about the Consensus Answer if SC was enabled and consensus was found
                     if self_consistency_enabled and consensus_answer is not None and str(consensus_answer).strip():
                          main_output_label = "Full Output (Chain 1) | Consensus Answer Available Below"
                     else:
                          main_output_label = "Full Output (Chain 1)" # Just label it as Full Output


                 else:
                     # Fallback if no chains were generated at all
                     main_output_to_show = "Generation completed, but no output was returned or could be processed."
                     main_output_label = "Status"


                 # Store the structured response data in session state (update keys to reflect new main display)
                 assistant_response_data = {
                     "prompt": user_input, # Store the text input used for this turn

                     # Data for the main chat bubble display (now Full Output)
                     "main_output_display": main_output_to_show, # Use new key name
                     "main_output_label": main_output_label, # Use new key name

                     # Data for the expandable details section (Individual Chains & Vote Details)
                     "details": {
                         "generated_results": generated_results_list, # This list has entries for all chains
                         "consensus_answer": consensus_answer, # Store the raw consensus answer again within details
                         "vote_counts": vote_counts,
                         "consensus_raw_answers": consensus_raw_answers,
                         # Potentially add more detailed status messages here if needed in the expander
                         "self_consistency_enabled": self_consistency_enabled, # Pass SC status for rendering details
                         "requested_chains": num_chains_requested # Pass number of chains requested for details display
                     },

                     # Keep all existing metadata at the top level
                     "generation_params": current_gen_params, # Store actual params used
                     "model_name": model_name_current, # Store model info with the response
                     "device": device_current,
                    "model_type": actual_model_class_name, # Store the loaded model architecture class name
                    "is_multimodal_compatible": is_multimodal_compatible_loaded, # Store capability flag
                    "generation_duration": duration,

                    "type": "cot_response", # Mark this as a structured CoT response for the renderer
                    "id": len(st.session_state.chat_history) # Assign new ID
                }
                 st.session_state.chat_history.append({"role": "assistant", "content": assistant_response_data, "type": "cot_response", "id": len(st.session_state.chat_history)})
                 # --- END Refactored Response Data Structure ---


                 status_box.update(label="✅ Generation Complete", state="complete") # Mark status as complete

             except Exception as e:
                 end_time = time.time()
                 duration = end_time - start_time
                 status_box.error(f"❌ An error occurred during generation after {duration:.2f} seconds.")
                 logger.error(f"Error during wrapper generation call: {e}")
                 update_telemetry() # Update telemetry after error
                 st.exception(e) # Display the full exception

                 # Add an error message to history
                 # Use type "error_message" for simple text error responses
                 st.session_state.chat_history.append({
                     "role": "assistant",
                     "content": f"🔴 An unexpected error occurred during reasoning. Details: {e}", # Include error detail
                     "type": "error_message", # Use error_message type
                     "id": len(st.session_state.chat_history) # Assign new ID
                 })
                 status_box.update(label="🔴 Generation Failed", state="complete") # Mark status as complete


             # Clear the "Thinking..." message placeholder now that generation is done
             thinking_message_placeholder.empty()

             # 7) Cleanup after generation attempt (success or failure)
             if torch.cuda.is_available():
                 try:
                     torch.cuda.empty_cache()
                     logger.debug("GPU memory cache cleared after generation attempt.")
                 except Exception as cleanup_e:
                      logger.warning(f"Error during cuda empty_cache after generation attempt: {cleanup_e}")
                      pass # Suppress this warning unless in debug mode
             gc.collect()
             logger.debug("Garbage collection performed after generation attempt.")

             # 8) Re-render the entire chat history to show the new response
             st.rerun() # Re-run the script to display the updated history correctly
