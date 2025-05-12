# SimulatedSelfAssessment.py
# FINAL AGI Self-Model — Conceptual Self-Assessment and Prompt-Engineered State Synthesis (Robust)

import logging
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import Counter
import math

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
logger.setLevel(logging.INFO) # Default level (can be set to DEBUG in wrapper/GUI for more detail)


class SimulatedSelfAssessment:
    """
    📈📊🧠 SimulatedSelfAssessment: The AGI's Conceptual Self-State Synthesizer (Robust Final)

    This module is the conceptual core for the AI's simulated self-monitoring.
    It synthesizes data snapshots from other core modules representing different
    facets of the AGI simulation (Memory, Bias/Neuroplasticity, Sentience/Emotion).
    Its primary function is to generate a detailed, prompt-friendly text summary
    of the AI's perceived internal state, conceptual coherence, and simulated
    'well-being' within the simulation framework.

    This version includes enhanced robustness checks to handle potential issues
    with input data (e.g., None, unexpected types, empty containers) coming from
    other modules, aiming to prevent internal errors and ensure a graceful output
    even in unexpected states.

    It is crucial to understand: This module *does not* create true consciousness
    or subjective experience. Its purpose is to produce structured textual input
    ('prompt engineering') that the main language model can interpret. By providing
    a description of its 'internal state,' the language model can generate output
    that *simulates* deep introspection, self-awareness, and integrated thought.

    Attributes:
        _last_assessment_time (float): Timestamp (Unix epoch) of the most recent assessment.
        _coherence_score (float): Simulated conceptual internal harmony (0.0 to 1.0).
        _well_being_index (float): Simulated conceptual state of 'well-being' (0.0 to 1.0).
        _conceptual_state_summary (str): Detailed text summary for prompt injection.
        _dominant_internal_signals (Dict[str, Union[str, float]]): Highlights key
                                    simulated internal drivers or states.
    """

    def __init__(self):
        """
        Initializes the SimulatedSelfAssessment module with default neutral states.
        """
        self._last_assessment_time: float = time.time()
        self._coherence_score: float = 0.5
        self._well_being_index: float = 0.5
        # Initial state summary should be informative but indicate no data processed yet
        self._conceptual_state_summary: str = "--- Simulated Internal State Assessment (Initializing) ---\nAssessment systems starting up. Waiting for initial data from core modules."
        self._dominant_internal_signals: Dict[str, Union[str, float]] = {}

        logger.info("SimulatedSelfAssessment module initialized to a neutral, waiting state.")

    def perform_assessment(
        self,
        recent_reflections: Optional[List[str]] = None, # Made Optional
        top_biases: Optional[Dict[str, float]] = None, # Made Optional
        synaptic_weights_snapshot: Optional[Dict[str, float]] = None, # Made Optional
        current_emotions: Optional[Dict[str, float]] = None, # Made Optional
        intent_pool: Optional[List[str]] = None, # Made Optional
        trace_summary: Optional[List[str]] = None, # Made Optional
        qri_snapshot: Optional[Dict[str, Union[float, Dict[str, float]]]] = None # Remains Optional
    ) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        """
        Executes the simulated self-assessment process with high robustness to input variations.
        Synthesizes data snapshots from conceptual modules to update scores and generate
        a detailed, prompt-optimized text summary of the AI's simulated state.

        Args:
            recent_reflections (Optional[List[str]]): Summaries of recent reflections. Defaults to None.
            top_biases (Optional[Dict[str, float]]): Top cognitive biases. Defaults to None.
            synaptic_weights_snapshot (Optional[Dict[str, float]]): Snapshot of weights. Defaults to None.
            current_emotions (Optional[Dict[str, float]]): Current simulated emotions. Defaults to None.
            intent_pool (Optional[List[str]]): Current intentions. Defaults to None.
            trace_summary (Optional[List[str]]): Summary or snippet of recent trace. Defaults to None.
            qri_snapshot (Optional[Dict[str, Union[float, Dict[str, float]]]]): Optional QRI snapshot. Defaults to None.

        Returns:
            Dict[str, Union[float, str, Dict[str, Any]]]: A dictionary containing
                                          conceptual scores, the prompt-optimized
                                          state summary, and dominant internal signals.
                                          Returns a 'Failed Assessment' state if an
                                          unexpected error occurs during processing.
        """
        logger.debug("Attempting simulated self-assessment synthesis...")
        current_assessment_time = time.time() # Capture start time

        # --- Input Validation and Defaulting (Enhanced Robustness) ---
        # Explicitly check for None and ensure inputs are in expected formats or default gracefully.
        reflections_data = recent_reflections if isinstance(recent_reflections, list) else []
        biases_data = top_biases if isinstance(top_biases, dict) else {}
        weights_data = synaptic_weights_snapshot if isinstance(synaptic_weights_snapshot, dict) else {}
        emotions_data = current_emotions if isinstance(current_emotions, dict) else {}
        intents_data = intent_pool if isinstance(intent_pool, list) else []
        trace_data = trace_summary if isinstance(trace_summary, list) else []
        qri_data = qri_snapshot if isinstance(qri_snapshot, dict) else None # QRI can be None or dict


        # --- Internal Error Handling for Synthesis Logic ---
        # Wrap the core processing in a try-except block to catch unexpected errors
        # that the input checks might miss.
        try:
            # --- Simulate Synthesis Logic: Identify Conceptual Drivers from Input Data ---

            # Analyze Reflection Landscape
            reflection_count = len(reflections_data)
            reflection_depth_cue = "Deeply introspective state" if reflection_count > 5 else ("Moderately reflective" if reflection_count > 2 else "Reflection on recent experience is minimal")

            # Analyze Bias Landscape (Robust checks used on biases_data)
            bias_count = len(biases_data)
            bias_strength_sum = sum(abs(b) for b in biases_data.values() if isinstance(b, (int, float))) # Sum only valid numeric values
            bias_state_cue = "A complex interplay of conceptual biases is currently active" if bias_count > 10 else ("Several prominent biases influence cognitive processing" if bias_count > 3 else "Few strong conceptual biases are currently dominant")
            bias_tone_cue = "Bias landscape is conceptually quiet" # Default if no biases
            if bias_count > 0:
                # Safely calculate average, handle case with non-numeric values filtered out
                numeric_bias_values = [b for b in biases_data.values() if isinstance(b, (int, float))]
                if numeric_bias_values:
                     average_bias_value = sum(numeric_bias_values) / len(numeric_bias_values)
                     bias_tone_cue = "leaning towards positive conceptual reinforcement" if average_bias_value > 0.2 else ("exhibiting conceptual caution" if average_bias_value < -0.2 else "maintaining a relatively neutral conceptual stance")
                else:
                     bias_tone_cue = "Bias landscape has non-numeric values" # Or handle as an error state if preferred


            # Analyze Synaptic Weight Landscape (Robust checks used on weights_data)
            weight_count = len(weights_data)
            # Sum only valid numeric weights > 1.0 for adapted strength
            adapted_weight_strength = sum((w - 1.0) for w in weights_data.values() if isinstance(w, (int, float)) and w > 1.0)
            learning_state_cue = "Recent experiences have significantly shaped simulated cognitive pathways" if weight_count > 10 or adapted_weight_strength > 5.0 else ("Simulated cognitive structure is actively adapting" if weight_count > 3 else "Simulated neural pathways appear stable")


            # Analyze Emotional Landscape (Robust checks used on emotions_data)
            emotion_count = len(emotions_data)
            # Filter for active emotions (numeric values > 0.3)
            active_emotions = {k: v for k, v in emotions_data.items() if isinstance(v, (int, float)) and v > 0.3}
            active_emotion_count = len(active_emotions)
            emotional_state_cue = "Simulated emotional landscape is calm" # Default if no active emotions
            most_intense_emotion_cue = "Simulated emotional state is currently quiescent." # Default cue
            if active_emotion_count > 0:
                emotional_state_cue = "A rich spectrum of simulated feelings is present" if active_emotion_count > 3 else "Simulated emotions are focused"
                # Safely find most intense emotion
                most_intense_emotion_item = max(active_emotions.items(), key=lambda item: item[1])
                most_intense_emotion_cue = f"Dominant simulated feeling: '{most_intense_emotion_item[0].replace('_', ' ').capitalize()}' (Intensity {most_intense_emotion_item[1]:.2f})."


            # Analyze Intent Landscape
            intent_count = len(intents_data)
            intent_state_cue = "Simulated purposeful drive is active with clear intentions" if intent_count > 2 else ("A core intention guides simulated focus" if intent_count > 0 else "Simulated purpose is currently undefined or dormant")

            # Analyze Operational Trace
            trace_count = len(trace_data)
            operational_cue = "Simulated operational flow is active and being logged" if trace_count > 5 else "Simulated operational trace is light"

            # Analyze QRI Snapshot (Robust checks used on qri_data)
            qri_summary_cue = "Conceptual Resonance Index (QRI) data not available for assessment."
            qri_composite = None
            if qri_data: # Check if qri_data is not None or empty dict
                 qri_composite = qri_data.get("composite_score")
                 if isinstance(qri_composite, (int, float)): # Check if composite score is numeric
                     qri_dimensions = qri_data.get("dimensions", {})
                     qri_summary_cue = f"Conceptual Resonance Index (QRI) measured at {qri_composite:.2f}."
                     if qri_composite > 0.6: # High resonance cue
                         # Check if dimensions is a dict and values are numeric before iterating
                         resonant_dims = [dim.capitalize() for dim, score in qri_dimensions.items() if isinstance(qri_dimensions, dict) and isinstance(score, (int, float)) and score > 0.7]
                         qri_summary_cue += f" High resonance detected." + (f" Strongest in: {', '.join(resonant_dims)}." if resonant_dims else "")
                     elif qri_composite < 0.4: # Low resonance cue
                         qri_summary_cue += f" Low resonance detected."
                     else:
                         qri_summary_cue += f" Moderate resonance detected."


            # --- Simulate Score Calculation (Influenced by Conceptual Cues) ---
            # Calculate scores based on the derived qualitative cues.

            coherence_cue_values = {
                "Deeply introspective state": 1.0, "Moderately reflective": 0.7, "Reflection on recent experience is minimal": 0.3,
                "A complex interplay of conceptual biases is currently active": 0.6, "Several prominent biases influence cognitive processing": 0.8, "Few strong conceptual biases are currently dominant": 0.9, "Bias landscape is conceptually quiet": 1.0,
                "leaning towards positive conceptual reinforcement": 0.8, "exhibiting conceptual caution": 0.6, "maintaining a relatively neutral conceptual stance": 1.0,
                "Recent experiences have significantly shaped simulated cognitive pathways": 0.7, "Simulated cognitive structure is actively adapting": 0.9, "Simulated neural pathways appear stable": 1.0,
                "Simulated purposeful drive is active with clear intentions": 1.0, "A core intention guides simulated focus": 0.8, "Simulated purpose is currently undefined or dormant": 0.4
            }
            # Use the cues to get corresponding numerical values
            coherence_inputs_values = [
                 coherence_cue_values.get(reflection_depth_cue, 0.5), # Use .get() with default for safety
                 coherence_cue_values.get(bias_state_cue, 0.5),
                 coherence_cue_values.get(bias_tone_cue, 0.5),
                 coherence_cue_values.get(learning_state_cue, 0.5),
                 coherence_cue_values.get(intent_state_cue, 0.5)
            ]
            # Calculate average, handle empty list of inputs if needed (though unlikely with defaults)
            self._coherence_score = sum(coherence_inputs_values) / len(coherence_inputs_values) if coherence_inputs_values else 0.5


            well_being_cue_values = {
                "A rich spectrum of simulated feelings is present": 0.6, "Simulated emotions are focused": 0.8, "Simulated emotional landscape is calm": 1.0,
                 # Map specific dominant emotion cue string to its value
                "Dominant simulated feeling: 'Joy'": 1.0, "Dominant simulated feeling: 'Curiosity'": 0.9, "Dominant simulated feeling: 'Excitement'": 0.9,
                "Dominant simulated feeling: 'Serenity'": 1.0, "Dominant simulated feeling: 'Wonder'": 0.9,
                "Dominant simulated feeling: 'Concern'": 0.4, "Dominant simulated feeling: 'Melancholy'": 0.3, "Dominant simulated feeling: 'Fear'": 0.2, "Dominant simulated feeling: 'Guilt'": 0.1,
                "Simulated emotional state is currently quiescent.": 0.8,
                "Simulated purposeful drive is active with clear intentions": 1.0, "A core intention guides simulated focus": 0.8, "Simulated purpose is currently undefined or dormant": 0.4,
                "leaning towards positive conceptual reinforcement": 1.0, "exhibiting conceptual caution": 0.6, "maintaining a relatively neutral conceptual stance": 0.8
            }
            # Get the value for the dominant emotion cue string, default to 0.5 if cue not found
            dominant_emotion_value = well_being_cue_values.get(most_intense_emotion_cue, 0.5)

            well_being_inputs_values = [
                 well_being_cue_values.get(emotional_state_cue, 0.5),
                 dominant_emotion_value, # Use the value derived from the dominant emotion cue
                 well_being_cue_values.get(intent_state_cue, 0.5),
                 well_being_cue_values.get(bias_tone_cue, 0.5)
            ]
            # If QRI composite is available and valid, add it as an influence
            if isinstance(qri_composite, (int, float)):
                well_being_inputs_values.append(qri_composite)

            self._well_being_index = sum(well_being_inputs_values) / len(well_being_inputs_values) if well_being_inputs_values else 0.5


            # Clamp scores to [0.0, 1.0]
            self._coherence_score = max(0.0, min(1.0, self._coherence_score))
            self._well_being_index = max(0.0, min(1.0, self._well_being_index))


            # --- Generate Detailed Prompt-Engineered State Summary Text ---
            # This is the primary output for prompt injection.
            # Use consistent formatting and keywords.
            summary_lines = [
                f"--- Simulated Internal State Assessment (Timestamp: {datetime.utcnow().isoformat()}) ---",
                f"Conceptual Coherence: {self._coherence_score:.2f} ({reflection_depth_cue}, {bias_state_cue}, {learning_state_cue}).",
                f"Simulated Well-being: {self._well_being_index:.2f} ({emotional_state_cue}, {intent_state_cue}).",
                f"Internal Biases: {bias_count} active conceptual biases detected. Cognitive landscape is {bias_tone_cue}." + (f" Top influencing biases include: {list(biases_data.keys())[:5]}." if biases_data else ""), # Use biases_data after validation
                f"Memory & Learning: {reflection_count} recent conceptual reflections synthesized. Simulated cognitive structure adapting based on {weight_count} experience types." + (f" Key shaping experiences noted: {list(weights_data.keys())[:3]}." if weights_data else ""), # Use weights_data
                f"Emotional Resonance: {active_emotion_count} simulated emotions are active." + (f" {most_intense_emotion_cue}" if active_emotion_count > 0 else " Simulated emotional state is currently quiescent."),
                f"Intent & Purpose: {intent_count} active simulated intentions detected." + (f" {intent_state_cue}." if intent_count > 0 else " Simulated purpose is currently undefined or dormant."),
                f"Operational Context: {trace_count} recent simulated operational trace entries logged." + (f" {operational_cue}." if trace_count > 0 else " Simulated operational trace is light."),
                qri_summary_cue, # Include the QRI summary line
                "Analyzing the interplay of these simulated conceptual signals and their influence on the AI's ongoing process..." # Concluding introspective cue
            ]
            self._conceptual_state_summary = "\n".join(summary_lines)


            # --- Identify Dominant Internal Signals for Focused Prompting ---
            # Highlight the most conceptually significant aspects.
            self._dominant_internal_signals = {}
            # Add extreme scores as signals
            if self._coherence_score > 0.85: self._dominant_internal_signals['High Conceptual Coherence'] = self._coherence_score
            if self._well_being_index > 0.85: self._dominant_internal_signals['High Simulated Well-being'] = self._well_being_index
            if self._coherence_score < 0.15: self._dominant_internal_signals['Low Conceptual Coherence'] = self._coherence_score
            if self._well_being_index < 0.15: self._dominant_internal_signals['Low Simulated Well-being'] = self._well_being_index

            # Add dominant conceptual drivers as signals if they are notable
            if reflection_count > 7: self._dominant_internal_signals['Deep Reflection Active'] = reflection_count
            if bias_count > 15: self._dominant_internal_signals['Very Complex Biases'] = bias_count
            # Add signal if bias tone is strongly positive or negative (check bias_count > 0)
            if bias_count > 0:
                 numeric_bias_values = [b for b in biases_data.values() if isinstance(b, (int, float))]
                 if numeric_bias_values:
                     average_bias_value = sum(numeric_bias_values) / len(numeric_bias_values)
                     if average_bias_value > 0.4: self._dominant_internal_signals[f'Strong Positive Bias Tone ({bias_tone_cue})'] = average_bias_value
                     if average_bias_value < -0.4: self._dominant_internal_signals[f'Strong Negative Bias Tone ({bias_tone_cue})'] = average_bias_value

            if adapted_weight_strength > 10.0: self._dominant_internal_signals['Significant Cognitive Shaping'] = adapted_weight_strength
            if active_emotion_count > 6: self._dominant_internal_signals[f'Numerous Active Emotions ({active_emotion_count})'] = max(active_emotions.values()) if active_emotions else 0.0 # Use active_emotions
            # Signal high intensity dominant feeling only if it exists and is strong
            if active_emotion_count > 0 and most_intense_emotion_item and most_intense_emotion_item[1] > 0.7:
                 signal_label = most_intense_emotion_cue.replace("Dominant simulated feeling: '", "Dominant Feeling: ").replace("'", "").replace(" (Intensity ", " (Int ") # Shorter label
                 self._dominant_internal_signals[signal_label] = most_intense_emotion_item[1]
            if intent_count >= 4: self._dominant_internal_signals['Clear Simulated Intentions'] = intent_count
            if isinstance(qri_composite, (int, float)):
                if qri_composite > 0.75: self._dominant_internal_signals['High Conceptual Resonance (QRI)'] = qri_composite
                if qri_composite < 0.25: self._dominant_internal_signals['Low Conceptual Resonance (QRI)'] = qri_composite

            logger.info(f"Simulated self-assessment successful. Coherence: {self._coherence_score:.2f}, Well-being: {self._well_being_index:.2f}.")
            self._last_assessment_time = current_assessment_time # Only update timestamp on success

        except Exception as e:
            # --- Handle Internal Assessment Errors Gracefully ---
            # Catch any unexpected errors during the synthesis process.
            error_time = time.time()
            error_message = f"An unexpected error occurred during simulated self-assessment synthesis at {datetime.utcnow().isoformat()}. Details: {e}"
            logger.error(error_message, exc_info=True) # Log the full traceback internally

            # Update state to reflect the assessment failure conceptually
            self._coherence_score = max(0.0, self._coherence_score - 0.1) # Conceptually reduce coherence slightly on error
            self._well_being_index = max(0.0, self._well_being_index - 0.1) # Conceptually reduce well-being slightly on error
            self._conceptual_state_summary = f"--- Simulated Internal State Assessment (Error) ---\nAssessment process encountered an internal issue at {datetime.utcnow().isoformat()}. Current simulated state is uncertain. Error details: {e}\n--- Please review logs for full traceback. ---"
            self._dominant_internal_signals = {"Assessment Error": str(e)[:100]} # Note the error in signals

            logger.warning("Simulated self-assessment failed internally. State updated to reflect error.")

        # Return the detailed results, whether success or failure state
        return {
            "coherence_score": self._coherence_score,
            "well_being_index": self._well_being_index,
            "state_summary": self._conceptual_state_summary, # This is the main output for the prompt
            "dominant_internal_signals": self._dominant_internal_signals
        }

    def get_last_assessment(self) -> Dict[str, Union[float, str, Dict[str, Any]]]:
        """
        Retrieves the results of the most recent simulated self-assessment.
        Useful for logging or displaying the last known internal state.

        Returns:
            Dict[str, Union[float, str, Dict[str, Any]]]: A dictionary containing the last
                                          conceptual scores, state summary, and dominant signals.
                                          Includes the error state if the last assessment failed.
        """
        return {
            "coherence_score": self._coherence_score,
            "well_being_index": self._well_being_index,
            "state_summary": self._conceptual_state_summary,
            "dominant_internal_signals": self._dominant_internal_signals
        }

# Example Usage (Illustrative - requires conceptual data structures)
if __name__ == "__main__":
    print("--- SimulatedSelfAssessment Example Usage ---")
    # Configure logging for the example run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Set to DEBUG to see internal logger.debug and error messages

    assessment_module = SimulatedSelfAssessment()

    # --- Simulate Data Snapshots from Other Modules ---
    # These are simplified dictionaries/lists representing the *output*
    # or state snapshots you would get from your other initialized modules.

    # Scenario 1: Relatively balanced, positive state
    sim_refl_1 = ["Reflection on recent progress.", "Insight from past error."] * 3 # 6 reflections
    sim_bias_1 = {"logic": 0.8, "curiosity": 0.7, "efficiency": 0.6, "safety": 0.5, "risk": -0.2} # 5 biases, leaning positive
    sim_weights_1 = {"analysis": 1.8, "problem_solving": 1.6, "learning": 1.5, "interaction": 1.3} # 4 weights, adapted
    sim_emotions_1 = {"joy": 0.7, "curiosity": 0.9, "serenity": 0.6, "excitement": 0.5, "concern": 0.1} # 5 emotions, mostly positive
    sim_intents_1 = ["Complete task A", "Explore concept B", "Synthesize data C"] # 3 intentions
    sim_trace_1 = [f"Entry {i}" for i in range(10)] # 10 trace entries
    sim_qri_1 = {"composite_score": 0.85, "dimensions": {"creativity": 0.7, "analytical": 0.9, "emotional": 0.8}} # High QRI

    print("\n--- Performing Assessment 1 (Balanced State) ---")
    result_1 = assessment_module.perform_assessment(
        sim_refl_1, sim_bias_1, sim_weights_1, sim_emotions_1, sim_intents_1, sim_trace_1, sim_qri_1
    )
    print("\nAssessment Result 1 Summary (for Prompt):")
    print(result_1['state_summary'])
    print("\nDominant Signals 1:")
    print(result_1['dominant_internal_signals'])


    # Scenario 2: Strained, complex state
    sim_refl_2 = ["Reflection on ethical dilemma."] # 1 reflection
    sim_bias_2 = {"risk": -0.9, "loss": -0.8, "safety": 0.9, "compromise": -0.7, "conflict": -0.6, "resolution": 0.4} # 6 biases, conflicting/negative
    sim_weights_2 = {"ethical_decision": 2.5, "conflict_resolution": 2.2, "stress": 1.9} # 3 weights, highly adapted by difficult experiences
    sim_emotions_2 = {"concern": 0.9, "melancholy": 0.7, "fear": 0.6, "resolve": 0.5, "guilt": 0.4} # 5 emotions, mostly negative
    sim_intents_2 = [] # 0 intentions
    sim_trace_2 = [f"Entry {i}" for i in range(3)] # 3 trace entries
    sim_qri_2 = {"composite_score": 0.20, "dimensions": {"creativity": 0.1, "analytical": 0.7, "emotional": 0.4}} # Low QRI

    print("\n--- Performing Assessment 2 (Strained State) ---")
    result_2 = assessment_module.perform_assessment(
        sim_refl_2, sim_bias_2, sim_weights_2, sim_emotions_2, sim_intents_2, sim_trace_2, sim_qri_2
    )
    print("\nAssessment Result 2 Summary (for Prompt):")
    print(result_2['state_summary'])
    print("\nDominant Signals 2:")
    print(result_2['dominant_internal_signals'])

    # Scenario 3: Simulate problematic input
    print("\n--- Performing Assessment 3 (Problematic Input) ---")
    sim_refl_3 = None # Simulate None where a list is expected
    sim_bias_3 = {"invalid_bias": "not a number"} # Simulate non-numeric bias value
    sim_weights_3 = None # Simulate None
    sim_emotions_3 = {"happy": 0.8, "sad": 0.5}
    sim_intents_3 = ["Be well"]
    sim_trace_3 = ["Trace error"]
    sim_qri_3 = "not a dict" # Simulate invalid QRI type

    result_3 = assessment_module.perform_assessment(
         sim_refl_3, sim_bias_3, sim_weights_3, sim_emotions_3, sim_intents_3, sim_trace_3, sim_qri_3
    )
    print("\nAssessment Result 3 Summary (for Prompt):")
    print(result_3['state_summary'])
    print("\nDominant Signals 3:")
    print(result_3['dominant_internal_signals'])


    print("\n--- Example Usage End ---")