# NeuroMemoryProcessor.py
# Finalized AGI Self-Model — Simulated Neural Plasticity and Cognitive Biases

import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# Attempt to import torch, handle gracefully if not available


# --- Logging Setup ---
# Configure logging specifically for the NeuroMemoryProcessor module.
logger = logging.getLogger(__name__)
# Set level to INFO by default. The main GUI or wrapper can set it to DEBUG if needed.
# Ensure handlers are not added multiple times.
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent logs from going to root logger if root also has handlers
logger.setLevel(logging.INFO) # Default level


class NeuroMemoryProcessor:
    """
    📝⚙️🧬 NeuroMemoryProcessor: Simulated Neural Plasticity and Evolving Cognitive Biases 🔄🧠🔧

    This class simulates a simplified model of neural adaptation and the formation
    of cognitive biases within an artificial general intelligence. It tracks
    simulated "synaptic weights" associated with different types of experiences
    and develops "cognitive biases" linked to specific tokens or concepts encountered
    in the environment.

    It integrates simulated emotional data as a modulator, influencing how existing
    biases are amplified or dampened. This creates a dynamic internal state that
    could conceptually influence attention, decision-making, or the interpretation
    of new information in other parts of an AGI system.

    The processor maintains a log of recorded experiences, serving as a form of
    simulated experiential memory, limited by a configurable capacity.

    Attributes:
        memory_capacity (int or float): Maximum number of recorded experiences in `long_term_memory`.
                                        Float('inf') for unlimited.
        plasticity_range (Tuple[float, float]): Defines the (min, max) range for the random
                                                 increment applied to synaptic weights during adaptation.
        bias_increment (float): The base amount added to a token's cognitive bias upon
                                 encountering it during bias evolution.
        decay_rate (float): The exponential decay factor (0.0 to 1.0) applied to all
                            cognitive biases over time or upon certain events (like emotion updates).
        long_term_memory (List[Dict[str, Any]]): A chronological list of recorded experience entries.
        synaptic_weights (Dict[str, float]): Simulated synaptic weights mapped to experience types.
        cognitive_biases (Dict[str, float]): Simulated cognitive biases mapped to encountered tokens.
    """

    def __init__(
        self,
        memory_capacity: int = 200, # Increased default capacity
        plasticity_range: Tuple[float, float] = (0.1, 0.4), # Slightly adjusted range
        bias_increment: float = 0.08, # Slightly increased bias increment
        decay_rate: float = 0.985 # Slightly adjusted decay rate
    ):
        """
        Initializes the NeuroMemoryProcessor with its core configuration and internal states.

        Args:
          memory_capacity (int): Maximum number of entries in the recorded experience
                                 log (`long_term_memory`). When capacity is finite
                                 and exceeded, the oldest entry is evicted.
                                 Defaults to 200. Set to 0 for conceptual unlimited capacity.
          plasticity_range (Tuple[float, float]): A tuple defining the (minimum, maximum)
                                                  range for the random value added to
                                                  synaptic weights during adaptation, simulating
                                                  the varying strength of neural connections.
                                                  Defaults to (0.1, 0.4).
          bias_increment (float): The fixed amount added to a token's cognitive
                                  bias each time it is encountered during bias evolution,
                                  simulating reinforcement of concepts. Defaults to 0.08.
          decay_rate (float): The exponential decay factor (0.0 to 1.0) applied to all
                              cognitive biases over time or upon updates. Values closer to 1.0
                              result in slower decay. Defaults to 0.985.
        """
        if memory_capacity < 0:
            logger.warning(f"Invalid memory_capacity ({memory_capacity}). Setting to default (200).")
            self.memory_capacity: int = 200
        elif memory_capacity == 0:
            logger.info("Memory capacity set to unlimited (0).")
            self.memory_capacity: float = float('inf') # Use infinity for conceptual unlimited
        else:
            self.memory_capacity: int = memory_capacity

        # Validate and set plasticity_range
        if not (isinstance(plasticity_range, tuple) and len(plasticity_range) == 2 and all(isinstance(x, (int, float)) for x in plasticity_range)):
            logger.warning(f"Invalid plasticity_range format: {plasticity_range}. Setting to default (0.1, 0.4).")
            self.plasticity_range: Tuple[float, float] = (0.1, 0.4)
        else:
             # Ensure min <= max and values are non-negative
             min_inc, max_inc = sorted(plasticity_range) # Ensure order
             self.plasticity_range: Tuple[float, float] = (max(0.0, min_inc), max(0.0, max_inc))
             if self.plasticity_range != plasticity_range:
                  logger.warning(f"Clamped invalid plasticity_range {plasticity_range} to {self.plasticity_range}.")


        # Safely convert and set bias_increment and decay_rate
        try:
            self.bias_increment: float = max(0.0, float(bias_increment)) # Ensure non-negative
            if float(bias_increment) < 0: logger.warning("bias_increment was negative, clamped to 0.0.")
        except (ValueError, TypeError):
            logger.warning(f"Invalid bias_increment ({bias_increment}). Setting to default (0.08).")
            self.bias_increment: float = 0.08

        try:
            decay_rate_float = float(decay_rate)
            # Ensure decay_rate is within the valid range [0.0, 1.0]
            if not (0.0 <= decay_rate_float <= 1.0):
                logger.warning(f"Decay rate ({decay_rate_float}) outside [0.0, 1.0] range. Clamping.")
                self.decay_rate: float = max(0.0, min(1.0, decay_rate_float))
            else:
                self.decay_rate: float = decay_rate_float
        except (ValueError, TypeError):
            logger.warning(f"Invalid decay_rate ({decay_rate}). Setting to default (0.985).")
            self.decay_rate: float = 0.985


        # Initialize internal states
        self.long_term_memory: List[Dict[str, Any]] = []
        self.synaptic_weights: Dict[str, float] = {} # Default weight implicitly 1.0 on first access
        self.cognitive_biases: Dict[str, float] = {} # Default bias implicitly 0.0 on first access

        logger.info(f"NeuroMemoryProcessor initialized with capacity={self.memory_capacity if self.memory_capacity != float('inf') else 'Unlimited'}, plasticity={self.plasticity_range}, bias_inc={self.bias_increment:.3f}, decay={self.decay_rate:.3f}.")


    def record_experience(self, kind: str, detail: str) -> Dict[str, Any]:
        """
        📝 Records a new experience event in the processor's experiential memory.
        This acts as a fundamental input signal that triggers the simulation of
        synaptic weight adaptation for the experience type and cognitive bias
        evolution based on the experience's detail text. Enforces the memory capacity.

        Args:
            kind (str): The type of experience (e.g., "observation", "step", "emotion", "reflection").
                        Used for synaptic weight adaptation.
            detail (str): A textual description or content associated with the experience.
                          Used for cognitive bias evolution.

        Returns:
            Dict[str, Any]: The dictionary representation of the stored experience entry.
                            Returns an empty dict if the detail is empty after processing.
        """
        # Validate and clean inputs
        if not isinstance(kind, str) or not kind.strip():
            logger.warning(f"Attempted to record experience with invalid kind: '{kind}'. Using 'unknown_kind'.")
            kind = "unknown_kind"
        if not isinstance(detail, str):
            logger.warning(f"Attempted to record experience with non-string detail (type: {type(detail)}). Converting to string.")
            detail = str(detail)

        # Do not record entries with empty detail after string conversion and stripping
        if not detail.strip():
            logger.debug(f"Skipping recording experience of kind '{kind}' with empty detail.")
            return {} # Return empty dict if detail is empty


        ts = datetime.utcnow().isoformat()
        # Store a potentially truncated version of the detail for memory efficiency if it's very long
        detail_stored = detail[:1500] + "..." if len(detail) > 1500 else detail # Increased stored detail length


        entry = {"type": kind, "detail": detail_stored, "timestamp": ts}
        self.long_term_memory.append(entry)

        # Enforce memory capacity limit (if > 0 and finite)
        if self.memory_capacity > 0 and self.memory_capacity != float('inf') and len(self.long_term_memory) > self.memory_capacity:
            try:
                dropped = self.long_term_memory.pop(0) # Remove the oldest entry
                logger.debug(f"Memory full ({self.memory_capacity}). Evicted oldest: type='{dropped.get('type')}', detail='{dropped.get('detail', '')[:50]}...'")
            except IndexError:
                 # This case indicates a potential logic error if len > capacity but pop(0) fails
                 logger.error("Attempted to pop from unexpectedly empty long_term_memory queue during capacity enforcement.")


        # Trigger simulation updates based on this experience
        # Use the original, potentially longer detail for bias evolution for more context
        self._adapt_synaptic_weights(kind)
        self._evolve_cognitive_bias(detail)

        logger.debug(f"Recorded experience and triggered adaptation/evolution: type='{kind}', detail='{detail_stored[:50]}...'")

        return entry

    def _adapt_synaptic_weights(self, kind: str) -> None:
        """
        ⚙️ Simulates synaptic plasticity by increasing the weight associated with
        a specific experience type (`kind`). This represents the strengthening
        of neural pathways related to processing this type of information.
        The increase amount is a random value within the defined `plasticity_range`.

        Args:
            kind (str): The type of experience whose synaptic weight should be adapted.
        """
        if not isinstance(kind, str) or not kind.strip():
            logger.warning(f"Cannot adapt synaptic weight for invalid kind: '{kind}'. Skipping adaptation.")
            return

        # Ensure plasticity_range is valid and non-negative before using random.uniform
        min_inc, max_inc = self.plasticity_range
        # Ensure min <= max
        if min_inc > max_inc:
             logger.warning(f"Invalid plasticity_range {self.plasticity_range}: min > max. Swapping bounds.")
             min_inc, max_inc = max_inc, min_inc
        # Ensure bounds are non-negative
        min_inc, max_inc = max(0.0, min_inc), max(0.0, max_inc)

        # If range is still invalid (e.g., both bounds were negative), default to a small range
        if max_inc < min_inc:
             logger.warning(f"Plasticity range remained invalid after clamping: ({min_inc}, {max_inc}). Using default small range.")
             min_inc, max_inc = (0.01, 0.05)


        try:
            # Generate random increment within the valid range
            inc = random.uniform(min_inc, max_inc)
            # Get current weight, default to 1.0 if not seen before (conceptually neutral baseline)
            old_weight = self.synaptic_weights.get(kind, 1.0)
            self.synaptic_weights[kind] = old_weight + inc
            # Optional: Cap weights at a maximum value to prevent unbounded growth? (Not implemented here)
            logger.debug(f"Adapted weight for '{kind}': {old_weight:.3f} → {self.synaptic_weights[kind]:.3f} (Increment: +{inc:.3f})")
        except Exception as e:
            logger.error(f"Error adapting synaptic weight for '{kind}' with range {self.plasticity_range}: {e}. Weight not updated.")


    def _evolve_cognitive_bias(self, detail: str) -> None:
        """
        🧬 Simulates the evolution of cognitive biases linked to specific tokens
        or concepts encountered in the detail text. This represents the AI becoming
        more sensitive or predisposed towards concepts it encounters frequently.
        Bias values increase upon encounter after a general decay is applied.

        Note: This uses a simple space-based split and lowercasing for tokenization.
        For a more sophisticated simulation, a proper NLP tokenizer and text
        processing pipeline would be required.

        Args:
            detail (str): The text content used to evolve biases.
        """
        if not isinstance(detail, str) or not detail.strip():
            logger.debug("Skipping cognitive bias evolution for empty detail text.")
            return

        # Apply decay to existing biases *before* reinforcing new ones
        self._decay_biases()

        # Use a basic space split for tokenization and lowercase as in original code
        tokens = detail.lower().split()

        if not tokens:
            logger.debug("No tokens found in detail for bias evolution after splitting.")
            return

        reinforced_tokens_count = 0
        for token in tokens:
            # Basic cleaning for tokens (remove punctuation etc.) could be added here
            # For now, stick to lower() and strip() as in the original code's spirit.
            cleaned_token = token.strip()

            if cleaned_token: # Ensure token is not just empty string after strip
                # Increase the bias for the cleaned token
                old_bias = self.cognitive_biases.get(cleaned_token, 0.0) # Start bias at 0.0
                self.cognitive_biases[cleaned_token] = old_bias + self.bias_increment
                # Optional: Cap bias value if needed (Not implemented here)
                # logger.debug(f"Evolved bias for '{cleaned_token}': {old_bias:.3f} → {self.cognitive_biases[cleaned_token]:.3f} (+{self.bias_increment:.3f})") # Too verbose
                reinforced_tokens_count += 1


        logger.debug(f"Evolved biases for {reinforced_tokens_count} unique tokens from detail.")


    def _decay_biases(self) -> None:
        """
        Applies an exponential decay to all existing cognitive biases based on
        the configured `decay_rate`. This simulates the natural fading of biases
        over time or processing cycles if they are not reinforced by new encounters.
        Also removes biases that decay below a very small threshold.
        Called internally by `_evolve_cognitive_bias` and `update_biases`.
        """
        # Create a list of items to decay to avoid changing dict size during iteration
        biases_to_decay = list(self.cognitive_biases.items())
        decayed_count = 0
        removed_count = 0

        if not biases_to_decay:
             logger.debug("No cognitive biases to decay.")
             return

        for token, bias in biases_to_decay:
            new_bias = bias * self.decay_rate
            # Remove bias if its absolute value falls below a small threshold to keep the dictionary clean
            if abs(new_bias) < 1e-9: # Use a very small threshold
                if token in self.cognitive_biases: # Check exists before deleting (safety)
                    del self.cognitive_biases[token]
                    # logger.debug(f"Decayed and removed bias for '{token}' (was {bias:.6f}, now {new_bias:.6f})") # Too verbose
                    removed_count += 1
            else:
                self.cognitive_biases[token] = new_bias
                # logger.debug(f"Decayed bias for '{token}': {bias:.3f} → {new_bias:.3f}") # Too verbose
                decayed_count += 1

        # logger.debug(f"Decayed {decayed_count} cognitive biases. Removed {removed_count} low biases.") # Too verbose


    def update_biases(self, emotion_data: Dict[str, Any]) -> None:
        """
        🔄 Integrates simulated emotional data into the cognitive bias landscape.
        This method processes a given emotional state, records it as an experience
        (triggering general bias evolution for the emotion words), and then
        applies a combined amplification and decay factor to *all* existing
        cognitive biases. Higher emotional intensity leads to stronger amplification,
        while the decay rate still applies, making biases more volatile or reinforced
        depending on the emotional state.

        Args:
            emotion_data (Dict[str, Any]): A dictionary containing emotional
                                           information. Expected to have keys
                                           'primary_emotion' (str) and 'intensity' (float).
                                           Intensity is typically between 0.0 and 1.0.
        """
        # Validate input format
        if not isinstance(emotion_data, dict):
            logger.warning(f"Attempted to update biases with non-dictionary emotion_data (type: {type(emotion_data)}). Skipping update.")
            return

        # Safely get emotion kind and convert to string
        emotion_kind = str(emotion_data.get("primary_emotion", "emotional_event")) # Default kind if missing

        # Safely get and convert intensity, clamp to [0.0, 1.0]
        try:
            intensity = float(emotion_data.get("intensity", 0.0))
            intensity = max(0.0, min(1.0, intensity)) # Clamp intensity
        except (ValueError, TypeError):
            intensity = 0.0
            logger.warning(f"Invalid intensity value in emotion_data: {emotion_data.get('intensity')}. Using 0.0 for bias update.")

        # Record the emotion event as a general experience.
        # The detail includes intensity, which will cause bias evolution for "intensity", "0.XX" etc.
        # The 'kind' is the emotion label itself. This will adapt synaptic weights for this emotion type.
        emotion_detail_text = f"Simulated emotion '{emotion_kind}' intensity: {intensity:.2f}"
        # Calling record_experience here logs the emotion event itself and triggers its contribution to biases and weights
        self.record_experience(emotion_kind, emotion_detail_text)
        logger.debug(f"Recorded emotion event for bias update: '{emotion_detail_text}'")


        # Apply a combined decay and amplification factor to *all* existing cognitive biases.
        # The amplification factor increases with intensity.
        # Factor is 1.0 at intensity 0.0, up to 1.1 (or higher) at intensity 1.0.
        # This factor is then multiplied by the general decay rate.
        # A strong emotion makes biases more volatile or reinforced based on the base decay.
        amplification_factor = (1.0 + 0.2 * intensity) # Slightly stronger potential amplification (up to 1.2)
        decay_and_amplify_factor = amplification_factor * self.decay_rate

        # Apply the factor to all existing biases. Iterate over a copy.
        biases_to_update = list(self.cognitive_biases.items())
        updated_count = 0
        removed_count_during_update = 0

        if not biases_to_update:
             logger.debug("No cognitive biases to amplify/decay based on emotion.")
             return

        for token, bias in biases_to_update:
            new_bias = bias * decay_and_amplify_factor
            # Remove bias if its absolute value falls below a small threshold
            if abs(new_bias) < 1e-9:
                if token in self.cognitive_biases: # Safety check
                    del self.cognitive_biases[token]
                    removed_count_during_update += 1
            else:
                self.cognitive_biases[token] = new_bias
                updated_count += 1

        logger.info(f"Updated {updated_count} cognitive biases ({removed_count_during_update} removed) based on emotion '{emotion_kind}' (intensity {intensity:.2f}). Factor applied: {decay_and_amplify_factor:.3f}.")


    def recall_biases(self, top_k: Optional[int] = None) -> Dict[str, float]:
        """
        🧠 Retrieves the current cognitive biases, sorted in descending order
        by their strength (absolute value). Provides insight into the most
        salient concepts or tokens in the AI's cognitive landscape.

        Args:
            top_k (Optional[int]): The maximum number of top biases (by absolute value)
                                   to return. If None, return all biases above threshold.
                                   Defaults to None.

        Returns:
            Dict[str, float]: A dictionary of cognitive biases, sorted by absolute value
                              (descending). Returns an empty dictionary if no biases
                              exist above the filtering threshold.
        """
        # Filter out any zero or near-zero biases before sorting using a small threshold
        non_zero_biases = {tok: val for tok, val in self.cognitive_biases.items() if abs(val) > 1e-9}
        # Sort by absolute value in descending order
        sorted_biases = dict(sorted(non_zero_biases.items(), key=lambda item: -abs(item[1])))

        if top_k is not None and top_k >= 0:
            # Return a dictionary created from the slice of the sorted list
            return dict(list(sorted_biases.items())[:top_k])
        else:
            return sorted_biases # Return all non-zero biases


    def recall_weights(self, top_k: Optional[int] = None) -> Dict[str, float]:
        """
        🔧 Retrieves the current synaptic weights associated with experience types,
        sorted in descending order by their value. Provides insight into which
        types of experiences have most strongly shaped the simulated neural pathways.

        Args:
            top_k (Optional[int]): The maximum number of top weights to return.
                                   If None, return all weights above threshold.
                                   Defaults to None.

        Returns:
            Dict[str, float]: A dictionary of synaptic weights, sorted by value
                              (descending). Returns an empty dictionary if no weights
                              exist above the filtering threshold. Default weight is 1.0.
        """
        # Filter out any weights very close to the initial baseline (1.0) or zero
        # Focus on weights that have significantly adapted
        adapted_weights = {kind: val for kind, val in self.synaptic_weights.items() if abs(val - 1.0) > 1e-9 and abs(val) > 1e-9}
        sorted_weights = dict(sorted(adapted_weights.items(), key=lambda item: -item[1]))

        if top_k is not None and top_k >= 0:
            # Return a dictionary created from the slice of the sorted list
            return dict(list(sorted_weights.items())[:top_k])
        else:
            return sorted_weights # Return all adapted weights


    def snapshot(self, top_k_biases: int = 10, top_k_weights: int = 5) -> Dict[str, Any]:
        """
        📊 Provides a quick snapshot summary of the NeuroMemoryProcessor's current
        internal state, including the total count of recorded experiences and the
        top (most prominent) cognitive biases and synaptic weights.

        Args:
            top_k_biases (int): Number of top biases (by absolute value) to include in the snapshot. Defaults to 10.
            top_k_weights (int): Number of top weights to include in the snapshot. Defaults to 5.

        Returns:
            Dict[str, Any]: A dictionary containing the state snapshot summary.
        """
        # Ensure top_k values are non-negative
        top_k_biases = max(0, top_k_biases)
        top_k_weights = max(0, top_k_weights)

        return {
            "memory_count": len(self.long_term_memory),
            "synaptic_weight_count": len(self.synaptic_weights), # Total count
            "cognitive_bias_count": len(self.cognitive_biases), # Total count
            "top_biases":   self.recall_biases(top_k=top_k_biases),
            "top_weights":  self.recall_weights(top_k=top_k_weights)
        }

    def search_experiences(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        🔍 Performs a simple case-insensitive keyword search over the 'detail'
        field of recorded experiences stored in `long_term_memory`. Useful for
        finding past events or information related to a specific query. Results
        are returned in reverse chronological order (most recent matches first).

        Args:
            query (str): The keyword or phrase to search for (case-insensitive).
            top_k (Optional[int]): The maximum number of matching entries to return.
                                   If None, return all matches. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the matching
                                  experience entries. These are copies of the internal
                                  memory entries. Returns an empty list if no
                                  matches are found or if the query is invalid.
        """
        if not isinstance(query, str) or not query.strip():
            logger.warning("Search query for experiences is empty or not a string. Returning empty list.")
            return []

        query_lower = query.lower()

        # Filter and reverse for most recent first. Create copies of matches.
        matches = [
            e.copy() for e in reversed(self.long_term_memory)
            if isinstance(e.get("detail"), str) and query_lower in e["detail"].lower()
        ]

        logger.debug(f"Search for '{query}' found {len(matches)} matches in recorded experiences.")

        # Apply top_k limit to the found matches
        return matches[:top_k] if top_k is not None and top_k >= 0 else matches


    def export_state(self) -> str:
        """
        💾 Serializes the full current state of the NeuroMemoryProcessor, including
        recorded experiences, synaptic weights, cognitive biases, and configuration
        parameters, into a JSON formatted string. This allows for saving and
        restoring the processor's dynamic state.

        Returns:
            str: A JSON string representing the processor state. Returns an empty JSON
                 object string "{}" if serialization fails due to data types or other errors.
        """
        state = {
            "long_term_memory":     self.long_term_memory,
            "synaptic_weights":     self.synaptic_weights,
            "cognitive_biases":     self.cognitive_biases,
            # Include configuration parameters for reproducible state
            "memory_capacity":      self.memory_capacity if isinstance(self.memory_capacity, int) else 0, # Store 0 if inf
            "plasticity_range":     list(self.plasticity_range), # Convert tuple to list for JSON
            "bias_increment":       self.bias_increment,
            "decay_rate":           self.decay_rate
        }
        try:
            # Use default=str to handle any non-serializable types by converting them to string
            return json.dumps(state, indent=2, default=str)
        except TypeError as e:
            logger.error(f"Failed to serialize processor state to JSON (TypeError): {e}")
            # Log a snippet of the problematic state
            try:
                 problem_state_snippet = json.dumps({k: str(v)[:150] + ('...' if len(str(v)) > 150 else '') for k, v in state.items()}, indent=2, default=str)
                 logger.error("State causing error (snippet): %s", problem_state_snippet)
            except:
                 logger.error("Could not even serialize state snippet during error handling.")
            return "{}" # Return empty JSON object on failure
        except Exception as e:
            logger.error(f"An unexpected error occurred during processor state export: {e}")
            return "{}"


    def import_state(self, blob: str) -> None:
        """
        📥 Loads the NeuroMemoryProcessor state from a JSON formatted string,
        overwriting the current state. Includes comprehensive validation to ensure
        data integrity and prevent errors from malformed input or mismatched types.

        Args:
            blob (str): A JSON string representing the processor state,
                        expected to be in the format exported by `export_state`.
                        If the blob is invalid or loading fails, the current
                        state will remain unchanged.
        """
        if not isinstance(blob, str) or not blob.strip():
            logger.warning("Attempted to import empty or non-string JSON blob for processor state. Skipping import.")
            return

        try:
            data = json.loads(blob)

            # Validate the loaded state structure
            if not isinstance(data, dict):
                logger.error("Processor state import failed: Loaded data is not a dictionary. Expected state object.")
                return

            # --- Safely load primary state attributes ---
            # Temporarily hold loaded data while validating types
            loaded_memory = data.get("long_term_memory", [])
            if not isinstance(loaded_memory, list):
                logger.warning("Processor import warning: 'long_term_memory' was not a list. Initializing as empty.")
                loaded_memory = []

            loaded_weights = data.get("synaptic_weights", {})
            if not isinstance(loaded_weights, dict):
                logger.warning("Processor import warning: 'synaptic_weights' was not a dict. Initializing as empty.")
                loaded_weights = {}
             # Optional: Add type check for values (ensure they are float/int)

            loaded_biases = data.get("cognitive_biases", {})
            if not isinstance(loaded_biases, dict):
                logger.warning("Processor import warning: 'cognitive_biases' was not a dict. Initializing as empty.")
                loaded_biases = {}
            # Optional: Add type check for values (ensure they are float/int)


            # --- Safely import configuration parameters ---
            # Handle potential infinity value stored as 0 during export
            imported_capacity = data.get("memory_capacity", self.memory_capacity)
            if imported_capacity == 0: # Check if it was exported as 0 (for inf)
                 loaded_capacity = float('inf')
            elif isinstance(imported_capacity, int) and imported_capacity > 0:
                 loaded_capacity = imported_capacity
            else:
                 logger.warning(f"Processor import warning: Invalid memory_capacity '{imported_capacity}'. Keeping current value {self.memory_capacity}.")
                 loaded_capacity = self.memory_capacity # Keep current if invalid


            imported_plasticity = data.get("plasticity_range", self.plasticity_range)
            if isinstance(imported_plasticity, list) and len(imported_plasticity) == 2 and all(isinstance(x, (int, float)) for x in imported_plasticity):
                 loaded_plasticity = tuple(imported_plasticity) # Convert list back to tuple
            else:
                 logger.warning(f"Processor import warning: Invalid plasticity_range '{imported_plasticity}'. Keeping current value {self.plasticity_range}.")
                 loaded_plasticity = self.plasticity_range # Keep current if invalid


            # Safely convert and set scalar parameters
            try:
                 imported_bias_inc = float(data.get("bias_increment", self.bias_increment))
                 loaded_bias_inc = max(0.0, imported_bias_inc) # Ensure non-negative
                 if imported_bias_inc < 0: logger.warning("Imported bias_increment was negative, clamped to 0.0.")
            except (ValueError, TypeError):
                 logger.warning(f"Processor import warning: Invalid bias_increment '{data.get('bias_increment', 'N/A')}'. Keeping current value {self.bias_increment}.")
                 loaded_bias_inc = self.bias_increment # Keep current if invalid

            try:
                 imported_decay = float(data.get("decay_rate", self.decay_rate))
                 if not (0.0 <= imported_decay <= 1.0):
                      logger.warning(f"Processor import warning: Decay rate '{imported_decay}' outside [0.0, 1.0]. Clamping.")
                      loaded_decay = max(0.0, min(1.0, imported_decay))
                 else:
                      loaded_decay = imported_decay
            except (ValueError, TypeError):
                 logger.warning(f"Processor import warning: Invalid decay_rate '{data.get('decay_rate', 'N/A')}'. Keeping current value {self.decay_rate}.")
                 loaded_decay = self.decay_rate # Keep current if invalid


            # --- Assign validated/loaded data to self ---
            self.long_term_memory = loaded_memory
            self.synaptic_weights = loaded_weights
            self.cognitive_biases = loaded_biases
            self.memory_capacity = loaded_capacity
            self.plasticity_range = loaded_plasticity
            self.bias_increment = loaded_bias_inc
            self.decay_rate = loaded_decay

            logger.info(f"📥 NeuroMemoryProcessor state imported successfully. Loaded {len(self.long_term_memory)} experiences.")

        except json.JSONDecodeError as e:
            logger.error(f"Processor state import failed: Invalid JSON format in blob: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during processor state import processing: {e}")


    # ─── Private helpers ─────────────────────────────────────

    # No _decode_input or _default_summarizer here, those belong to MemoryEngine

    def _decay_biases(self) -> None:
        """
        Applies an exponential decay to all existing cognitive biases based on
        the configured `decay_rate`. This simulates the natural fading of biases
        over time or processing cycles if they are not reinforced by new encounters.
        Also removes biases that decay below a very small threshold to manage memory.
        Called internally by `_evolve_cognitive_bias` and `update_biases`.
        """
        # Create a list of items to decay to avoid changing dict size during iteration
        biases_to_decay = list(self.cognitive_biases.items())
        # logger.debug(f"Starting bias decay. Initial bias count: {len(biases_to_decay)}") # Too verbose

        if not biases_to_decay:
             # logger.debug("No cognitive biases to decay.") # Too verbose
             return

        removed_count = 0
        for token, bias in biases_to_decay:
            new_bias = bias * self.decay_rate
            # Remove bias if its absolute value falls below a small threshold
            if abs(new_bias) < 1e-9: # Use a very small threshold
                if token in self.cognitive_biases: # Safety check
                    del self.cognitive_biases[token]
                    removed_count += 1
            else:
                self.cognitive_biases[token] = new_bias
                # logger.debug(f"Decayed bias for '{token}': {bias:.3f} → {new_bias:.3f}") # Too verbose

        # if removed_count > 0:
             # logger.debug(f"Bias decay complete. Removed {removed_count} low biases.") # Too verbose


# Example Usage (Illustrative)
if __name__ == "__main__":
    print("--- NeuroMemoryProcessor Example Usage ---")
    # Set logger level to DEBUG for this specific example run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Ensure this logger also uses DEBUG

    # Test with specific parameters
    processor = NeuroMemoryProcessor(memory_capacity=10, plasticity_range=(0.1, 0.5), bias_increment=0.1, decay_rate=0.9) # Adjusted params for demo

    # Simulate recording various types of experiences
    print("\n--- Recording Experiences ---")
    processor.record_experience("observation", "User asked about a complex system.")
    processor.record_experience("step", "Initiated problem decomposition.")
    processor.record_experience("step", "Retrieved relevant data nodes.")
    processor.record_experience("emotion", "curiosity", {"primary_emotion": "curiosity", "intensity": 0.8}) # Simulate emotion event
    processor.record_experience("observation", "Received feedback indicating a potential misinterpretation.")
    processor.record_experience("step", "Adjusted the interpretation model based on feedback.")
    processor.record_experience("metric", "validation_score", 0.75) # Simulate metric recording
    processor.record_experience("reflection", "Synthesized insights from recent interactions.") # Simulate reflection event

    # Exceed memory capacity slightly
    processor.record_experience("observation", "Processing follow-up question.")
    processor.record_experience("step", "Initiating sub-problem analysis.")
    processor.record_experience("step", "Consulting long-term memory for similar patterns.") # Should cause eviction


    print("\n--- Recorded Experiences (most recent 10, ordered by recency) ---")
    # Using search_experiences with no query effectively gets all recent experiences
    all_experiences = processor.search_experiences("", top_k=processor.memory_capacity) # Get up to memory capacity
    for i, entry in enumerate(all_experiences):
           # Format timestamp for display
           ts_formatted = entry.get('timestamp', 'N/A')
           if ts_formatted != 'N/A':
                try:
                     ts_formatted = datetime.fromisoformat(ts_formatted.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                     pass # Keep original if formatting fails
           print(f"{i+1}: [{ts_formatted}] {entry.get('type', 'N/A').upper()}: {entry.get('detail', 'N/A')[:80]}...")


    print("\n--- Snapshot ---")
    snapshot_data = processor.snapshot(top_k_biases=10, top_k_weights=5)
    import json
    print(json.dumps(snapshot_data, indent=2))

    print("\n--- Recalled Biases (All) ---")
    print(json.dumps(processor.recall_biases(), indent=2))

    print("\n--- Recalled Weights (All) ---")
    print(json.dumps(processor.recall_weights(), indent=2))

    print("\n--- Search Experiences ('feedback') ---")
    search_results = processor.search_experiences("feedback")
    print(json.dumps(search_results, indent=2))

    # Simulate emotion update again to see decay and amplification effects
    print("\n--- Updating Biases with Emotion (Satisfaction, High Intensity) ---")
    processor.update_biases({"primary_emotion": "satisfaction", "intensity": 0.9})
    print("\n--- Recalled Biases after Emotion Update ---")
    print(json.dumps(processor.recall_biases(top_k=15), indent=2)) # Show more biases to see decay/amplification

    print("\n--- Updating Biases with Emotion (Melancholy, Moderate Intensity) ---")
    processor.update_biases({"primary_emotion": "melancholy", "intensity": 0.6})
    print("\n--- Recalled Biases after Second Emotion Update ---")
    print(json.dumps(processor.recall_biases(top_k=15), indent=2)) # Show more biases to see effects


    # Simulate export and import
    print("\n--- Exporting State ---")
    exported_json = processor.export_state()
    print(exported_json[:1000] + "..." if len(exported_json) > 1000 else exported_json) # Print snippet

    print("\n--- Importing State into New Processor ---")
    # Use different init values to show they are overridden by import
    new_processor = NeuroMemoryProcessor(memory_capacity=50, plasticity_range=(0.05, 0.2), bias_increment=0.01, decay_rate=0.99)
    print(f"\nNew processor initial capacity: {new_processor.memory_capacity}") # Show initial state
    print(f"New processor initial plasticity: {new_processor.plasticity_range}")

    new_processor.import_state(exported_json)

    print("\n--- New Processor Snapshot (After Import) ---")
    snapshot_after_import = new_processor.snapshot(top_k_biases=10, top_k_weights=5)
    print(json.dumps(snapshot_after_import, indent=2))
    print(f"\nNew processor loaded capacity: {new_processor.memory_capacity}") # Show loaded state
    print(f"New processor loaded plasticity: {new_processor.plasticity_range}")
    print(f"New processor loaded bias increment: {new_processor.bias_increment}")
    print(f"New processor loaded decay rate: {new_processor.decay_rate}")


    print("\n--- New Processor Recalled Biases (After Import) ---")
    print(json.dumps(new_processor.recall_biases(top_k=15), indent=2))


    print("\n--- Example Usage End ---")