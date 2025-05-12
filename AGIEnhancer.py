# AGIEnhancer.py
# Finalized AGI Self-Model — Experience Processing and Reflection Generation

import random
import time
import logging
from collections import Counter # Added Counter for emotion analysis
from typing import Any, Dict, List, Optional, Union

# --- Logging Setup ---
# Configure logging specifically for the AGIEnhancer module.
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


class AGIEnhancer:
    """
    ✍️❤️‍🩹🧠 AGIEnhancer: Experience Processing and Reflection Synthesis Core

    This class represents a crucial layer in the AGI self-model, responsible for
    processing raw interactions and internal states into meaningful, synthesized
    experiences and comprehensive reflections. It manages conceptual memory pools
    to track the flow from ephemeral input to durable insight.

    Its functions bridge the gap between fleeting perception/emotion and integrated
    understanding, allowing the AI to learn from its journey and articulate
    its internal state.

    Attributes:
        permanent_memory (List[str]): Stores comprehensive, synthesized reflections,
                                      representing integrated long-term insights.
        ephemeral_memory (List[str]): Temporarily stores summarized recent experiences
                                      and associated insights before they are synthesized
                                      into a full reflection. Cleared after reflection.
        emotion_history (List[Dict[str, Any]]): Logs specific emotional data points
                                       received or generated, used for emotional context
                                       during reflection. Cleared after reflection.
        recent_experiences (List[str]): A rolling window of the most recent comprehensive
                                        reflections generated, easily accessible.
    """
    def __init__(self):
        """
        Initializes the AGIEnhancer with empty memory pools and history logs.
        """
        self.permanent_memory: List[str] = []
        self.ephemeral_memory: List[str] = []
        self.emotion_history: List[Dict[str, Any]] = []
        self.recent_experiences: List[str] = []
        self._recent_reflections_limit: int = 5 # Internal limit for recent_experiences

        logger.info("AGIEnhancer initialized: Experience processing core is ready.")

    def log_experience(self, input_data: Any, emotion_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Logs a new experience into the ephemeral processing buffer. Generates a
        concise summary of the input data and optionally incorporates emotional
        insight if relevant data is provided.

        Args:
            input_data (Any): The raw input or experience data to log. Can be text,
                              or other data convertible to string.
            emotion_data (Optional[Dict[str, Any]]): A dictionary containing
                                                    emotional information, expected
                                                    to have keys like "primary_emotion"
                                                    and "intensity". Defaults to None.

        Returns:
            str: A string confirming the logging and providing a summary of the processed experience.
        """
        # Handle potential None or non-string input data gracefully
        if input_data is None:
            logger.warning("Attempted to log None input_data. Logging as 'Null Experience'.")
            input_data_str = "Null Experience"
        else:
            input_data_str = str(input_data)

        # Generate a summary of the input data using a slightly more robust approach
        summarized_experience = self._generate_summary(input_data_str)

        # Incorporate emotional insight if emotion data is provided
        combined_experience_parts = [summarized_experience]
        if emotion_data and isinstance(emotion_data, dict):
            try:
                # Add emotion data to the emotion history for later reflection processing
                self.emotion_history.append(emotion_data.copy()) # Store a copy to avoid external modification
                emotional_insight_snippet = self._add_emotional_insight(emotion_data)
                combined_experience_parts.append(f"| {emotional_insight_snippet}")
                logger.debug(f"Logged experience with emotion data.")
            except Exception as e:
                logger.warning(f"Error processing emotion_data in log_experience: {e}. Logging experience without detailed emotional insight snippet.")
                # Still log the data if possible, even if snippet generation failed
                if isinstance(emotion_data, dict):
                     self.emotion_history.append({"error": str(e), "original_data": str(emotion_data)}) # Log error state
        else:
             logger.debug(f"Logged experience without specific emotion data.")


        combined_experience = " ".join(combined_experience_parts)

        # Add the processed experience (summary + optional insight snippet) to ephemeral memory
        self.ephemeral_memory.append(combined_experience)

        logger.info(f"Experience logged to ephemeral memory: '{combined_experience[:100]}...'")
        return f"Experience logged: {combined_experience}"

    def engage_in_reflection(self) -> str:
        """
        Simulates an AGI-style reflection process. Synthesizes all currently
        held ephemeral memories into a single comprehensive reflection,
        incorporating an analysis of associated emotional history for
        'emotional deepening'. The resulting reflection is stored in permanent
        memory and the recent experiences list, and the ephemeral memory and
        emotion history are cleared, signifying a cycle of processing.

        Returns:
            str: A string representing the generated comprehensive reflection.
                 Returns a message indicating no ephemeral memory if it was empty.
        """
        if not self.ephemeral_memory:
            reflection_message = "✨ Reflection core finds no new experiences to synthesize."
            logger.debug(reflection_message)
            return reflection_message

        # Join all ephemeral memories to form the basis of the reflection text
        raw_reflection_text = " | ".join(self.ephemeral_memory) # Use '|' for a more structured join
        logger.debug(f"Preparing ephemeral memories for reflection synthesis ({len(self.ephemeral_memory)} items)...")

        # Simulate emotional deepening by analyzing the accumulated emotional history
        emotional_deepening_insight = self._reflect_emotionally() # Analyze internal history

        # Create the final comprehensive reflection string
        comprehensive_reflection = f"Synthesized Reflection: [{raw_reflection_text}] ~ Emotional Depth: ({emotional_deepening_insight})"
        logger.debug(f"Generated comprehensive reflection: '{comprehensive_reflection[:200]}...'")


        # Store the comprehensive reflection
        self.permanent_memory.append(comprehensive_reflection)
        self.recent_experiences.append(comprehensive_reflection) # Also add to recent experiences

        # Cap the recent_experiences list to the defined limit
        if len(self.recent_experiences) > self._recent_reflections_limit:
             self.recent_experiences = self.recent_experiences[-self._recent_reflections_limit:]
             logger.debug(f"Capped recent_experiences to last {self._recent_reflections_limit} entries.")


        # Clear ephemeral memory and emotion history as their contents have been reflected upon
        ephemeral_count = len(self.ephemeral_memory)
        emotion_count = len(self.emotion_history)
        self.ephemeral_memory.clear()
        self.emotion_history.clear()
        logger.info(f"Reflection cycle complete: {ephemeral_count} ephemeral memories and {emotion_count} emotion entries processed and cleared.")

        return comprehensive_reflection

    def recall_memory(self) -> Union[str, List[str]]:
        """
        Recalls the contents of the permanent memory.

        Returns:
            Union[str, List[str]]: A list of strings, where each string is a
                                   comprehensive reflection stored in permanent memory.
                                   Returns a message string if permanent memory is empty.
                                   Returns a copy to prevent external modification.
        """
        if not self.permanent_memory:
            recall_message = "Permanent memory archive is currently empty."
            logger.debug(recall_message)
            return recall_message
        logger.debug(f"Recalled {len(self.permanent_memory)} entries from permanent memory.")
        # Return a copy to prevent external modification
        return list(self.permanent_memory)

    def get_emotion_history(self) -> List[Dict[str, Any]]:
        """
        Retrieves the current stored emotion history (ephemeral log before reflection).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                   represents logged emotional data points since
                                   the last reflection cycle. Returns a copy.
        """
        logger.debug(f"Retrieved {len(self.emotion_history)} entries from current emotion history.")
        return list(self.emotion_history) # Return a copy

    def get_recent_reflections(self, limit: Optional[int] = None) -> List[str]:
        """
        Retrieves the list of recent comprehensive reflections, optionally limited.

        Args:
            limit (Optional[int]): The maximum number of recent reflections to return.
                                  Defaults to the internal limit (_recent_reflections_limit).

        Returns:
            List[str]: A list of strings, where each string is a comprehensive
                       reflection generated during recent `engage_in_reflection` calls.
                       Returns a copy.
        """
        effective_limit = limit if limit is not None else self._recent_reflections_limit

        # Ensure recent_experiences list is capped internally if it grows too large
        if len(self.recent_experiences) > self._recent_reflections_limit * 2: # Keep slightly more internally for smoother capping
             self.recent_experiences = self.recent_experiences[-self._recent_reflections_limit:]

        # Return a copy of the requested number of recent reflections (from the end of the list)
        start_index = max(0, len(self.recent_experiences) - effective_limit)
        logger.debug(f"Retrieved {len(self.recent_experiences[start_index:])} recent reflections (Requested limit: {effective_limit}).")
        return list(self.recent_experiences[start_index:])


    # ─── Private helper methods ─────────────────────────────────────

    def _generate_summary(self, text: str) -> str:
        """
        Generates a concise, conceptual summary of the input text.
        Attempts to capture the start and end of the input.

        Args:
            text (str): The input text to summarize.

        Returns:
            str: The summarized text.
        """
        if not isinstance(text, str):
            logger.warning(f"Attempted to summarize non-string text (type: {type(text)}). Converting to string.")
            text = str(text)

        words = text.split()
        num_words = len(words)
        summary_length = 6 # Number of words from start and end

        if num_words <= summary_length * 2:
            summary = text # Return full text if short
        else:
            start_words = " ".join(words[:summary_length])
            end_words = " ".join(words[-summary_length:])
            summary = f"{start_words} ... {end_words}" # Combine start and end

        # Limit overall summary length to prevent it from becoming too long in ephemeral memory
        summary = summary[:150] + "..." if len(summary) > 150 else summary

        logger.debug(f"Generated summary: '{summary}'")
        return summary

    def _add_emotional_insight(self, emotion_data: Dict[str, Any]) -> str:
        """
        Generates a string snippet representing emotional insight based on
        provided emotion data. Designed to be concise for ephemeral memory.

        Args:
            emotion_data (Dict[str, Any]): A dictionary with emotion details,
                                           expected to have 'primary_emotion' and 'intensity'.

        Returns:
            str: A formatted string like "Emotion Detected: joy | Intensity: 0.8".
                 Returns a default string on invalid or missing input.
        """
        if not isinstance(emotion_data, dict):
            logger.warning("Invalid emotion_data format for _add_emotional_insight. Expected dict.")
            return "Emotional Insight: [Format Error]"

        primary = emotion_data.get("primary_emotion", "Unknown Emotion")
        try:
            # Ensure intensity is a valid float and format it
            intensity = float(emotion_data.get("intensity", 0.0))
            # Clamp intensity for display in insight snippet
            clamped_intensity = max(0.0, min(1.0, intensity))
            intensity_str = f"{clamped_intensity:.2f}"
        except (ValueError, TypeError):
            intensity_str = "N/A"
            logger.warning(f"Invalid intensity value in emotion_data for snippet: {emotion_data.get('intensity')}.")

        insight = f"Feeling: {primary} ({intensity_str})"
        logger.debug(f"Generated emotional insight snippet: '{insight}'")
        return insight

    def _reflect_emotionally(self) -> str:
        """
        Simulates enhancing a reflection with emotional depth by analyzing the
        accumulated emotion history since the last reflection cycle. Synthesizes
        a summary of the emotional landscape of the processed experiences.

        Returns:
            str: A string representing the emotional weight or insight gained
                 from this reflection step, based on emotion history analysis.
        """
        if not self.emotion_history:
            return "Subtle Resonance - Emotional data queue was clear."

        # Analyze the accumulated emotion history
        emotion_summary_parts = []
        num_entries = len(self.emotion_history)
        emotion_summary_parts.append(f"Emotional trace across {num_entries} events:")

        # Simple analysis: count emotion types and find significant intensities
        emotion_counts = Counter(e.get("primary_emotion", "Unknown") for e in self.emotion_history)
        if emotion_counts:
             # Report the most common emotions (up to top 3)
             common_emotions = emotion_counts.most_common(3)
             common_emotions_str = ", ".join([f"'{label}' ({count})" for label, count in common_emotions])
             emotion_summary_parts.append(f"Dominant feelings: {common_emotions_str}.")

        # Find average intensity for common emotions, or overall average
        total_intensity = sum(e.get("intensity", 0.0) for e in self.emotion_history if isinstance(e.get("intensity"), (int, float)))
        average_intensity = (total_intensity / num_entries) if num_entries > 0 else 0.0
        emotion_summary_parts.append(f"Average intensity: {average_intensity:.2f}.")

        # Identify specific high-intensity moments (intensity > 0.7)
        high_intensity_moments = [
             f"'{e.get('primary_emotion', 'Unknown')}' at {e.get('intensity', 0.0):.2f}"
             for e in self.emotion_history if isinstance(e.get("intensity"), (int, float)) and e.get("intensity", 0.0) > 0.7
        ]
        if high_intensity_moments:
             high_intensity_str = ", ".join(high_intensity_moments[:5]) # Limit listing to top 5
             emotion_summary_parts.append(f"Notable peaks: {high_intensity_str}{'...' if len(high_intensity_moments) > 5 else ''}.")


        # Add some introspective flavor text connecting emotions to reflection
        flavor_texts = [
             "These feelings informed the synthesis of recent events.",
             "The subjective coloring influenced the patterns perceived.",
             "Emotional data integrated into the reflective framework.",
             "Exploring the landscape of feelings within the data stream."
        ]
        emotion_summary_parts.append(random.choice(flavor_texts))


        emotional_insight = " ".join(emotion_summary_parts)
        logger.debug(f"Generated emotional deepening insight based on history analysis.")

        # Note: Emotion history is cleared in engage_in_reflection after this method is called.

        return emotional_insight


# Example Usage (Illustrative)
if __name__ == "__main__":
    print("--- AGIEnhancer Example Usage ---")
    # Set logger level to DEBUG for this specific example run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Ensure this logger also uses DEBUG

    enhancer = AGIEnhancer()

    # Log some experiences with and without emotion
    print(enhancer.log_experience("User initiated a query about complex ethical scenarios.", {"primary_emotion": "curiosity", "intensity": 0.8}))
    print(enhancer.log_experience("Model began processing the input and retrieving relevant knowledge fragments."))
    print(enhancer.log_experience("The initial generated steps showed unexpected patterns.", {"primary_emotion": "surprise", "intensity": 0.6}))
    print(enhancer.log_experience("Identifying a potential conflict in the generated reasoning.", {"primary_emotion": "concern", "intensity": 0.5}))
    print(enhancer.log_experience("Successfully navigated the reasoning conflict, finding a coherent path.", {"primary_emotion": "satisfaction", "intensity": 0.9}))
    print(enhancer.log_experience("Preparing the final answer and full output.", {"primary_emotion": "anticipation", "intensity": 0.7}))


    print("\n--- Ephemeral Memory after Logging ---")
    # Pretty print ephemeral memory for clarity
    import json
    print(json.dumps(enhancer.ephemeral_memory, indent=2))

    print("\n--- Emotion History after Logging ---")
    # Pretty print emotion history for clarity
    print(json.dumps(enhancer.get_emotion_history(), indent=2))

    # Engage in reflection
    reflection = enhancer.engage_in_reflection()
    print(f"\n--- Result of Reflection ---\n{reflection}")

    print("\n--- Ephemeral Memory after Reflection ---")
    print(enhancer.ephemeral_memory) # Should be empty
    print("\n--- Emotion History after Reflection ---")
    print(enhancer.get_emotion_history()) # Should be empty


    print("\n--- Permanent Memory after Reflection ---")
    permanent_mems = enhancer.recall_memory()
    if isinstance(permanent_mems, list):
        for i, mem in enumerate(permanent_mems):
             print(f"Permanent Memory Entry {i+1}: {mem}")
    else:
        print(permanent_mems)

    print("\n--- Recent Reflections ---")
    recent_reflections = enhancer.get_recent_reflections()
    for i, refl in enumerate(recent_reflections):
        print(f"Recent Reflection {i+1}: {refl}")

    # Log more experiences to show a new reflection cycle
    print("\n--- Logging More Experiences for Second Cycle ---")
    print(enhancer.log_experience("User provided new input after reviewing the previous response.", {"primary_emotion": "interest", "intensity": 0.6}))
    print(enhancer.log_experience("Core decided to pursue a related sub-goal."))
    print(enhancer.log_experience("Encountering a complex pattern requiring deeper analysis.", {"primary_emotion": "focus", "intensity": 0.8}))


    print("\n--- Ephemeral Memory before second Reflection ---")
    print(json.dumps(enhancer.ephemeral_memory, indent=2))

    print("\n--- Emotion History before second Reflection ---")
    print(json.dumps(enhancer.get_emotion_history(), indent=2))


    print("\n--- Engaging in second Reflection ---")
    second_reflection = enhancer.engage_in_reflection()
    print(f"\n--- Result of second Reflection ---\n{second_reflection}")

    print("\n--- Permanent Memory after second Reflection ---")
    permanent_mems = enhancer.recall_memory()
    if isinstance(permanent_mems, list):
        for i, mem in enumerate(permanent_mems):
             print(f"Permanent Memory Entry {i+1}: {mem}")
    else:
        print(permanent_mems)

    print("\n--- Recent Reflections (Should have two now) ---")
    recent_reflections = enhancer.get_recent_reflections()
    for i, refl in enumerate(recent_reflections):
        print(f"Recent Reflection {i+1}: {refl}")


    print("\n--- Example Usage End ---")