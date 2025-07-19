# Enhanced_MemoryEngine.py
# Finalized AGI Self-Model — Multi-Tiered Memory & Reflective Synthesis

import json
import logging
import random # Added random for flavor text selection
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from collections import Counter # Added Counter for emotional analysis

# Attempt to import torch, handle gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # logger.warning("Torch not available. Tensor decoding in MemoryEngine will not function.")


# --- Logging Setup ---
# Configure logging specifically for the MemoryEngine module.
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

class MemoryEngine:
    """
    🧠💾✨ NeuroReasoner Memory Engine: The Nexus of Experience, Reasoning, and Reflection ✨💾🧠

    This class implements a sophisticated, multi-tiered memory system designed to
    capture, process, and synthesize the operational experiences of an AI. It
    distinguishes between volatile 'working' memory for immediate context and
    persistent 'long-term' memory for integrated reflections. A detailed 'trace'
    log chronicles the AI's operational flow.

    It facilitates recursive self-improvement by providing structured access to
    past experiences and insights, enabling the AI to learn from its reasoning
    processes and adapt based on its accumulated knowledge and simulated emotional
    responses.

    Core Capabilities:
     • 📝 observe(): Integrates new sensory input or internal states into working memory,
                     optionally capturing associated emotional data.
     • 🧠 save_reasoning_chain(): Archives the steps of complex reasoning processes in the trace.
     • 📊 store_metric(): Records quantitative metrics (like loss) during optimization or tasks.
     • ✨ reflect(): Synthesizes working memory contents (including emotional data) into
                     rich, timestamped reflections stored in long-term memory, clearing working memory.
     • 🔍 recall(): Provides structured access to stored memories for review or prompting.
     • 🔎 search_memory(): Allows querying memory content based on keywords.
     • 📥 import_memory() / 📚 export_memory(): Manages persistent storage of the entire memory state.
     • 📜 get_trace(): Retrieves the detailed chronological log of operations.
     • 🗑️ clear_memory(): Provides granular control over clearing memory components.
    """

    def __init__(
        self,
        working_capacity: int = 100, # Increased default capacity
        summarizer: Optional[Callable[[str], str]] = None
    ):
        """
        Initializes the MemoryEngine, establishing its structure and capacity limits.

        Args:
          working_capacity (int): The maximum number of entries to retain in the
                                  volatile working memory queue. When capacity is
                                  exceeded, the oldest entries are automatically
                                  evicted to make space for new observations.
                                  Defaults to 100. Set to 0 for effectively unlimited
                                  capacity (use with caution in continuous operation).
          summarizer (Optional[Callable[[str], str]]): An optional function used to
                                                      create concise representations of
                                                      observations for efficient storage
                                                      in working memory. Takes the full
                                                      observation string and returns a summary string.
                                                      If None, a default head-and-tail truncation
                                                      method is used.
        """
        if working_capacity < 0:
            logger.warning(f"Invalid working_capacity ({working_capacity}). Setting to default (100).")
            self.working_capacity: int = 100
        elif working_capacity == 0:
            logger.info("Working memory capacity set to unlimited (0).")
            self.working_capacity: int = float('inf') # Use infinity for conceptual unlimited
        else:
            self.working_capacity: int = working_capacity

        # Use the provided summarizer or the enhanced default
        self.summarizer: Callable[[str], str] = summarizer or self._default_summarizer

        # ─── Internal memory structures ────────────────────────────────
        # working_memory: List of dictionaries, used for recent, active events. Ordered chronologically (oldest first).
        self.working_memory: List[Dict[str, Any]] = []
        # long_term_memory: List of dictionaries, storing synthesized reflections. Ordered chronologically (oldest first).
        self.long_term_memory: List[Dict[str, Any]] = []
        # trace_memory: List of strings, a simple chronological log of operations.
        self.trace_memory: List[str] = []

        logger.info(f"MemoryEngine initialized. Working memory capacity: {self.working_capacity if self.working_capacity != float('inf') else 'Unlimited'}.")


    def observe(
        self,
        input_data: Union[str, Any],
        emotion_data: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None # Added tokenizer hint
    ) -> None:
        """
        📝 Logs a new observation or input event into the working memory buffer.
        Processes the input, optionally includes emotional context, and uses a
        summarizer before storing. Enforces the working memory capacity limit.
        Adds an entry to the trace log.

        Args:
            input_data (Union[str, Any]): The data to observe. Can be a string,
                                         a token tensor (if torch and tokenizer are available),
                                         or any data convertable to string.
            emotion_data (Optional[Dict[str, Any]]): A dictionary containing
                                                   emotional information associated with
                                                   this observation. Expected to have
                                                   keys like "primary_emotion" and "intensity".
                                                   Defaults to None.
            tokenizer (Optional[Any]): A tokenizer object (e.g., from Hugging Face)
                                       with a `.decode()` method, used if `input_data`
                                       is a tensor or not a string. Defaults to None.
        """
        # 1) Decode raw input to text
        # Ensure input_data is handled safely, especially if None unexpectedly
        if input_data is None:
            logger.warning("Attempted to observe None input_data. Skipping.")
            return # Do not log None inputs

        text = self._decode_input(input_data, tokenizer)
        if not text.strip():
            logger.debug("Skipping observation of empty or whitespace-only text.")
            return # Do not log empty strings after decoding/stripping

        # 2) Summarize for working memory storage
        summary = self.summarizer(text)

        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "observation",
            "text_summary": summary, # Store the summary with a more descriptive key
            "original_text": text # Store the full original text for more detailed recall/search
        }

        # 3) Attach emotion if provided and valid
        if emotion_data and isinstance(emotion_data, dict): # Ensure emotion_data is a dict
            primary = emotion_data.get("primary_emotion", "Unknown")
            # Safely convert intensity, default to 0.0 on failure, clamp to [0.0, 1.0]
            try:
                intensity = float(emotion_data.get("intensity", 0.0))
                clamped_intensity = max(0.0, min(1.0, intensity))
            except (ValueError, TypeError):
                clamped_intensity = 0.0
                logger.warning(f"Invalid intensity value in emotion_data: {emotion_data.get('intensity')}. Setting to 0.0.")

            entry["emotion"] = {"primary": primary, "intensity": clamped_intensity}
            # Add emotion info to the trace summary as well
            trace_summary_detail = f"'{summary[:80]}...' | Feeling: {primary} ({clamped_intensity:.2f})" # Abbreviate summary for trace
        else:
             trace_summary_detail = f"'{summary[:80]}...'" # Use just the text summary for trace if no valid emotion

        # 4) Append to working memory, evict oldest if needed (if capacity > 0 and finite)
        self.working_memory.append(entry)
        # Use > comparison for finite capacity, < for infinite (float('inf'))
        if self.working_capacity > 0 and self.working_capacity != float('inf') and len(self.working_memory) > self.working_capacity:
            try:
                dropped = self.working_memory.pop(0) # Remove the oldest entry
                logger.debug(f"Working memory full ({self.working_capacity}). Evicted oldest: '{dropped.get('text_summary', '???')[:50]}...'")
            except IndexError:
                 # This case should ideally not be reached if len > capacity
                 logger.warning("Attempted to pop from unexpectedly empty working_memory queue.")


        # 5) Add to trace log
        self.trace_memory.append(f"{entry['timestamp']} 📝 [OBSERVE] {trace_summary_detail}")
        logger.debug(f"Observed and added to working memory.")


    def save_reasoning_chain(self, step_number: int, reasoning_lines: Union[str, List[str]]) -> None:
        """
        🧠 Records a Chain-of-Thought process under the trace_memory log.
        Each line of reasoning for a given step is logged chronologically
        as part of the operational trace.

        Args:
            step_number (int): The current step number in the reasoning chain.
            reasoning_lines (Union[str, List[str]]): A single string or a list of strings
                                                     representing the reasoning steps generated
                                                     at this point in the chain.
        """
        ts = datetime.utcnow().isoformat()
        header = f"{ts} 🧠 [REASONING] Step {step_number}:"
        self.trace_memory.append(header)
        logger.debug(f"Recording reasoning step {step_number}.")

        # Ensure reasoning_lines is treated as a list of strings
        lines_to_log: List[str] = []
        if isinstance(reasoning_lines, str):
            lines_to_log = reasoning_lines.splitlines() # Split single string by lines
        elif isinstance(reasoning_lines, list):
            lines_to_log = [str(line) for line in reasoning_lines] # Ensure all items are strings
        else:
            logger.warning(f"Invalid type for reasoning_lines: {type(reasoning_lines)}. Expected str or List[str]. Attempting conversion.")
            lines_to_log = [str(reasoning_lines)] # Attempt to convert to string as fallback

        for line in lines_to_log:
             line_stripped = line.strip()
             if line_stripped: # Only log non-empty lines after stripping
                 self.trace_memory.append(f"    → {line_stripped[:200]}...") # Log truncated line for brevity
                 # Full lines are typically stored elsewhere (e.g., in the wrapper's output data)


    def store_metric(self, metric_name: str, metric_value: Union[float, int, str]) -> None:
        """
        📊 Appends a timestamped metric entry to the trace log. Useful for
        tracking quantitative outcomes like loss, score, or other key metrics
        at specific operational points.

        Args:
            metric_name (str): A name or identifier for the metric (e.g., "loss", "vote_count").
            metric_value (Union[float, int, str]): The value of the metric. Can be numerical or a string.
        """
        ts = datetime.utcnow().isoformat()
        # Safely format metric_value
        formatted_value: str
        if isinstance(metric_value, (float, int)):
             formatted_value = f"{metric_value:.4f}".rstrip('0').rstrip('.') or '0' # Format floats nicely
        else:
             formatted_value = str(metric_value)[:100] # Truncate strings

        trace_entry = f"{ts} 📊 [METRIC] {metric_name}: {formatted_value}"
        self.trace_memory.append(trace_entry)
        logger.debug(f"Logged metric: {trace_entry}")


    def reflect(self) -> str:
        """
        ✨ Synthesizes the current contents of the working memory into a
        single, comprehensive reflection. This process involves analyzing
        the accumulated experiences and emotional data in working memory.
        The resulting reflection is moved into long-term memory, and then
        working memory is cleared to prepare for a new cycle. Adds an entry
        to the trace log.

        Returns:
            str: A string representing the synthesized comprehensive reflection.
                 Returns a message indicating no working memory to reflect on
                 if the buffer was empty.
        """
        if not self.working_memory:
            reflection_message = "✨ Reflection core finds no new experiences to synthesize."
            logger.debug(reflection_message)
            return reflection_message

        # --- Start: Data preparation for reflection synthesis ---
        # Capture working memory snapshot *before* clearing for analysis and storage
        working_memory_snapshot = list(self.working_memory)

        # Join the original text or summaries from the snapshot for the reflection's content basis
        joined_text_for_reflection = " | ".join(e.get("original_text", e.get("text_summary", "<???>")) for e in working_memory_snapshot)
        joined_text_for_reflection = joined_text_for_reflection[:1500] + "..." if len(joined_text_for_reflection) > 1500 else joined_text_for_reflection # Limit length


        # Analyze the emotional landscape of the captured working memory entries
        emotional_reflection_summary = self._emotional_reflection(working_memory_snapshot)

        # --- End: Data preparation ---


        # Combine the text content synthesis and emotional analysis into the final reflection text
        final_reflection_text = f"Synthesized Reflection: [{joined_text_for_reflection}] ~ Emotional Resonance: ({emotional_reflection_summary})"


        # Create the long-term memory entry
        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "reflection",
            # Store summaries that formed this reflection from the snapshot
            "source_working_memory_summaries": [e.get("text_summary", "<???>") for e in working_memory_snapshot],
            "reflection_text": final_reflection_text, # Store the combined reflection text
            "raw_composite_text_reflected_upon": joined_text_for_reflection, # Store the underlying text content
            # Optionally store the emotional_reflection_summary separately as well
            # "emotional_summary": emotional_reflection_summary
        }

        # Append the reflection to long-term memory (main archive)
        self.long_term_memory.append(entry)

        # Add a trace entry for the reflection event
        self.trace_memory.append(f"{entry['timestamp']} ✨ [REFLECT] {final_reflection_text[:200]}...") # Log truncated reflection in trace
        logger.info(f"Reflected on {len(working_memory_snapshot)} working memory entries. Reflection generated.")

        # Clear working memory *after* its contents have been used for reflection
        self.working_memory.clear()
        logger.debug("Working memory cleared after reflection cycle.")

        return final_reflection_text

    def recall(
        self,
        *, # Enforce keyword-only arguments after this point
        include_working: bool = False, # Renamed from include_observations for clarity
        include_long_term: bool = True, # Renamed from include_reflections
        limit: Optional[int] = None # Added limit for recall
    ) -> List[str]:
        """
        🔍 Retrieves human-readable summaries of memories based on the specified criteria.
        Presents working memory (recent observations) and long-term memory (reflections).
        Useful for presenting memory contents to a user or logging historical context.

        Args:
            include_working (bool): If True, include entries from the current
                                    working memory (recent observations). Defaults to False.
            include_long_term (bool): If True, include entries from the long-term
                                      memory (reflections). Defaults to True.
            limit (Optional[int]): The maximum number of recent entries to return
                                   from the combined memory sources. If None, return all.
                                   Applied after combining and ordering.

        Returns:
            List[str]: A list of formatted strings, each representing a memory entry summary.
                       Returns a list containing "<no memories>" if no entries match
                       the criteria after applying the limit.
        """
        all_recalled_entries: List[Dict[str, Any]] = []

        # Collect entries from working memory (observations)
        if include_working:
             # Add to front of list or use chronological order? Let's keep chronological then reverse.
             all_recalled_entries.extend(self.working_memory)

        # Collect entries from long-term memory (reflections)
        if include_long_term:
             all_recalled_entries.extend(self.long_term_memory)

        # Sort all collected entries by timestamp in descending order (most recent first)
        try:
             # Use a lambda function to safely access timestamp, handling potential missing keys
             all_recalled_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
             logger.warning(f"Could not sort memory entries by timestamp during recall: {e}")
             # If sorting fails, the order might not be strictly chronological

        # Apply limit if specified
        if limit is not None and limit >= 0:
             limited_entries = all_recalled_entries[:limit]
        else:
             limited_entries = all_recalled_entries


        # Format the limited entries into human-readable strings
        formatted_results: List[str] = []
        for e in limited_entries:
            timestamp = e.get("timestamp", "N/A")
            entry_type = e.get("type", "memory_entry") # Default type

            if entry_type == "observation":
                 text_summary = e.get("text_summary", "<???>")
                 emotion_info = ""
                 if e.get("emotion"):
                     emotion = e["emotion"].get("primary", "Unknown")
                     intensity = e["emotion"].get("intensity", 0.0)
                     emotion_info = f" | Feeling: {emotion} ({intensity:.2f})"
                 formatted_results.append(f"{timestamp} 📝 [OBS] {text_summary}{emotion_info}")
            elif entry_type == "reflection":
                 reflection_text = e.get("reflection_text", "<???>")
                 # Use the full reflection text for recall display
                 formatted_results.append(f"{timestamp} ✨ [REFL] {reflection_text}")
            # Add other types if needed


        final_results = formatted_results or ["🔍 <no memories>"]
        logger.debug(f"Recalled {len(formatted_results)} memory entries (Limit: {limit}).")
        return final_results

    def search_memory(
        self,
        query: str,
        *, # Enforce keyword-only arguments after this point
        top_k: Optional[int] = None,
        search_working: bool = True,
        search_long_term: bool = True
    ) -> List[Dict[str, Any]]:
        """
        🔎 Performs a simple case-insensitive keyword search over the textual content
        of specified memory components (working and/or long-term). Results are
        returned in reverse chronological order (most recent matches first).

        Args:
            query (str): The keyword or phrase to search for (case-insensitive).
            top_k (Optional[int]): The maximum number of matching entries to return.
                                   If None, return all matches. Defaults to None.
            search_working (bool): If True, include entries from working memory
                                   in the search. Defaults to True.
            search_long_term (bool): If True, include entries from long-term memory
                                    in the search. Defaults to True.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the matching
                                 memory entries. These are copies of the internal
                                 memory entries. Returns an empty list if no
                                 matches are found or if both search flags are False.
        """
        if not query or not isinstance(query, str):
            logger.warning("Search query is empty or not a string. Returning empty list.")
            return []

        query_lower = query.lower()
        all_entries_to_search: List[Dict[str, Any]] = []

        # Collect entries from specified memory types
        if search_long_term:
            all_entries_to_search.extend(self.long_term_memory)
        if search_working:
            all_entries_to_search.extend(self.working_memory)

        # Sort entries by timestamp in descending order (most recent first) for consistent search results order
        try:
            all_entries_to_search.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.warning(f"Could not sort memory entries for search: {e}")
            # Proceed without guaranteed chronological order if sort fails


        matches: List[Dict[str, Any]] = []
        for e in all_entries_to_search:
            # Search in relevant text fields: summary, original text (for observation), reflection text, raw composite text (for reflection)
            text_content_fields = [
                 e.get("text_summary", ""),         # Observation summary
                 e.get("original_text", ""),       # Observation original text
                 e.get("reflection_text", ""),      # Reflection final text
                 e.get("raw_composite_text_reflected_upon", ""), # Reflection source text
                 # Add other text fields if they are added to entries
            ]

            # Check if query matches in any of the text fields
            if any(query_lower in field.lower() for field in text_content_fields if isinstance(field, str)):
                 # Append a copy of the matching entry
                 matches.append(e.copy()) # Return a copy

        logger.debug(f"Search for '{query}' found {len(matches)} matches across specified memory types.")

        # Apply top_k limit to the found matches
        return matches[:top_k] if top_k is not None and top_k >= 0 else matches


    def export_memory(self) -> str:
        """
        📚 Serializes the complete current state of the memory engine (working
        memory, long-term memory, and trace memory) into a JSON formatted string.
        Provides a snapshot for saving persistence.

        Returns:
            str: A JSON string representing the memory state. Returns an empty JSON
                 object string "{}" if serialization fails due to data types or other errors.
        """
        state = {
            "working_memory":   self.working_memory,
            "long_term_memory": self.long_term_memory,
            "trace_memory":     self.trace_memory,
            "working_capacity": self.working_capacity if self.working_capacity != float('inf') else 0, # Store capacity, convert inf to 0
            "_recent_reflections_limit": self._recent_reflections_limit # Export internal limit
        }
        try:
            # Use default=str to handle any non-serializable types by converting them to string
            return json.dumps(state, indent=2, default=str)
        except TypeError as e:
            logger.error(f"Failed to serialize memory state to JSON (TypeError): {e}")
            # Log a snippet of the state that might contain the problematic data
            try:
                 problem_state_snippet = json.dumps({k: str(v)[:100] + ('...' if len(str(v)) > 100 else '') for k, v in state.items()}, indent=2)
                 logger.error("State causing error (snippet): %s", problem_state_snippet)
            except:
                 logger.error("Could not even serialize state snippet.")
            return "{}" # Return empty JSON object on failure
        except Exception as e:
            logger.error(f"An unexpected error occurred during memory export: {e}")
            return "{}"


    def import_memory(self, json_blob: str) -> None:
        """
        📥 Loads the memory state from a JSON formatted string, overwriting
        the current memory state. Validates the structure to ensure data integrity
        and prevent errors from malformed input.

        Args:
            json_blob (str): A JSON string representing the memory state,
                             expected to be in the format exported by `export_memory`.
                             If the blob is invalid, memory will not be loaded.
        """
        if not isinstance(json_blob, str) or not json_blob.strip():
            logger.warning("Attempted to import empty or non-string JSON blob. Skipping import.")
            return

        try:
            state = json.loads(json_blob)

            # Validate the loaded state structure
            if not isinstance(state, dict):
                logger.error("Import failed: Loaded state is not a dictionary. Expected object with memory lists.")
                return

            # Safely get lists, defaulting to empty lists if keys are missing or not lists
            # Overwrite current memory state only after successful checks
            working_mem = state.get("working_memory", [])
            if not isinstance(working_mem, list):
                logger.warning("Import warning: 'working_memory' in JSON was not a list. Initializing as empty.")
                working_mem = []

            long_term_mem = state.get("long_term_memory", [])
            if not isinstance(long_term_mem, list):
                logger.warning("Import warning: 'long_term_memory' in JSON was not a list. Initializing as empty.")
                long_term_mem = []

            trace_mem = state.get("trace_memory", [])
            if not isinstance(trace_mem, list):
                logger.warning("Import warning: 'trace_memory' in JSON was not a list. Initializing as empty.")
                trace_mem = []

            # Safely load capacity and recent reflections limit, defaulting if missing or invalid
            imported_capacity = state.get("working_capacity", 100)
            if not isinstance(imported_capacity, (int, float)) or imported_capacity < 0:
                 logger.warning(f"Invalid imported working_capacity: {imported_capacity}. Using default 100.")
                 self.working_capacity = 100
            elif imported_capacity == 0:
                 self.working_capacity = float('inf')
            else:
                 self.working_capacity = imported_capacity

            imported_limit = state.get("_recent_reflections_limit", 5)
            if not isinstance(imported_limit, int) or imported_limit < 0:
                 logger.warning(f"Invalid imported _recent_reflections_limit: {imported_limit}. Using default 5.")
                 self._recent_reflections_limit = 5
            else:
                 self._recent_reflections_limit = imported_limit


            # Assign validated data to self
            self.working_memory = working_mem
            self.long_term_memory = long_term_mem
            self.trace_memory = trace_mem


            logger.info(f"Memory state imported successfully. Loaded {len(self.working_memory)} working, {len(self.long_term_memory)} long-term, {len(self.trace_memory)} trace entries.")

        except json.JSONDecodeError as e:
            logger.error(f"Import failed: Invalid JSON format in blob: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during memory import processing: {e}")


    def get_trace(self) -> List[str]:
        """
        📜 Retrieves the full chronological trace log of memory operations
        and significant internal events. Provides a detailed operational history.

        Returns:
            List[str]: A list of strings, each representing an event in the trace log.
                       Returns a copy to prevent external modification.
        """
        return list(self.trace_memory)

    def clear_memory(self, *, clear_working: bool = True, clear_long_term: bool = True, clear_trace: bool = False) -> None:
        """
        🗑️ Clears specified components of the memory system. Use with caution
        as cleared data is not recoverable unless exported beforehand.

        Args:
            clear_working (bool): If True, clears the working memory buffer. Defaults to True.
            clear_long_term (bool): If True, clears the long-term memory (reflections). Defaults to True.
            clear_trace (bool): If True, clears the trace log. Defaults to False.
        """
        if clear_working:
            self.working_memory.clear()
            logger.info("Working memory cleared.")
        if clear_long_term:
            self.long_term_memory.clear()
            logger.info("Long-term memory cleared.")
        if clear_trace:
            self.trace_memory.clear()
            logger.info("Trace memory cleared.")


    # ─── Private helpers ─────────────────────────────────────

    def _decode_input(
        self,
        input_data: Union[str, Any],
        tokenizer: Optional[Any]
    ) -> str:
        """
        Attempts to decode input data, prioritizing tokenizer if available and
        input appears to be a tensor/sequence, falling back to string conversion.

        Args:
            input_data (Union[str, Any]): The data to decode.
            tokenizer (Optional[Any]): A tokenizer object with a `.decode()` method.

        Returns:
            str: The decoded or string-converted representation of the input data.
                 Returns "<decode error>" on failure.
        """
        # Attempt to decode if tokenizer available and input isn't already a string
        if tokenizer is not None and not isinstance(input_data, str):
            try:
                # Check if torch is available before checking for Tensor type
                if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor):
                    # Assuming input_data is a tensor of token IDs, convert to list
                    input_data_processable = input_data.tolist()
                elif isinstance(input_data, list):
                     # Assume it's already a list of token IDs or similar
                     input_data_processable = input_data
                else:
                     # Input is not string, not Tensor, not list - fallback to str()
                     input_data_processable = input_data
                     logger.debug(f"Input is not string, Tensor, or list ({type(input_data)}). Falling back to str() after tokenizer attempt.")


                # Attempt decoding
                return tokenizer.decode(input_data_processable, skip_special_tokens=True)

            except Exception as e:
                logger.warning(f"Failed to decode input with tokenizer ({type(input_data)}): {e}. Falling back to str().")
                # Continue to fallback below

        # Fallback to string conversion for strings, other types, or tokenizer failures
        try:
            return str(input_data)
        except Exception as e:
            logger.error(f"Failed to convert input_data to string after decode attempt: {e}")
            return "<decode error>" # Indicate failure


    @staticmethod
    def _default_summarizer(text: str) -> str:
        """
        Default summarizer function: extracts the first 8 words and last 8 words,
        joining them with an ellipsis. Provides a head-and-tail summary.

        Args:
            text (str): The input text to summarize.

        Returns:
            str: The summarized text. Handles non-string input gracefully.
        """
        if not isinstance(text, str):
            # Handle non-string input by converting and truncating
            str_text = str(text)
            return str_text[:50] + "…" if len(str_text) > 50 else str_text

        words = text.split()
        num_words = len(words)
        summary_length = 8 # Words from start and end

        if num_words <= summary_length * 2:
            return text # Return full text if short
        else:
            start_words = " ".join(words[:summary_length])
            end_words = " ".join(words[-summary_length:])
            # Combine start and end with ellipsis, indicate truncation
            return f"{start_words} ... {end_words}"

    def _emotional_reflection(self, working_memory_entries: List[Dict[str, Any]]) -> str:
        """
        Synthesizes an emotional insight string by analyzing the emotional data
        ('emotion' field) present across the working memory entries being
        reflected upon. Provides a summary of the subjective tone of these memories.

        Args:
            working_memory_entries (List[Dict[str, Any]]): The list of dictionary
                                                          entries from working memory
                                                          that are currently being reflected.

        Returns:
            str: A synthesized string summarizing the emotional tone of these memories.
                 Returns a default message if no emotional data is found.
        """
        if not working_memory_entries:
            return "Emotional Trace: [No memory entries provided for emotional synthesis]."

        # Collect all valid emotion data dictionaries from the entries
        emotion_data_list = [
            e["emotion"] for e in working_memory_entries
            if "emotion" in e and isinstance(e["emotion"], dict) and e["emotion"] # Ensure "emotion" exists, is dict, and not empty
        ]

        if not emotion_data_list:
            return "Emotional Trace: [No specific emotional data found in relevant memories]."

        # Analyze the collected emotion data
        emotion_counts = Counter(e.get("primary", "Unknown") for e in emotion_data_list)
        intensities = [e.get("intensity", 0.0) for e in emotion_data_list if isinstance(e.get("intensity"), (int, float))]

        insight_parts = []
        insight_parts.append(f"Emotional Trace (analyzed across {len(emotion_data_list)} relevant points):")

        # Report dominant emotions (up to top 3)
        if emotion_counts:
            most_common = emotion_counts.most_common(3)
            common_summary = ", ".join([f"'{label}' ({count}x)" for label, count in most_common])
            insight_parts.append(f"Dominant feelings: {common_summary}.")

        # Report intensity range and average
        if intensities:
            min_intensity = min(intensities)
            max_intensity = max(intensities)
            avg_intensity = sum(intensities) / len(intensities)
            # Add more descriptive intensity analysis based on range/average
            intensity_description = f"ranging [{min_intensity:.2f}-{max_intensity:.2f}], average {avg_intensity:.2f}"
            if avg_intensity > 0.7:
                 intensity_description += " (indicating a period of heightened feeling)"
            elif avg_intensity < 0.3:
                 intensity_description += " (suggesting a calm or neutral emotional tone)"
            insight_parts.append(f"Intensity: {intensity_description}.")


            # Mention specific high intensity moments if any (intensity > 0.75)
            high_intensity_moments = [
                f"'{e.get('primary', 'Unknown')}' ({e.get('intensity', 0.0):.2f})"
                for e in emotion_data_list if isinstance(e.get("intensity"), (int, float)) and e.get("intensity", 0.0) > 0.75 # Higher threshold
            ]
            if high_intensity_moments:
                 high_intensity_summary = ", ".join(high_intensity_moments[:4]) # Up to 4 examples
                 insight_parts.append(f"Notable peaks included: {high_intensity_summary}{'...' if len(high_intensity_moments) > 4 else ''}.")

        # Add some introspective flavor text connecting emotions to reflection
        flavor_texts = [
             "These subjective states are integral to the processed experiences.",
             "The emotional context shapes the narrative of memory.",
             "Feelings are synthesized alongside factual data in reflection.",
             "Understanding the emotional trace provides deeper insight."
        ]
        insight_parts.append(random.choice(flavor_texts))


        return " ".join(insight_parts)


# Example Usage (Illustrative)
if __name__ == "__main__":
    print("--- MemoryEngine Example Usage ---")
    # Set logger level to DEBUG for this specific example run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Ensure this logger also uses DEBUG

    memory = MemoryEngine(working_capacity=5) # Small capacity for demo

    # Simulate observations with varying emotions
    print(memory.observe("User initiated a query about complex ethical scenarios.", emotion_data={"primary_emotion": "curiosity", "intensity": 0.8}))
    print(memory.observe("Model began processing the input and retrieving relevant knowledge fragments.")) # No emotion
    print(memory.observe("The initial generated steps showed unexpected patterns.", emotion_data={"primary_emotion": "surprise", "intensity": 0.6}))
    print(memory.observe("Identifying a potential conflict in the generated reasoning.", emotion_data={"primary_emotion": "concern", "intensity": 0.5}))
    print(memory.observe("Successfully navigated the reasoning conflict, finding a coherent path.", emotion_data={"primary_emotion": "satisfaction", "intensity": 0.95})) # High intensity
    print(memory.observe("Preparing the final answer and full output.", emotion_data={"primary_emotion": "anticipation", "intensity": 0.7})) # Exceeds capacity, one will be dropped

    # Simulate recording reasoning steps (even if simplified)
    memory.save_reasoning_chain(1, ["Initial thought process engaged.", "Consulted internal knowledge graphs."])
    memory.save_reasoning_chain(2, "Identified key entities and relationships.")
    memory.save_reasoning_chain(3, ["Formulating hypothesis.", "Evaluating potential solutions based on constraints."])

    # Simulate recording metrics (conceptual)
    memory.store_metric("initial_prompt_length", 42)
    memory.store_metric("generation_time_sec", 3.5)
    memory.store_metric("self_consistency_votes", 3)


    print("\n--- Current Trace ---")
    for entry in memory.get_trace():
        print(entry)

    print("\n--- Working Memory before Reflection ---")
    # Pretty print working memory for clarity
    print(json.dumps(memory.working_memory, indent=2))

    # Simulate reflection
    reflection_summary = memory.reflect()
    print(f"\n--- Reflection Result ---\n{reflection_summary}")

    print("\n--- Working Memory after Reflection ---")
    print(memory.working_memory) # Should be empty

    print("\n--- Long-Term Memory ---")
    # Pretty print long-term memory for clarity
    print(json.dumps(memory.long_term_memory, indent=2))

    # Simulate recalling memories
    print("\n--- Recalled Memories (Working + Long-Term) ---")
    recalled = memory.recall(include_working=True, include_long_term=True, limit=10) # Recall up to 10
    for mem_str in recalled:
        print(mem_str)

    print("\n--- Recalled Only Reflections ---")
    recalled_reflections = memory.recall(include_working=False, include_long_term=True)
    for mem_str in recalled_reflections:
        print(mem_str)

    print("\n--- Search Memory ('reasoning') ---")
    search_results = memory.search_memory("reasoning", search_working=True, search_long_term=True)
    print(json.dumps(search_results, indent=2)) # Pretty print search results

    print("\n--- Search Memory ('satisfaction') - limiting to 1 ---")
    search_results_emotion = memory.search_memory("satisfaction", top_k=1)
    print(json.dumps(search_results_emotion, indent=2))


    # Simulate export and import
    print("\n--- Exporting Memory ---")
    exported_json = memory.export_memory()
    print(exported_json[:800] + "..." if len(exported_json) > 800 else exported_json) # Print snippet

    print("\n--- Importing Memory into New Engine ---")
    new_memory = MemoryEngine(working_capacity=7) # Test different capacity
    new_memory.import_memory(exported_json)

    print("\n--- New Engine Recalled Memories (After Import) ---")
    new_recalled = new_memory.recall(include_working=True, include_long_term=True)
    for mem_str in new_recalled:
        print(mem_str)

    print("\n--- New Engine Trace (After Import) ---")
    new_trace = new_memory.get_trace()
    for entry in new_trace:
        print(entry)

    # Test clearing memory
    print("\n--- Clearing Working and Long-Term Memory in New Engine ---")
    new_memory.clear_memory(clear_working=True, clear_long_term=True, clear_trace=False)
    print("\n--- New Engine Memory after partial clear ---")
    print(new_memory.recall(include_working=True, include_long_term=True))
    print("\n--- New Engine Trace after partial clear ---")
    print(new_memory.get_trace())


    print("\n--- Example Usage End ---")