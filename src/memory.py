"""
Three-layer memory module for cross-episode action-pattern learning.

Layer 1 & 2: RawMemoryStore — per-step success/failure entries with entity-overlap retrieval.
Layer 3: InsightStore — model-extracted structured patterns with condition-based matching.
InsightExtractor — periodically calls an LLM to analyze raw memories into insights.
MemoryAugmenter — formats matched insights and raw examples into a prompt hint.
"""

import json
import os
import re
import time
from collections import namedtuple

from . import constants

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

MemoryEntry = namedtuple("MemoryEntry", [
    "id", "question", "step", "entities", "success",
    "predicted_candidates", "correct_action", "action_type",
    "prev_action", "prev_action_type", "analyzed", "timestamp",
])


def _make_entry(eid, question, step, entities, success,
                predicted_candidates, correct_action, action_type,
                prev_action, prev_action_type, analyzed=False, timestamp=None):
    return MemoryEntry(
        id=eid,
        question=question,
        step=step,
        entities=list(entities) if entities else [],
        success=success,
        predicted_candidates=list(predicted_candidates) if predicted_candidates else [],
        correct_action=correct_action,
        action_type=action_type,
        prev_action=prev_action,
        prev_action_type=prev_action_type,
        analyzed=analyzed,
        timestamp=timestamp or time.time(),
    )


# ---------------------------------------------------------------------------
# RawMemoryStore
# ---------------------------------------------------------------------------

class RawMemoryStore:
    """Stores, retrieves, and persists per-step success/failure memory entries."""

    def __init__(self, storage_path=None, max_entries=None):
        self.storage_path = storage_path or constants.memory_store_path
        self.max_entries = max_entries or constants.memory_max_entries
        self.entries = []          # list of MemoryEntry
        self._next_id = 0
        self._load()

    # ---- public API -------------------------------------------------------

    def add(self, question, step, entities, success, predicted_candidates,
            correct_action, action_type, prev_action, prev_action_type):
        entry = _make_entry(
            eid=self._next_id,
            question=question,
            step=step,
            entities=entities,
            success=success,
            predicted_candidates=predicted_candidates,
            correct_action=correct_action,
            action_type=action_type,
            prev_action=prev_action,
            prev_action_type=prev_action_type,
        )
        self._next_id += 1
        self.entries.append(entry)
        self._prune()

    def retrieve(self, question, step, prev_action=None,
                 k_success=None, k_failure=None):
        """Return (success_entries, failure_entries) for the given context."""
        if not self.entries or not constants.memory_enabled:
            return [], []

        k_success = k_success or constants.memory_top_k_success
        k_failure = k_failure or constants.memory_top_k_failure

        query_entities = set(self._extract_entities(question))
        query_prev_type = self._get_action_type(prev_action)

        scored = []
        for entry in self.entries:
            score = self._compute_score(query_entities, step, query_prev_type, entry)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        successes = []
        failures = []
        seen_success = set()
        seen_failure = set()

        for _score, entry in scored:
            if entry.success:
                if entry.correct_action not in seen_success:
                    successes.append(entry)
                    seen_success.add(entry.correct_action)
            else:
                if entry.correct_action not in seen_failure:
                    failures.append(entry)
                    seen_failure.add(entry.correct_action)

        return successes[:k_success], failures[:k_failure]

    def get_unanalyzed(self):
        return [e for e in self.entries if not e.analyzed]

    def mark_analyzed(self, entry_ids):
        ids = set(entry_ids)
        for i, e in enumerate(self.entries):
            if e.id in ids:
                self.entries[i] = e._replace(analyzed=True)

    def get_stats(self):
        total = len(self.entries)
        success = sum(1 for e in self.entries if e.success)
        failure = total - success
        unanalyzed = sum(1 for e in self.entries if not e.analyzed)
        return {"total": total, "success": success, "failure": failure, "unanalyzed": unanalyzed}

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "next_id": self._next_id,
            "entries": [e._asdict() for e in self.entries],
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def __len__(self):
        return len(self.entries)

    # ---- internal ---------------------------------------------------------

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            self._next_id = data.get("next_id", 0)
            self.entries = [MemoryEntry(**item) for item in data.get("entries", [])]
        except (json.JSONDecodeError, TypeError):
            self.entries = []
            self._next_id = 0

    @staticmethod
    def _extract_entities(text):
        if not text:
            return []
        proper = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        topic_words = re.findall(
            r'\b(year|born|died|city|country|capital|founded|ceo|president|'
            r'author|director|population|located|name|known|famous|'
            r'what|who|where|when|how|why|which)\b',
            text.lower(),
        )
        return list(set(proper + topic_words))

    @staticmethod
    def _get_action_type(action):
        if not action:
            return None
        al = action.lower()
        if al.startswith("search"):
            return "Search"
        elif al.startswith("lookup"):
            return "Lookup"
        elif al.startswith("finish"):
            return "Finish"
        return None

    def _compute_score(self, query_entities, query_step, query_prev_type, entry):
        w_entity = constants.memory_entity_overlap_weight
        w_step = constants.memory_step_match_weight
        w_prev = constants.memory_prev_action_match_weight
        w_action = constants.memory_action_type_match_weight

        entry_entities = set(entry.entities)
        if not query_entities or not entry_entities:
            entity_score = 0.0
        else:
            intersection = len(query_entities & entry_entities)
            union = len(query_entities | entry_entities)
            entity_score = intersection / union if union > 0 else 0.0

        step_score = 1.0 if query_step == entry.step else 0.0
        prev_score = 1.0 if (query_prev_type and query_prev_type == entry.prev_action_type) else 0.0
        action_score = 1.0 if (query_prev_type and query_prev_type == entry.action_type) else 0.0

        return (w_entity * entity_score + w_step * step_score +
                w_prev * prev_score + w_action * action_score)

    def _prune(self):
        while len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.timestamp)
            self.entries.pop(0)


# ---------------------------------------------------------------------------
# InsightStore
# ---------------------------------------------------------------------------

class InsightStore:
    """Stores, matches, and persists structured insights extracted from raw memories."""

    def __init__(self, insights_path=None):
        self.insights_path = insights_path or constants.memory_insights_path
        self.insights = []         # list of insight dicts
        self._load()

    # ---- public API -------------------------------------------------------

    def add_insights(self, new_insights):
        if not new_insights:
            return
        existing_patterns = [i.get("pattern", "") for i in self.insights]
        for ni in new_insights:
            if not self._is_duplicate(ni["pattern"], existing_patterns):
                ni["timestamp"] = time.time()
                self.insights.append(ni)
                existing_patterns.append(ni["pattern"])

    def match(self, question, step, prev_action_type, threshold=None):
        threshold = threshold or constants.insight_match_threshold
        scored = []
        for ins in self.insights:
            s = self._compute_match_score(ins, question, step, prev_action_type)
            if s >= threshold:
                scored.append((s, ins))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ins for _, ins in scored]

    def get_all(self):
        return list(self.insights)

    def save(self):
        os.makedirs(os.path.dirname(self.insights_path), exist_ok=True)
        with open(self.insights_path, "w") as f:
            json.dump(self.insights, f, indent=2)

    def __len__(self):
        return len(self.insights)

    # ---- internal ---------------------------------------------------------

    def _load(self):
        if not os.path.exists(self.insights_path):
            return
        try:
            with open(self.insights_path, "r") as f:
                self.insights = json.load(f)
        except (json.JSONDecodeError, TypeError):
            self.insights = []

    def _compute_match_score(self, insight, question, step, prev_action_type):
        score = 0.0
        conditions = insight.get("conditions", {})

        keywords = conditions.get("question_keywords", [])
        if keywords:
            qlower = question.lower()
            hits = sum(1 for kw in keywords if kw.lower() in qlower)
            if hits > 0:
                score += 0.5 * (hits / len(keywords))

        cond_step = conditions.get("step")
        if cond_step is not None and cond_step == step:
            score += 0.3

        cond_prev = conditions.get("prev_action_type")
        if cond_prev is not None and cond_prev == prev_action_type:
            score += 0.2

        return score

    @staticmethod
    def _is_duplicate(new_pattern, existing_patterns):
        """Simple overlap-based dedup. Two patterns are duplicates if their
        word sets have Jaccard > 0.7."""
        new_words = set(new_pattern.lower().split())
        if not new_words:
            return False
        for ep in existing_patterns:
            ep_words = set(ep.lower().split())
            if not ep_words:
                continue
            inter = len(new_words & ep_words)
            union = len(new_words | ep_words)
            if union > 0 and inter / union > 0.7:
                return True
        return False


# ---------------------------------------------------------------------------
# InsightExtractor
# ---------------------------------------------------------------------------

class InsightExtractor:
    """Calls an LLM to extract structured insights from raw memory entries."""

    @staticmethod
    def extract(raw_entries, existing_insights, llm_client):
        prompt = InsightExtractor._build_extraction_prompt(raw_entries, existing_insights)
        response = llm_client.call(prompt) or ""
        return InsightExtractor._parse_insights(response)

    @staticmethod
    def _build_extraction_prompt(raw_entries, existing_insights):
        from .prompts import PromptTemplates

        lines = []
        for e in raw_entries:
            marker = "+" if e.success else "-"
            cand_str = ", ".join(e.predicted_candidates[:3])
            if e.success:
                line = (f"{marker} Q: \"{e.question[:120]}\" | "
                        f"Step {e.step} | {e.correct_action}")
            else:
                line = (f"{marker} Q: \"{e.question[:120]}\" | "
                        f"Step {e.step} | Predicted: {cand_str} | "
                        f"Correct: {e.correct_action}")
            lines.append(line)

        if existing_insights:
            existing_str = json.dumps(existing_insights, indent=2)
        else:
            existing_str = "(none yet)"

        return PromptTemplates.INSIGHT_EXTRACTION_PROMPT.format(
            raw_entries="\n".join(lines),
            existing_insights=existing_str,
        )

    @staticmethod
    def _parse_insights(response):
        if not response:
            return []

        # Try to extract a JSON array from the response
        json_str = response.strip()

        # Strip markdown code fences if present
        if json_str.startswith("```"):
            json_str = re.sub(r'^```(?:json)?\s*\n', '', json_str)
            json_str = re.sub(r'\n```\s*$', '', json_str)

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return InsightExtractor._validate_insights(parsed)
            elif isinstance(parsed, dict):
                return InsightExtractor._validate_insights([parsed])
        except json.JSONDecodeError:
            pass

        # Fallback: try to find a JSON array with regex
        m = re.search(r'\[.*\]', json_str, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    return InsightExtractor._validate_insights(parsed)
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _validate_insights(insights):
        valid = []
        for ins in insights:
            if not isinstance(ins, dict):
                continue
            if not ins.get("pattern"):
                continue
            if ins.get("support_count", 0) < constants.insight_min_support:
                # Still accept but mark low confidence
                ins["confidence"] = min(ins.get("confidence", 0.5), 0.5)
            # Ensure required fields
            ins.setdefault("conditions", {})
            ins.setdefault("guidance", "")
            ins.setdefault("preferred_actions", [])
            ins.setdefault("avoid_actions", [])
            ins.setdefault("confidence", 0.5)
            ins.setdefault("support_count", 0)
            valid.append(ins)
        return valid


# ---------------------------------------------------------------------------
# MemoryAugmenter
# ---------------------------------------------------------------------------

class MemoryAugmenter:
    """Formats matched insights and raw memory entries into a prompt hint string."""

    @staticmethod
    def format_memory_hint(matched_insights, raw_successes, raw_failures):
        from .prompts import PromptTemplates

        insight_section = MemoryAugmenter._format_insight_section(matched_insights)
        example_section = MemoryAugmenter._format_example_section(raw_successes, raw_failures)

        if not insight_section and not example_section:
            return ""

        return PromptTemplates.MEMORY_HINT_PROMPT.format(
            insight_section=insight_section,
            example_section=example_section,
        )

    @staticmethod
    def _format_insight_section(insights):
        if not insights:
            return ""
        lines = ["### Relevant Patterns (learned from past experience):"]
        for i, ins in enumerate(insights, 1):
            pref = ", ".join(ins.get("preferred_actions", [])) or "N/A"
            avoid = ", ".join(ins.get("avoid_actions", [])) or "N/A"
            guidance = ins.get("guidance", "")
            lines.append(f"{i}. {ins['pattern']}")
            if guidance:
                lines.append(f"   Guidance: {guidance}")
            lines.append(f"   -> Prefer: {pref} | Avoid: {avoid}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _format_example_section(successes, failures):
        if not successes and not failures:
            return ""
        lines = ["### Similar Past Examples:"]
        for e in successes:
            lines.append(f"+ Q: \"{e.question[:120]}\" | Step {e.step} | {e.correct_action}")
        for e in failures:
            cand_str = ", ".join(e.predicted_candidates[:3])
            lines.append(f"- Q: \"{e.question[:120]}\" | Step {e.step} | "
                         f"Predicted: {cand_str} | Correct: {e.correct_action}")
        return "\n".join(lines) + "\n\n"
