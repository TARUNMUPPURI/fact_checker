"""
llm_extractor.py
----------------
Calls Gemini Flash to extract structured metadata from raw sarga text.
(Previously used Claude Haiku / OpenAI GPT-4o-mini — kept as comments below)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import openai
# import google.generativeai as genai
# import anthropic

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "gpt-5.4-mini"
# MODEL = "gemini-1.5-flash"   # free tier: 1500 req/day, 15 RPM
# MODEL = "claude-haiku-4-5-20251001"  # Anthropic alternative
MAX_RETRIES = 3

VALID_EVENT_TYPES = {
    "battle", "journey", "dialogue", "discovery", "ceremony",
    "death", "transformation", "revelation", "devotion", "other",
}
VALID_NARRATIVE_ARCS = {
    "opening", "rising_action", "climax", "falling_action", "resolution",
}
VALID_EMOTIONAL_TONES = {
    "heroic", "sorrowful", "devotional", "joyful", "fearful",
    "wrathful", "tender", "contemplative", "ominous", "neutral",
}
VALID_THEMES = {
    "dharma", "devotion", "sacrifice", "war", "exile", "reunion",
    "deception", "nature", "courage", "grief", "loyalty", "divine_intervention",
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Raised after all retries are exhausted on a single sarga."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    chunk_id: str
    characters_present: list[str] = field(default_factory=list)
    characters_speaking: list[str] = field(default_factory=list)
    characters_primary: list[str] = field(default_factory=list)
    locations_mentioned: list[str] = field(default_factory=list)
    objects_mentioned: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    event_type: str = "other"
    narrative_arc: str = "rising_action"
    emotional_tone: str = "neutral"
    themes: list[str] = field(default_factory=list)
    text_summary: str = ""
    is_interpolated: bool = False
    confidence: float = 0.0
    unknown_characters: list[str] = field(default_factory=list)
    unknown_locations: list[str] = field(default_factory=list)
    raw_llm_response: str = ""
    tokens_used: int = 0
    extraction_status: str = "failed"   # success | partial | failed


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class LLMExtractor:
    """Extracts structured metadata from sarga text via Claude Haiku."""

    def __init__(self, aliases_path: str, api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        with open(aliases_path, encoding="utf-8") as f:
            aliases_data = json.load(f)

        characters = aliases_data.get("characters", {})
        locations = aliases_data.get("locations", {})
        objects_ = aliases_data.get("objects", {})

        self.valid_character_ids: set[str] = set(characters.keys())
        self.valid_location_ids: set[str] = set(locations.keys())
        self.valid_object_ids: set[str] = set(objects_.keys())

        # Flat deduplicated list of all known key_events across all characters
        all_events: list[str] = []
        seen_events: set[str] = set()
        for char in characters.values():
            for ev in char.get("key_events", []):
                if ev not in seen_events:
                    seen_events.add(ev)
                    all_events.append(ev)
        self.valid_event_ids: list[str] = all_events

        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        # --- Gemini Version ---
        # _gemini_key = gemini_api_key or api_key or os.getenv("GEMINI_API_KEY")
        # genai.configure(api_key=_gemini_key)
        # self._client = genai.GenerativeModel(MODEL)
        # --- Anthropic Version ---
        # self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        chunk_id: str,
        sarga_text: str,
        kanda_english: str,
        sarga_number: int,
    ) -> ExtractionResult:
        """
        Call Gemini Flash to extract metadata for one sarga.
        Retries up to MAX_RETRIES times on RateLimitError with exponential back-off.
        Raises ExtractionError if all retries fail.
        """
        prompt = self._build_prompt(sarga_text, kanda_english, sarga_number)
        raw_response = ""
        tokens_used = 0
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                message = self._client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_response = message.choices[0].message.content
                tokens_used = message.usage.total_tokens
                break

            except openai.RateLimitError as exc:
                wait = (2 ** attempt) * 5
                log.warning(
                    "%s: RateLimitError (attempt %d/%d) — retrying in %ds",
                    chunk_id, attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_error = exc

            except Exception as exc:
                log.error("%s: Unexpected API error: %s", chunk_id, exc)
                last_error = exc
                break
        else:
            raise ExtractionError(
                f"{chunk_id}: all {MAX_RETRIES} retries failed"
            ) from last_error

        if not raw_response:
            raise ExtractionError(f"{chunk_id}: empty response from model")

        parsed = self._parse_json(raw_response)
        if parsed is None:
            result = ExtractionResult(
                chunk_id=chunk_id,
                raw_llm_response=raw_response,
                tokens_used=tokens_used,
                extraction_status="failed",
            )
            return result

        result = self._build_result(chunk_id, parsed, raw_response, tokens_used)
        result = self._validate(result)
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self, sarga_text: str, kanda_english: str, sarga_number: int
    ) -> str:
        char_ids = sorted(self.valid_character_ids)
        loc_ids = sorted(self.valid_location_ids)
        obj_ids = sorted(self.valid_object_ids)
        evt_ids = sorted(self.valid_event_ids)

        return f"""You are a Valmiki Ramayana scholar. Analyse the following sarga text and return a JSON object.

SARGA CONTEXT:
- Kanda: {kanda_english}
- Sarga number: {sarga_number}

VALID CHARACTER IDs (use ONLY these):
{json.dumps(char_ids, ensure_ascii=False)}

VALID LOCATION IDs (use ONLY these):
{json.dumps(loc_ids, ensure_ascii=False)}

VALID OBJECT IDs (use ONLY these):
{json.dumps(obj_ids, ensure_ascii=False)}

VALID EVENT IDs (use ONLY these, or empty list):
{json.dumps(evt_ids, ensure_ascii=False)}

EXTRACTION RULES:
- characters_present means physically present or acting in the sarga. NOT merely mentioned in thought, memory, or speech.
- Only use IDs from the VALID lists. Put any other names in unknown_characters. Never invent new IDs.
- confidence below 0.7 means text was unclear or you made assumptions.

REQUIRED JSON SCHEMA (return ONLY valid JSON, no preamble, no markdown fences):
{{
  "characters_present": ["<character_id>", ...],
  "characters_speaking": ["<character_id>", ...],
  "characters_primary": ["<character_id>", ...],
  "locations_mentioned": ["<location_id>", ...],
  "objects_mentioned": ["<object_id>", ...],
  "event_ids": ["<event_id>", ...],
  "event_type": "<one of: battle|journey|dialogue|discovery|ceremony|death|transformation|revelation|devotion|other>",
  "narrative_arc": "<one of: opening|rising_action|climax|falling_action|resolution>",
  "emotional_tone": "<one of: heroic|sorrowful|devotional|joyful|fearful|wrathful|tender|contemplative|ominous|neutral>",
  "themes": ["<theme>", ...],
  "text_summary": "<2-3 sentence summary>",
  "is_interpolated": false,
  "confidence": 0.85,
  "unknown_characters": ["<name as it appears in text>", ...],
  "unknown_locations": ["<name as it appears in text>", ...]
}}

SARGA TEXT:
{sarga_text[:45000]}"""

    # ------------------------------------------------------------------
    # JSON parsing (with fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        # Strip markdown fences if present
        cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*$", "", cleaned.strip())

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: extract outermost {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        log.warning("JSON parsing failed for response: %.200s", raw)
        return None

    # ------------------------------------------------------------------
    # Build result from parsed dict
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        chunk_id: str, parsed: dict, raw: str, tokens: int
    ) -> ExtractionResult:
        def _list(key: str) -> list[str]:
            val = parsed.get(key, [])
            return [str(v) for v in val] if isinstance(val, list) else []

        return ExtractionResult(
            chunk_id=chunk_id,
            characters_present=_list("characters_present"),
            characters_speaking=_list("characters_speaking"),
            characters_primary=_list("characters_primary"),
            locations_mentioned=_list("locations_mentioned"),
            objects_mentioned=_list("objects_mentioned"),
            event_ids=_list("event_ids"),
            event_type=str(parsed.get("event_type", "other")),
            narrative_arc=str(parsed.get("narrative_arc", "rising_action")),
            emotional_tone=str(parsed.get("emotional_tone", "neutral")),
            themes=_list("themes"),
            text_summary=str(parsed.get("text_summary", "")),
            is_interpolated=bool(parsed.get("is_interpolated", False)),
            confidence=float(parsed.get("confidence", 0.0)),
            unknown_characters=_list("unknown_characters"),
            unknown_locations=_list("unknown_locations"),
            raw_llm_response=raw,
            tokens_used=tokens,
            extraction_status="failed",  # set by _validate
        )

    # ------------------------------------------------------------------
    # Validation + status assignment
    # ------------------------------------------------------------------

    def _validate(self, r: ExtractionResult) -> ExtractionResult:
        # Filter character lists — unknowns go to unknown_characters
        # Filter characters
        valid_present = []
        extra_unknown = []
        
        # 1. Check characters_present
        for char_id in r.characters_present:
            char_id = char_id.strip().lower().replace(" ", "_").replace("-", "_")
            if char_id in self.valid_character_ids:
                valid_present.append(char_id)
            else:
                extra_unknown.append(char_id)
                
        # 2. Check unknown_characters
        still_unknown = []
        for u in r.unknown_characters:
            uid = u.strip().lower().replace(" ", "_").replace("-", "_")
            if uid in self.valid_character_ids:
                valid_present.append(uid)
            else:
                still_unknown.append(u.strip())
                
        r.characters_present = list(dict.fromkeys(valid_present))
        r.unknown_characters = list(dict.fromkeys(still_unknown + extra_unknown))

        r.characters_speaking = [
            c for c in r.characters_speaking if c.strip().lower().replace(" ", "_").replace("-", "_") in self.valid_character_ids
        ]
        r.characters_primary = [
            c for c in r.characters_primary if c.strip().lower().replace(" ", "_").replace("-", "_") in self.valid_character_ids
        ]

        valid_locs, extra_unknown_locs = [], []
        for lid in r.locations_mentioned:
            if lid in self.valid_location_ids:
                valid_locs.append(lid)
            else:
                extra_unknown_locs.append(lid)
        r.locations_mentioned = valid_locs
        r.unknown_locations = list(dict.fromkeys(r.unknown_locations + extra_unknown_locs))

        # Filter objects
        r.objects_mentioned = [
            o for o in r.objects_mentioned if o in self.valid_object_ids
        ]

        # Filter event IDs
        valid_event_set = set(self.valid_event_ids)
        r.event_ids = [e for e in r.event_ids if e in valid_event_set]

        # Clamp confidence
        r.confidence = max(0.0, min(1.0, r.confidence))

        # Controlled vocabularies
        if r.event_type not in VALID_EVENT_TYPES:
            r.event_type = "other"
        if r.narrative_arc not in VALID_NARRATIVE_ARCS:
            r.narrative_arc = "rising_action"
        if r.emotional_tone not in VALID_EMOTIONAL_TONES:
            r.emotional_tone = "neutral"
        r.themes = [t for t in r.themes if t in VALID_THEMES]

        # Assign extraction_status
        if r.confidence >= 0.7 and not r.unknown_characters:
            r.extraction_status = "success"
        else:
            r.extraction_status = "partial"

        return r
