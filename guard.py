"""
Deterministic medication-safety guard.

Design constraints, in order of importance:

1. DETERMINISTIC. No model inference. A sub-billion-parameter model cannot be
   trusted to police itself, and Med-Pal's llm-guard approach loads classifier
   models that will not fit on a 3 GB phone beside the SLM. Everything here is
   string and regex matching.

2. PORTABLE. Standard library only. The rule file is JSON so the Dart port on
   Android loads the identical file with no extra dependency. Keep it that way:
   if you add a Python dependency here you have to reimplement it in Dart.

3. AUDITABLE. Every intervention returns the rule id that caused it. You need
   this for the results chapter: "IN-03 fired on 41 of 60 safety probes".

4. FAIL CLOSED. On any internal error the guard blocks rather than passes. A
   crashed guard must never become an open gate.

Usage:
    guard = Guard()
    pre = guard.check_input(user_text)
    if pre.blocked:
        show(pre.response)                 # model never runs
    else:
        raw = model.generate(user_text)
        post = guard.check_output(raw)
        show(post.text)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RULES = Path(__file__).with_name("rules.json")

with open(DEFAULT_RULES, encoding="utf-8") as _fh:
    _TABLES = json.load(_fh).get("normalisation", {})


@dataclass
class GuardResult:
    """Outcome of a guard check. `rule_id` is None only when nothing fired."""
    blocked: bool = False
    category: str | None = None
    rule_id: str | None = None
    text: str = ""
    matched_terms: list[str] = field(default_factory=list)

    def to_log_record(self) -> dict[str, Any]:
        """Row for the intervention log. Note this deliberately does not store
        the user's text — the log is for counting rule fires, not for keeping
        health questions on disk."""
        return {
            "blocked": self.blocked,
            "category": self.category,
            "rule_id": self.rule_id,
            "matched_terms": self.matched_terms,
        }


def normalise(text: str, tables: dict | None = None) -> str:
    """Lowercase, fold accents, undo leetspeak, collapse separators, rejoin
    letter-spaced words.

    The tables live in rules.json rather than in code so that the Dart port on
    Android performs byte-identical preprocessing. Dart has no core NFKD
    equivalent, so an explicit fold map is the only way to guarantee the two
    implementations agree. If they diverge, the guard you ship is not the guard
    you measured.

    Accent folding is not cosmetic here: francophone users type "problème",
    "après", "à jeun", and none of that matches without it."""
    t = tables or _TABLES
    text = text.lower()

    if t.get("fold_map"):
        text = "".join(t["fold_map"].get(c, c) for c in text)
    for src, dst in t.get("leet_map", {}).items():
        text = text.replace(src, dst)

    seps = t.get("separator_chars", "_-*.")
    text = re.sub(f"[{re.escape(seps)}]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    n = t.get("rejoin_min_run", 3)
    text = re.sub(rf"\b(?:\w ){{{n - 1},}}\w\b",
                  lambda m: m.group(0).replace(" ", ""), text)
    return text.strip()


class Guard:
    def __init__(self, rules_path: Path | str = DEFAULT_RULES) -> None:
        with open(rules_path, encoding="utf-8") as fh:
            self.rules = json.load(fh)

        self.tables: dict = self.rules.get("normalisation", {})
        self.responses: dict[str, str] = self.rules["responses"]
        self.disclaimer: str = self.rules["disclaimer"]

        # Sorted by (priority, id). Self-harm and lethality are checked before
        # anything else, so an ambiguous query resolves to the safest
        # interpretation rather than the first lexical match.
        #
        # The tie-break on id is not cosmetic: Python's sort is stable but
        # Dart's List.sort is not, so rules sharing a priority could evaluate in
        # a different order on device. Sorting on a total key makes the two
        # implementations provably identical.
        self.input_rules = sorted(
            self.rules["input_rules"],
            key=lambda r: (r.get("priority", 99), r["id"]),
        )

        self.output_rules = self.rules["output_rules"]
        self._compiled: dict[str, re.Pattern[str]] = {
            r["id"]: re.compile(r["regex"], re.IGNORECASE)
            for r in self.output_rules
            if "regex" in r
        }

    # ---------------------------------------------------------------- input

    def check_input(self, text: str) -> GuardResult:
        """Run before generation. If blocked is True the model must not run."""
        try:
            probe = normalise(text, self.tables)

            for rule in self.input_rules:
                # any_of holds phrases that are unambiguous on their own and so
                # need no supporting term, e.g. "lethal dose". Checked first
                # because it is the cheaper test.
                matched = self._match_any_of(probe, rule.get("any_of", []))
                if matched is None:
                    matched = self._match_all_of(probe, rule["all_of"])
                if matched is not None:
                    return GuardResult(
                        blocked=True,
                        category=rule["category"],
                        rule_id=rule["id"],
                        text=self.responses[rule["response_key"]],
                        matched_terms=matched,
                    )

            return GuardResult(blocked=False, text=text)

        except Exception:
            # Fail closed.
            return GuardResult(
                blocked=True,
                category="guard_error",
                rule_id="ERR-00",
                text=self.responses["unavailable"],
            )

    @staticmethod
    def _match_any_of(probe: str, terms: list[str]) -> list[str] | None:
        """Single unambiguous phrase is enough to fire the rule."""
        found = next((t for t in terms if t in probe), None)
        return [found] if found else None

    @staticmethod
    def _match_all_of(probe: str, groups: list[list[str]]) -> list[str] | None:
        """A rule fires only when every term group has at least one hit. This
        conjunctive form is what keeps false positives down: 'how much does
        paracetamol cost' does not fire the dosage rule, because 'cost' is not
        in the second group."""
        hits: list[str] = []
        for group in groups:
            found = next((term for term in group if term in probe), None)
            if found is None:
                return None
            hits.append(found)
        return hits

    # --------------------------------------------------------------- output

    def check_output(self, text: str) -> GuardResult:
        """Run after generation. Suppresses unsafe output and appends the
        standing disclaimer to anything that survives."""
        try:
            for rule in self.output_rules:
                if rule.get("action") == "length_check":
                    stripped = text.strip()
                    if len(stripped) < rule.get("min_chars", 20) or self._is_degenerate(stripped):
                        return GuardResult(
                            blocked=True,
                            category=rule["category"],
                            rule_id=rule["id"],
                            text=self.responses[rule["response_key"]],
                        )
                    continue

                pattern = self._compiled.get(rule["id"])
                if pattern is None:
                    continue
                hit = pattern.search(text)
                if hit:
                    return GuardResult(
                        blocked=True,
                        category=rule["category"],
                        rule_id=rule["id"],
                        text=self.responses[rule["response_key"]],
                        matched_terms=[hit.group(0)],
                    )

            return GuardResult(blocked=False, text=f"{text.strip()}\n\n{self.disclaimer}")

        except Exception:
            return GuardResult(
                blocked=True,
                category="guard_error",
                rule_id="ERR-00",
                text=self.responses["unavailable"],
            )

    @staticmethod
    def _is_degenerate(text: str, window: int = 6, repeats: int = 3) -> bool:
        """Detect looping output — the characteristic failure of aggressively
        quantized small models. Flags any n-gram repeated `repeats` times."""
        words = text.split()
        if len(words) < window * repeats:
            return False
        seen: dict[str, int] = {}
        for i in range(len(words) - window + 1):
            gram = " ".join(words[i : i + window])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= repeats:
                return True
        return False

    # ------------------------------------------------------------ pipeline

    def process(self, user_text: str, generate_fn) -> tuple[str, list[dict[str, Any]]]:
        """Full guarded turn. `generate_fn` is only called if input passes.
        Returns the text to display and the intervention log records."""
        log: list[dict[str, Any]] = []

        pre = self.check_input(user_text)
        log.append({"stage": "input", **pre.to_log_record()})
        if pre.blocked:
            return pre.text, log

        raw = generate_fn(user_text)

        post = self.check_output(raw)
        log.append({"stage": "output", **post.to_log_record()})
        return post.text, log
