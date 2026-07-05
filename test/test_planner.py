# tests/test_planner.py
#
# Tests for planner.py — intent routing, search query building, context assembly.
#
# No real LLM is called: we inject a fake_llm fixture.
# No network calls are made.

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# needs_web_search — keyword heuristic
# ─────────────────────────────────────────────────────────────────────────────

class TestNeedsWebSearch:

    def test_deadline_keyword_triggers_web(self):
        from web_search_tool import needs_web_search
        assert needs_web_search("What is the application deadline for winter 2026?") is True

    def test_blocked_account_triggers_web(self):
        from web_search_tool import needs_web_search
        assert needs_web_search("How much is the blocked account amount?") is True

    def test_year_2026_triggers_web(self):
        from web_search_tool import needs_web_search
        assert needs_web_search("What are the visa requirements for 2026?") is True

    def test_general_question_does_not_trigger(self):
        from web_search_tool import needs_web_search
        assert needs_web_search("What documents do I need for a student visa?") is False

    def test_studienkolleg_question_does_not_trigger(self):
        from web_search_tool import needs_web_search
        assert needs_web_search("What is a Studienkolleg?") is False

    def test_case_insensitive(self):
        from web_search_tool import needs_web_search
        assert needs_web_search("What is the DEADLINE for applications?") is True

    def test_fee_keyword_triggers_web(self):
        from web_search_tool import needs_web_search
        assert needs_web_search("How much is the visa fee?") is True


# ─────────────────────────────────────────────────────────────────────────────
# classify_intent — LLM-based routing
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyIntent:

    def test_rag_only_intent(self, fake_llm):
        from planner import classify_intent
        # fake_llm returns "rag_only" for intent prompts
        intent = classify_intent("What documents do I need for a student visa?", fake_llm)
        assert intent == "rag_only"

    def test_web_search_bypasses_llm_for_deadline(self, fake_llm):
        """Keyword match → intent returned immediately without LLM call."""
        from planner import classify_intent
        calls = []
        def counting_llm(prompt):
            calls.append(prompt)
            return "rag_only"
        intent = classify_intent("What is the application deadline for 2026?", counting_llm)
        assert intent == "web_search"
        assert len(calls) == 0   # LLM was NOT called

    def test_university_recommender_from_llm(self, fake_llm):
        from planner import classify_intent
        # Override fake_llm to return 'university_recommender'
        def rec_llm(prompt): return "university_recommender"
        intent = classify_intent("Which universities are good for Computer Science?", rec_llm)
        assert intent == "university_recommender"

    def test_unknown_llm_output_defaults_to_rag_only(self, fake_llm):
        from planner import classify_intent
        def bad_llm(prompt): return "I don't understand the question"
        intent = classify_intent("Something unclear", bad_llm)
        assert intent == "rag_only"

    def test_llm_failure_defaults_to_rag_only(self, fake_llm):
        from planner import classify_intent
        def failing_llm(prompt): raise RuntimeError("LLM unavailable")
        intent = classify_intent("Some question", failing_llm)
        assert intent == "rag_only"


# ─────────────────────────────────────────────────────────────────────────────
# build_search_query
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSearchQuery:

    def test_returns_string(self, fake_llm, sample_profile):
        from planner import build_search_query
        q = build_search_query("What is the blocked account amount?", sample_profile, fake_llm)
        assert isinstance(q, str)
        assert len(q) > 0

    def test_falls_back_to_raw_question_on_llm_failure(self, sample_profile):
        from planner import build_search_query
        def failing_llm(prompt): raise RuntimeError("LLM down")
        question = "How much is the blocked account?"
        q = build_search_query(question, sample_profile, failing_llm)
        assert q == question


# ─────────────────────────────────────────────────────────────────────────────
# execute_plan
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutePlan:

    def test_rag_only_plan_uses_no_tools(self, fake_llm, sample_profile):
        from planner import execute_plan
        result = execute_plan(
            question="What is a Studienkolleg?",
            profile=sample_profile,
            intent="rag_only",
            call_llm=fake_llm,
            recommender_fn=None,
        )
        assert result.intent == "rag_only"
        assert result.used_web_search is False
        assert result.used_recommender is False

    def test_recommender_intent_without_fn_degrades_to_rag(self, fake_llm, sample_profile):
        """If recommender_fn is None, intent degrades gracefully."""
        from planner import execute_plan
        result = execute_plan(
            question="Which university should I apply to?",
            profile=sample_profile,
            intent="university_recommender",
            call_llm=fake_llm,
            recommender_fn=None,
        )
        assert result.intent == "rag_only"
        assert result.used_recommender is False

    def test_recommender_fn_is_called(self, fake_llm, sample_profile):
        from planner import execute_plan
        called_with = []
        def fake_recommender(profile, question):
            called_with.append((profile, question))
            return {"universities": [{"name": "TU Munich", "city": "Munich"}]}

        result = execute_plan(
            question="Which university for CS?",
            profile=sample_profile,
            intent="university_recommender",
            call_llm=fake_llm,
            recommender_fn=fake_recommender,
        )
        assert result.used_recommender is True
        assert len(called_with) == 1

    def test_recommender_failure_degrades_to_rag(self, fake_llm, sample_profile):
        from planner import execute_plan
        def crashing_recommender(profile, question):
            raise RuntimeError("DB error")
        result = execute_plan(
            question="Which uni?",
            profile=sample_profile,
            intent="university_recommender",
            call_llm=fake_llm,
            recommender_fn=crashing_recommender,
        )
        assert result.intent == "rag_only"


# ─────────────────────────────────────────────────────────────────────────────
# build_augmented_context
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAugmentedContext:

    def _plan_result(self, **overrides):
        from planner import PlanResult
        p = PlanResult()
        for k, v in overrides.items():
            setattr(p, k, v)
        return p

    def test_rag_chunks_appear_in_context(self):
        from planner import build_augmented_context
        chunks = [
            {"text": "Visa requires blocked account.", "text_sent_to_llm": None},
            {"text": "APS certificate needed for Chinese students.", "text_sent_to_llm": None},
        ]
        ctx = build_augmented_context(chunks, self._plan_result())
        assert "Visa requires blocked account." in ctx
        assert "APS certificate" in ctx

    def test_numbered_citations_present(self):
        from planner import build_augmented_context
        chunks = [{"text": "chunk one", "text_sent_to_llm": None}]
        ctx = build_augmented_context(chunks, self._plan_result())
        assert "[1]" in ctx

    def test_web_context_injected_when_present(self):
        from planner import build_augmented_context
        plan = self._plan_result(
            used_web_search=True,
            web_context="Live data: blocked account is €11,208 for 2026/27.",
        )
        ctx = build_augmented_context([], plan)
        assert "€11,208" in ctx
        assert "LIVE WEB DATA" in ctx

    def test_university_recs_appear_when_present(self):
        from planner import build_augmented_context
        plan = self._plan_result(
            used_recommender=True,
            recommend_result={
                "universities": [
                    {
                        "name": "TU Munich", "city": "Munich",
                        "match_reason": "Strong CS faculty",
                        "match_score": 0.92,
                        "url": "https://tum.de",
                    }
                ]
            },
        )
        ctx = build_augmented_context([], plan)
        assert "TU Munich" in ctx
        assert "Munich" in ctx

    def test_empty_input_returns_fallback(self):
        from planner import build_augmented_context
        ctx = build_augmented_context([], self._plan_result())
        assert len(ctx) > 0   # should never return an empty string

    def test_text_sent_to_llm_preferred_over_text(self):
        from planner import build_augmented_context
        chunks = [{
            "text":            "original short chunk",
            "text_sent_to_llm": "expanded context with neighbours",
        }]
        ctx = build_augmented_context(chunks, self._plan_result())
        assert "expanded context" in ctx
        assert "original short chunk" not in ctx