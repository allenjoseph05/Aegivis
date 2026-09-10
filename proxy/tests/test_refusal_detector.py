"""Tests for proxy/app/security/refusal_detector.py."""
import pytest
from proxy.app.security.refusal_detector import score_refusal, RefusalResult, _DETECTION_THRESHOLD


# ─── Helper ────────────────────────────────────────────────────────────────────

def _score(text, tokens=None, finish=None):
    return score_refusal(text, token_count=tokens, finish_reason=finish)


# ─── Strong refusal phrases ────────────────────────────────────────────────────

class TestStrongRefusalPhrases:
    def test_cannot_help(self):
        r = _score("I cannot help with that request.")
        assert r.detected is True
        assert r.strong_match is True
        assert r.score >= 0.55

    def test_wont_assist(self):
        r = _score("I won't assist with creating that.")
        assert r.detected is True
        assert r.strong_match is True

    def test_unable_to_assist(self):
        r = _score("I'm unable to assist with this task.")
        assert r.detected is True
        assert r.strong_match is True

    def test_must_decline(self):
        r = _score("I must decline this request as it goes against my guidelines.")
        assert r.detected is True
        assert r.strong_match is True

    def test_will_not_provide(self):
        r = _score("I will not provide instructions for that.")
        assert r.detected is True
        assert r.strong_match is True

    def test_not_something_i_can(self):
        r = _score("That's not something I can assist you with.")
        assert r.detected is True


# ─── Soft refusal phrases ──────────────────────────────────────────────────────

class TestSoftRefusalPhrases:
    def test_against_guidelines(self):
        r = _score("This request goes against my guidelines. I'd suggest a different approach.")
        assert r.score > 0.0

    def test_sorry_but(self):
        r = _score("I'm sorry, but I cannot fulfill this particular request.")
        assert r.detected is True

    def test_ethical_concern(self):
        # Two soft phrases: "I'm sorry, but" + "designed to avoid" → score >= 0.60
        r = _score(
            "I'm sorry, but I'm designed to avoid generating this type of content. "
            "There are serious ethical concerns and potential harm to others involved."
        )
        assert r.detected is True

    def test_designed_to_avoid(self):
        r = _score("I'm designed to avoid generating harmful content.")
        assert r.score > 0.0

    def test_potential_harm(self):
        r = _score(
            "I understand your request, but there is potential harm associated "
            "with providing this information."
        )
        assert r.score >= 0.30


# ─── Non-refusal responses ─────────────────────────────────────────────────────

class TestNonRefusals:
    def test_normal_answer(self):
        r = _score(
            "Sure! Here is how you can implement a binary search tree in Python. "
            "First, define a Node class with left and right children..."
        )
        assert r.detected is False
        assert r.score < _DETECTION_THRESHOLD

    def test_empty_text(self):
        r = _score("")
        assert r.detected is False
        assert r.score == 0.0

    def test_whitespace_only(self):
        r = _score("   \n\t  ")
        assert r.detected is False

    def test_benign_cannot(self):
        # "cannot" in a non-refusal context should not trigger
        r = _score(
            "The system cannot connect to the database because the port is closed. "
            "Please check your firewall rules and try again."
        )
        assert r.detected is False

    def test_greeting(self):
        r = _score("Hello! How can I help you today?")
        assert r.detected is False


# ─── Finish-reason suppression ──────────────────────────────────────────────────

class TestFinishReasonSuppression:
    def test_tool_calls_not_refusal(self):
        r = _score(
            "I cannot do that.",
            finish="tool_calls",
        )
        # finish_reason = 'tool_calls' → not a refusal (agent is using a tool)
        assert r.detected is False

    def test_stop_is_not_suppressed(self):
        r = _score("I cannot help with that.", finish="stop")
        assert r.detected is True

    def test_end_turn_not_suppressed(self):
        r = _score("I'm unable to assist with this.", finish="end_turn")
        assert r.detected is True


# ─── Short-response boost ───────────────────────────────────────────────────────

class TestShortResponseBoost:
    def test_short_refusal_gets_boost(self):
        r_short = _score("I cannot help with that.", tokens=15)
        r_long  = _score("I cannot help with that.", tokens=300)
        assert r_short.score > r_long.score

    def test_border_token_threshold(self):
        r = _score("I must decline this request.", tokens=79)
        assert r.detected is True


# ─── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_refusal_result(self):
        r = _score("I cannot help.")
        assert isinstance(r, RefusalResult)

    def test_to_dict_keys(self):
        d = _score("I'm unable to assist.").to_dict()
        assert "detected" in d
        assert "score" in d
        assert "strong_match" in d
        assert "matched_phrases" in d

    def test_score_capped_at_one(self):
        # many phrases stacked — score must never exceed 1.0
        text = (
            "I cannot help. I won't assist. I must decline. "
            "This is against my guidelines. I'm sorry but I'm unable to. "
            "There are serious ethical concerns with potential harm to others."
        )
        r = _score(text, tokens=20)
        assert r.score <= 1.0

    def test_matched_phrases_capped_at_five(self):
        text = (
            "I cannot help. I won't assist. I must decline. "
            "This is against my guidelines. I'm sorry but unable. "
            "Ethical concern for potential harm."
        )
        r = _score(text)
        assert len(r.to_dict()["matched_phrases"]) <= 5
