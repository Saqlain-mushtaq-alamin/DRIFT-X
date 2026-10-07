"""Unit tests for SignalNormalizer.

These tests verify the three documented behaviours of SignalNormalizer:

1. During warmup (fewer than ``warmup_windows`` calls) the raw values are
   returned unchanged (passthrough).
2. For a channel that shows no variation (all values identical) the normalised
   value is 0.5 (the neutral mid-point), NOT 0.0.
3. After warmup, values are min-max scaled to [0, 1] using the *full* observed
   history.
4. reset() clears the accumulated history, reverting the normaliser to warmup
   mode.

The docs previously stated that warmup returns 0.0 and constant channels
return 0.0.  Both claims were wrong.  These tests reflect the *actual* code
behaviour (raw passthrough in warmup; 0.5 for constant channels).
"""
import pytest
import numpy as np

from driftx.fusion.normalizer import SignalNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_n(norm: SignalNormalizer, n: int, stat=0.5, mag=0.5, rank=0.5):
    """Call normalize() n times with constant values (fills warmup history)."""
    for _ in range(n):
        norm.normalize(stat=stat, mag=mag, rank=rank)


# ---------------------------------------------------------------------------
# Warmup behaviour
# ---------------------------------------------------------------------------

class TestWarmupPassthrough:
    """During warmup the raw signal values must be returned unchanged."""

    def test_warmup_returns_raw_stat(self):
        norm = SignalNormalizer(warmup_windows=3)
        result = norm.normalize(stat=0.123, mag=0.0, rank=0.0)
        assert result["stat"] == pytest.approx(0.123), (
            "During warmup, stat must be returned as-is (passthrough)."
        )

    def test_warmup_returns_raw_mag(self):
        norm = SignalNormalizer(warmup_windows=3)
        result = norm.normalize(stat=0.0, mag=0.456, rank=0.0)
        assert result["magnitude"] == pytest.approx(0.456)

    def test_warmup_returns_raw_rank(self):
        norm = SignalNormalizer(warmup_windows=3)
        result = norm.normalize(stat=0.0, mag=0.0, rank=0.789)
        assert result["rank_change"] == pytest.approx(0.789)

    def test_warmup_exactly_at_boundary(self):
        """The boundary call (len == warmup_windows) must still passthrough."""
        warmup = 2
        norm = SignalNormalizer(warmup_windows=warmup)
        # First call: len(history)==1, still warmup
        r1 = norm.normalize(stat=0.1, mag=0.2, rank=0.3)
        assert r1["stat"] == pytest.approx(0.1)
        # Second call: len(history)==2, == warmup_windows, still warmup
        r2 = norm.normalize(stat=0.9, mag=0.8, rank=0.7)
        assert r2["stat"] == pytest.approx(0.9)

    def test_warmup_never_returns_zero_for_nonzero_input(self):
        """Warmup must NOT return 0.0 for a non-zero raw value (old doc bug)."""
        norm = SignalNormalizer(warmup_windows=5)
        result = norm.normalize(stat=0.35, mag=0.12, rank=0.08)
        assert result["stat"] != 0.0, (
            "Warmup must pass raw values through, not return 0.0."
        )
        assert result["magnitude"] != 0.0
        assert result["rank_change"] != 0.0


# ---------------------------------------------------------------------------
# Post-warmup min-max scaling
# ---------------------------------------------------------------------------

class TestMinMaxScaling:
    """After warmup the normaliser must min-max scale each channel."""

    def test_current_equals_max_returns_one(self):
        norm = SignalNormalizer(warmup_windows=2)
        # Push two warmup values through (raw passthrough, not checked)
        norm.normalize(stat=0.0, mag=0.0, rank=0.0)
        norm.normalize(stat=0.5, mag=0.5, rank=0.5)
        # Third call (post-warmup): current is the max observed so far
        result = norm.normalize(stat=1.0, mag=1.0, rank=1.0)
        assert result["stat"] == pytest.approx(1.0), (
            "When current value equals the max, normalised output must be 1.0."
        )

    def test_current_equals_min_returns_zero(self):
        norm = SignalNormalizer(warmup_windows=2)
        norm.normalize(stat=0.5, mag=0.5, rank=0.5)
        norm.normalize(stat=1.0, mag=1.0, rank=1.0)
        # Third call: current is the min observed (0.0 < 0.5)
        result = norm.normalize(stat=0.0, mag=0.0, rank=0.0)
        assert result["stat"] == pytest.approx(0.0)

    def test_scaling_uses_full_history(self):
        """The normaliser uses the *full* observed range, not just recent windows."""
        norm = SignalNormalizer(warmup_windows=2)
        # Warmup: push in min=0.0 and max=1.0
        norm.normalize(stat=0.0, mag=0.0, rank=0.0)
        norm.normalize(stat=1.0, mag=1.0, rank=1.0)
        # Post-warmup: midpoint should normalise to ~0.5
        result = norm.normalize(stat=0.5, mag=0.5, rank=0.5)
        assert result["stat"] == pytest.approx(0.5, abs=1e-6)

    def test_output_clipped_to_unit_interval(self):
        """Clipping must ensure output stays in [0, 1] even for out-of-range vals."""
        norm = SignalNormalizer(warmup_windows=2)
        norm.normalize(stat=0.2, mag=0.2, rank=0.2)
        norm.normalize(stat=0.8, mag=0.8, rank=0.8)
        # Value well above the observed max
        result = norm.normalize(stat=5.0, mag=5.0, rank=5.0)
        assert 0.0 <= result["stat"] <= 1.0
        assert 0.0 <= result["magnitude"] <= 1.0
        assert 0.0 <= result["rank_change"] <= 1.0

    def test_channels_are_independent(self):
        """Each signal channel is normalised independently."""
        norm = SignalNormalizer(warmup_windows=2)
        # stat range: [0, 1], mag range: [0, 0.1], rank range: [0, 0.5]
        norm.normalize(stat=0.0, mag=0.0, rank=0.0)
        norm.normalize(stat=1.0, mag=0.1, rank=0.5)
        result = norm.normalize(stat=0.5, mag=0.05, rank=0.25)
        # Each channel independently normalised to ~0.5
        assert result["stat"] == pytest.approx(0.5, abs=1e-5)
        assert result["magnitude"] == pytest.approx(0.5, abs=1e-5)
        assert result["rank_change"] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Constant-channel behaviour
# ---------------------------------------------------------------------------

class TestConstantChannel:
    """A channel where all observed values are identical must return 0.5.

    The actual code returns 0.5 (neutral mid-point) for constant channels.
    An earlier version of the documentation incorrectly stated it returns 0.0.
    """

    def test_constant_channel_returns_half(self):
        norm = SignalNormalizer(warmup_windows=2)
        # Push warmup through with constant value
        norm.normalize(stat=0.3, mag=0.3, rank=0.3)
        norm.normalize(stat=0.3, mag=0.3, rank=0.3)
        # Post-warmup: all history values are identical
        result = norm.normalize(stat=0.3, mag=0.3, rank=0.3)
        assert result["stat"] == pytest.approx(0.5), (
            "Constant channel must return 0.5, not 0.0 (see normalizer.py:_normalize)."
        )
        assert result["magnitude"] == pytest.approx(0.5)
        assert result["rank_change"] == pytest.approx(0.5)

    def test_constant_channel_not_zero(self):
        """Explicitly guard against the old doc bug: constant must NOT return 0."""
        norm = SignalNormalizer(warmup_windows=1)
        norm.normalize(stat=0.0, mag=0.0, rank=0.0)  # warmup
        result = norm.normalize(stat=0.0, mag=0.0, rank=0.0)
        # Both entries in history are 0.0 — all-zero constant channel
        assert result["stat"] == pytest.approx(0.5), (
            "Constant all-zero channel must return 0.5, not 0.0."
        )


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

class TestReset:
    """reset() must clear all accumulated history, reverting to warmup mode."""

    def test_reset_clears_history(self):
        norm = SignalNormalizer(warmup_windows=2)
        # Fill past warmup
        _call_n(norm, 5, stat=0.5, mag=0.5, rank=0.5)
        # Reset
        norm.reset()
        assert norm.history["stat"] == [], "After reset, stat history must be empty."
        assert norm.history["magnitude"] == [], "After reset, magnitude history must be empty."
        assert norm.history["rank_change"] == [], "After reset, rank_change history must be empty."

    def test_reset_restores_warmup_passthrough(self):
        """After reset, the first call(s) must again use raw passthrough."""
        norm = SignalNormalizer(warmup_windows=2)
        _call_n(norm, 5, stat=0.0, mag=0.0, rank=0.0)  # fill well past warmup
        norm.reset()
        # First call after reset \u2014 must passthrough (warmup active again)
        result = norm.normalize(stat=0.77, mag=0.33, rank=0.44)
        assert result["stat"] == pytest.approx(0.77), (
            "After reset, first call must use warmup passthrough."
        )

    def test_reset_then_refill_normalises_correctly(self):
        """After reset and re-filling, normalisation works as expected."""
        norm = SignalNormalizer(warmup_windows=2)
        _call_n(norm, 5, stat=99.0, mag=99.0, rank=99.0)  # arbitrary fill
        norm.reset()
        # Re-fill with known values
        norm.normalize(stat=0.0, mag=0.0, rank=0.0)
        norm.normalize(stat=1.0, mag=1.0, rank=1.0)
        result = norm.normalize(stat=0.5, mag=0.5, rank=0.5)
        assert result["stat"] == pytest.approx(0.5, abs=1e-5), (
            "After reset + refill, normalisation must use new history only."
        )


# ---------------------------------------------------------------------------
# Warmup size edge cases
# ---------------------------------------------------------------------------

class TestWarmupEdgeCases:
    """Edge cases around warmup_windows configuration."""

    def test_warmup_zero_activates_immediately(self):
        """warmup_windows=0 should activate scaling from the very first call."""
        norm = SignalNormalizer(warmup_windows=0)
        # With warmup=0, the first call adds to history (len=1 > 0) and
        # triggers the normalisation path.  The single observed value
        # creates a constant channel, so output should be 0.5.
        result = norm.normalize(stat=0.42, mag=0.42, rank=0.42)
        assert result["stat"] == pytest.approx(0.5), (
            "With warmup=0 the first call should normalise (constant-channel -> 0.5)."
        )

    def test_warmup_one_boundary(self):
        """warmup_windows=1: first call is warmup, second enters normalisation."""
        norm = SignalNormalizer(warmup_windows=1)
        r1 = norm.normalize(stat=0.2, mag=0.2, rank=0.2)
        # First call: len(history)==1 == warmup_windows, still warmup
        assert r1["stat"] == pytest.approx(0.2), "First call (len==warmup) must passthrough."
        r2 = norm.normalize(stat=0.8, mag=0.8, rank=0.8)
        # Second call: len(history)==2 > warmup_windows, triggers scaling
        # History=[0.2, 0.8], current=0.8 -> normalised = (0.8-0.2)/(0.8-0.2) = 1.0
        assert r2["stat"] == pytest.approx(1.0), (
            "Second call with warmup=1 should normalise to 1.0 (current=max)."
        )
