"""No-go evidence: count-weighted stacks cannot reconstruct wall time."""

import pytest


def speedscope_count_profile(stacks, rate):
    # The reviewed py-spy serializer uses these fields, not elapsed timestamps.
    return {
        "type": "sampled",
        "unit": "seconds",
        "startValue": 0,
        "endValue": len(stacks) / rate,
        "samples": stacks,
        "weights": [1 / rate] * len(stacks),
    }


@pytest.mark.parametrize("rate", [50, 100])
def test_same_profile_and_start_anchor_cannot_locate_sample_in_agent_span(rate):
    active_span = (0.0, 0.05)
    schedule_a = (0.0, 1 / rate)
    schedule_b = (0.0, 2.0)  # idle/failed samples removed before serialization
    stacks = [[0], [1, 0]]
    a = speedscope_count_profile(stacks, rate)
    b = speedscope_count_profile(stacks, rate)
    assert a == b
    assert schedule_a[0] == schedule_b[0]  # even an explicit start anchor is insufficient
    assert active_span[0] <= schedule_a[1] <= active_span[1]
    assert not active_span[0] <= schedule_b[1] <= active_span[1]
