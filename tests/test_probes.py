"""Host-independent tests for the quick probes: acceptance, streams and determinism."""

import unittest
from unittest import mock

from sonar import acceptance as sweep, determinism, streams

METRICS = """# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 40.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 25.0
vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="m"} 5.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 9.0
vllm:spec_decode_num_drafts_created 1.7e9
"""


class AcceptanceSweepTest(unittest.TestCase):
    def test_counters_sum_engines_and_skip_other_series(self) -> None:
        self.assertEqual(sweep.counters(METRICS), {
            "vllm:spec_decode_num_drafts_total": 10.0,
            "vllm:spec_decode_num_draft_tokens_total": 40.0,
            "vllm:spec_decode_num_accepted_tokens_total": 30.0,
        })

    def test_summary_means_over_prompts(self) -> None:
        rows = [{"accepted_per_draft": 1.0, "tps": 80.0}, {"accepted_per_draft": 2.0, "tps": 100.0}]
        result = sweep.summary(rows)
        self.assertEqual(result["summary"], "acceptance sweep")
        self.assertEqual(result["prompts"], 2)
        self.assertEqual(result["accepted_per_draft_mean"], 1.5)
        self.assertEqual(result["tps_mean"], 90.0)
        self.assertGreater(result["accepted_ci95_pct"], 0)

    def test_uses_the_distinct_stream_prompts(self) -> None:
        self.assertEqual(len(sweep.PROMPTS), 16)
        self.assertEqual(determinism.PROMPTS, streams.PROMPTS[:8])

    def test_streams_summary(self) -> None:
        result = streams.summary(8, [100.0, 104.0])
        self.assertEqual((result["summary"], result["tps_mean"], result["samples"]), ("c8 distinct prompts", 102.0, 2))

    def test_determinism_digests_and_compare(self) -> None:
        a = {"tokens": ["x", "y"], "logprobs": [-0.1, -0.2], "text": "xy"}
        b = {"tokens": ["x", "z"], "logprobs": [-0.1, -0.3], "text": "xz"}
        self.assertEqual(determinism.compare(a, dict(a)), {"first_token_diff": None, "logprobs_equal": True,
                                                           "first_logprob_diff": None})
        self.assertEqual(determinism.compare(b, a)["first_token_diff"], 1)
        self.assertNotEqual(determinism.digest([a]), determinism.digest([b]))
        self.assertEqual(determinism.text_digest([a]), determinism.text_digest([{"tokens": None, "text": "xy"}]))
        self.assertEqual(determinism.long_prompt(3), determinism.long_prompt(3))


class StreamsForeignRequestsTest(unittest.TestCase):
    def test_requests_beyond_the_samples_own_are_foreign(self) -> None:
        """A sample counts the server's finished requests: its own streams are expected,
        anything more was someone else's load on the server (a production client reaching a
        test server once made a gate read 3-4x slow)."""
        def fake_stream(base, model, prompt, tokens, out, extra_body=None):
            out.update(start=0.0, first=1.0, last=3.0, tokens=tokens, offset=0.0)

        for counts, foreign in (([10.0, 14.0], 0), ([10.0, 21.0], 7), ([None, 14.0], None)):
            with mock.patch.object(streams, "stream", fake_stream), \
                    mock.patch.object(streams, "finished_requests", side_effect=counts):
                result = streams.sample("http://x", "m", streams.PROMPTS[:4], 16)
            self.assertEqual(result["foreign_requests"], foreign)

    def test_prompts_are_distinct(self) -> None:
        self.assertEqual(len(set(streams.PROMPTS)), len(streams.PROMPTS))
        self.assertGreaterEqual(len(streams.PROMPTS), 16)


if __name__ == "__main__":
    unittest.main()
