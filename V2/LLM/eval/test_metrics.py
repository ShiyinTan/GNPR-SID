import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import compute_metrics, extract_sid, sid_rank
from prompts import build_prompt


class ExtractSidTest(unittest.TestCase):
    def test_plain_sid(self):
        self.assertEqual(extract_sid("<a_57><b_31><c_9>"), ("<a_57>", "<b_31>", "<c_9>"))

    def test_collision_suffix_is_part_of_the_id(self):
        self.assertEqual(
            extract_sid("<a_60><b_51><c_27><d_0>"),
            ("<a_60>", "<b_51>", "<c_27>", "<d_0>"),
        )
        self.assertNotEqual(
            extract_sid("<a_60><b_51><c_27><d_0>"),
            extract_sid("<a_60><b_51><c_27>"),
        )

    def test_ignores_surrounding_text(self):
        text = "The next place is <a_50><b_15><c_62>. Thanks."
        self.assertEqual(extract_sid(text), ("<a_50>", "<b_15>", "<c_62>"))

    def test_stops_at_the_next_sid(self):
        text = "<a_1><b_2><c_3> then <a_4><b_5><c_6>"
        self.assertEqual(extract_sid(text), ("<a_1>", "<b_2>", "<c_3>"))


class MetricTest(unittest.TestCase):
    def test_hit_at_rank_two(self):
        golds = ["<a_1><b_2><c_3>"]
        predictions = [[
            "noise <a_9><b_9><c_9>",
            "<a_9><b_9><c_9>",
            "<a_1><b_2><c_3>",
        ]]
        self.assertEqual(sid_rank(golds[0], predictions[0]), 2)
        metrics = compute_metrics(golds, predictions)
        self.assertEqual(metrics["num_samples"], 1)
        self.assertEqual(metrics["Acc@1"], 0.0)
        self.assertEqual(metrics["Acc@5"], 1.0)
        self.assertEqual(metrics["Acc@10"], 1.0)
        self.assertEqual(metrics["MRR"], 0.5)
        self.assertAlmostEqual(metrics["NDCG@5"], 1.0 / math.log2(3))
        self.assertAlmostEqual(metrics["NDCG@10"], 1.0 / math.log2(3))

    def test_miss_is_zero(self):
        metrics = compute_metrics(
            ["<a_1><b_2><c_3>"],
            [["<a_8><b_8><c_8>"]],
        )
        self.assertEqual(metrics["Acc@1"], 0.0)
        self.assertEqual(metrics["Acc@10"], 0.0)
        self.assertEqual(metrics["MRR"], 0.0)
        self.assertEqual(metrics["NDCG@10"], 0.0)

    def test_skips_gold_without_sid(self):
        metrics = compute_metrics(["not a sid"], [["<a_1><b_2><c_3>"]])
        self.assertEqual(metrics["num_samples"], 0)
        self.assertEqual(metrics["num_skipped"], 1)


class PromptTest(unittest.TestCase):
    def test_qwen_prompt_matches_llamafactory_template(self):
        prompt = build_prompt("Do the task.", "history here", "qwen")
        self.assertIn(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n",
            prompt,
        )
        self.assertIn("<|im_start|>user\nDo the task.\nhistory here<|im_end|>\n<|im_start|>assistant\n", prompt)

    def test_llama3_prompt_has_no_system_block(self):
        prompt = build_prompt("task", "history", "llama3")
        self.assertTrue(prompt.startswith("<|begin_of_text|><|start_header_id|>user<|end_header_id|>"))
        self.assertIn("task\nhistory", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>\n\n", prompt)


if __name__ == "__main__":
    unittest.main()
