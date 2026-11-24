"""terminal/tools/bench_text.py: the bench report's prompt text is the same every time, prose, and long enough."""
import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_text  # noqa: E402


class FakeTokenizer:
    def encode(self, text):
        return [len(w) for w in text.split()]


class BenchText(unittest.TestCase):
    def test_same_text_every_time(self):
        a, b = bench_text.words(3000), bench_text.words(3000)
        self.assertEqual(a, b)
        self.assertNotEqual(a, bench_text.words(3000, seed=2))
        self.assertGreaterEqual(len(a.split()), 3000)
        # the text of VERSION 2: a change to the lists or the generator must change VERSION too
        self.assertEqual(bench_text.VERSION, 2)
        self.assertEqual(hashlib.sha256(bench_text.words(500).encode()).hexdigest()[:12], "9597a732396f")

    def test_prose(self):
        t = bench_text.words(400)
        self.assertTrue(t[0].isupper())
        self.assertIn(". ", t)
        self.assertIn("\n\n", t)
        for bad in ("/", "\\", "@", "http"):
            self.assertNotIn(bad, t)

    def test_tokens_long_enough(self):
        ids = bench_text.tokens(FakeTokenizer(), 5000)
        self.assertGreaterEqual(len(ids), 5000)


if __name__ == "__main__":
    unittest.main()
