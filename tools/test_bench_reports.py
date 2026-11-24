"""tools/bench_reports.py: the privacy scan, the table per card and the outlier flags; and, where the engine's
poly_bench_report_test is built (BENCH_REPORT_TEST_BIN), the report it writes passes the scan."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_reports as br  # noqa: E402


def report(card="NVIDIA GeForce RTX 2070 SUPER", prompt2k=650.0, decode=20.0, h2d=12.0, model="Inkling 2 Mini"):
    runs = [{"name": "prompt 302", "prompt_tok_s": 300.0, "decode_tok_s": decode},
            {"name": "prompt 2048", "prompt_tok_s": prompt2k, "decode_tok_s": decode * 0.95},
            {"name": "prompt 8192", "prompt_tok_s": prompt2k * 1.3, "decode_tok_s": decode * 0.9}]
    return {"kind": "strata-bench-report", "schema": 1,
            "model": {"name": model, "expert_types": "iq4_nl/iq4_nl"},
            "hardware": {"gpu": {"name": card, "pcie_gen": 3, "pcie_width": 16}, "cpu": {"model": "AMD Ryzen 5 3600XT"}},
            "probes": {"h2d_pinned_gbs": h2d, "ram_read_gbs": 38.0, "gpu_mem_gbs": 400.0,
                       "cpu_dot": [{"gbs_all_threads": 30.0}]},
            "cached_experts": 982, "total_experts": 3072, "runs": runs, "long": []}


class Privacy(unittest.TestCase):
    def test_clean_text_passes(self):
        text = (json.dumps(report()) + " tok/s GB/s gen3 x16 Linux 6.8.0-149-generic driver 580.178.04 sse4.2 q8_0/iq4_nl"
                " Linux 6.6.87.2-microsoft-standard-WSL2")
        self.assertEqual(br.privacy_problems(text), [])

    def test_each_kind_of_private_detail_is_found(self):
        for text in ["/home/alice/models/m.gguf", "C:\\Users\\bob\\m.gguf", "{\"p\": \"D:\\\\models\"}",
                     "\\\\server\\share", "me@example.com", "GPU-1a2b3c4d-1111-2222-3333-444455556666",
                     "00:1A:2b:3C:4d:5E", "192.168.11.93", "0000:01:00.0", "sk-abcdefgh1234", "~/models",
                     "/work/models/m.gguf", "/workspace/x", "/Volumes/disk/m"]:
            self.assertTrue(br.privacy_problems(text), text)

    def test_private_words_only_whole(self):
        self.assertTrue(br.privacy_problems("ran by alice", ["alice"]))
        self.assertEqual(br.privacy_problems("malice and alice_2", ["alice"]), [])

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d, "good.json"), Path(d, "bad.json")
            good.write_text(json.dumps(report()), encoding="utf-8")
            r = report()
            r["settings"] = {"args": [["--expert-profile", "/home/alice/p.txt"]]}
            bad.write_text(json.dumps(r), encoding="utf-8")
            self.assertEqual(br.main(["--check-privacy", str(good)]), 0)
            self.assertEqual(br.main(["--check-privacy", str(bad)]), 1)
            self.assertEqual(br.main(["--check-privacy", "--private", "Ryzen", str(good)]), 1)


class Table(unittest.TestCase):
    def test_outliers_against_the_same_cards_median(self):
        rows = []
        for i, (p, h2d) in enumerate([(650, 12), (640, 12.2), (660, 11.9), (300, 3.0)]):
            f = br.figures(report(prompt2k=p, h2d=h2d))
            f["source"] = f"r{i}"
            rows.append(f)
        other = br.figures(report(card="NVIDIA GeForce RTX 3060", prompt2k=300))
        other["source"] = "r3060"
        rows.append(other)
        br.flag_outliers(rows)
        self.assertEqual([r["source"] for r in rows if r["flags"]], ["r3"])
        self.assertTrue(any("prompt_2048" in f for f in rows[3]["flags"]))
        self.assertTrue(any("h2d_pinned_gbs" in f for f in rows[3]["flags"]))
        md = br.table(rows)
        self.assertIn("### NVIDIA GeForce RTX 2070 SUPER (4 reports)", md)
        self.assertIn("### NVIDIA GeForce RTX 3060 (1 report)", md)
        self.assertIn("Far from the same card's other reports", md)

    def test_another_model_is_no_peer_for_speeds(self):
        a = br.figures(report(prompt2k=650))
        b = br.figures(report(prompt2k=651))
        c = br.figures(report(prompt2k=200, model="Qwen3.6"))
        for i, f in enumerate((a, b, c)):
            f["source"] = str(i)
        br.flag_outliers([a, b, c])
        self.assertEqual(c["flags"], [])

    def test_bad_fields_and_markdown(self):
        r = report(card="Evil | <img src=x> # card\n![x](y)")
        r["probes"] = {"h2d_pinned_gbs": "fast", "cpu_dot": "nope"}
        r["runs"] = [{"name": "prompt 2048", "prompt_tok_s": [1]}, "junk"]
        f = br.figures(r)
        self.assertIsNone(f["h2d_pinned_gbs"])
        self.assertIsNone(f["prompt_2048"])
        f["source"] = "x"
        br.flag_outliers([f])
        t = br.table([f])
        self.assertNotIn("<img", t)
        self.assertNotIn("\n![", t)
        self.assertIn("Evil \\| &lt;img", t)

    def test_one_bad_file_leaves_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d, "good.json"), Path(d, "bad.json")
            good.write_text(json.dumps(report()), encoding="utf-8")
            bad.write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(br.main([str(bad), str(good)]), 0)

    def test_load_refuses_other_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "x.json")
            p.write_text("{\"kind\": \"other\"}", encoding="utf-8")
            with self.assertRaises(ValueError):
                br.load(p)


@unittest.skipUnless(os.environ.get("BENCH_REPORT_TEST_BIN"), "set BENCH_REPORT_TEST_BIN to poly_bench_report_test")
class EngineReport(unittest.TestCase):
    def test_the_engines_report_passes_the_scan(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d, "report.json")
            env = dict(os.environ, BENCH_REPORT_TEST_DUMP=str(out))
            subprocess.run([os.environ["BENCH_REPORT_TEST_BIN"]], env=env, check=True, capture_output=True)
            text = out.read_text(encoding="utf-8")
            json.loads(text)
            self.assertEqual(br.privacy_problems(text, ["zq_tester_9", "zq-host-box"]), [])


if __name__ == "__main__":
    unittest.main()
