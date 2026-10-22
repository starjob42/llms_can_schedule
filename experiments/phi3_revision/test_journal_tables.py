import unittest

from .journal_tables import reflow_result_tables


def fixture(label="tab:phi3_training_budget", specification="lrrrr", scaled=True):
    data = r"Updates & Feasible & Feasibility [95\% CI] & Mean reference gap & Median reference gap \\ \midrule" + "\n"
    data += r"1844 (100\%, 100.00 h) & 216/216 & 100.00\% [98.25\%, 100.00\%] & -1.25\% & -- \\" + "\n"
    inner = r"\begin{tabular}{"+specification+r"}\toprule"+data+r"\bottomrule\end{tabular}"
    if scaled:
        inner = r"\resizebox{\textwidth}{!}{"+inner+"}"
    return r"\begin{table}[!ht]\caption{SYNTHETIC ENGINEERING ONLY}\label{"+label+"}"+inner+r"\end{table}", data


class JournalTablesTest(unittest.TestCase):
    def test_preserves_every_cell_caption_label_and_outside_text(self):
        text, data = fixture()
        result = reflow_result_tables("before\n"+text+"\nafter")
        self.assertIn(data, result)
        self.assertIn(r"\caption{SYNTHETIC ENGINEERING ONLY}\label{tab:phi3_training_budget}", result)
        self.assertTrue(result.startswith("before\n"))
        self.assertTrue(result.endswith("\nafter"))
        self.assertNotIn(r"\resizebox", result)
        self.assertIn(r"\dimexpr0.32\linewidth-2.56\tabcolsep\relax", result)
        self.assertEqual(result, reflow_result_tables(result))

    def test_unscaled_auto_validation_table_is_supported(self):
        text, data = fixture("tab:phi3_auto_prompt_validation", "lrr", False)
        result = reflow_result_tables(text)
        self.assertIn(data, result)
        self.assertIn(r"\dimexpr0.49\linewidth-1.96\tabcolsep\relax", result)

    def test_unknown_scaled_layout_and_changed_shape_fail_closed(self):
        for label, shape in [("tab:new_unreviewed", "lrrrr"),
                             ("tab:phi3_training_budget", "lrr")]:
            with self.subTest(label=label, shape=shape):
                with self.assertRaises(ValueError):
                    reflow_result_tables(fixture(label, shape)[0])

    def test_unrelated_unscaled_table_is_unchanged(self):
        text, _ = fixture("tab:other", "lrrrr", False)
        self.assertEqual(text, reflow_result_tables(text))


if __name__ == "__main__":
    unittest.main()
