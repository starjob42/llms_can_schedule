"""Presentation-only reflow of verified generated tables for Springer sn-jnl.

Call only after the scientific reporters have verified their own evidence.
This creates a derivative string; it never edits reporter sources or results.
"""
from __future__ import annotations

import re

EXTRA_PREAMBLE = r"\usepackage{array}"
_MARKER = "% JOURNAL_RESULT_TABLE_REFLOW_V1"
_TABLE = re.compile(r"\\begin\{table\}(?:\[[^\]]*\])?[\s\S]*?\\end\{table\}")
_LAYOUTS = {
    "tab:phi3_training_budget": ("lrrrr", (.32, .13, .23, .16, .16), 1),
    "tab:phi3_prompt_wording": ("lrrrr", (.32, .13, .23, .16, .16), 1),
    "tab:phi3_auto_prompt": ("lrrrr", (.32, .13, .23, .16, .16), 1),
    "tab:phi3_auto_prompt_validation": ("lrr", (.26, .25, .49), 1),
    "tab:phi3_supplementary_baselines": ("lrrrrr", (.26, .12, .16, .15, .15, .16), 1),
    "tab:phi3_baseline_paired_gaps": ("llrrrr", (.17, .24, .12, .16, .14, .17), 2),
}


def _payload(table: str) -> str:
    match = re.search(r"\\toprule([\s\S]*?)\\bottomrule", table)
    if match is None:
        raise ValueError("Expected one booktabs header/data region")
    return match.group(1)


def _columns(fractions, left_columns):
    # @{} removes outer padding; n columns retain 2*(n-1) tabcolsep gaps.
    gaps = 2 * (len(fractions) - 1)
    parts = []
    for index, fraction in enumerate(fractions):
        alignment = r"\raggedright" if index < left_columns else r"\centering"
        width = (rf"\dimexpr{fraction:.2f}\linewidth-"
                 rf"{fraction*gaps:.2f}\tabcolsep\relax")
        parts.append(r">{" + alignment + r"\arraybackslash}p{" + width + "}")
    return "@{}" + "".join(parts) + "@{}"


def reflow_result_tables(text: str) -> str:
    """Wrap known result columns without changing any header, row or value.

    Unknown scaled tables and unexpected generated formats raise ValueError;
    a newly added scientific table needs an explicit reviewed layout.
    """
    def transform(match):
        table = match.group()
        labels = re.findall(r"\\label\{([^}]+)\}", table)
        if len(labels) != 1 or labels[0] not in _LAYOUTS:
            if r"\resizebox" in table:
                raise ValueError("No reviewed Springer layout for scaled table: " + repr(labels))
            return table
        if _MARKER in table:
            if r"\resizebox" in table:
                raise ValueError("Reflow marker cannot coexist with resizebox")
            return table
        specification, fractions, left_columns = _LAYOUTS[labels[0]]
        opener = r"\begin{tabular}{" + specification + "}"
        if table.count(opener) != 1 or table.count(r"\end{tabular}") != 1:
            raise ValueError("Unexpected generated tabular shape: " + labels[0])
        before = _payload(table)
        scaled = r"\resizebox{\textwidth}{!}{"
        if r"\resizebox" in table:
            if table.count(scaled) != 1:
                raise ValueError("Unexpected scaling wrapper: " + labels[0])
            pattern = re.escape(scaled) + r"\s*" + re.escape(opener)
            replacement = (_MARKER + "\n" + r"\setlength{\tabcolsep}{3pt}"
                           + r"\renewcommand{\arraystretch}{1.15}" + "\n"
                           + r"\begin{tabular}{" + _columns(fractions, left_columns) + "}")
            table, count = re.subn(pattern, lambda _: replacement, table)
            if count != 1:
                raise ValueError("Scaling wrapper is not adjacent to tabular: " + labels[0])
            table, count = re.subn(re.escape(r"\end{tabular}") + r"\s*\}",
                                   lambda _: r"\end{tabular}", table)
            if count != 1:
                raise ValueError("Cannot close scaling wrapper safely: " + labels[0])
        else:
            replacement = (_MARKER + "\n" + r"\setlength{\tabcolsep}{3pt}"
                           + r"\renewcommand{\arraystretch}{1.15}" + "\n"
                           + r"\begin{tabular}{" + _columns(fractions, left_columns) + "}")
            table = table.replace(opener, replacement, 1)
        if _payload(table) != before:
            raise AssertionError("Presentation transformation changed result cells")
        return table
    return _TABLE.sub(transform, text)
