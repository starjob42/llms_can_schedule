"""Build the journal manuscript without changing any scientific run or protocol.

The pending build contains methods and an explicit results-pending notice.
The separate journal_publication bridge is responsible for accepting completed
scientific evidence; this module only compiles, inspects, and packages sources.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "LLMs_can_Schedule_paper"
ART = ROOT / "experiments/phi3_revision/artifacts/submission_guidelines"
AUTHORS = [
    {"given": "Henrik", "family": "Abgaryan", "email": "henrik.abgaryan@dauphine.eu"},
    {"given": "Ararat", "family": "Harutyunyan", "email": "ararat.harutyunyan@dauphine.eu"},
    {"given": "Tristan", "family": "Cazenave", "email": "tristan.cazenave@dauphine.eu"},
]
TITLE = "Large Language Models for Job Shop Scheduling"
SOURCES = [
    "manuscript.tex", "journal_authors.tex", "journal_abstract.tex",
    "journal_intro.tex", "journal_body.tex", "journal_results.tex", "journal_original_comparison.tex",
    "journal_conclusion.tex", "journal_limitations.tex",
    "journal_declarations.tex", "journal_title_acknowledgments.tex", "journal_status.tex", "main.bib",
    "sn-jnl.cls", "sn-mathphys-num.bst", "llms-sn-mathphys-num.bst",
    "historical_original_box_plot.png", "matched_gap_boxplot.pdf", "matched_gap_boxplot.png",
    "matched_gap_boxplot.svg", "matched_gap_plot.py", "training_budget_plot.pdf",
    "training_budget_plot.png", "training_budget_plot.svg", "training_budget_plot.py",
    "plot_requirements.txt",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def escape(value):
    return "".join({"&": r"\&", "%": r"\%", "_": r"\_",
                    "#": r"\#", "{": r"\{", "}": r"\}"}.get(c, c)
                   for c in value)


def author_files(paper=PAPER, final=False):
    metadata = json.loads((ART / "AUTHOR_DECLARATIONS.json").read_text())
    facts = metadata["author_metadata"]
    corresponding = facts.get("corresponding_email")
    if corresponding and corresponding not in {a["email"] for a in AUTHORS}:
        raise ValueError("Corresponding email is not one of the source authors")
    lines = []
    for author in AUTHORS:
        star = "*" if corresponding == author["email"] else ""
        lines += [r"\author" + star + "[1]{\\fnm{" + author["given"] +
                  "} \\sur{" + author["family"] + "}}",
                  r"\email{" + author["email"] + "}"]
    lines.append(r"\affil[1]{\orgdiv{LAMSADE, CNRS}, "
                 r"\orgname{Universit\'e Paris Dauphine--PSL}, "
                 r"\orgaddress{\city{Paris}, \country{France}}}")
    (paper / "journal_authors.tex").write_text("\n".join(lines) + "\n")
    funding = (ART / "confirmed_declarations.tex").read_text()
    funding_marker = "\\subsection*{Funding}\n"
    competing_marker = "\\subsection*{Competing interests}"
    if funding.count(funding_marker) != 1 or funding.count(competing_marker) != 1:
        raise ValueError("Confirmed support fragment has unexpected headings")
    title_support = funding.split(funding_marker, 1)[1].split(competing_marker, 1)[0].strip()
    if not title_support:
        raise ValueError("Confirmed title-page support statement is empty")
    (paper / "journal_title_acknowledgments.tex").write_text(title_support + "\n")
    declarations = r"\section*{Statements and Declarations}" + "\n" + funding
    declarations += r"""
\subsection*{Ethics and consent}
The study uses generated scheduling instances and does not involve human
participants or animals. Ethics approval and participant consent are not applicable.

\subsection*{Data and code availability}
The public scheduling dataset is available at
\url{https://github.com/starjob42/datasetjsp}.
The audited release contains 120,000 training-source instances and 1,800
held-out test instances. Frozen split definitions, prompts, source snapshots,
and reproduction instructions accompany this manuscript as Online Resource~1
(\texttt{ESM\_1.zip}).
"""
    declarations += ("The completed checkpoint and prompt evaluation records accompany that resource.\n" if final else
                     "The fixed-instruction checkpoint and baseline results are reported in the manuscript. Raw records for the completed update-922, update-1383, and update-1844 evaluations, together with the final instruction and refinement results, will be added to that resource after verification.\n")
    declarations += "The resource does not claim to reproduce an unrecovered historical evaluation cohort.\n"
    roles = facts.get("contribution_roles")
    if roles:
        if not isinstance(roles, str):
            raise ValueError("Contribution statement must be an author-supplied string")
        declarations += "\n\\subsection*{Author contributions}\n" + escape(roles) + "\n"
    (paper / "journal_declarations.tex").write_text(declarations)
    return {"corresponding_author_supplied": bool(corresponding),
            "contribution_roles_supplied": bool(roles),
            "names_order_affiliation_emails_source": "original manuscript author block",
            "author_declarations_sha256": digest(ART / "AUTHOR_DECLARATIONS.json")}


def prepare_pending(paper=PAPER):
    # Never erase completed measurements with a pending draft.
    if (paper / "build/JOURNAL_RESULT_INTEGRATION.json").exists():
        raise ValueError("Verified journal results exist; refusing pending reset")
    for stem in ("abstract", "intro", "conclusion"):
        shutil.copyfile(paper / f"journal_{stem}_pending.tex", paper / f"journal_{stem}.tex")
    (paper / "journal_results.tex").write_text(r"""\section{Results}
\label{sec:journal_results}
\subsection{Controlled Phi-3 training and prompt experiments}
\label{sec:phi3_revision}
The full training run and its checkpoint, instruction-wording, and automatic
refinement evaluations are in progress. Their measured results will be inserted
here after the complete output records and selection chronology have passed
verification. No final trained-model performance or improvement is asserted.
\label{sec:phi3_auto_prompt}
""")
    (paper / "journal_status.tex").write_text(r"""\begin{center}
\fbox{\parbox{0.94\textwidth}{\textbf{Experimental results pending.}
The methods and reporting protocol are prepared; the controlled training and
evaluation study is still running. This draft makes no final performance claim.}}
\end{center}
""")
    return author_files(paper)


def native_compile(paper, build, host="ourasi", documents=("manuscript", "rebuttal")):
    paper, build = Path(paper).resolve(), Path(build).resolve()
    build.mkdir(parents=True, exist_ok=True)
    if not documents or any(name not in {"manuscript", "rebuttal"} for name in documents):
        raise ValueError("Unrecognized compilation target")
    payload = {"paper": str(paper), "build": str(build), "documents": list(documents)}
    script = "import json,os,pathlib,subprocess,sys\np=json.loads(" + repr(json.dumps(payload)) + ")\n"
    script += """
paper=pathlib.Path(p['paper']);build=pathlib.Path(p['build']);build.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,BIBINPUTS=str(paper)+os.pathsep,BSTINPUTS=str(paper)+os.pathsep)
stages=[]
for name in p['documents']:
    sequence=[('pdflatex1',['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory='+str(build),name+'.tex'],paper)]
    if name=='manuscript':sequence.append(('bibtex',['bibtex',name],build))
    sequence += [(stage,['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory='+str(build),name+'.tex'],paper) for stage in ['pdflatex2','pdflatex3']]
    for stage,cmd,cwd in sequence:
        r=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=600)
        (build/(name+'.'+stage+'.stdout.txt')).write_text(r.stdout+r.stderr)
        stages.append(dict(document=name,stage=stage,returncode=r.returncode))
        (build/'native_compile.json').write_text(json.dumps(stages,indent=2)+'\\n')
        if r.returncode:
            print(json.dumps(dict(failed=stages[-1],tail=(r.stdout+r.stderr)[-4500:])),flush=True)
            sys.exit(r.returncode)
print(json.dumps(dict(status='compiled',stages=stages)),flush=True)
"""
    if host not in {"ourasi", "kaisertrot", "localhost"}:
        raise ValueError("Unrecognized compilation host")
    command = ["python3", "-"] if host == "localhost" else [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o",
        "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no", "-o",
        "ControlMaster=no", "-o", "ControlPath=none", host, "python3", "-"]
    result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=1800)
    (build / "remote_compile.stdout.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Native compile failed: " + (result.stdout + result.stderr)[-4500:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def expanded_source(paper, name, active=None):
    active = set() if active is None else active
    path = paper / name
    if path in active:
        raise ValueError("Recursive manuscript input")
    text = path.read_text()
    active = active | {path}
    return re.sub(r"\\input\{([^}]+)\}",
                  lambda m: expanded_source(paper, m[1] if m[1].endswith(".tex") else m[1]+".tex", active), text)


def inspect(paper, build):
    from pypdf import PdfReader
    checks = {}
    for name in ("manuscript", "rebuttal"):
        log = (build / (name + ".log")).read_text(errors="replace")
        prohibited = [r"undefined references", r"undefined citations", r"Reference .* undefined",
                      r"Citation .* undefined", r"Missing character", r"Overfull \\hbox", r"Overfull \\vbox"]
        bad = [pattern for pattern in prohibited if re.search(pattern, log, re.I)]
        if bad:
            raise ValueError(name + " final log needs correction: " + ", ".join(bad))
        pdf = build / (name + ".pdf")
        reader = PdfReader(str(pdf))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        for token in ("??", "Not Specified", "Anonymous submission", "Anonymous Authors", "formatting instructions"):
            if token.lower() in text.lower():
                raise ValueError(name + " PDF contains unresolved text: " + token)
        if not all(page.strip() for page in pages):
            raise ValueError(name + " contains an empty/unextractable page")
        (build / (name + ".extracted.txt")).write_text(text)
        checks[name] = {"pages": len(pages), "sha256": digest(pdf),
                        "final_log_sha256": digest(build / (name + ".log")),
                        "undefined_references_or_glyphs": False, "overfull_boxes": False}
    figure_text = {
        "matched_gap_boxplot.pdf": ("Matched best-of-10", "feasible reference", "Method"),
        "training_budget_plot.pdf": ("(a) Feasibility", "(b) Gap", "Training budget"),
    }
    for filename, required_text in figure_text.items():
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(paper / filename)).pages)
        missing = [token for token in required_text if token not in text]
        if missing:
            raise ValueError(f"{filename} is missing selectable text: " + ", ".join(missing))
        checks[filename] = {"selectable_text": True, "required_labels": list(required_text)}
    abstract = (paper / "journal_abstract.tex").read_text()
    words = len(abstract.split())
    if not 150 <= words <= 250:
        raise ValueError(f"Abstract has {words} words, expected150–250")
    source = expanded_source(paper, "manuscript.tex")
    if re.search(r"\\(?:paragraph|subparagraph)\*?\{", source):
        raise ValueError("Fourth-level manuscript headings remain")
    for required in ("Statements and Declarations", "Data and code availability", "Competing interests", "Funding"):
        if required not in source:
            raise ValueError("Required manuscript section is absent: " + required)
    checks["abstract_words"] = words
    checks["source_headings_and_required_statements"] = True
    return checks


def supplementary(paper, final=False):
    try:
        from .si_package import build_supplementary
    except ImportError:
        from si_package import build_supplementary
    return build_supplementary(paper, ROOT, TITLE, AUTHORS,
                               ART / "AUTHOR_DECLARATIONS.json", final=final)


def package(paper, build, final=False):
    name = "LLMs_can_Schedule_journal_sources.zip" if final else "LLMs_can_Schedule_journal_sources_pending_results.zip"
    target = paper / name
    required = list(SOURCES)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in required:
            path = paper / name
            if not path.is_file():
                raise ValueError("Submission dependency missing: " + name)
            archive.write(path, name)
        for name in ("manuscript.pdf", "manuscript.bbl"):
            archive.write(build / name, name)
        archive.writestr("README.txt", "Compile manuscript.tex using PDFLaTeX, BibTeX, PDFLaTeX twice.\n"
                         "All LaTeX dependencies are in this directory. Upload ESM_1.zip separately as Online Resource1.\n" +
                         ("Completed measured results included.\n" if final else
                          "Completed fixed-instruction checkpoint and baseline results are included; final selected-instruction and prompt/refinement results are pending. This is not the final submission.\n"))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any("/" in name for name in archive.namelist()):
            raise ValueError("Submission ZIP is corrupt or not flat")
    return {"path": str(target), "sha256": digest(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-pending", action="store_true")
    parser.add_argument("--host", default="ourasi")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    bridge = None
    if args.final:
        try:
            from .journal_publication import json_hash
        except ImportError:
            from journal_publication import json_hash
        path = PAPER / "build/JOURNAL_RESULT_INTEGRATION.json"
        if not path.is_file():
            raise ValueError("Final package requires verified journal result bridge")
        bridge = json.loads(path.read_text())
        if (bridge.get("status") != "verified_journal_results_integrated_pdf_rebuild_required" or
                bridge.get("receipt_sha256") != json_hash({k: v for k, v in bridge.items() if k != "receipt_sha256"})):
            raise ValueError("Journal result bridge completion receipt is invalid")
        if any(digest(PAPER / name) != expected for name, expected in bridge["outputs_sha256"].items()):
            raise ValueError("Integrated scientific result text changed after verification")
    if args.prepare_pending and args.final:
        raise ValueError("A final build cannot reset measured results to pending")
    metadata = prepare_pending() if args.prepare_pending else author_files(final=args.final)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build = args.build or PAPER / "build" / ("journal_" + stamp)
    compile_receipt = native_compile(PAPER, build, args.host)
    checks = inspect(PAPER, build)
    resource = supplementary(PAPER, args.final)
    archive = package(PAPER, build, args.final)
    for name in ("manuscript", "rebuttal"):
        shutil.copy2(build / (name + ".pdf"), PAPER / (name + ".pdf"))
    receipt = {"status": "compiled_and_checked", "experimental_results_pending": not args.final,
               "built_at_utc": datetime.now(timezone.utc).isoformat(), "host": args.host,
               "build_dir": str(build), "compiler": compile_receipt, "pdf_checks": checks,
               "author_metadata": metadata, "supplementary": resource, "source_package": archive,
               "source_sha256": {name: digest(PAPER / name) for name in SOURCES},
               "rebuttal_source_sha256": digest(PAPER / "rebuttal.tex"),
               "visual_inspection_required": True}
    if bridge:
        receipt["upstream_journal_integration_receipt_sha256"] = digest(PAPER / "build/JOURNAL_RESULT_INTEGRATION.json")
        receipt["frozen_publisher_receipt_file_sha256"] = bridge["publisher_receipt_file_sha256"]
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    write_json(build / "JOURNAL_BUILD.json", receipt)
    write_json(PAPER / "JOURNAL_BUILD.json", receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
