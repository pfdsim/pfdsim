"""The ``pfdsim fit`` command; numerical and export behavior live in the API."""

from __future__ import annotations

import argparse
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.request import HTTPCookieProcessor, Request, build_opener

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .thermodynamics_models.interaction_fitting import (
        export_fit,
        fit_interactions,
        inspect_observations,
    )
    from .activity_fit_store import ActivityFitStore, DEFAULT_ACTIVITY_FITS_PATH
else:
    from thermodynamics_models.interaction_fitting import (
        export_fit,
        fit_interactions,
        inspect_observations,
    )
    from activity_fit_store import ActivityFitStore, DEFAULT_ACTIVITY_FITS_PATH


def _parser():
    parser = argparse.ArgumentParser(
        prog="pfdsim fit",
        description="Fit binary NRTL/UNIQUAC experimental data and export PFD parameters.",
    )
    parser.add_argument(
        "file",
        help="fit request JSON, or JSON/Markdown/CSV/TSV observations; '-' reads stdin",
    )
    parser.add_argument(
        "--config", type=Path, help="JSON fit settings for a table input"
    )
    parser.add_argument("--components", nargs=2, metavar=("COMPONENT1", "COMPONENT2"))
    parser.add_argument(
        "--import-options",
        help="JSON import settings, including mapping, units and common conditions",
    )
    parser.add_argument(
        "--preview-import",
        action="store_true",
        help="inspect the table and missing choices without fitting",
    )
    parser.add_argument(
        "--pressure", type=float, help="common absolute pressure for a pasted table"
    )
    parser.add_argument(
        "--pressure-unit",
        choices=("bar", "atm", "kpa", "pa", "mpa", "mmhg", "torr", "psi"),
    )
    parser.add_argument("--temperature-unit", choices=("C", "K", "F"))
    parser.add_argument(
        "--composition-basis",
        choices=("mole_fraction", "mole_percent", "mass_fraction", "mass_percent"),
    )
    parser.add_argument("--composition-component", type=int, choices=(1, 2))
    parser.add_argument(
        "--data-kind",
        choices=("VLE", "LLE", "HE", "GAMMA_INF", "AZEOTROPE", "VLLE", "UCST", "LCST"),
    )
    parser.add_argument("--model", choices=("NRTL", "UNIQUAC"))
    parser.add_argument(
        "--vapor", help="IDEAL, RK, PR, VDM, TSONOPOULOS, PITZER-CURL, ABBOTT or HOC"
    )
    parser.add_argument(
        "--form",
        help="constant, inverse, constant_inverse, constant_inverse_anchored, constant_inverse_linear, constant_inverse_anchored_linear (ABCD) or full",
    )
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--fit-alpha", action="store_true", default=None)
    parser.add_argument(
        "--weights", help='JSON objective weights, e.g. {"VLE":1,"HE":2}'
    )
    parser.add_argument("--rq", help='JSON [{"r":...,"q":...},{"r":...,"q":...}]')
    parser.add_argument(
        "--cv", choices=("none", "kfold", "leave_temperature_out", "leave_group_out")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--source", help="source citation, required for submission")
    parser.add_argument("--online-lookup", action="store_true", default=None)
    parser.add_argument(
        "--pfd-input", type=Path, help="existing PFD supplying component definitions"
    )
    parser.add_argument(
        "--scope", default="global", help="scope for definitions and PFD application"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write complete JSON fit report; otherwise print it",
    )
    parser.add_argument(
        "--entry", type=Path, help="write individual INTERACTION_PARAMETERS entries"
    )
    parser.add_argument(
        "--pfd-output",
        type=Path,
        help="write a fitted mixture PFD, or update --apply-to into this new file",
    )
    parser.add_argument(
        "--apply-to",
        type=Path,
        help="existing PFD to receive the fit; requires --pfd-output",
    )
    parser.add_argument(
        "--component-map", help="JSON mapping fitted symbols to destination PFD symbols"
    )
    parser.add_argument(
        "--submission-bundle", type=Path, help="write a sourced review bundle"
    )
    parser.add_argument(
        "--submit-url",
        help="submit for review to a running PFDSim server (e.g. http://localhost:5000)",
    )
    parser.add_argument("--source-url")
    parser.add_argument("--doi")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacing output files, preserving a numbered .bak copy",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _write(path, text, force):
    if path.exists():
        if not force:
            raise ValueError(f"Output {path} exists. Use a new path or --force.")
        backup = Path(str(path) + ".bak")
        count = 1
        while backup.exists():
            backup = Path(str(path) + f".bak.{count}")
            count += 1
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}", file=sys.stderr)


def _submit(url, result, source):
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    base = url.rstrip("/")
    with opener.open(base + "/api/session", timeout=30) as response:
        token = json.load(response)["csrf_token"]
    data = json.dumps({"result": result, "source": source}, allow_nan=False).encode()
    request = Request(
        base + "/api/fitting/submit",
        data=data,
        headers={"Content-Type": "application/json", "X-CSRF-Token": token},
    )
    with opener.open(request, timeout=30) as response:
        return json.load(response)


def _review_submissions(argv):
    parser = argparse.ArgumentParser(
        prog="pfdsim fit submissions",
        description="Read the local sourced-fit review queue; never modifies parameter databases.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
    )
    parser.add_argument("--id", help="retrieve one complete submission bundle")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    database = (
        (args.directory.expanduser().resolve() / "user_activity_fits.sqlite")
        if args.directory
        else Path(
            os.environ.get("PFDSIM_ACTIVITY_FITS_PATH", DEFAULT_ACTIVITY_FITS_PATH)
        )
    )
    if not database.is_file():
        parser.error("No user_activity_fits.sqlite exists at this location.")
    if args.output and args.output.expanduser().resolve() == database:
        parser.error("Output must differ from the review database.")
    try:
        store = ActivityFitStore(database)
        report = store.get(args.id) if args.id else store.list()
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            _write(args.output.expanduser(), text, args.force)
        else:
            print(text, end="")
        return 0
    except (ValueError, OSError) as error:
        print(f"pfdsim fit submissions: error: {error}", file=sys.stderr)
        return 1


def _submit_report(argv):
    parser = argparse.ArgumentParser(
        prog="pfdsim fit submit",
        description="Submit a saved fit report or review bundle without rerunning regression.",
    )
    parser.add_argument("file", type=Path)
    parser.add_argument("--server", "--submit-url", required=True)
    parser.add_argument("--source")
    parser.add_argument("--source-url")
    parser.add_argument("--doi")
    parser.add_argument("--notes")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.file.expanduser().read_text(encoding="utf-8"))
        result = payload.get("result", payload)
        export_fit(result)  # Validate the reusable export contract, never refit.
        if not result.get("success"):
            raise ValueError(
                "A fit needing review cannot be submitted for general inclusion."
            )
        source = (
            dict(payload.get("source", {}))
            if "result" in payload
            else {"citation": result["request"].get("source", "")}
        )
        for field, value in (
            ("citation", args.source),
            ("url", args.source_url),
            ("doi", args.doi),
            ("notes", args.notes),
        ):
            if value is not None:
                source[field] = value
        if not source.get("citation", "").strip():
            raise ValueError(
                "Supply --source or a source citation in the saved bundle."
            )
        response = _submit(args.server, result, source)
        print(f"Submitted for review: {response['submission']['id']}")
        return 0
    except (ValueError, OSError, UnicodeError, KeyError) as error:
        print(f"pfdsim fit submit: error: {error}", file=sys.stderr)
        return 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "submissions":
        return _review_submissions(argv[1:])
    if argv and argv[0] == "submit":
        return _submit_report(argv[1:])
    parser = _parser()
    args = parser.parse_args(argv)
    if args.apply_to and not args.pfd_output:
        parser.error(
            "--apply-to requires --pfd-output; the input PFD is never overwritten"
        )
    paths = [
        path.expanduser().resolve()
        for path in (args.output, args.entry, args.pfd_output, args.submission_bundle)
        if path
    ]
    inputs = [
        path.expanduser().resolve()
        for path in (
            Path(args.file) if args.file != "-" else None,
            args.config,
            args.pfd_input,
            args.apply_to,
        )
        if path
    ]
    if len(set(paths)) != len(paths) or set(paths) & set(inputs):
        parser.error("All output paths must be distinct and different from input paths")
    if not args.force and any(path.exists() for path in paths):
        parser.error("An output path already exists; use a new path or --force")
    try:
        text = (
            sys.stdin.read()
            if args.file == "-"
            else Path(args.file).expanduser().read_text(encoding="utf-8")
        )
        request = (
            json.loads(args.config.expanduser().read_text(encoding="utf-8"))
            if args.config
            else {}
        )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        if isinstance(payload, dict) and "components" in payload:
            request.update(payload)
        else:
            request["observations"] = payload
        for name in (
            "components",
            "model",
            "vapor",
            "form",
            "alpha",
            "fit_alpha",
            "online_lookup",
            "source",
        ):
            value = getattr(args, name)
            if value is not None:
                request[name] = value
        for name in ("weights", "rq"):
            if getattr(args, name):
                request[name] = json.loads(getattr(args, name))
        imported = (
            json.loads(args.import_options)
            if args.import_options
            else dict(request.get("import_options", {}))
        )
        for name in (
            "pressure",
            "pressure_unit",
            "temperature_unit",
            "composition_basis",
            "composition_component",
        ):
            value = getattr(args, name)
            if value is not None:
                imported[name] = value
        if args.data_kind:
            imported["kind"] = args.data_kind
        if imported:
            request["import_options"] = imported
        if args.preview_import:
            proposal = inspect_observations(
                request.get("observations"),
                import_options=imported,
                components=request.get("components"),
            )
            preview_text = json.dumps(proposal, indent=2, allow_nan=False) + "\n"
            if args.output:
                _write(args.output.expanduser(), preview_text, args.force)
            else:
                print(preview_text, end="")
            return 0 if proposal["ready"] else 2
        if args.cv:
            request["cv"] = {"method": args.cv, "folds": args.folds}
        if args.pfd_input:
            request.update(
                pfd_text=args.pfd_input.expanduser().read_text(encoding="utf-8"),
                scope=args.scope,
            )
        source = {
            "citation": args.source or request.get("source", ""),
            "url": args.source_url or "",
            "doi": args.doi or "",
            "notes": args.notes,
        }
        if (args.submit_url or args.submission_bundle) and not source[
            "citation"
        ].strip():
            parser.error(
                "--source or a source citation in the request is required for submission"
            )
        progress = (
            (lambda message: print(message, file=sys.stderr, flush=True))
            if args.verbose
            else None
        )
        try:
            result = fit_interactions(request, progress=progress)
        except Exception as error:
            # Numerical/model exceptions share one command-line error boundary;
            # KeyboardInterrupt/SystemExit retain their usual behavior.
            raise ValueError(str(error)) from error
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            _write(args.output.expanduser(), text, args.force)
        else:
            print(text, end="")
        if args.entry:
            _write(args.entry.expanduser(), result["entry"] + "\n", args.force)
        if args.pfd_output:
            exported = export_fit(
                result,
                pfd_text=args.apply_to.expanduser().read_text(encoding="utf-8")
                if args.apply_to
                else None,
                scope=args.scope if args.apply_to else "global",
                component_map=json.loads(args.component_map)
                if args.component_map
                else None,
            )
            _write(args.pfd_output.expanduser(), exported["pfd_text"], args.force)
        if args.submission_bundle:
            _write(
                args.submission_bundle.expanduser(),
                json.dumps(
                    {"source": source, "result": result}, indent=2, allow_nan=False
                )
                + "\n",
                args.force,
            )
        if args.submit_url:
            response = _submit(args.submit_url, result, source)
            print(
                f"Submitted for review: {response['submission']['id']}", file=sys.stderr
            )
        return 0 if result["success"] else 1
    except (ValueError, OSError, UnicodeError, RuntimeError) as error:
        print(f"pfdsim fit: error: {error}", file=sys.stderr)
        return 1
