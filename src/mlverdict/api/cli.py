"""Command line: fit a dataset, print a readable verdict, save or serve an artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import DecisionStatus
from mlverdict.production.artifact import ModelArtifact
from mlverdict.production.service import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlverdict",
        description="Evidence-based model selection. Prints a readable verdict.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Train, compare models, print the verdict.")
    fit.add_argument("data", help="Path to a CSV file")
    fit.add_argument("--target", default=None, help="Target column for supervised Phase 1")
    fit.add_argument(
        "--task",
        default=None,
        help="Unsupervised objective: clustering | anomaly_detection | dimensionality_reduction",
    )
    fit.add_argument("--save", default=None, help="Path to write the deployable artifact (joblib)")
    fit.add_argument("--no-hpo", action="store_true", help="Skip hyperparameter search")
    fit.add_argument("--fast", action="store_true", help="Smaller CV/HPO budget (demos and tests)")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--problem-type", default=None)
    fit.add_argument("--primary-metric", default=None)
    fit.add_argument("--group-col", default=None)
    fit.add_argument("--time-col", default=None)
    fit.add_argument("--report", action="store_true", help="Also print the full markdown report")
    fit.set_defaults(func=cmd_fit)

    serve_p = sub.add_parser("serve", help="Serve a saved artifact as a local JSON API.")
    serve_p.add_argument("artifact", help="Path to a saved .joblib artifact")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)
    serve_p.set_defaults(func=cmd_serve)

    return parser


def _config(args: argparse.Namespace) -> VerdictConfig:
    if args.fast:
        return VerdictConfig(
            random_state=args.seed,
            test_size=0.2,
            cv_splits=3,
            max_hpo_trials=2,
            max_hpo_time_seconds=20,
            n_hpo_candidates=1,
            latency_probe_rows=16,
        )
    return VerdictConfig(random_state=args.seed)


def cmd_fit(args: argparse.Namespace) -> int:
    from mlverdict.api.verdict import Verdict

    data = args.data
    path = Path(data)
    if not path.exists():
        print(f"File not found: {data}", file=sys.stderr)
        return 2

    engine = Verdict(config=_config(args), enable_hpo=not args.no_hpo)
    run = engine.fit(
        str(path),
        args.target,
        task=args.task,
        problem_type=args.problem_type,
        primary_metric=args.primary_metric,
        group_col=args.group_col,
        time_col=args.time_col,
    )
    print(run)
    if args.report:
        print()
        print(run.report())
    if args.save:
        artifact = run.artifact()
        if artifact is None:
            print("No artifact to save - run did not produce a deployable model.", file=sys.stderr)
            return 2
        saved = artifact.save(args.save)
        print(f"Saved artifact -> {saved}")
    if run.status == DecisionStatus.DECIDED:
        return 0
    return 1


def cmd_serve(args: argparse.Namespace) -> int:
    path = Path(args.artifact)
    if not path.exists():
        print(f"Artifact not found: {path}", file=sys.stderr)
        return 2
    # Validate load before binding the port.
    ModelArtifact.load(path)
    server = serve(path, host=args.host, port=args.port)
    print(f"MLVerdict serving {path}")
    print(f"GET  http://{args.host}:{args.port}/health")
    print(f"POST http://{args.host}:{args.port}/predict   {{\"records\": [{{...}}]}}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
