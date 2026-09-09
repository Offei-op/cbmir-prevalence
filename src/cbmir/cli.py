"""Explicit stages; importing cbmir never launches data loading or training."""

import argparse
import json
from pathlib import Path
from .io import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare")
    validate = sub.add_parser("validate")
    validate.add_argument("--check-images", action="store_true")
    sub.add_parser("content-audit")
    train = sub.add_parser("train")
    train.add_argument("--model", required=True, choices=MODELS)
    embed = sub.add_parser("embed")
    embed.add_argument("--model", required=True, choices=MODELS)
    embed.add_argument("--checkpoint")
    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("--model", required=True, choices=MODELS)
    legacy.add_argument("--cache", required=True)
    legacy.add_argument("--index", required=True)
    sub.add_parser("evaluate")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--results")
    sub.add_parser("diagnostics")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    # Paths are relative to the current working directory, documented in README.
    if args.stage == "prepare":
        from .prepare import run

        result = run(config)
    elif args.stage == "validate":
        from .validation import validate_artifacts

        result = validate_artifacts(config["artifacts_dir"], config, args.check_images)
    elif args.stage == "content-audit":
        from .validation import content_audit

        result = content_audit(config["artifacts_dir"])
    elif args.stage == "train":
        from .train import run

        result = run(config, args.model)
    elif args.stage == "embed":
        from .embeddings import run

        result = run(config, args.model, args.checkpoint)
    elif args.stage == "import-legacy":
        from .embeddings import import_legacy

        result = import_legacy(config, args.model, args.cache, args.index)
    elif args.stage == "evaluate":
        from .retrieval import run

        result = run(config)
    elif args.stage == "analyze":
        from .analysis import run

        result = run(config, args.results)
    else:
        from .diagnostics import run

        result = run(config)
    if result is not None:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
