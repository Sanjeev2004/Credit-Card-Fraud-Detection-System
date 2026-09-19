"""Command-line entry points; training imports are lazy for inference deployments."""

import argparse
import shutil
from pathlib import Path


def download_dataset(destination: str) -> Path:
    import kagglehub

    target = Path(destination)
    if target.exists():
        raise ValueError("Destination already exists; refusing to overwrite the dataset.")
    directory = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
    source = directory / "creditcard.csv"
    if not source.is_file():
        raise FileNotFoundError("Kaggle download did not contain creditcard.csv.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target.resolve()


def main():
    parser = argparse.ArgumentParser(description="Credit card fraud detection benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the original ULB Kaggle dataset")
    download.add_argument("--output", default="data/creditcard.csv")
    training = commands.add_parser("train", help="Fast staged training (default) or full grid search")
    training.add_argument("--data", default="data/creditcard.csv")
    training.add_argument("--output", default="artifacts")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--jobs", type=int, default=2)
    training.add_argument("--min-precision", type=float, default=0.80)
    training.add_argument("--quick", action="store_true", help="Use fewer trees, not fewer data rows")
    training.add_argument("--mode", choices=["fast", "grid"], default="fast")
    training.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    training.add_argument("--models", nargs="+", default=None,
                          choices=["logistic_regression", "xgboost", "lightgbm"])
    training.add_argument("--max-trees", type=int, default=300)
    training.add_argument("--patience", type=int, default=30)
    training.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict", help="Score a CSV in bounded-memory chunks")
    predict.add_argument("--model", default="artifacts/model.joblib")
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()
    if args.command == "download":
        print(download_dataset(args.output))
    elif args.command == "train":
        if args.mode == "grid":
            if args.device != "cpu" or args.models or args.resume:
                parser.error("Grid mode is CPU-only and does not support --models or --resume.")
            from fraud_detection.training import train
            train(args.data, args.output, args.seed, args.jobs, args.quick, args.min_precision)
        else:
            from fraud_detection.fast_training import FAST_MODELS, train_fast
            train_fast(args.data, args.output, seed=args.seed, jobs=args.jobs,
                       min_precision=args.min_precision, device=args.device,
                       models=args.models or FAST_MODELS,
                       max_trees=40 if args.quick else args.max_trees,
                       patience=args.patience, resume=args.resume)
    else:
        from fraud_detection.prediction import predict_csv
        count = predict_csv(args.model, args.data, args.output, args.chunk_size)
        print(f"Scored {count} transactions into {args.output}")


if __name__ == "__main__":
    main()
