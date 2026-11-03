"""CLI: одна конфигурация или последовательная серия, с одинаковыми folds."""

import argparse

from omegaconf import OmegaConf

from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--cache-root", help="Read-only source feature cache for trees")
    parser.add_argument("--fold-root", help="Existing LR fold assignments for trees")
    parser.add_argument(
        "--threads", help="Tree CPU threads: auto, -1 or positive integer"
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "fixed",
            "cache",
            "benchmark",
            "baseline",
            "screen",
            "search",
            "refit",
        ],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Nine train subjects, two val subjects; no experiments.csv entry",
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Only prepare/reuse raw feature cache"
    )
    args = parser.parse_args()
    for path in args.configs:
        print(f"Запуск {path}", flush=True)
        if OmegaConf.load(path).model.type in ("lightgbm", "catboost", "auto"):
            from src.training.boosting import run_tree

            run_tree(
                path,
                stage="cache" if args.prepare else args.stage,
                data_root=args.data_root,
                output_root=args.output_root,
                cache_root=args.cache_root,
                fold_root=args.fold_root,
                threads=args.threads,
                smoke=args.smoke,
            )
        else:
            if args.stage != "all" or args.cache_root or args.fold_root or args.threads:
                parser.error(
                    "--stage/--cache-root/--threads are tree-specific; LR settings remain in its YAML"
                )
            run_experiment(
                path,
                data_root=args.data_root,
                output_root=args.output_root,
                smoke=args.smoke,
                prepare=args.prepare,
            )


if __name__ == "__main__":
    main()
