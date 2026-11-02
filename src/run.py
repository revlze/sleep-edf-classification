"""CLI: одна конфигурация или последовательная серия, с одинаковыми folds."""

import argparse

from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
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
        run_experiment(
            path,
            data_root=args.data_root,
            output_root=args.output_root,
            smoke=args.smoke,
            prepare=args.prepare,
        )


if __name__ == "__main__":
    main()
