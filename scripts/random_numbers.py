#!/usr/bin/env python3
"""Generate random numbers and print them.

Examples:
    python scripts/random_numbers.py
    python scripts/random_numbers.py --count 20 --min 1 --max 100
    python scripts/random_numbers.py --count 5 --seed 42
"""

import argparse
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and print random numbers."
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=10,
        help="How many numbers to generate (default: 10).",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=0,
        dest="low",
        help="Inclusive lower bound (default: 0).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=100,
        dest="high",
        help="Inclusive upper bound (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.count < 0:
        raise SystemExit("--count must be non-negative")
    if args.low > args.high:
        raise SystemExit("--min must be less than or equal to --max")

    rng = random.Random(args.seed)
    numbers = [rng.randint(args.low, args.high) for _ in range(args.count)]

    for number in numbers:
        print(number)


if __name__ == "__main__":
    main()
