"""
DevSquad — 4-Agent Autonomous Engineering Team
Milestone 10 | AlgoProfessor AI R&D Internship 2026
Entry point: accepts a task specification and drives agents → Docker → GitHub
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from crew import DevSquadCrew

load_dotenv()


def validate_env() -> None:
    """Raise early if required secrets are missing."""
    missing = [k for k in ("GROQ_API_KEY",) if not os.getenv(k)]
    if missing:
        print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
        print("Create a .env file — see README for the exact format.")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DevSquad: 4-Agent Autonomous Engineering Team"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Build a Python CLI tool that fetches weather data from the "
                "Open-Meteo API (no key required) for a given city and prints "
                "current temperature, wind speed, and weather code.",
        help="Natural-language task description for the engineering squad.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push generated code to GitHub after QA passes.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run generated code inside an isolated Docker container for QA.",
    )
    return parser.parse_args()


def main() -> None:
    validate_env()
    args = parse_args()

    print("\n" + "=" * 60)
    print("  DevSquad — Autonomous Engineering Team")
    print("=" * 60)
    print(f"  Task   : {args.task[:80]}{'...' if len(args.task) > 80 else ''}")
    print(f"  Docker : {'enabled' if args.docker else 'disabled (mock mode)'}")
    print(f"  GitHub : {'push enabled' if args.push else 'local only'}")
    print("=" * 60 + "\n")

    crew = DevSquadCrew(
        task_description=args.task,
        use_docker=args.docker,
        push_to_github=args.push,
    )
    result = crew.run()

    print("\n" + "=" * 60)
    print("  FINAL SQUAD OUTPUT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
