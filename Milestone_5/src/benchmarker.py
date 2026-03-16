"""
benchmarker.py — Hyperfine Statistical Benchmark Runner
========================================================
AlgoProfessor AI R&D Internship | Day 30 | Milestone 5

Uses Hyperfine to statistically benchmark Python vs C++ execution time.
Hyperfine runs each program N times, computes mean/stddev/min/max, and
exports results as JSON for analysis.

Why Hyperfine over Python's timeit?
- Handles process startup overhead correctly
- Computes statistics across multiple runs (not just one measurement)
- Provides warmup runs to eliminate cache cold-start effects
- Can benchmark ANY executable, not just Python code

Usage:
    from benchmarker import Benchmarker
    bench = Benchmarker()
    result = bench.run("script.py", "/tmp/my_binary", "my_program")
    print(f"Speedup: {result['speedup']}x")
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


class Benchmarker:
    """
    Statistical benchmarker using Hyperfine.

    Compares execution time of a Python script vs a compiled C++ binary.
    Reports mean, stddev, min, max, and speedup factor.

    Example:
        bench = Benchmarker(warmup=3, runs=10)
        result = bench.run("bubble_sort.py", "/tmp/bubble_sort", "bubble_sort")
        print(f"{result['speedup']:.0f}x faster in C++")
    """

    def __init__(self, warmup: int = 3, runs: int = 10):
        """
        Args:
            warmup: Number of warmup runs (not counted in statistics).
                    Warmup eliminates OS page cache and CPU branch predictor effects.
            runs:   Number of timed measurement runs.
                    10 runs gives ~95% confidence interval on the mean.
        """
        if not shutil.which("hyperfine"):
            raise EnvironmentError(
                "hyperfine not found. Install with:\n"
                "  Linux: wget .../hyperfine_1.18.0_amd64.deb && dpkg -i ...\n"
                "  Mac:   brew install hyperfine\n"
                "  Colab: re-run Cell 1 in the notebook"
            )
        self.warmup = warmup
        self.runs   = runs

    def run(self, py_path: str, cpp_binary: str, program_name: str) -> dict:
        """
        Benchmark a Python script against a C++ binary.

        Uses Hyperfine to run both programs multiple times and compute
        precise statistics. Results exported to JSON and parsed automatically.

        Args:
            py_path:      Path to the Python script (.py file).
            cpp_binary:   Path to the compiled C++ binary.
            program_name: Name used in output dict and logging.

        Returns:
            dict with timing statistics:
                program      - name identifier
                py_mean_ms   - Python mean execution time (milliseconds)
                py_std_ms    - Python standard deviation (ms)
                py_min_ms    - Python fastest run (ms)
                py_max_ms    - Python slowest run (ms)
                cpp_mean_ms  - C++ mean execution time (ms)
                cpp_std_ms   - C++ standard deviation (ms)
                cpp_min_ms   - C++ fastest run (ms)
                cpp_max_ms   - C++ slowest run (ms)
                speedup      - py_mean_ms / cpp_mean_ms
                runs         - number of measurement runs
        """
        # Temporary file for Hyperfine JSON export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        cmd = [
            "hyperfine",
            "--warmup",      str(self.warmup),
            "--runs",        str(self.runs),
            "--export-json", json_path,
            "--style",       "none",          # No ANSI colours (cleaner output)
            f"python3 {py_path}",             # Command 1: Python
            cpp_binary,                        # Command 2: C++ binary
        ]

        print(f"  Benchmarking {program_name}...")
        print(f"    Python : python3 {py_path}")
        print(f"    C++    : {cpp_binary}")
        print(f"    Config : {self.warmup} warmup + {self.runs} runs")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,   # 10-minute timeout for slow Python programs
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Hyperfine failed for {program_name}:\n{result.stderr[:400]}"
            )

        # Parse JSON results
        with open(json_path) as f:
            data = json.load(f)

        # results[0] = Python, results[1] = C++
        py  = data["results"][0]
        cpp = data["results"][1]

        def ms(seconds: float) -> float:
            return round(seconds * 1000, 4)

        py_mean_ms  = ms(py["mean"])
        cpp_mean_ms = ms(cpp["mean"])

        # Avoid division by zero for extremely fast programs
        speedup = py_mean_ms / max(cpp_mean_ms, 0.001)

        return {
            "program":     program_name,
            "py_mean_ms":  py_mean_ms,
            "py_std_ms":   ms(py["stddev"]),
            "py_min_ms":   ms(py["min"]),
            "py_max_ms":   ms(py["max"]),
            "cpp_mean_ms": cpp_mean_ms,
            "cpp_std_ms":  ms(cpp["stddev"]),
            "cpp_min_ms":  ms(cpp["min"]),
            "cpp_max_ms":  ms(cpp["max"]),
            "speedup":     round(speedup, 1),
            "runs":        self.runs,
        }

    def run_batch(self, jobs: list[dict]) -> list[dict]:
        """
        Benchmark multiple programs in sequence.

        Args:
            jobs: List of dicts, each with keys:
                    py_path      - path to Python script
                    cpp_binary   - path to C++ binary
                    program_name - identifier string

        Returns:
            List of result dicts (same format as run()).
        """
        results = []
        for job in jobs:
            try:
                result = self.run(
                    job["py_path"],
                    job["cpp_binary"],
                    job["program_name"],
                )
                results.append(result)
                print(f"  Result: {result['py_mean_ms']:.1f}ms (Python) vs "
                      f"{result['cpp_mean_ms']:.3f}ms (C++) = {result['speedup']:.0f}x speedup")
            except Exception as e:
                print(f"  Benchmark failed for {job.get('program_name', '?')}: {e}")
        return results

    @staticmethod
    def format_table(results: list[dict]) -> str:
        """
        Format benchmark results as a readable ASCII table.

        Args:
            results: List of result dicts from run() or run_batch().

        Returns:
            Formatted string table.
        """
        header = f"{'Program':<20} {'Python (ms)':>12} {'C++ (ms)':>10} {'Speedup':>10} {'Grade':>14}"
        sep    = "-" * 70
        rows   = []

        for r in results:
            s = r["speedup"]
            grade = (
                "EXTREME (>1000x)" if s > 1000 else
                "EXCELLENT (>100x)" if s > 100 else
                "VERY GOOD (>10x)" if s > 10 else
                "GOOD (>1x)"
            )
            rows.append(
                f"{r['program']:<20} {r['py_mean_ms']:>12.1f} "
                f"{r['cpp_mean_ms']:>10.3f} {s:>9.0f}x  {grade}"
            )

        return "\n".join([sep, header, sep] + rows + [sep])
