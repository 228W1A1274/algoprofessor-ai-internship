"""
transpiler.py — Two-Pass GPT-4o/Groq Python-to-C++ Transpiler
==============================================================
AlgoProfessor AI R&D Internship | Day 30 | Milestone 5

Converts Python source code to optimised, parallelised C++ using
a two-pass LLM pipeline:
  Pass 1: Python → C++ (semantic correctness)
  Pass 2: C++ → OpenMP C++ (multi-core parallelism)

Usage:
    python transpiler.py examples/bubble_sort.py
    python transpiler.py examples/prime_sieve.py --output output.cpp
"""

import os
import sys
import time
import argparse
from pathlib import Path

from groq import Groq


# ── System Prompts ────────────────────────────────────────────────────────────

PASS1_SYSTEM = """
You are a world-class C++17 developer and compiler engineer.

Your task: Convert the given Python code to equivalent, production-grade C++ code.

STRICT RULES:
1. SEMANTIC EQUIVALENCE: The C++ must produce IDENTICAL output to the Python
2. Use modern C++17 features: auto, range-based for, structured bindings
3. Use STL containers: std::vector, std::string, std::unordered_map
4. Include ALL necessary #include statements
5. Include a complete main() function that runs the same computation
6. Handle integer overflow: use long long for large numbers
7. Use <random> with mt19937 seeded at 42 for reproducibility
8. Code must compile with: clang++ -O3 -std=c++17

OUTPUT FORMAT: Raw C++ code ONLY. No markdown. No explanation.
""".strip()

PASS2_SYSTEM = """
You are an expert in OpenMP parallel programming and high-performance computing.

Your task: Add OpenMP parallelisation to the given C++ code to maximise performance.

STRICT RULES:
1. Add: #include <omp.h>
2. Add: omp_set_num_threads(omp_get_max_threads()); at start of main()
3. Add #pragma omp parallel for to ALL loops with NO loop-carried dependencies
4. For accumulation loops use: reduction(+:variable)
5. For single-variable updates use: #pragma omp atomic
6. DO NOT parallelise loops with dependencies (e.g. swap-based sorts)
7. The parallelised code must produce IDENTICAL results to the sequential version
8. Code must compile with: clang++ -O3 -fopenmp -march=native -std=c++17

OUTPUT FORMAT: Raw C++ code ONLY. No markdown. No explanation.
""".strip()


# ── Helper Functions ──────────────────────────────────────────────────────────

def clean_cpp_output(raw: str) -> str:
    """Strip markdown code fences that LLMs sometimes add despite instructions."""
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        # Find closing fence
        start, end = 1, len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        code = "\n".join(lines[start:end])
    return code.strip()


# ── Main Transpiler Class ─────────────────────────────────────────────────────

class CodeXcelerator:
    """
    Two-pass Python-to-C++ transpiler using Groq LLM.

    Example:
        transpiler = CodeXcelerator(api_key="gsk_...")
        result = transpiler.transpile(python_code)
        print(result['cpp_openmp'])
    """

    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialise the transpiler.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model:   Groq model to use. Default: llama-3.3-70b-versatile
        """
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key provided. "
                "Set GROQ_API_KEY environment variable or pass api_key argument. "
                "Get a free key at: console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def _call_llm(self, system_prompt: str, user_content: str) -> tuple[str, int]:
        """
        Single LLM API call with timing and token tracking.

        Returns:
            (response_text, total_tokens_used)
        """
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,       # Low temperature = deterministic code
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        )
        elapsed = time.time() - t0
        tokens = response.usage.total_tokens
        text = clean_cpp_output(response.choices[0].message.content)
        print(f"     LLM call complete: {elapsed:.1f}s | {tokens} tokens")
        return text, tokens

    def pass1_convert(self, python_code: str) -> tuple[str, int]:
        """
        Pass 1: Convert Python to semantically equivalent C++.
        Focus: correctness, not performance.
        """
        print("  Pass 1: Python -> C++ (correctness)...")
        cpp, tokens = self._call_llm(
            PASS1_SYSTEM,
            f"Convert this Python code to C++:\n\n{python_code}"
        )
        return cpp, tokens

    def pass2_parallelise(self, cpp_code: str) -> tuple[str, int]:
        """
        Pass 2: Add OpenMP parallelism to already-correct C++.
        Focus: performance via multi-core execution.
        """
        print("  Pass 2: C++ -> OpenMP C++ (parallelism)...")
        cpp_parallel, tokens = self._call_llm(
            PASS2_SYSTEM,
            f"Add OpenMP parallelisation to this C++ code:\n\n{cpp_code}"
        )
        return cpp_parallel, tokens

    def transpile(self, python_code: str, name: str = "program") -> dict:
        """
        Full two-pass transpilation pipeline.

        Args:
            python_code: The Python source code string to transpile.
            name:        Identifier for logging purposes.

        Returns:
            dict with keys:
                name         - program identifier
                python       - original Python code
                cpp_basic    - Pass 1 output (correct C++)
                cpp_openmp   - Pass 2 output (parallelised C++)
                total_tokens - total tokens consumed
        """
        print(f"\nTranspiling: {name}")
        print("  " + "-" * 44)

        cpp_basic,    t1 = self.pass1_convert(python_code)
        cpp_openmp,   t2 = self.pass2_parallelise(cpp_basic)

        total = t1 + t2
        print(f"  Done. Total tokens: {total} | Cost: $0.00 (Groq free tier)")

        return {
            "name":         name,
            "python":       python_code,
            "cpp_basic":    cpp_basic,
            "cpp_openmp":   cpp_openmp,
            "total_tokens": total,
        }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CodeXcelerate: Python-to-C++ AI Transpiler (Groq/Llama 3.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transpiler.py examples/bubble_sort.py
  python transpiler.py examples/prime_sieve.py --output out/prime_sieve.cpp
  python transpiler.py examples/matrix_multiply.py --pass1-only
        """
    )
    parser.add_argument("input",         help="Path to Python source file")
    parser.add_argument("--output",  "-o", help="Output path for C++ file (default: <input>.cpp)")
    parser.add_argument("--pass1-only",   action="store_true", help="Only run Pass 1 (no OpenMP)")
    parser.add_argument("--model",        default="llama-3.3-70b-versatile", help="Groq model to use")
    args = parser.parse_args()

    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    python_code = input_path.read_text(encoding="utf-8")
    name = input_path.stem

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".cpp")

    # Run transpilation
    transpiler = CodeXcelerator(model=args.model)

    if args.pass1_only:
        cpp, tokens = transpiler.pass1_convert(python_code)
        print(f"\nPass 1 complete. Tokens used: {tokens}")
    else:
        result = transpiler.transpile(python_code, name=name)
        cpp = result["cpp_openmp"]
        print(f"\nTranspilation complete. Total tokens: {result['total_tokens']}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cpp, encoding="utf-8")
    print(f"C++ saved to: {output_path}")
    print(f"\nTo compile:")
    print(f"  clang++ -O3 -fopenmp -march=native -std=c++17 {output_path} -o {output_path.stem}")


if __name__ == "__main__":
    main()
