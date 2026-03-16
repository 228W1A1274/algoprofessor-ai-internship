"""
compiler.py — C++ Compilation Wrapper with Auto-Retry
======================================================
AlgoProfessor AI R&D Internship | Day 30 | Milestone 5

Compiles C++ code using clang++ with -O3, OpenMP, and native CPU instructions.
Includes LLM-assisted self-healing: if compilation fails, Groq reads the
compiler error and attempts to fix the code automatically.

Usage:
    from compiler import CppCompiler
    compiler = CppCompiler()
    result = compiler.compile(cpp_code, "my_program")
    if result['success']:
        print(f"Binary at: {result['binary']}")
"""

import os
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

from groq import Groq


# ── Compiler Detection ────────────────────────────────────────────────────────

def find_clang() -> str:
    """
    Find the clang++ binary on this system.
    Tries: clang++ symlink → any versioned clang++ in /usr/bin → error.

    Returns:
        Full path to working clang++ binary.

    Raises:
        EnvironmentError if no clang++ found.
    """
    # 1. Check if generic symlink exists in PATH
    if shutil.which("clang++"):
        return "clang++"

    # 2. Scan /usr/bin for any versioned clang++ (clang++-14, clang++-15, etc.)
    candidates = sorted(glob.glob("/usr/bin/clang++*"), reverse=True)
    for candidate in candidates:
        r = subprocess.run([candidate, "--version"], capture_output=True)
        if r.returncode == 0:
            return candidate

    # 3. Check common Mac/Homebrew paths
    for path in ["/opt/homebrew/bin/clang++", "/usr/local/bin/clang++"]:
        if os.path.exists(path):
            return path

    raise EnvironmentError(
        "clang++ not found. Install with:\n"
        "  Linux: sudo apt-get install -y clang\n"
        "  Mac:   brew install llvm\n"
        "  Colab: re-run Cell 1 in the notebook"
    )


# ── Compiler Class ────────────────────────────────────────────────────────────

class CppCompiler:
    """
    Compiles C++ code with clang++ and optional LLM-assisted error recovery.

    Example:
        compiler = CppCompiler(groq_api_key="gsk_...")
        result, final_code = compiler.compile_with_retry(cpp_code, "my_prog")
        if result['success']:
            subprocess.run([result['binary']])
    """

    # Default flags for maximum performance
    DEFAULT_FLAGS = ["-O3", "-fopenmp", "-march=native", "-std=c++17"]

    def __init__(
        self,
        output_dir: str = None,
        groq_api_key: str = None,
        flags: list = None,
    ):
        """
        Args:
            output_dir:    Directory for compiled binaries. Default: /tmp/codexcelerate
            groq_api_key:  Groq API key for auto-fix feature. If None, reads from env.
            flags:         Compiler flags list. Default: -O3 -fopenmp -march=native -std=c++17
        """
        self.compiler   = find_clang()
        self.output_dir = Path(output_dir or "/tmp/codexcelerate")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flags      = flags or self.DEFAULT_FLAGS

        # Groq client for auto-fix (optional)
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.groq = Groq(api_key=api_key) if api_key else None

        print(f"Compiler: {self.compiler}")
        print(f"Flags:    {' '.join(self.flags)}")

    def compile(self, cpp_code: str, program_name: str) -> dict:
        """
        Compile C++ source code to a binary.

        Args:
            cpp_code:     C++ source code as string.
            program_name: Name for the output binary (no extension).

        Returns:
            dict:
                success    (bool)   - whether compilation succeeded
                binary     (str)    - path to compiled binary (if success)
                source     (str)    - path to saved .cpp file
                stderr     (str)    - compiler error output
                returncode (int)    - compiler return code
        """
        src_path = self.output_dir / f"{program_name}.cpp"
        bin_path = self.output_dir / program_name

        # Save source to file
        src_path.write_text(cpp_code, encoding="utf-8")

        # Build and run compile command
        cmd = [self.compiler] + self.flags + [str(src_path), "-o", str(bin_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        return {
            "success":    result.returncode == 0,
            "binary":     str(bin_path),
            "source":     str(src_path),
            "stderr":     result.stderr,
            "stdout":     result.stdout,
            "returncode": result.returncode,
        }

    def _fix_with_llm(self, cpp_code: str, error_message: str) -> str:
        """
        Ask Groq to fix a compilation error.
        Self-healing feature — feeds the compiler error back to the LLM.

        Args:
            cpp_code:      The broken C++ source code.
            error_message: The compiler's stderr output.

        Returns:
            Fixed C++ code string.
        """
        if not self.groq:
            raise RuntimeError("Groq API key required for auto-fix feature")

        print("  Auto-fix: sending compiler error to Groq LLM...")

        fix_prompt = (
            "This C++ code has a compilation error. Fix ONLY the error shown. "
            "Do not change anything else. Return ONLY the fixed C++ code.\n\n"
            f"COMPILER ERROR:\n{error_message[:800]}\n\n"
            f"C++ CODE:\n{cpp_code}"
        )

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=4096,
            messages=[{"role": "user", "content": fix_prompt}],
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            start = 1
            end = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"), len(lines))
            raw = "\n".join(lines[start:end])
        return raw.strip()

    def compile_with_retry(
        self, cpp_code: str, program_name: str, max_retries: int = 2
    ) -> tuple[dict, str]:
        """
        Compile with automatic LLM-assisted error recovery.

        Tries to compile. If it fails, asks Groq to fix the error and retries.
        Maximum max_retries fix attempts.

        Args:
            cpp_code:     C++ source code.
            program_name: Output binary name.
            max_retries:  Number of fix attempts before giving up.

        Returns:
            (compile_result_dict, final_cpp_code)
        """
        for attempt in range(max_retries + 1):
            result = self.compile(cpp_code, program_name)

            if result["success"]:
                if attempt > 0:
                    print(f"  Fixed after {attempt} attempt(s)")
                return result, cpp_code

            # Show first 3 error lines
            error_preview = "\n".join(result["stderr"].strip().split("\n")[:3])
            print(f"  Attempt {attempt + 1} failed:\n    {error_preview}")

            if attempt < max_retries and self.groq:
                cpp_code = self._fix_with_llm(cpp_code, result["stderr"])
            else:
                print(f"  Could not fix after {max_retries} attempt(s)")

        return result, cpp_code

    def test_binary(self, binary_path: str, timeout: int = 30) -> dict:
        """
        Run the compiled binary to verify it executes without error.

        Returns:
            dict with success, stdout, stderr, returncode
        """
        result = subprocess.run(
            [binary_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success":    result.returncode == 0,
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
        }
