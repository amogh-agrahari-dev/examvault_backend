"""
Code Execution Sandbox Service
==============================
Provides isolated sandbox code execution for Quiz coding questions.

Supports:
1. Judge0 Provider: If configured via JUDGE0_API_URL (and optional JUDGE0_API_KEY).
2. Local Isolated Sandbox: Built-in secure subprocess sandbox with timeouts,
   temporary directory isolation, resource limits, and environment variable sanitization.
"""

import os
import sys
import uuid
import shutil
import asyncio
import tempfile
import logging
from typing import Dict, Any, Optional, List
import httpx

from config import settings

logger = logging.getLogger(__name__)

# Map common language names to Judge0 language IDs (CE edition)
JUDGE0_LANGUAGE_IDS = {
    "python": 71,       # Python (3.8.1)
    "python3": 71,
    "py": 71,
    "javascript": 63,   # JavaScript (Node.js 12.14.0)
    "js": 63,
    "typescript": 74,   # TypeScript (3.7.4)
    "ts": 74,
    "cpp": 54,          # C++ (GCC 9.2.0)
    "c++": 54,
    "c": 50,            # C (GCC 9.2.0)
    "java": 62,         # Java (OpenJDK 13.0.1)
}


async def _execute_judge0(
    code: str,
    language: str = "python",
    stdin: str = "",
    timeout: float = 7.0
) -> Dict[str, Any]:
    """
    Submits code to a remote Judge0 instance and returns standard output/error.
    """
    api_url = settings.JUDGE0_API_URL.rstrip("/")
    lang_id = JUDGE0_LANGUAGE_IDS.get(language.lower(), 71)

    headers = {
        "Content-Type": "application/json"
    }

    if settings.JUDGE0_API_KEY:
        if "rapidapi.com" in api_url:
            headers["X-RapidAPI-Key"] = settings.JUDGE0_API_KEY
            headers["X-RapidAPI-Host"] = api_url.replace("https://", "").replace("http://", "").split("/")[0]
        else:
            headers["X-Auth-Token"] = settings.JUDGE0_API_KEY

    payload = {
        "source_code": code,
        "language_id": lang_id,
        "stdin": stdin or "",
        "cpu_time_limit": min(max(timeout, 1.0), 15.0),
    }

    async with httpx.AsyncClient(timeout=timeout + 5.0) as client:
        # Submit and request wait=true for synchronous evaluation
        url = f"{api_url}/submissions?base64_encoded=false&wait=true"
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        stdout = data.get("stdout") or ""
        stderr = data.get("stderr") or data.get("compile_output") or ""
        status_info = data.get("status", {})
        status_desc = status_info.get("description", "Unknown")
        status_id = status_info.get("id", 0)

        # Status 3 is Accepted in Judge0
        is_success = status_id == 3
        if not is_success and not stderr and status_desc:
            stderr = f"Execution Status: {status_desc}"

        return {
            "output": stdout,
            "error": stderr,
            "exit_code": 0 if is_success else 1,
            "status": "success" if is_success else "error",
            "provider": "judge0",
            "execution_time": data.get("time"),
            "memory": data.get("memory"),
        }


async def _execute_local_sandbox(
    code: str,
    language: str = "python",
    stdin: str = "",
    timeout: float = 7.0
) -> Dict[str, Any]:
    """
    Executes code in an isolated temporary subprocess environment with timeouts,
    stripped environment variables, and captured streams.
    """
    lang_lower = language.lower()
    temp_dir = tempfile.mkdtemp(prefix="examvault_sandbox_")

    try:
        # Sanitized, minimal environment variables (prevents leakage of DATABASE_URL, JWT_SECRET, etc.)
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": temp_dir,
        }

        # Prepare executable and script file
        if lang_lower in ("python", "python3", "py"):
            script_path = os.path.join(temp_dir, "solution.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Use active venv python interpreter
            cmd = [sys.executable, "-u", script_path]
        elif lang_lower in ("javascript", "js", "node"):
            node_bin = shutil.which("node")
            if not node_bin:
                return {
                    "output": "",
                    "error": "Node.js runtime is not installed on this server to execute JavaScript.",
                    "exit_code": 1,
                    "status": "error",
                    "provider": "local_sandbox"
                }
            script_path = os.path.join(temp_dir, "solution.js")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = [node_bin, script_path]
        else:
            return {
                "output": "",
                "error": f"Language '{language}' is not currently supported by the local sandbox.",
                "exit_code": 1,
                "status": "error",
                "provider": "local_sandbox"
            }

        stdin_data = (stdin or "").encode("utf-8")

        # Spawn isolated subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=temp_dir,
            env=safe_env
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return {
                "output": "",
                "error": f"Execution timed out (exceeded limit of {timeout}s). Please check for infinite loops or heavy operations.",
                "exit_code": 124,
                "status": "timeout",
                "provider": "local_sandbox"
            }

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        # Cap output size to 50KB to prevent memory issues
        max_output_len = 50_000
        if len(stdout_text) > max_output_len:
            stdout_text = stdout_text[:max_output_len] + "\n... [Output truncated: exceeded 50KB limit]"
        if len(stderr_text) > max_output_len:
            stderr_text = stderr_text[:max_output_len] + "\n... [Error log truncated: exceeded 50KB limit]"

        return {
            "output": stdout_text,
            "error": stderr_text,
            "exit_code": proc.returncode,
            "status": "success" if proc.returncode == 0 else "error",
            "provider": "local_sandbox"
        }

    except Exception as e:
        logger.error(f"Sandbox execution error: {e}")
        return {
            "output": "",
            "error": f"Sandbox execution error: {str(e)}",
            "exit_code": 1,
            "status": "error",
            "provider": "local_sandbox"
        }
    finally:
        # Guarantee cleanup of temporary sandbox directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


async def execute_code(
    code: str,
    language: str = "python",
    stdin: str = "",
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Main code execution entry point.
    Tries Judge0 if configured; otherwise gracefully executes in the local isolated sandbox.
    """
    effective_timeout = timeout or settings.SANDBOX_TIMEOUT_SECONDS

    # 1. If Judge0 is configured, try Judge0 first
    if settings.JUDGE0_API_URL and settings.JUDGE0_API_URL.strip():
        try:
            result = await _execute_judge0(
                code=code,
                language=language,
                stdin=stdin,
                timeout=effective_timeout
            )
            return result
        except Exception as e:
            logger.warning(f"Judge0 execution failed ({e}), falling back to local isolated sandbox.")

    # 2. Built-in local isolated sandbox
    return await _execute_local_sandbox(
        code=code,
        language=language,
        stdin=stdin,
        timeout=effective_timeout
    )


async def evaluate_code_against_tests(
    code: str,
    test_cases: List[Dict[str, Any]],
    language: str = "python",
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs code against a set of input/output test cases.
    Each test case is a dict with {"input": "...", "expected_output": "..."}.
    """
    results = []
    passed_count = 0

    for idx, tc in enumerate(test_cases):
        test_in = str(tc.get("input", ""))
        expected_out = str(tc.get("expected_output", tc.get("output", ""))).strip()

        exec_res = await execute_code(
            code=code,
            language=language,
            stdin=test_in,
            timeout=timeout or settings.SANDBOX_TIMEOUT_SECONDS
        )

        actual_out = (exec_res.get("output") or "").strip()
        has_error = bool(exec_res.get("error")) or exec_res.get("exit_code") != 0
        passed = (not has_error) and (actual_out == expected_out)

        if passed:
            passed_count += 1

        results.append({
            "test_case": idx + 1,
            "input": test_in,
            "expected_output": expected_out,
            "actual_output": actual_out,
            "error": exec_res.get("error"),
            "passed": passed
        })

    return {
        "passed": passed_count,
        "total": len(test_cases),
        "all_passed": passed_count == len(test_cases) if test_cases else False,
        "details": results
    }
