"""Test CLI invocation."""

import subprocess
import sys
import json


def test_cli_execution_via_module():
    cmd = [
        sys.executable,
        "-m",
        "arn.cli",
        "turn off the study room heater",
        "--json"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"CLI exited with error: {res.stderr}"

    # Parse stdout as valid JSON
    output_json = json.loads(res.stdout)
    assert isinstance(output_json, dict)
    assert "tools" in output_json
    assert "arguments" in output_json
