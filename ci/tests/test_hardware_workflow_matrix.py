"""Execute workflow matrix generation without allocating GPU runners."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def run_step(step, env):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("ci", "scripts", ".devcontainer", "requirements.txt", "pyproject.toml"):
            (root / name).symlink_to(ROOT / name)
        output = root / "output"
        subprocess.run(
            ["bash", "-euo", "pipefail", "-c", step["run"]],
            cwd=root,
            env={**os.environ, **env, "GITHUB_OUTPUT": str(output)},
            check=True,
            capture_output=True,
            text=True,
        )
        return dict(line.split("=", 1) for line in output.read_text().splitlines())


class HardwareWorkflowMatrixTest(unittest.TestCase):
    def pr_matrix(self, server, **overrides):
        step = next(s for s in workflow("pr-test.yml")["jobs"]["setup"]["steps"] if s.get("id") == "matrix")
        config = json.loads((ROOT / "ci/cuda-versions.json").read_text())
        env = {
            "CI_SERVER_URL": server,
            "EVENT_NAME": "schedule",
            "H100_ON_PRS": "false",
            "SKIP_AOT": "false",
            "SKIP_GPU": "false",
            "REQUESTED_TEST_PATH": "",
            "RUNTIME_CUDA_MATRIX": json.dumps(config["runtime"]),
            **overrides,
        }
        result = run_step(step, env)
        rows = json.loads(result["matrix"])["include"]
        self.assertEqual(int(result["count"]), len(rows))
        self.assertFalse(any(set(row["runner"]) & {"sm75", "sm86"} for row in rows))
        return rows

    def test_pr_gitea_keeps_sm120_and_both_cpu_architectures(self):
        rows = self.pr_matrix("https://gitea.example")
        self.assertEqual(sum("sm120" in row["runner"] for row in rows), 5)
        self.assertTrue(any(row["kind"] == "aot" for row in rows))
        self.assertTrue(any("arm64" in row["runner"] for row in rows))
        self.assertTrue(any("x64" in row["runner"] and row["kind"] == "aot" for row in rows))
        self.assertFalse(any("h100" in row["runner"] for row in rows))

    def test_github_keeps_arm64_and_h100(self):
        rows = self.pr_matrix("https://github.com")
        self.assertTrue(any("arm64" in row["runner"] for row in rows))
        self.assertTrue(any("h100" in row["runner"] for row in rows))
        self.assertEqual(sum("sm120" in row["runner"] for row in rows), 5)

    def test_pr_skip_inputs_still_apply(self):
        rows = self.pr_matrix("https://gitea.example", SKIP_AOT="true")
        self.assertTrue(all(row["kind"] == "gpu" for row in rows))
        rows = self.pr_matrix("https://gitea.example", SKIP_GPU="true")
        self.assertTrue(all(row["kind"] == "aot" for row in rows))

    def test_nightly_keeps_arm_on_both_platforms(self):
        jobs = workflow("nightly-release.yml")["jobs"]
        step = next(s for s in jobs["setup"]["steps"] if s.get("id") == "cuda-versions")
        gitea = run_step(step, {"CI_SERVER_URL": "https://gitea.example"})
        github = run_step(step, {"CI_SERVER_URL": "https://github.com"})
        for kind in ("providers", "shims"):
            all_rows = json.loads(github[kind])
            rows = json.loads(gitea[kind])
            self.assertTrue(any(row["cpu_architecture"] == "aarch64" for row in all_rows))
            self.assertEqual(rows, all_rows)
        self.assertIn("sm120", jobs["test-nightly-build"]["runs-on"])


if __name__ == "__main__":
    unittest.main()
