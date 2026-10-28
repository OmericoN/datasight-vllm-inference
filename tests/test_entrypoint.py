import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASH = next((str(p) for p in (
    Path("C:/Program Files/Git/bin/bash.exe"), Path("C:/msys64/usr/bin/bash.exe")
) if p.exists()), shutil.which("bash"))


@unittest.skipUnless(BASH, "Bash is needed for entrypoint checks")
class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.capture = self.directory / "capture"
        stub_dir = self.directory / "bin"
        stub_dir.mkdir()
        for name, content in {
            "vllm": 'printf "%s\\n" "$@" > "$CAPTURE_PATH"',
            "python": 'cat > "${CAPTURE_PATH}.stdin"; printf "%s\\n" "$VLLM_MODEL" "$VLLM_REVISION" "$HF_HUB_CACHE" > "$CAPTURE_PATH"',
            "tail": 'echo hold > "$CAPTURE_PATH"',
        }.items():
            stub = stub_dir / name
            stub.write_text("#!/usr/bin/env bash\n" + content + "\n", encoding="utf-8", newline="\n")
            stub.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "HF_", "HUGGINGFACE_", "TRANSFORMERS_", "SERVED_MODEL_"))}
        self.env.update({
            "PATH": stub_dir.as_posix() + (";" if os.name == "nt" else ":") + os.environ["PATH"],
            "HF_HOME": (self.directory / "cache").as_posix(),
            "HOME": self.directory.as_posix(),
            "PYTHONPYCACHEPREFIX": (self.directory / "pycache").as_posix(),
            "CAPTURE_PATH": self.capture.as_posix(),
            "STUB_DIR": stub_dir.as_posix(),
        })
        for key in ("XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            self.env.pop(key, None)

    def invoke(self, mode="serve", script=None):
        self.env["VLLM_START_MODE"] = mode
        stub_path = '$(cygpath -u "$STUB_DIR")' if os.name == "nt" else "$STUB_DIR"
        return subprocess.run(
            [BASH, "-c", f'export PATH="{stub_path}:$PATH"; exec bash "$1"', "test", (script or ROOT / "entrypoint-vllm.sh").as_posix()],
            env=self.env, capture_output=True, text=True, timeout=10,
        )

    def test_serving_arguments_and_json_boundary(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertEqual(args[:2], ["serve", "Qwen/Qwen3.8-27B"])
        for flag, expected in {
            "--revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "--tokenizer-revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "--dtype": "bfloat16", "--max-model-len": "16384",
            "--max-num-seqs": "1", "--tensor-parallel-size": "1",
            "--download-dir": self.env["HF_HOME"] + "/hub",
        }.items():
            self.assertEqual(args[args.index(flag) + 1], expected)
        self.assertEqual(json.loads(args[args.index("--default-chat-template-kwargs") + 1]), {"enable_thinking": False})

    def test_download_uses_same_revision_and_cache_without_vllm(self):
        result = self.invoke("download")
        self.assertEqual(result.returncode, 0, result.stderr)
        model, revision, cache = self.capture.read_text().splitlines()
        hub = ModuleType("huggingface_hub")
        calls = []
        hub.snapshot_download = lambda **kwargs: calls.append(kwargs) or "snapshot"
        with patch.dict("sys.modules", {"huggingface_hub": hub, "vllm": None, "torch": None}), patch.dict(os.environ, {
            "VLLM_MODEL": model, "VLLM_REVISION": revision, "HF_HUB_CACHE": cache,
        }), contextlib.redirect_stdout(io.StringIO()):
            exec(Path(str(self.capture) + ".stdin").read_text(), {})
        self.assertEqual(calls, [{"repo_id": model, "revision": revision, "cache_dir": self.env["HF_HOME"] + "/hub"}])

    def test_hold_does_not_start_python_or_vllm(self):
        result = self.invoke("hold")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.capture.read_text().strip(), "hold")
        self.assertFalse(Path(str(self.capture) + ".stdin").exists())

    def test_invalid_mode_fails_before_startup(self):
        result = self.invoke("invalid")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture.exists())

    def test_unwritable_cache_fails_without_temporary_fallback(self):
        blocked = self.directory / "blocked"
        blocked.write_text("not a directory")
        self.env["HF_HOME"] = blocked.as_posix()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Fix the PVC", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_image_assembly_preserves_dockerfile_environment_and_file_modes(self):
        source = self.directory / "source"
        source.mkdir()
        for name in ("Dockerfile", "entrypoint-vllm.sh"):
            shutil.copyfile(ROOT / name, source / name)
        account = self.directory / "account"
        account.mkdir()
        (account / "token").write_text("fake-test-token")
        (account / "service-ca.crt").write_text("fake-test-ca")
        posix_directory = self.directory.as_posix()
        if os.name == "nt":
            posix_directory = subprocess.check_output([BASH, "-c", 'cygpath -u "$1"', "test", posix_directory], text=True).strip()
        script = (ROOT / "scripts/build-image.sh").read_text()
        script = script.replace("/tmp", posix_directory + "/work").replace("/source", posix_directory + "/source")
        script = script.replace("/var/run/secrets/kubernetes.io/serviceaccount", posix_directory + "/account")
        script = script.replace("/etc/ssl/certs/ca-certificates.crt", posix_directory + "/account/service-ca.crt")
        script_path = self.directory / "build.sh"
        script_path.write_text(script, encoding="utf-8", newline="\n")
        stub = self.directory / "bin/crane"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CAPTURE_PATH"\nif [[ "$1" == auth ]]; then cat > /dev/null; fi\n', encoding="utf-8", newline="\n")
        stub.chmod(0o755)
        self.env["NAMESPACE"] = "test-namespace"
        result = self.invoke(script=script_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertIn("docker.io/vllm/vllm-openai:v0.32.0", args)
        self.assertIn("image-registry.openshift-image-registry.svc:5000/test-namespace/datasight-vllm-gpu:qwen3.8-27b-v2", args)
        self.assertNotIn("fake-test-token", self.capture.read_text() + result.stdout + result.stderr)
        for line in (ROOT / "Dockerfile").read_text().splitlines():
            if line.startswith("ENV "):
                self.assertIn(line[4:], args)
        with tarfile.open(self.directory / "work/layer.tar") as layer:
            self.assertEqual(layer.getmember("./entrypoint-vllm.sh").mode, 0o755)
            if os.name != "nt":  # Git Bash cannot represent Unix directory write bits on NTFS.
                self.assertEqual(layer.getmember("./hf-cache").mode, 0o777)


if __name__ == "__main__":
    unittest.main()
