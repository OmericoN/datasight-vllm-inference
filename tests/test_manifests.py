from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_image_assembly_job_uses_existing_builder_account_and_no_gpu(self):
        build = yaml.safe_load((ROOT / "dsri/image-build-job.yaml").read_text())["spec"]
        self.assertEqual(build["backoffLimit"], 0)
        self.assertEqual(build["ttlSecondsAfterFinished"], 86400)
        self.assertEqual(build["template"]["spec"]["serviceAccountName"], "builder")
        container = build["template"]["spec"]["containers"][0]
        self.assertEqual(container["image"], "gcr.io/go-containerregistry/crane/debug:v0.22.1")
        for kind in ("requests", "limits"):
            self.assertEqual(set(container["resources"][kind]), {"cpu", "memory", "ephemeral-storage"})

    def test_deployment_and_download_share_persistent_model_without_extra_gpu(self):
        deployment = yaml.safe_load((ROOT / "dsri/deployment.yaml").read_text())
        job = yaml.safe_load((ROOT / "dsri/download-job.yaml").read_text())
        service = yaml.safe_load((ROOT / "dsri/service.yaml").read_text())
        pvc = yaml.safe_load((ROOT / "dsri/pvc.yaml").read_text())
        self.assertEqual(deployment["spec"]["replicas"], 0)
        self.assertEqual(deployment["spec"]["strategy"]["type"], "Recreate")
        self.assertGreater(deployment["spec"]["progressDeadlineSeconds"], 1830)
        gpu = deployment["spec"]["template"]["spec"]["containers"][0]
        download = job["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(gpu["image"], download["image"])
        self.assertTrue(gpu["image"].startswith("image-registry.openshift-image-registry.svc:5000/ub-ub-datasight/"))
        self.assertNotIn(":latest", gpu["image"])
        envs = [{item["name"]: item["value"] for item in c["env"]} for c in (gpu, download)]
        for key in ("VLLM_MODEL", "VLLM_REVISION", "HF_HOME"):
            self.assertEqual(envs[0][key], envs[1][key])
        self.assertEqual(envs[0]["VLLM_MODEL"], "Qwen/Qwen3.8-27B")
        self.assertEqual(envs[1]["VLLM_START_MODE"], "download")
        for controller, container in ((deployment, gpu), (job, download)):
            spec = controller["spec"]["template"]["spec"]
            claim = next(v for v in spec["volumes"] if v["name"] == "hf-cache")
            self.assertEqual(claim["persistentVolumeClaim"]["claimName"], pvc["metadata"]["name"])
            mount = next(v for v in container["volumeMounts"] if v["name"] == "hf-cache")
            self.assertEqual(mount["mountPath"], envs[0]["HF_HOME"])
            for kind in ("requests", "limits"):
                resources = container["resources"][kind]
                self.assertTrue({"cpu", "memory", "ephemeral-storage"} <= resources.keys())
                self.assertEqual(str(resources.get("nvidia.com/gpu", 0)), "1" if container is gpu else "0")
        self.assertEqual(job["spec"]["template"]["spec"]["restartPolicy"], "Never")
        self.assertEqual(job["spec"]["backoffLimit"], 1)
        self.assertEqual(job["spec"]["activeDeadlineSeconds"], 7200)
        self.assertEqual(job["spec"]["ttlSecondsAfterFinished"], 86400)
        self.assertEqual(service["spec"]["selector"], deployment["spec"]["selector"]["matchLabels"])
        self.assertEqual(service["spec"]["ports"][0]["targetPort"], gpu["ports"][0]["containerPort"])
        for path in (ROOT / "dsri").glob("*.yaml"):
            self.assertNotEqual(yaml.safe_load(path.read_text())["kind"], "Route")


if __name__ == "__main__":
    unittest.main()
