import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DatasetDownloadShellContractTests(unittest.TestCase):
    def test_downloader_uses_official_sources_and_validates_full_datasets(self):
        script = (REPOSITORY_ROOT / "download_halueval_boolq.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("b7253db3cdaa0ab2c382f92b26b390109174f77e", script)
        self.assertIn("storage.googleapis.com/boolq/train.jsonl", script)
        self.assertIn("storage.googleapis.com/boolq/dev.jsonl", script)
        self.assertIn('"halueval_qa": 10000', script)
        self.assertIn('"boolq_train": 9427', script)
        self.assertIn('"boolq_dev": 3270', script)
        self.assertIn("dataset_manifest.json", script)
        self.assertIn("sha256", script)
        self.assertIn("os.replace", script)


class FeaturePilotShellContractTests(unittest.TestCase):
    def test_launcher_checks_gpu_and_runs_datasets_sequentially_without_training(self):
        script = (
            REPOSITORY_ROOT / "run_halueval_boolq_feature_pilots.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("nvidia-smi", script)
        self.assertIn("GPU_WITNESS", script)
        self.assertIn("run_dataset halueval_qa", script)
        self.assertIn("run_dataset boolq", script)
        self.assertIn("RUN_TRAINING=0", script)
        self.assertNotIn("&\n", script)
        self.assertIn("download_halueval_boolq.sh", script)
        self.assertIn("flock -n", script)
        self.assertIn("PYTHONUNBUFFERED=1", script)
        self.assertIn("OFFLINE_ONLY=1", script)
        self.assertIn('POSTPROCESS_DEVICE="${POSTPROCESS_DEVICE:-auto}"', script)
        self.assertIn('RETAIN_DENSE_ATTENTION="${RETAIN_DENSE_ATTENTION:-0}"', script)
        self.assertIn('OMP_NUM_THREADS="${CPU_THREADS}"', script)

    def test_dataset_pilot_forwards_device_and_dtype_to_both_model_loads(self):
        script = (
            REPOSITORY_ROOT / "run_unsupervised_token_graph_pilot.sh"
        ).read_text(encoding="utf-8")

        self.assertGreaterEqual(script.count('--device "${DEVICE}"'), 2)
        self.assertGreaterEqual(script.count('--dtype "${DTYPE}"'), 2)
        self.assertIn('--postprocess-device "${POSTPROCESS_DEVICE}"', script)
        self.assertIn("--discard-dense-attention", script)
        self.assertIn(
            'if [[ "${RETAIN_DENSE_ATTENTION}" == "0" ]]', script
        )
        self.assertIn(
            'EXTRACTION_STORAGE_ARGS+=(--discard-dense-attention)', script
        )
        self.assertIn('"${EXTRACTION_STORAGE_ARGS[@]}"', script)


if __name__ == "__main__":
    unittest.main()
