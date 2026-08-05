from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "run_ragtruth_cept_pilot.sh"


def _runner_source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_installs_cept_dependencies_only_inside_a_dedicated_venv():
    source = _runner_source()

    assert 'CEPT_VENV="${CEPT_VENV:-' in source
    assert '-m venv --system-site-packages "${CEPT_VENV}"' in source
    assert 'CEPT_PYTHON="${CEPT_VENV}/bin/python"' in source
    assert '"${CEPT_PYTHON}" -m pip install' in source
    assert '"${PYTHON_BIN}" -m pip install' not in source


def test_runner_reuses_cuda_torch_but_requires_the_exact_transformers_runtime():
    source = _runner_source()

    assert 'transformers.__version__ != "4.52.3"' in source
    assert "transformers_path.relative_to(venv_prefix)" in source
    assert "--ignore-installed --no-deps transformers==4.52.3" in source
    assert "torch.cuda.is_available()" in source
    assert "sys.prefix == sys.base_prefix" in source
    assert "torch.__file__" in source
    assert "transformers.__file__" in source


def test_runner_preserves_existing_overrides_and_adds_a_venv_override():
    source = _runner_source()

    for variable in (
        "PYTHON_BIN",
        "CEPT_VENV",
        "DATA_ROOT",
        "DATASET_DIR",
        "MODEL_PATH",
        "FEATURE_ROOT",
        "SEED",
        "NUM_SAMPLES",
        "MAX_SEQUENCE_TOKENS",
        "MAX_COUNTERFACTUALS",
        "HISTORY_BLOCK_SIZE",
        "MAX_HISTORY_BLOCKS",
        "GPU_ID",
        "DEVICE",
        "DTYPE",
        "RUN_TAG",
        "OUTPUT_DIR",
        "INSTALL_DEPS",
        "RESUME",
    ):
        assert f"${{{variable}:-" in source


def test_runner_uses_atomic_new_output_and_verified_resume_mode():
    source = _runner_source()

    assert 'if [[ "${RESUME}" == "1" ]]' in source
    assert '"${OUTPUT_DIR}/manifest.json"' in source
    assert "RESUME_ARGS=(--resume)" in source
    assert 'if ! mkdir "${OUTPUT_DIR}"' in source
    assert 'tee -a "${OUTPUT_DIR}/run.log"' in source
    assert 'flock -n 8' in source


def test_runner_serializes_venv_install_and_binds_its_base_identity():
    source = _runner_source()

    assert 'flock -x 9' in source
    assert 'CEPT_VENV}.bootstrap.lock' in source
    assert '.cept-base-identity.json' in source
    assert '"python": str(Path(sys.executable).resolve())' in source
    assert '"torch_version": torch.__version__' in source
    assert '"torch_cuda_version": torch.version.cuda' in source


def test_readme_documents_one_command_and_isolated_runtime_override():
    source = (ROOT / "counterfactual_grounding" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "bash ./run_ragtruth_cept_pilot.sh" in source
    assert "CEPT_VENV" in source
    assert "--system-site-packages" in source
    assert "does not modify" in source.casefold()
