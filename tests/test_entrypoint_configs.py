import sys
import types

from conftest import REPO_ROOT, load_module_from_path


def load_train_module(monkeypatch):
    fake_model_module = types.ModuleType("decifer.decifer_model")
    fake_model_module.Decifer = object
    fake_model_module.DeciferConfig = object
    fake_dataset_module = types.ModuleType("decifer.decifer_dataset")
    fake_dataset_module.DeciferDataset = object
    fake_utility_module = types.ModuleType("decifer.utility")
    fake_utility_module.discrete_to_continuous_xrd = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "decifer.decifer_model", fake_model_module)
    monkeypatch.setitem(sys.modules, "decifer.decifer_dataset", fake_dataset_module)
    monkeypatch.setitem(sys.modules, "decifer.utility", fake_utility_module)

    return load_module_from_path(REPO_ROOT / "bin" / "train.py", "test_bin_train_module")


def load_run_protocol_module(monkeypatch):
    sys.modules.pop("decifer.workflows.run_protocol", None)
    fake_pipeline_module = types.ModuleType("decifer.experimental")
    fake_pipeline_module.DeciferPipeline = object
    monkeypatch.setitem(sys.modules, "decifer.experimental", fake_pipeline_module)

    return load_module_from_path(REPO_ROOT / "bin" / "run_protocol.py", "test_bin_run_protocol_module")


def load_experimental_pipeline_module(monkeypatch):
    sys.modules.pop("decifer.experimental", None)
    fake_pipeline_module = types.ModuleType("decifer.experimental")
    fake_pipeline_module.DeciferPipeline = object
    monkeypatch.setitem(sys.modules, "decifer.experimental", fake_pipeline_module)

    return load_module_from_path(
        REPO_ROOT / "bin" / "experimental_pipeline.py",
        "test_bin_experimental_pipeline_module",
    )


def test_train_parse_config_uses_shared_loader(monkeypatch, tmp_path):
    module = load_train_module(monkeypatch)
    config_path = tmp_path / "train.yaml"
    config_path.write_text("dataset: data/root\nout_dir: train_out\n")

    config = module.parse_config(["--config", str(config_path)])

    assert config.dataset == "data/root"
    assert config.out_dir == "train_out"


def test_run_protocol_parse_config_merges_yaml_and_cli(monkeypatch, tmp_path):
    module = load_run_protocol_module(monkeypatch)
    config_path = tmp_path / "protocol.yaml"
    config_path.write_text("model_path: ckpt.pt\nzip_path: scans.zip\nn_trials: 5\n")

    config = module.parse_config(
        ["--config", str(config_path), "--n-trials", "9", "--suffix", "smoke"]
    )

    assert config.model_path == "ckpt.pt"
    assert config.zip_path == "scans.zip"
    assert config.n_trials == 9
    assert config.suffix == "smoke"


def test_experimental_pipeline_wrapper_reexports_workflow_pipeline(monkeypatch):
    module = load_experimental_pipeline_module(monkeypatch)

    assert module.DeciferPipeline is object
