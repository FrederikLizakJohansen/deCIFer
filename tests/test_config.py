import pytest

from decifer.config import (
    AblationConfig,
    EvaluateConfig,
    RunProtocolConfig,
    TrainWorkflowConfig,
    load_dataclass_config,
)


def test_load_dataclass_config_merges_yaml_and_cli_overrides(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_ckpt: from_yaml.pt",
                "dataset_path: from_yaml.h5",
                "add_composition: true",
                "num_reps: 4",
            ]
        )
    )

    config = load_dataclass_config(
        EvaluateConfig,
        config_path=str(config_path),
        overrides={
            "num_reps": 2,
            "add_composition": None,
            "temperature": 0.7,
        },
    )

    assert config.model_ckpt == "from_yaml.pt"
    assert config.dataset_path == "from_yaml.h5"
    assert config.add_composition is True
    assert config.num_reps == 2
    assert config.temperature == 0.7


def test_load_dataclass_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("unknown_key: 1\n")

    with pytest.raises(ValueError, match="Unknown config keys"):
        load_dataclass_config(AblationConfig, config_path=str(config_path))


def test_train_and_protocol_configs_load_from_yaml(tmp_path):
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text(
        "\n".join(
            [
                "dataset: data/root",
                "out_dir: model_out",
                "condition: true",
            ]
        )
    )
    train_config = load_dataclass_config(TrainWorkflowConfig, config_path=str(train_config_path))
    assert train_config.dataset == "data/root"
    assert train_config.out_dir == "model_out"
    assert train_config.condition is True

    protocol_config_path = tmp_path / "protocol.yaml"
    protocol_config_path.write_text(
        "\n".join(
            [
                "model_path: ckpt.pt",
                "zip_path: scans.zip",
                "n_trials: 7",
            ]
        )
    )
    protocol_config = load_dataclass_config(RunProtocolConfig, config_path=str(protocol_config_path))
    assert protocol_config.model_path == "ckpt.pt"
    assert protocol_config.zip_path == "scans.zip"
    assert protocol_config.n_trials == 7
