import queue
import sys
import types

import torch

from conftest import REPO_ROOT, load_module_from_path
from decifer.tokenizer import Tokenizer


class FakeDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class FakeModel:
    def __init__(self, token_ids):
        self.device = "cpu"
        self.token_ids = token_ids
        self.calls = 0
        self.config = types.SimpleNamespace(condition=True)

    def generate(
        self,
        idx,
        max_new_tokens,
        cond_vec=None,
        start_indices_batch=None,
        temperature=1.0,
        top_k=None,
        disable_pbar=False,
    ):
        self.calls += 1
        return self.token_ids.clone()


def load_evaluate_module(monkeypatch):
    fake_model_module = types.ModuleType("decifer.decifer_model")
    fake_model_module.Decifer = object
    fake_model_module.DeciferConfig = object

    fake_dataset_module = types.ModuleType("decifer.decifer_dataset")
    fake_dataset_module.DeciferDataset = object

    fake_utility_module = types.ModuleType("decifer.utility")
    fake_utility_module.get_rmsd = lambda *args, **kwargs: None
    fake_utility_module.replace_symmetry_loop_with_P1 = lambda cif: cif
    fake_utility_module.extract_space_group_symbol = lambda cif: "P 1"
    fake_utility_module.reinstate_symmetry_loop = lambda cif, sg: cif
    fake_utility_module.is_sensible = lambda cif: True
    fake_utility_module.extract_numeric_property = lambda *args, **kwargs: None
    fake_utility_module.get_unit_cell_volume = lambda *args, **kwargs: None
    fake_utility_module.extract_volume = lambda *args, **kwargs: None
    fake_utility_module.is_space_group_consistent = lambda cif: True
    fake_utility_module.is_atom_site_multiplicity_consistent = lambda cif: True
    fake_utility_module.is_formula_consistent = lambda cif: True
    fake_utility_module.bond_length_reasonableness_score = lambda cif: 1.0
    fake_utility_module.extract_species = lambda cif: []
    fake_utility_module.space_group_symbol_to_number = lambda symbol: 1

    def fake_discrete_to_continuous_xrd(batch_q, batch_iq, **kwargs):
        return {
            "q": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "iq": torch.tensor([[1.0, 0.5]], dtype=torch.float32),
        }

    fake_utility_module.discrete_to_continuous_xrd = fake_discrete_to_continuous_xrd
    fake_utility_module.generate_continuous_xrd_from_cif = lambda *args, **kwargs: None

    fake_train_module = types.ModuleType("bin.train")
    fake_train_module.TrainConfig = object

    monkeypatch.setitem(sys.modules, "decifer.decifer_model", fake_model_module)
    monkeypatch.setitem(sys.modules, "decifer.decifer_dataset", fake_dataset_module)
    monkeypatch.setitem(sys.modules, "decifer.utility", fake_utility_module)
    monkeypatch.setitem(sys.modules, "bin.train", fake_train_module)

    return load_module_from_path(
        REPO_ROOT / "bin" / "evaluate.py",
        "test_bin_evaluate_module",
    )


def test_process_dataset_counts_requested_and_submitted_tasks(monkeypatch, tmp_path):
    module = load_evaluate_module(monkeypatch)
    tokenizer = Tokenizer()
    start_id = tokenizer.token_to_id["data_"]
    newline_id = tokenizer.token_to_id["\n"]

    fake_items = [
        {
            "cif_name": "sample0.cif",
            "cif_tokens": torch.tensor([start_id, newline_id], dtype=torch.long),
            "xrd.q": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "xrd.iq": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "cif_string": "data_sample0\n",
            "spacegroup": "P 1",
        },
        {
            "cif_name": "sample1.cif",
            "cif_tokens": torch.tensor([start_id, newline_id], dtype=torch.long),
            "xrd.q": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "xrd.iq": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "cif_string": "data_sample1\n",
            "spacegroup": "P 1",
        },
    ]

    module.DeciferDataset = lambda path, keys: FakeDataset(fake_items)

    generated = torch.tensor([[start_id, newline_id, newline_id]], dtype=torch.long)
    model = FakeModel(generated)
    task_queue = queue.Queue()

    existing_file = tmp_path / "sample0_1.pkl.gz"
    existing_file.write_text("done")

    requested, submitted = module.process_dataset(
        dataset_path="ignored.h5",
        dataset_name="test-dataset",
        model=model,
        model_name="fake-model",
        input_queue=task_queue,
        eval_files_dir=str(tmp_path),
        num_workers=0,
        override=False,
        num_repetitions=3,
        add_composition=False,
        add_spacegroup=False,
        max_new_tokens=5,
        debug_max=2,
        debug=False,
    )

    queued_tasks = []
    while not task_queue.empty():
        queued_tasks.append(task_queue.get())

    assert requested == 6
    assert submitted == 5
    assert model.calls == 5
    assert len(queued_tasks) == 5
    assert all(task["prompt_string"].startswith("data_") for task in queued_tasks)
    assert all(task["prompt_token_count"] >= 1 for task in queued_tasks)
    assert all(task["prompt_flags"]["add_composition"] is False for task in queued_tasks)
    assert {(task["cif_name"], task["rep"]) for task in queued_tasks} == {
        ("sample0", 0),
        ("sample0", 2),
        ("sample1", 0),
        ("sample1", 1),
        ("sample1", 2),
    }


def test_parse_config_merges_yaml_with_cli_overrides(monkeypatch, tmp_path):
    module = load_evaluate_module(monkeypatch)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_ckpt: checkpoint.pt",
                "dataset_path: dataset.h5",
                "add_composition: true",
                "num_reps: 5",
            ]
        )
    )

    config = module.parse_config(
        [
            "--config",
            str(config_path),
            "--num-reps",
            "2",
            "--temperature",
            "0.6",
        ]
    )

    assert config.model_ckpt == "checkpoint.pt"
    assert config.dataset_path == "dataset.h5"
    assert config.add_composition is True
    assert config.num_reps == 2
    assert config.temperature == 0.6
