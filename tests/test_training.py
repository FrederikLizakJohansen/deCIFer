import sys
import types

from conftest import REPO_ROOT, load_module_from_path


def load_training_runtime_module(monkeypatch):
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

    return load_module_from_path(REPO_ROOT / "decifer" / "training.py", "test_decifer_training_module")


def test_move_batch_tensor_skips_pin_memory_on_cpu(monkeypatch):
    module = load_training_runtime_module(monkeypatch)
    calls = []

    class FakeTensor:
        def pin_memory(self):
            calls.append(("pin_memory",))
            return self

        def to(self, device, non_blocking=False):
            calls.append(("to", device, non_blocking))
            return {"device": device, "non_blocking": non_blocking}

    result = module.move_batch_tensor(FakeTensor(), "cpu")

    assert result == {"device": "cpu", "non_blocking": False}
    assert calls == [("to", "cpu", False)]


def test_move_batch_tensor_uses_pin_memory_on_cuda(monkeypatch):
    module = load_training_runtime_module(monkeypatch)
    calls = []

    class FakeTensor:
        def pin_memory(self):
            calls.append(("pin_memory",))
            return self

        def to(self, device, non_blocking=False):
            calls.append(("to", device, non_blocking))
            return {"device": device, "non_blocking": non_blocking}

    result = module.move_batch_tensor(FakeTensor(), "cuda:0")

    assert result == {"device": "cuda:0", "non_blocking": True}
    assert calls == [("pin_memory",), ("to", "cuda:0", True)]


def test_build_training_block_batch_returns_none_when_tokens_do_not_fill_a_block(monkeypatch):
    module = load_training_runtime_module(monkeypatch)
    short_sequence = [module.torch.tensor([module.START_ID, 11, 12, 13], dtype=module.torch.long)]

    batch = module.build_training_block_batch(
        sequences=short_sequence,
        cond_sequences=[],
        block_size=8,
        batch_size=1,
        condition=False,
    )

    assert batch is None


def test_build_training_block_batch_skips_startless_blocks_and_keeps_valid_start_indices(monkeypatch):
    module = load_training_runtime_module(monkeypatch)
    start = module.START_ID
    sequences = [
        module.torch.tensor([start, 21, 22, 23, 24, 25], dtype=module.torch.long),
        module.torch.tensor([26, 27, 28, 29, start, 31], dtype=module.torch.long),
    ]

    batch = module.build_training_block_batch(
        sequences=sequences,
        cond_sequences=[],
        block_size=6,
        batch_size=2,
        condition=False,
    )

    assert batch is not None
    x_batch, y_batch, cond_batch, start_indices_list = batch
    assert tuple(x_batch.shape) == (2, 5)
    assert tuple(y_batch.shape) == (2, 5)
    assert cond_batch is None
    assert [indices.tolist() for indices in start_indices_list] == [[0], [4]]
