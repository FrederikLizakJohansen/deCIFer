import sys
import types

from conftest import REPO_ROOT, load_module_from_path


def load_collect_module(monkeypatch):
    fake_evaluation_module = types.ModuleType("decifer.evaluation")
    fake_evaluation_module.summarize_successful_evaluation_row = lambda row: {"ok": True}
    monkeypatch.setitem(sys.modules, "decifer.evaluation", fake_evaluation_module)

    return load_module_from_path(
        REPO_ROOT / "bin" / "collect_evaluations.py",
        "test_bin_collect_evaluations_module",
    )


def test_process_file_returns_none_for_invalid_pickle(monkeypatch, tmp_path):
    module = load_collect_module(monkeypatch)
    bad_file = tmp_path / "broken.pkl.gz"
    bad_file.write_text("not a gzip file")

    assert module.process_file(str(bad_file)) is None
