import gzip
import pickle
import types
from pathlib import Path

from conftest import REPO_ROOT, load_module_from_path


def load_monitor_module():
    return load_module_from_path(REPO_ROOT / "apps" / "local_monitor.py", "test_local_monitor_module")


def test_local_monitor_resolves_training_and_evaluation_run_dirs_from_settings(tmp_path):
    module = load_monitor_module()
    settings = {
        "train": {"out_dir": str(tmp_path / "demo-train")},
        "eval": {
            "model_ckpt": str(tmp_path / "demo-train" / "ckpt.pt"),
            "out_folder": str(tmp_path / "demo-eval"),
            "dataset_name": "val-preview",
            "model_name": "demo-model",
        },
    }

    assert module.resolve_training_run_dir_from_settings(settings).endswith("demo-train")
    assert module.resolve_evaluation_run_dir_from_settings(settings).endswith("demo-eval")


def test_local_monitor_persists_settings_and_generates_config_paths(tmp_path):
    module = load_monitor_module()

    app = module.MonitorApplication(
        train_config_path=None,
        eval_config_path=None,
        python_executable="python",
        state_path=str(tmp_path / "monitor-state.json"),
    )
    app.update_settings(
        {
            "train": {
                "dataset": "data/demo",
                "out_dir": str(tmp_path / "train-run"),
            },
            "eval": {
                "dataset_path": "data/demo",
                "out_folder": str(tmp_path / "eval-run"),
                "model_ckpt": str(tmp_path / "custom.pt"),
            },
        }
    )

    snapshot = app.snapshot()

    assert Path(app.state_path).exists()
    assert snapshot["paths"]["train_run_dir"].endswith("train-run")
    assert snapshot["paths"]["eval_run_dir"].endswith("eval-run")
    assert snapshot["paths"]["checkpoint_path"].endswith("custom.pt")


def test_local_monitor_builds_comparison_record_from_eval_file(tmp_path):
    module = load_monitor_module()

    eval_row = {
        "cif_name": "sample0",
        "rep": 0,
        "status": ["task", "syntax", "sensible", "statistics", "success"],
        "prompt_string": "data_sample0\n_chemical_formula_sum Ce1 O2\n",
        "prompt_flags": {
            "add_composition": True,
            "add_spacegroup": False,
            "condition": True,
        },
        "rmsd": 0.25,
        "validity": {
            "formula": True,
            "site_multiplicity": True,
            "bond_length": True,
            "spacegroup": False,
        },
        "spacegroup_sample": "P 1",
        "spacegroup": "P 1",
        "seq_len_sample": 12,
        "seq_len_gen": 14,
        "cif_string_sample": "data_sample0\n",
        "cif_string_completion_raw": "_generated\n",
        "cif_string_gen_raw": "data_sample0\n_generated\n",
        "cif_string_gen": "data_generated0\n",
        "xrd_overlay_ready": True,
        "xrd_error": None,
        "xrd_clean_sample": {
            "q": [0.0, 1.0, 2.0],
            "iq": [1.0, 0.5, 0.25],
        },
        "xrd_clean_gen": {
            "q": [0.0, 1.0, 2.0],
            "iq": [0.9, 0.45, 0.3],
        },
    }

    file_path = tmp_path / "sample0_0.pkl.gz"
    with gzip.open(file_path, "wb") as handle:
        pickle.dump(eval_row, handle)

    comparison = module.build_comparison_record(str(file_path))

    assert comparison["success"] is True
    assert comparison["cif_name"] == "sample0"
    assert comparison["rmsd"] == 0.25
    assert comparison["is_valid"] is False
    assert comparison["prompt_cif"].startswith("data_sample0")
    assert comparison["generated_completion_raw"] == "_generated\n"
    assert comparison["generated_cif_raw"].startswith("data_sample0")
    assert comparison["xrd_overlay_ready"] is True
    assert comparison["plot_data_uri"].startswith("data:image/png;base64,")


def test_local_monitor_choose_native_path_uses_tk_file_dialog(monkeypatch, tmp_path):
    module = load_monitor_module()
    selected_dir = tmp_path / "selected-dir"
    selected_file = tmp_path / "selected.pt"

    class FakeRoot:
        def withdraw(self):
            return None

        def attributes(self, *_args):
            return None

        def destroy(self):
            return None

    fake_tk = types.ModuleType("tkinter")
    fake_filedialog = types.ModuleType("tkinter.filedialog")

    def fake_tk_factory():
        return FakeRoot()

    def fake_askdirectory(initialdir=None, mustexist=False):
            assert initialdir is not None
            assert mustexist is False
            return str(selected_dir)

    def fake_askopenfilename(initialdir=None):
        assert initialdir is not None
        return str(selected_file)

    fake_tk.Tk = fake_tk_factory
    fake_tk.filedialog = fake_filedialog
    fake_filedialog.askdirectory = fake_askdirectory
    fake_filedialog.askopenfilename = fake_askopenfilename

    monkeypatch.setitem(module.sys.modules, "tkinter", fake_tk)
    monkeypatch.setitem(module.sys.modules, "tkinter.filedialog", fake_filedialog)

    assert module.choose_native_path("dir", str(tmp_path)) == str(selected_dir.resolve())
    assert module.choose_native_path("file", str(tmp_path / "current.pt")) == str(selected_file.resolve())


def test_local_monitor_ui_saves_form_before_running_actions():
    module = load_monitor_module()

    assert 'onclick="startTraining()"' in module.INDEX_HTML
    assert 'onclick="startEvaluation()"' in module.INDEX_HTML
    assert "attachAutoSaveHandlers();" in module.INDEX_HTML
    assert "if (!formDirty) fillSettingsForm(state.settings);" in module.INDEX_HTML
    assert "saveSettings({refreshAfter: false, successMessage: \"Settings saved. Starting training...\"})" in module.INDEX_HTML
    assert "if (signature === lastComparisonsSignature) return;" in module.INDEX_HTML
    assert "<strong>Model Completion</strong>" in module.INDEX_HTML
    assert "<strong>Generated Full Sequence</strong>" in module.INDEX_HTML


def test_local_monitor_reports_xrd_overlay_error_when_arrays_missing(tmp_path):
    module = load_monitor_module()

    eval_row = {
        "cif_name": "sample1",
        "rep": 0,
        "status": ["task", "syntax", "sensible", "statistics", "success"],
        "cif_string_sample": "data_sample1\n",
        "cif_string_gen": "data_generated1\n",
        "xrd_overlay_ready": False,
        "xrd_error": "XRD generation returned no plottable arrays.",
        "xrd_clean_sample": None,
        "xrd_clean_gen": None,
    }

    file_path = tmp_path / "sample1_0.pkl.gz"
    with gzip.open(file_path, "wb") as handle:
        pickle.dump(eval_row, handle)

    comparison = module.build_comparison_record(str(file_path))

    assert comparison["plot_data_uri"] is None
    assert comparison["plot_error"] == "XRD generation returned no plottable arrays."
