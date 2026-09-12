import json

from scripts.build_cross_phase_results import _collect_rows, _to_details_row, _to_results_row


def _summary(model_name, epochs, fp32_top1, **extra):
    return {"model_name": model_name, "epochs": epochs, "fp32_top1": fp32_top1, **extra}


def test_collect_rows_dedupes_same_phase_model_keeping_more_epochs(tmp_path, monkeypatch):
    import scripts.build_cross_phase_results as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "PHASE_DIRS", [("Phase 1 — Baselines", "phase_1_baseline")])
    monkeypatch.setattr(mod, "PHASE_8_CSV", tmp_path / "nonexistent.csv")

    phase_dir = tmp_path / "phase_1_baseline"
    phase_dir.mkdir()
    (phase_dir / "alexnet_tv_summary.json").write_text(json.dumps(_summary("alexnet_tv", 79, 32.89)))
    # a stray second file for the same (phase, model) with fewer epochs must lose the dedupe
    (phase_dir / "alexnet_tv_rerun_summary.json").write_text(json.dumps(_summary("alexnet_tv", 20, 10.0)))
    # missing required field -> skipped, not crashed on
    (phase_dir / "broken_summary.json").write_text(json.dumps({"model_name": "broken"}))
    # different schema -> skipped
    (phase_dir / "experiment_summary.json").write_text(json.dumps({"anything": True}))
    (phase_dir / "alexnet_x_compression_summary.json").write_text(json.dumps({"anything": True}))

    rows = _collect_rows()
    assert list(rows.keys()) == [("Phase 1 — Baselines", "alexnet_tv")]
    assert rows[("Phase 1 — Baselines", "alexnet_tv")]["epochs"] == 79


def test_row_mappers_produce_the_documented_schema():
    s = _summary(
        "alexnet_tv", 79, 32.89, fp32_top5=58.2, int8_top5=50.0, int8_top1=30.0,
        quantization_drop_top1=2.89, fp32_loss=1.1, int8_loss=1.5, params_m=57.8, fp32_size_mb=220.6,
    )
    results_row = _to_results_row("Phase 1 — Baselines", s)
    assert results_row["qat_top5_drop_%"] == 58.2 - 50.0
    assert results_row["top1_%_FP32"] == 32.89
    assert results_row["size_MB_INT8"] == ""  # not set on s -> blank, not KeyError

    details_row = _to_details_row("Phase 1 — Baselines", s)
    assert details_row["model_name"] == "alexnet_tv"
    assert details_row["epochs"] == 79
    assert details_row["avg_epoch_time_s"] == ""  # not set on s -> blank
