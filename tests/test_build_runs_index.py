import json

from scripts.build_runs_index import build_index


def test_build_index_covers_all_four_layouts(tmp_path):
    root = tmp_path / "outputs"

    # A) classification (scripts/train.py)
    cls_dir = root / "local" / "phase_8_efficient_vit" / "swin_pico_w2" / "results"
    cls_dir.mkdir(parents=True)
    (cls_dir / "swin_pico_w2_summary.json").write_text(json.dumps({
        "config": {"provenance": {"hostname": "tupi5", "git_hash": "abc123"}},
        "mode": "fp32/qat/int8", "best_val_top1": 41.2, "int8_size_mb": 1.9, "fp32_size_mb": 2.4,
    }))

    # B) detection (scripts/train_det_seg.py)
    det_dir = root / "pcad" / "phase_7_detection_segmentation" / "ssd_alexnet_fire_fp32_phase_7_detection"
    det_dir.mkdir(parents=True)
    (det_dir / "config.yaml").write_text("data: {}\n")
    (det_dir / "git_hash.txt").write_text("deadbeef\n")
    (det_dir / "metrics.json").write_text(json.dumps({"best_val_mAP": 0.31, "summary": {"true_size_mb": 3.1}}))

    # C) notebooks
    nb_dir = root / "notebooks" / "phase_2_kernel_restriction"
    nb_dir.mkdir(parents=True)
    (nb_dir / "alexnet_2x2_gap_meta.json").write_text(json.dumps({"epoch": 49, "best_val_acc": 33.24}))

    # D) profiling
    prof_dir = root / "pcad" / "phase_6_hardware_profiling" / "runs"
    prof_dir.mkdir(parents=True)
    with (prof_dir / "rtx4090_profile.json").open("w") as f:
        f.write(json.dumps({"kind": "metadata", "device_name": "RTX 4090"}) + "\n")
        f.write(json.dumps({"kind": "layer"}) + "\n")

    rows = build_index(root)
    by_model = {r["model"]: r for r in rows}

    assert len(rows) == 4
    assert by_model["swin_pico_w2"]["phase"] == "phase_8"
    assert by_model["swin_pico_w2"]["git_hash"] == "abc123"
    assert by_model["alexnet_fire"]["top1"] == 0.31
    assert by_model["alexnet_fire"]["stage"] == "fp32"
    assert by_model["alexnet_2x2_gap"]["top1"] == 33.24
    assert by_model["(sweep)"]["host"] == "RTX 4090"
