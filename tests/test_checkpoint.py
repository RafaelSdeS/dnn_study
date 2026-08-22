"""save_checkpoint/load_resume_state round-trip — the safety net the requeue design depends on."""
import torch
import torch.nn as nn

from ml.checkpoint import auto_resume_path, load_resume_state, save_checkpoint


def _make_model_optim():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return model, optimizer, scheduler


def test_save_and_load_resume_state_round_trip(tmp_path):
    model, optimizer, scheduler = _make_model_optim()
    with torch.no_grad():
        model.weight.fill_(0.5)
    path = tmp_path / "run_resume.pth"

    save_checkpoint(
        path, model, optimizer, scheduler, epoch=3, metrics={"val_acc": 42.0},
        best_val_acc=42.0, best_val_top5=70.0,
        history={"train_loss": [1.0, 0.5]}, wandb_run_id="abc123",
        patience_counter=2, elapsed_time_s=123.4,
    )
    assert path.exists()

    model2, optimizer2, scheduler2 = _make_model_optim()
    state = load_resume_state(path, model2, optimizer2, scheduler2, device="cpu")

    assert state["epoch"] == 3
    assert state["best_val_acc"] == 42.0
    assert state["best_val_top5"] == 70.0
    assert state["history"] == {"train_loss": [1.0, 0.5]}
    assert state["wandb_run_id"] == "abc123"
    assert state["patience_counter"] == 2
    assert state["elapsed_time_s"] == 123.4
    assert torch.equal(model2.weight, model.weight)


def test_reset_scheduler_skips_stale_t_max(tmp_path):
    # Checkpoint saved with a short schedule (T_max=10, as if cfg.epochs=10 already finished).
    model, optimizer, scheduler = _make_model_optim()
    for _ in range(10):
        scheduler.step()
    path = tmp_path / "run_resume.pth"
    save_checkpoint(path, model, optimizer, scheduler, epoch=9, metrics={"val_acc": 1.0})

    # Resuming into a bigger budget: a freshly-built scheduler already has the new T_max=100.
    model2, optimizer2 = nn.Linear(4, 2), None
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)
    load_resume_state(path, model2, optimizer2, scheduler2, device="cpu", reset_scheduler=True)
    assert scheduler2.T_max == 100  # untouched by the checkpoint's stale T_max=10

    # Default behavior (reset_scheduler=False) still clobbers it, for the ordinary
    # same-budget preemption-resume case.
    model3, optimizer3 = nn.Linear(4, 2), None
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=100)
    load_resume_state(path, model3, optimizer3, scheduler3, device="cpu")
    assert scheduler3.T_max == 10


def test_auto_resume_path_only_returns_existing_files(tmp_path):
    assert auto_resume_path(tmp_path, "missing_run") is None

    (tmp_path / "present_run_resume.pth").touch()
    found = auto_resume_path(tmp_path, "present_run")
    assert found == tmp_path / "present_run_resume.pth"
