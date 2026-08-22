"""Phase 8 H4 — hard-label knowledge distillation (DeiT-Tiny only).

Subclasses only the training step, keeping Trainer's fit()/evaluate()/benchmark()
loop, checkpointing, and AMP/grad-clip/logging scaffolding untouched -- same
"subclass only the step" precedent as ml/det_seg_trainer.py's DetectionTrainer.
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .trainer import Trainer


class DistillationTrainer(Trainer):
    """Hard-label distillation (Touvron et al. 2021, DeiT): student's cross-entropy
    against the teacher's argmax prediction, not a soft KL term -- the DeiT paper's
    own finding that hard distillation outperforms soft-KL for this setup (Table 4),
    and it avoids a temperature hyperparameter.
    """

    def __init__(self, *args, teacher: nn.Module, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval().to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.alpha = alpha

    def _train_one_epoch(self, model, optimizer, scaler, criterion) -> tuple[float, float, float | None]:
        model.train()
        cfg = self.cfg
        total_loss = correct = total = 0
        total_norm = 0.0

        for data, target in (bar := tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_labels = self.teacher(data).argmax(dim=1)

            if cfg.use_amp and scaler:
                with torch.amp.autocast("cuda"):
                    out = model(data)
                    loss = (1 - self.alpha) * criterion(out, target) + self.alpha * criterion(out, teacher_labels)
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    total_norm += nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(data)
                loss = (1 - self.alpha) * criterion(out, target) + self.alpha * criterion(out, teacher_labels)
                loss.backward()
                if cfg.grad_clip_norm:
                    total_norm += nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()
                optimizer.step()

            total_loss += loss.item() * target.size(0)
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")

        avg_norm = total_norm / len(self.train_loader) if cfg.grad_clip_norm else None
        return total_loss / total, 100 * correct / total, avg_norm


def demo() -> None:
    """Assert-based self-check (Task 4 validation): distillation loss is non-zero and
    the teacher's parameters never receive gradients. Not run automatically -- invoke
    directly (`python -m ml.distillation_trainer`)."""
    import tempfile

    from .config import TrainerConfig

    torch.manual_seed(0)

    class _Tiny(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.fc = nn.Linear(8, n)

        def forward(self, x):
            return self.fc(x.flatten(1))

    student, teacher = _Tiny(), _Tiny()
    x = torch.randn(6, 8)
    y = torch.randint(0, 4, (6,))
    loader = [(x, y)]

    with tempfile.TemporaryDirectory() as tmp:
        trainer = DistillationTrainer(
            student, loader, loader,
            cfg=TrainerConfig(epochs=1, use_amp=False),
            device=torch.device("cpu"),
            save_dir=tmp, run_name="distill_demo",
            teacher=teacher, alpha=0.5,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        teacher_params_before = [p.clone() for p in teacher.parameters()]
        loss, acc, _ = trainer._train_one_epoch(student, optimizer, None, criterion)

        assert loss > 0, "distillation loss should be non-zero on a random init"
        for p in teacher.parameters():
            assert p.grad is None, "teacher must never receive gradients"
        for before, after in zip(teacher_params_before, teacher.parameters()):
            assert torch.equal(before, after), "teacher weights must not change"
        print(f"DistillationTrainer: OK, loss={loss:.4f} acc={acc:.1f}% teacher frozen")


if __name__ == "__main__":
    demo()
