from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from configs.loader import load_config


def _load_yaml(path_or_name: str, subdir: str) -> dict:
    candidate = Path(path_or_name)
    if candidate.exists():
        return yaml.safe_load(candidate.read_text()) or {}
    return load_config(f"{subdir}/{path_or_name}.yaml")


def _build_sbatch_command(runtime_cfg: dict, slurm_cfg: dict, experiment: str, runtime: str, device: str | None = None) -> list[str]:
    script = Path(__file__).resolve().parent / "slurm" / "train.sbatch"
    output_root = Path(runtime_cfg.get("root", "outputs/pcad")).expanduser().resolve()
    log_dir = output_root / "logs" / experiment
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["sbatch"]
    if slurm_cfg.get("job_name"):
        cmd += ["--job-name", slurm_cfg["job_name"]]
    if slurm_cfg.get("partition"):
        cmd += ["--partition", slurm_cfg["partition"]]
    if slurm_cfg.get("account"):
        cmd += ["--account", slurm_cfg["account"]]
    if slurm_cfg.get("qos"):
        cmd += ["--qos", slurm_cfg["qos"]]
    if slurm_cfg.get("gres"):
        cmd += ["--gres", slurm_cfg["gres"]]
    if slurm_cfg.get("nodes"):
        cmd += ["--nodes", str(slurm_cfg["nodes"])]
    if slurm_cfg.get("ntasks"):
        cmd += ["--ntasks", str(slurm_cfg["ntasks"])]
    if slurm_cfg.get("cpus_per_task"):
        cmd += ["--cpus-per-task", str(slurm_cfg["cpus_per_task"])]
    if slurm_cfg.get("mem"):
        cmd += ["--mem", str(slurm_cfg["mem"])]
    if slurm_cfg.get("time"):
        cmd += ["--time", str(slurm_cfg["time"])]
    cmd += ["--output", str(log_dir / "%x-%j.out")]
    cmd += ["--error", str(log_dir / "%x-%j.err")]

    export_vars = [
        f"TRAIN_REPO_ROOT={Path(__file__).resolve().parents[1]}",
        f"CONDA_ENV_NAME={runtime_cfg.get('conda_env', 'alexnet_rafael')}",
        f"TRAIN_RUNTIME={runtime}",
        f"TRAIN_EXPERIMENT={experiment}",
    ]
    dataset_root = runtime_cfg.get("dataset_root")
    if dataset_root and "${" not in str(dataset_root):
        export_vars.append(f"PCAD_DATASET_ROOT={dataset_root}")
    if device:
        export_vars.append(f"TRAIN_DEVICE={device}")

    cmd += ["--export", ",".join(["ALL"] + export_vars), str(script), "--experiment", experiment, "--runtime", runtime]
    if device:
        cmd += ["--device", device]
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage PCAD Slurm submissions for training runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit a new training job")
    submit.add_argument("--experiment", default="default")
    submit.add_argument("--runtime", default="pcad")
    submit.add_argument("--slurm", default="single_gpu")
    submit.add_argument("--device", default=None)

    status = sub.add_parser("status", help="Show job status")
    status.add_argument("job_id")

    cancel = sub.add_parser("cancel", help="Cancel a job")
    cancel.add_argument("job_id")

    resume = sub.add_parser("resume", help="Resubmit using a saved run config")
    resume.add_argument("run_dir")
    resume.add_argument("--runtime", default="pcad")
    resume.add_argument("--slurm", default="single_gpu")
    resume.add_argument("--device", default=None)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "status":
        subprocess.run(["squeue", "-j", str(args.job_id), "-o", "%.18i %.9P %.20j %.8T %.10M %.6D %R"], check=False)
        return 0
    if args.command == "cancel":
        subprocess.run(["scancel", str(args.job_id)], check=False)
        return 0

    if args.command == "submit":
        runtime_cfg = _load_yaml(args.runtime, "runtime")
        slurm_cfg = _load_yaml(args.slurm, "slurm")
        cmd = _build_sbatch_command(runtime_cfg, slurm_cfg, args.experiment, args.runtime, args.device)
        print(subprocess.check_output(cmd, text=True).strip())
        return 0

    if args.command == "resume":
        run_dir = Path(args.run_dir).expanduser().resolve()
        config_path = run_dir / "resolved_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing resolved_config.json in {run_dir}")
        resolved = json.loads(config_path.read_text())
        experiment_cfg = resolved.get("experiment", {})
        experiment_path = run_dir / "_resume_experiment.yaml"
        experiment_path.write_text(yaml.safe_dump(experiment_cfg, sort_keys=False))
        runtime_cfg = _load_yaml(args.runtime, "runtime")
        slurm_cfg = _load_yaml(args.slurm, "slurm")
        cmd = _build_sbatch_command(runtime_cfg, slurm_cfg, str(experiment_path), args.runtime, args.device)
        print(subprocess.check_output(cmd, text=True).strip())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
