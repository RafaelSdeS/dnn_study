"""scripts/pcad/feed_queue.sh runs unattended for days on PCAD: a submit that hits the QOS limit must go back to
the head of the queue (not be lost), a real failure must land in .failed, and a success in .done."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pcad" / "feed_queue.sh"


def test_feed_queue_requeues_on_qos_limit_and_records_outcomes(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text("#!/bin/sh\necho 1\n")  # 1 job queued, always under MAX
    (bin_dir / "squeue").chmod(0o755)
    marker = tmp_path / "qos_hit"
    qos_once = f"if [ -e {marker} ]; then echo 'Submitted batch job 2'; else touch {marker}; " \
               f"echo 'sbatch: error: QOSMaxSubmitJobPerUserLimit'; exit 1; fi"
    queue = tmp_path / "queue.txt"
    queue.write_text(f"echo 'Submitted batch job 1'\n{qos_once}\nexit 3\n")

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLEEP": "0", "GAP": "0", "USER": "tester"}
    subprocess.run(["bash", str(SCRIPT), str(queue)], env=env, check=True, timeout=60, capture_output=True)

    assert queue.read_text() == ""
    assert len(Path(f"{queue}.done").read_text().splitlines()) == 2  # job 1, and the QOS one on its retry
    assert Path(f"{queue}.failed").read_text().strip() == "exit 3"


def test_feed_queue_waits_out_a_slurm_controller_outage(tmp_path):
    """2026-10-07 15:55: the controller was unreachable for ~1 min. squeue's failure counted as 0 jobs, so the feeder
    kept popping lines, and each sbatch's 'Unable to contact slurm controller' sent it to .failed -- 3 jobs dropped."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state, down_once = tmp_path / "slurm_state", tmp_path / "squeue_failed"
    # squeue fails on its first call (records "down"), then answers (records "up")
    (bin_dir / "squeue").write_text(
        f"#!/bin/sh\nif [ -e {down_once} ]; then echo up > {state}; echo 1; else touch {down_once}; echo down > {state}; "
        f"echo 'slurm_load_jobs error: Unable to contact slurm controller (connect failure)' >&2; exit 1; fi\n")
    (bin_dir / "squeue").chmod(0o755)
    # a submit run while squeue was down fails for good; a submit whose sbatch can't reach the controller fails once
    needs_up = f"grep -q up {state} && echo 'Submitted batch job 1' || {{ echo boom; exit 1; }}"
    marker = tmp_path / "sbatch_failed"
    unreachable_once = f"if [ -e {marker} ]; then echo 'Submitted batch job 2'; else touch {marker}; " \
                       f"echo 'sbatch: error: Batch job submission failed: Unable to contact slurm controller " \
                       f"(connect failure)' >&2; exit 1; fi"
    queue = tmp_path / "queue.txt"
    queue.write_text(f"{needs_up}\n{unreachable_once}\n")

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLEEP": "0", "GAP": "0", "USER": "tester"}
    subprocess.run(["bash", str(SCRIPT), str(queue)], env=env, check=True, timeout=60, capture_output=True)

    assert queue.read_text() == ""
    assert Path(f"{queue}.done").read_text().splitlines() == [needs_up, unreachable_once]
    assert not Path(f"{queue}.failed").exists()
