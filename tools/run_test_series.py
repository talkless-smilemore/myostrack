import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Fill this list, then run:
#   python tools/run_test_series.py --threads 8 --num-gpus 1
#
# config can be either the yaml file name or the name without ".yaml".
# dataset must match lib/test/evaluation/datasets.py, for example:
# anti_uav, anti_uav_ir, anti_uav_rgb, anti_uav410, anti_uav600, anti_uav300,
# uav, lasot, got10k_test, trackingnet.
TEST_JOBS = [
    {
        "config":"asc_lora_ablation_F0_full",
        "dataset": "anti_uav_ir",
        "display_name": "F0_full",
    },
    {
        "config":"asc_lora_ablation_F0_full300",
        "dataset": "anti_uav300_ir",
        "display_name": "F0_full",
    },
     {
        "config":"asc_lora_ablation_F0_full410",
        "dataset": "anti_uav410",
        "display_name": "F0_full",
    },
    {
        "config":"asc_lora_ablation_F1_no_channel_gate",
        "dataset": "anti_uav_ir",
        "display_name": "F1_no_channel_gate",
    },
    {
        "config":"asc_lora_ablation_F1_no_channel_gate300",
        "dataset": "anti_uav300_ir",
        "display_name": "F1_no_channel_gate",
    },
    {
        "config":"asc_lora_ablation_F1_no_channel_gate410",
        "dataset": "anti_uav410",
        "display_name": "F1_no_channel_gate",
    },
    {
        "config":"asc_lora_ablation_F2_no_spectral_complement",
        "dataset": "anti_uav_ir",
        "display_name": "F2_no_spectral_complement",
    },
    {
        "config":"asc_lora_ablation_F2_no_spectral_complement300",
        "dataset": "anti_uav300_ir",
        "display_name": "F2_no_spectral_complement",
    },
    {
        "config":"asc_lora_ablation_F2_no_spectral_complement410",
        "dataset": "anti_uav410",
        "display_name": "F2_no_spectral_complement",
    },
    {
        "config":"asc_lora_ablation_F3_hard_spectral_complement",
        "dataset": "anti_uav_ir",
        "display_name": "F3_hard_spectral_complement",
    },
    {
        "config":"asc_lora_ablation_F3_hard_spectral_complement300",
        "dataset": "anti_uav300_ir",
        "display_name": "F3_hard_spectral_complement",
    },
    {
        "config":"asc_lora_ablation_F3_hard_spectral_complement410",
        "dataset": "anti_uav410",
        "display_name": "F3_hard_spectral_complement",
    },
    {
        "config":"asc_lora_ablation_F4_no_complement_constraint",
        "dataset": "anti_uav_ir",
        "display_name": "F4_no_complement_constraint",
    },
    {
        "config":"asc_lora_ablation_F4_no_complement_constraint300",
        "dataset": "anti_uav_ir",
        "display_name": "F4_no_complement_constraint",
    },
    {
        "config":"asc_lora_ablation_F4_no_complement_constraint410",
        "dataset": "anti_uav410",
        "display_name": "F4_no_complement_constraint",
    },
    {
        "config":"asc_lora_ablation_F5_standard_lora",
        "dataset": "anti_uav_ir",
        "display_name": "F5_standard_lora",
    },
    {
        "config":"asc_lora_ablation_F5_standard_lora300",
        "dataset": "anti_uav300_ir",
        "display_name": "F5_standard_lora",
    },
    {
        "config":"asc_lora_ablation_F5_standard_lora410",
        "dataset": "anti_uav410",
        "display_name": "F5_standard_lora",
    },
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_config(config: str) -> str:
    return Path(config).name.removesuffix(".yaml")


def run_and_log(cmd, cwd: Path, log_file: Path, env) -> int:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            fh.write(line)
        return process.wait()


def run_analysis(repo: Path, job, merge_results: bool, force_evaluation: bool):
    sys.path.insert(0, str(repo))

    from lib.test.analysis.plot_results import plot_results, print_results
    from lib.test.evaluation import get_dataset, trackerlist

    config = normalize_config(job["config"])
    dataset_name = job["dataset"]
    display_name = job.get("display_name") or config

    trackers = trackerlist(
        name="ostrack",
        parameter_name=config,
        dataset_name=dataset_name,
        run_ids=job.get("run_ids"),
        display_name=display_name,
    )
    dataset = get_dataset(*[x.strip() for x in dataset_name.split(",") if x.strip()])

    plot_results(
        trackers,
        dataset,
        dataset_name,
        merge_results=merge_results,
        plot_types=("success", "norm_prec", "prec"),
        force_evaluation=force_evaluation,
    )
    print_results(
        trackers,
        dataset,
        dataset_name,
        merge_results=merge_results,
        plot_types=("success", "norm_prec", "prec"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run OSTrack test jobs one by one.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run tracking/test.py.")
    parser.add_argument("--threads", type=int, default=2, help="Number of test threads.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs used by tracking/test.py.")
    parser.add_argument("--gpu-ids", default=None, help='Optional CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--debug", type=int, default=0, help="Debug level passed to tracking/test.py.")
    parser.add_argument("--runid", type=int, default=None, help="Optional run id passed to tracking/test.py.")
    parser.add_argument("--skip-analysis", action="store_true", help="Only run tests, do not print/plot results.")
    parser.add_argument("--force-evaluation", action="store_true", help="Recompute analysis cache.")
    parser.add_argument("--no-merge-results", action="store_true", help="Do not merge multiple runs during analysis.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with later jobs if one job fails.")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = repo_root()
    logs_dir = repo / "output" / "logs" / "test_series" / datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    if args.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    failed = []
    for index, job in enumerate(TEST_JOBS, start=1):
        config = normalize_config(job["config"])
        dataset = job["dataset"]
        print(f"\n[{index}/{len(TEST_JOBS)}] config={config}, dataset={dataset}", flush=True)

        cmd = [
            args.python,
            "tracking/test.py",
            "ostrack",
            config,
            "--dataset_name",
            dataset,
            "--threads",
            str(args.threads),
            "--num_gpus",
            str(args.num_gpus),
            "--debug",
            str(args.debug),
        ]
        if args.runid is not None:
            cmd.extend(["--runid", str(args.runid)])

        return_code = run_and_log(cmd, repo, logs_dir / f"{index:02d}_{config}_{dataset}.log", env)
        if return_code != 0:
            failed.append((config, dataset, return_code))
            print(f"[failed] config={config}, dataset={dataset}, return_code={return_code}", flush=True)
            if not args.continue_on_error:
                break
            continue

        if not args.skip_analysis:
            print(f"[analysis] config={config}, dataset={dataset}", flush=True)
            run_analysis(repo, job, merge_results=not args.no_merge_results, force_evaluation=args.force_evaluation)

    if failed:
        print("\nFailed jobs:")
        for config, dataset, return_code in failed:
            print(f"  config={config}, dataset={dataset}, return_code={return_code}")
        raise SystemExit(1)

    print(f"\nAll jobs finished. Logs: {logs_dir}")


if __name__ == "__main__":
    main()
