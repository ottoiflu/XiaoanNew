import subprocess
log = "/root/otto/XiaoanNew/outputs/benchmark_output/v2/opro_iter/run3.log"
cmd = [
    "uv", "run", "python",
    "/root/otto/XiaoanNew/scripts/tool/prompt_iterate.py",
    "--prompt", "/root/otto/XiaoanNew/assets/prompts/cv_enhanced_v2_new4dim_opro_base.yaml",
    "--results-csv", "/root/otto/XiaoanNew/outputs/benchmark_output/v2/model_matrix/qwen30b/results.csv",
    "--scoring-config", "/root/otto/XiaoanNew/assets/configs/scoring_new4d_gs_best_opro_base.yaml",
    "--rounds", "8",
    "--mode", "cv",
    "--workers", "32",
    "--model", "qwen/qwen3-vl-30b-a3b-instruct",
    "--log-dir", "outputs/benchmark_output/v2/opro_iter",
]
f = open(log, "a", buffering=1)
proc = subprocess.Popen(cmd, cwd="/root/otto/XiaoanNew", stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
print(f"OPRO launched: PID={proc.pid}", flush=True)
