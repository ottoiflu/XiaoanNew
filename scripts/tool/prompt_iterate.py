"""OPRO 式 Prompt 自迭代优化框架

流程：
1. 输入：当前 prompt 文件 + benchmark split + 上一轮失败案例 (FP/FN)
2. 用 LLM 读失败案例 + 当前 prompt，生成 2-3 个候选改写
3. 每个候选跑 train 子集 benchmark，打分 (balanced_acc + 违规召回≥0.6 约束)
4. val 上验证不退化才保留，否则回滚
5. 输出：最优 prompt + 对应权重阈值 + 每轮日志

用法：
    uv run python scripts/tool/prompt_iterate.py \\n        --prompt assets/prompts/xxx.yaml \\n        --results-csv outputs/.../results.csv \\n        --scoring-config assets/configs/scoring_new4d_gs_best.yaml \\n        --rounds 5
"""

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

_PROJECT_ROOT = "/root/otto/XiaoanNew"
sys.path.insert(0, _PROJECT_ROOT)

from modules.config.settings import get_settings
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.parser import normalize_label
from modules.vlm.retry import chat_completion_with_retry

SPLIT_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/split.json")

# ──────────────────────────── 加载 split ────────────────────────────

with open(SPLIT_PATH, encoding="utf-8") as f:
    SPLIT_MAP = json.load(f)


def load_results(csv_path: str) -> list[dict]:
    """加载 benchmark results.csv，注入 split 信息"""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["split"] = SPLIT_MAP.get(r.get("id", ""), "test")
            rows.append(r)
    return rows


def filter_by_split(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r.get("split") == split]


# ──────────────────────────── 评分 ────────────────────────────


def evaluate(rows: list[dict], scoring_engine) -> dict:
    """对一批结果统一评分，返回指标"""
    tp = tn = fp = fn = 0
    for r in rows:
        statuses = [r.get("position_status", ""), r.get("medium_status", ""),
                     r.get("angle_status", ""), r.get("state_status", "")]
        sr = scoring_engine.score(*statuses)
        pred = "yes" if sr.is_compliant else "no"
        gt = normalize_label(r["gt"])
        if gt == "yes":
            tp += pred == "yes"
            fn += pred != "yes"
        else:
            tn += pred == "no"
            fp += pred != "no"
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    pre = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    vr = tn / (tn + fp) if (tn + fp) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    bal_acc = (rec + spec) / 2
    return {
        "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4),
        "f1": round(f1, 4), "violation_recall": round(vr, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "total": total,
    }


def balanced_accuracy(tp, tn, fp, fn):
    rec = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    return (rec + spec) / 2


# ──────────────────────────── 失败案例采样 ────────────────────────────


def sample_failures(rows: list[dict], scoring_engine, n_fp=8, n_fn=8) -> str:
    """从 train 集采样 FP/FN 案例，格式化为 LLM 提示"""
    fp_cases = []
    fn_cases = []
    for r in rows:
        if r["split"] != "train":
            continue
        statuses = [r.get("position_status", ""), r.get("medium_status", ""),
                     r.get("angle_status", ""), r.get("state_status", "")]
        sr = scoring_engine.score(*statuses)
        pred = "yes" if sr.is_compliant else "no"
        gt = normalize_label(r["gt"])
        if gt == "no" and pred == "yes":
            fp_cases.append(r)
        elif gt == "yes" and pred == "no":
            fn_cases.append(r)

    # 采样
    import random
    rng = random.Random(42)
    rng.shuffle(fp_cases)
    rng.shuffle(fn_cases)

    lines = []
    lines.append("## 失败案例（FP: 实违规判合规）\n")
    for r in fp_cases[:n_fp]:
        lines.append(f"- id={r.get('id','?')[:12]} score={r.get('final_score','?')}")
        for dim in ["position_status", "medium_status", "angle_status", "state_status"]:
            lines.append(f"  {dim}: {r.get(dim, '?')}")
        lines.append(f"  reason: {str(r.get('vlm_reason',''))[:200]}")
        lines.append("")

    lines.append("\n## 失败案例（FN: 实合规判违规）\n")
    for r in fn_cases[:n_fn]:
        lines.append(f"- id={r.get('id','?')[:12]} score={r.get('final_score','?')}")
        for dim in ["position_status", "medium_status", "angle_status", "state_status"]:
            lines.append(f"  {dim}: {r.get(dim, '?')}")
        lines.append(f"  reason: {str(r.get('vlm_reason',''))[:200]}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────── LLM Prompt 改写 ────────────────────────────


GENERATE_PROMPT_TEMPLATE = """你是一名停车合规检测系统的 Prompt Engineer。当前系统使用以下 prompt 让 VLM 对图片进行四维（position/medium/angle/state）合规判定。

## 当前 Prompt
```yaml
{prompt_content}
```

## 当前评分配置（仅参考，不改评分）
- weights: {weights}
- threshold: {threshold}

## 性能指标（当前 train 集）
- accuracy: {metrics[accuracy]}
- balanced_accuracy: {metrics[balanced_accuracy]}
- f1: {metrics[f1]}
- violation_recall: {metrics[violation_recall]}

## 失败案例
{failure_cases}

## 任务
分析失败案例的模式，找出当前 prompt 不足。生成 2-3 个候选改写 prompt。

改写方向参考：
1. medium 判定：盲道检测依赖 CV 硬事实（CV 检测到盲道且有红色高亮接触时必须判盲道违规），绿化带由 VLM 视觉自行识别（绿色植被质感、不规则边缘）。现 medium 过高敏感（盲道误报率偏高），加约束：CV 未检出盲道时 VLM 仅凭视觉判绿化带/禁停区。
2. position 判定：以线判优先（检测到停车线时重叠率>0.3判合规），无缘时降级路缘内侧合理带判合规，仅当线、缘、邻车三者皆不可见时才判[无参照]。特别关注：不要过度收窄[无参照]导致 VLM 注意力从 medium 维度偏移。
3. 平衡约束：position 和 medium 两维度此消彼长是当前摆钟根因。锁定 medium 权重高（0.45）约束下，任何 candidate 不可让 medium 误报恶化（盲道FP<=当前基线），优先改进 medium 的绿化/禁停区视觉判断。

## 输出格式
严格按以下 JSON 格式，每条 prompt_content 只输出 content: 字段的内容（不含外层 YAML 元数据）：

```json
[
  {{
    "prompt_content": "改写后的完整 prompt content 部分"
  }},
  ...
]
```
"""


def generate_candidates(prompt_path: str, failure_text: str, metrics: dict,
                        weights: dict, threshold: float, client, model: str) -> list[str]:
    """用 LLM 生成候选 prompt 改写"""
    prompt_content = load_prompt_from_file(prompt_path)
    prompt_text = GENERATE_PROMPT_TEMPLATE.format(
        prompt_content=prompt_content,
        failure_cases=failure_text,
        weights=json.dumps(weights, ensure_ascii=False),
        threshold=threshold,
        metrics=metrics,
    )

    resp = chat_completion_with_retry(
        client, model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=8192, temperature=0.7, top_p=0.95,
    )
    text = resp.choices[0].message.content

    # 解析 JSON
    json_match = re.search(r"```json\n(.+?)\n```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\[.*?\])", text, re.DOTALL)
    if not json_match:
        print(f"[WARN] 无法解析 LLM 输出，使用原 prompt")
        return [load_prompt_from_file(prompt_path)]

    try:
        candidates = json.loads(json_match.group(1))
        contents = [c["prompt_content"] for c in candidates]
        print(f"  -> 生成 {len(contents)} 个候选")
        return contents
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[WARN] JSON 解析失败: {e}")
        return [load_prompt_from_file(prompt_path)]


def load_prompt_from_file(prompt_path: str) -> str:
    """从 YAML 文件读取 prompt content 字段"""
    import yaml
    with open(prompt_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["content"]


def write_candidate_prompt(orig_path: str, content: str, idx: int, log_dir: str) -> str:
    """将候选 prompt 写入临时文件，同时注册到 assets/prompts/"""
    import yaml
    with open(orig_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["content"] = content
    # 保存副本到 log_dir
    cand_path = os.path.join(log_dir, f"candidate_{idx}.yaml")
    with open(cand_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    # 复制到 assets/prompts/ 供 load_prompt 加载
    prompt_name = f"candidate_{os.path.basename(log_dir)}_{idx}"
    assets_path = os.path.join(_PROJECT_ROOT, "assets/prompts", f"{prompt_name}.yaml")
    with open(cand_path) as src, open(assets_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    return cand_path, prompt_name


# ──────────────────────────── Benchmark 运行 ────────────────────────────


def run_benchmark(prompt_id: str, scoring_config: str, mode: str, workers: int, out_dir: str,
                  smoke: int = 0) -> str:
    """调用 run_benchmark_v2.py 跑一次 benchmark，返回 results.csv 路径"""
    scoring_name = os.path.splitext(os.path.basename(scoring_config))[0]
    cmd = [
        "uv", "run", "python", "scripts/run_benchmark_v2.py",
        "--mode", mode,
        "--workers", str(workers),
        "--out", out_dir,
        "--prompt", prompt_id,
        "--scoring", scoring_name,
    ]
    if smoke > 0:
        cmd.extend(["--smoke", str(smoke)])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"  [ERROR] Benchmark failed:\n{result.stderr[-500:]}")
        return None

    results_path = os.path.join(_PROJECT_ROOT, out_dir, "results.csv")
    if not os.path.exists(results_path):
        print(f"  [ERROR] results.csv not found at {results_path}")
        return None
    return results_path


# ──────────────────────────── 主循环 ────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="OPRO Prompt 自迭代优化")
    parser.add_argument("--prompt", required=True, help="当前 prompt YAML 路径")
    parser.add_argument("--results-csv", required=True, help="上一轮 results.csv 路径")
    parser.add_argument("--scoring-config", default="assets/configs/scoring_optimized_cv_p4.yaml",
                        help="评分配置 YAML")
    parser.add_argument("--rounds", type=int, default=5, help="迭代轮数")
    parser.add_argument("--mode", choices=["pure", "cv"], default="pure", help="benchmark 模式")
    parser.add_argument("--workers", type=int, default=16, help="benchmark 并发数")
    parser.add_argument("--model", default="qwen/qwen3-vl-30b-a3b-instruct", help="LLM 改写模型")
    parser.add_argument("--smoke", type=int, default=0, help="冒烟测试张数（调试用）")
    parser.add_argument("--log-dir", default="outputs/prompt_iterate_logs", help="日志目录")
    args = parser.parse_args()

    # 初始化
    settings = get_settings()
    from openai import OpenAI
    client = OpenAI(base_url=settings.API_BASE_URL, api_key=settings.VLM_API_KEY)
    scoring_engine = ScoringEngine.from_yaml(os.path.join(_PROJECT_ROOT, args.scoring_config))

    # 加载数据
    all_rows = load_results(args.results_csv)
    train_rows = filter_by_split(all_rows, "train")
    val_rows   = filter_by_split(all_rows, "val")
    test_rows  = filter_by_split(all_rows, "test")

    print(f"[Prompt Iterate] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # 初始评估
    train_metrics = evaluate(train_rows, scoring_engine)
    val_metrics   = evaluate(val_rows, scoring_engine)
    print(f"[Init] train: BA={train_metrics['balanced_accuracy']} VR={train_metrics['violation_recall']}")
    print(f"[Init] val:   BA={val_metrics['balanced_accuracy']} VR={val_metrics['violation_recall']}")

    # 日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(_PROJECT_ROOT, args.log_dir, f"iter_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "iter_log.jsonl")
    log_entries = []

    # 保留最佳
    best_prompt_path = os.path.join(_PROJECT_ROOT, args.prompt)
    best_train_ba = train_metrics["balanced_accuracy"]
    best_val_ba   = val_metrics["balanced_accuracy"]
    best_train_vr = train_metrics["violation_recall"]

    current_prompt_path = os.path.join(_PROJECT_ROOT, args.prompt)

    # 迭代
    for round_idx in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_idx}/{args.rounds}")
        print(f"="*60)

        # 1. 采样失败案例
        failure_text = sample_failures(
            all_rows, scoring_engine,
            n_fp=4 if args.smoke else 8,
            n_fn=4 if args.smoke else 8,
        )

        # 2. 生成候选
        weights = scoring_engine.config.weights
        threshold = scoring_engine.config.threshold
        candidates = generate_candidates(
            current_prompt_path, failure_text, train_metrics,
            weights, threshold, client, args.model,
        )

        # 3. 跑每个候选的 train benchmark
        cand_scores = []
        for idx, content in enumerate(candidates):
            cand_path, prompt_id = write_candidate_prompt(current_prompt_path, content, idx, log_dir)
            out_tag = f"round{round_idx}_cand{idx}_{timestamp}"
            cand_out = os.path.join(args.log_dir, out_tag) if args.smoke == 0 else f"/tmp/{out_tag}"

            smoke_val = args.smoke if args.smoke > 0 else 0
            results_path = run_benchmark(prompt_id, args.scoring_config, args.mode, args.workers, cand_out, smoke=smoke_val)
            if results_path is None:
                print(f"  -> 候选 {idx}: benchmark 失败，跳过")
                continue

            # 评分
            cand_rows = load_results(results_path)
            cand_train = filter_by_split(cand_rows, "train")
            cand_val   = filter_by_split(cand_rows, "val")
            t_metrics = evaluate(cand_train, scoring_engine)
            v_metrics = evaluate(cand_val, scoring_engine)

            print(f"  -> 候选 {idx}: train BA={t_metrics['balanced_accuracy']} VR={t_metrics['violation_recall']} | val BA={v_metrics['balanced_accuracy']} VR={v_metrics['violation_recall']}")

            cand_scores.append((idx, cand_path, t_metrics, v_metrics, results_path))

        if not cand_scores:
            print("  无有效候选，跳过本轮")
            continue

        # 4. 选最优：balanced_acc 最高且 val 不退化
        cand_scores.sort(key=lambda x: (-x[2]["balanced_accuracy"], -x[2]["violation_recall"]))

        best_cand = None
        for idx, cand_path, t_metrics, v_metrics, results_path in cand_scores:
            if t_metrics["violation_recall"] < 0.6:
                print(f"  候选 {idx}: 违规召回 {t_metrics['violation_recall']} < 0.6，跳过")
                continue
            if v_metrics["balanced_accuracy"] < best_val_ba * 0.95:
                print(f"  候选 {idx}: val BA {v_metrics['balanced_accuracy']} 退化（best={best_val_ba}），跳过")
                continue
            best_cand = (idx, cand_path, t_metrics, v_metrics, results_path)
            break

        if best_cand is None:
            print("  无满足约束的候选，本轮跳过（保留当前 prompt）")
            continue

        idx, cand_path, t_metrics, v_metrics, results_path = best_cand

        # 5. 更新最佳
        if t_metrics["balanced_accuracy"] > best_train_ba:
            best_prompt_path = cand_path
            best_train_ba = t_metrics["balanced_accuracy"]
            best_val_ba   = v_metrics["balanced_accuracy"]
            best_train_vr = t_metrics["violation_recall"]
            current_prompt_path = cand_path
            all_rows = load_results(results_path)  # 更新结果集
            train_rows = filter_by_split(all_rows, "train")
            val_rows = filter_by_split(all_rows, "val")

            print(f"  >>> 新最佳: 候选 {idx} (BA={best_train_ba}, VR={best_train_vr})")
        else:
            print(f"  候选 {idx} 未超越当前最佳，跳过")

        # 日志
        entry = {
            "round": round_idx,
            "best_prompt": best_prompt_path,
            "train_ba": best_train_ba,
            "train_vr": best_train_vr,
            "val_ba": best_val_ba,
            "candidates": [
                {"idx": c[0], "train_ba": c[2]["balanced_accuracy"],
                 "train_vr": c[2]["violation_recall"],
                 "val_ba": c[3]["balanced_accuracy"] if len(c) > 3 else None}
                for c in cand_scores
            ],
            "timestamp": datetime.now().isoformat(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 最终报告
    print(f"\n{'='*60}")
    print("迭代完成")
    print(f"{'='*60}")
    print(f"最佳 prompt: {best_prompt_path}")
    print(f"Train: BA={best_train_ba} VR={best_train_vr}")

    # 在 test 上做最终评估
    if len(test_rows) > 0:
        test_metrics = evaluate(test_rows, scoring_engine)
        print(f"Test:  BA={test_metrics['balanced_accuracy']} VR={test_metrics['violation_recall']}")

    print(f"日志: {log_path}")


if __name__ == "__main__":
    main()