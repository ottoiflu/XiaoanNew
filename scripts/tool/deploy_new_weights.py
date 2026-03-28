"""训练完成后自动部署权重并触发实验

用法:
    uv run scripts/tool/deploy_new_weights.py
    uv run scripts/tool/deploy_new_weights.py --watch  # 监控训练，完成后自动部署
    uv run scripts/tool/deploy_new_weights.py --skip-experiment  # 只部署不跑实验
"""

import argparse
import os
import shutil
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_OUTPUT = os.path.join(_PROJECT_ROOT, "outputs/train_outputs/yolov8x_seg_bike")
BEST_WEIGHT = os.path.join(TRAIN_OUTPUT, "weights/best.pt")
DEPLOY_PATH = os.path.join(_PROJECT_ROOT, "assets/weights/best.pt")
BACKUP_DIR = os.path.join(_PROJECT_ROOT, "assets/weights/archive")


def deploy_weights():
    """部署训练产出的权重文件"""
    if not os.path.exists(BEST_WEIGHT):
        print(f"[ERROR] best.pt 不存在: {BEST_WEIGHT}")
        return False

    # 备份旧权重
    if os.path.exists(DEPLOY_PATH):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        mtime = os.path.getmtime(DEPLOY_PATH)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(mtime))
        backup = os.path.join(BACKUP_DIR, f"best_{ts}.pt")
        shutil.copy2(DEPLOY_PATH, backup)
        print(f"旧权重备份: {backup}")

    shutil.copy2(BEST_WEIGHT, DEPLOY_PATH)
    size_mb = os.path.getsize(DEPLOY_PATH) / (1024 * 1024)
    print(f"新权重已部署: {DEPLOY_PATH} ({size_mb:.1f}MB)")
    return True


def check_training_done():
    """检查训练是否完成"""
    results_csv = os.path.join(TRAIN_OUTPUT, "results.csv")
    if not os.path.exists(results_csv):
        return False, 0
    with open(results_csv, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
    epoch_count = max(0, len(lines) - 1)
    return epoch_count >= 500, epoch_count


def watch_training(interval=60):
    """监控训练进度，完成后返回"""
    print(">>> 监控训练进度...")
    while True:
        done, epochs = check_training_done()
        if done:
            print(f"\n>>> 训练完成! ({epochs} epochs)")
            return True
        print(f"  epoch {epochs}/500", end="\r", flush=True)
        time.sleep(interval)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="部署训练权重")
    parser.add_argument("--watch", action="store_true", help="监控训练完成后自动部署")
    parser.add_argument("--skip-experiment", action="store_true", help="只部署不跑实验")
    args = parser.parse_args()

    if args.watch:
        watch_training()

    if not deploy_weights():
        sys.exit(1)

    if not args.skip_experiment:
        print("\n>>> 启动对比实验...")
        exp_script = os.path.join(_PROJECT_ROOT, "scripts/run_contrast_batch_v2.py")
        os.system(f"cd {_PROJECT_ROOT} && uv run python {exp_script}")


if __name__ == "__main__":
    main()
