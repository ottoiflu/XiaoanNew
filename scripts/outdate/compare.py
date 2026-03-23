import csv
import os

LABELS_PATH = "/root/XiaoanNew/App_collected_dataset/test/2025-12-31/labels.txt"
MODEL_CSV_PATH = "/root/XiaoanNew/App_collected_dataset/test/qwen3-vl-235b-a22b-instruct_vlm_test_results_compressed.csv"
OUT_CSV_PATH = "/root/XiaoanNew/App_collected_dataset/test/compare_results_with_match_rate.csv"


def norm_yesno(x: str) -> str:
    """normalize yes/no string; return '' if invalid."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "yes"
    if s in ("no", "n", "false", "0"):
        return "no"
    return ""


def read_labels(path: str) -> dict:
    """labels.txt: each line like 'xxx.jpg, yes'."""
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split by first comma
            if "," not in line:
                continue
            name, lab = line.split(",", 1)
            name = name.strip()
            lab = norm_yesno(lab)
            if name and lab:
                labels[name] = lab
    return labels


def read_model_results(csv_path: str) -> dict:
    """model csv should have columns: image_name, result (others ignored)."""
    results = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # try common column names
        # your script outputs 'image_name' and 'result'
        for row in reader:
            img = row.get("image_name") or row.get("filename") or row.get("image") or ""
            pred = row.get("result") or row.get("pred") or row.get("prediction") or ""
            img = str(img).strip()
            pred = norm_yesno(pred)
            if img and pred:
                results[img] = pred
    return results


def main():
    labels = read_labels(LABELS_PATH)
    model = read_model_results(MODEL_CSV_PATH)

    # union of filenames (so你能看到缺失项)
    all_names = sorted(set(labels.keys()) | set(model.keys()))

    total_compared = 0
    matched = 0

    with open(OUT_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "labels", "大模型结果"])

        for name in all_names:
            lab = labels.get(name, "")
            pred = model.get(name, "")
            writer.writerow([name, lab, pred])

            # only count when both exist
            if lab and pred:
                total_compared += 1
                if lab == pred:
                    matched += 1

        match_rate = (matched / total_compared) if total_compared else 0.0
        writer.writerow(["MATCH_RATE", "", f"{match_rate:.4f} ({match_rate*100:.2f}%)"])

    print("Done!")
    print(f"Compared: {total_compared}, Matched: {matched}, Match rate: {match_rate*100:.2f}%")
    print(f"Saved to: {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()
