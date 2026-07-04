"""阶段2 v2: 真实CV特征 + 逻辑回归训练

从 feature_fusion_data/results.csv 读取真实 CV 几何+深度特征。
"""
import argparse, csv, json, os, sys, warnings
import numpy as np, joblib
warnings.filterwarnings("ignore")

_PROJECT_ROOT = "/root/otto/XiaoanNew"
sys.path.insert(0, _PROJECT_ROOT)
SPLIT_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/split.json")
GT_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/fourdim_gt_v2.json")

LABEL_VALUES = {
    "position": ["[合规]", "[基本合规-压线]", "[不合规-超界]", "[无参照]"],
    "medium": ["[合规]", "[不合规-盲道]", "[不合规-绿化]", "[不合规-禁停区]"],
    "angle": ["[合规]", "[不合规-斜停]", "[N/A]"],
    "state": ["[正立]", "[倒伏]"],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/benchmark_output/v2/feature_fusion_data/results.csv")
    parser.add_argument("--out", default="outputs/feature_fusion/v2")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(GT_PATH) as f: gt_map = {d["id"]: d for d in json.load(f)}
    with open(SPLIT_PATH) as f: split_map = json.load(f)

    rows = list(csv.DictReader(open(os.path.join(_PROJECT_ROOT, args.results), encoding="utf-8-sig")))

    X, y, splits = [], [], []
    for r in rows:
        g = gt_map.get(r["id"], {})
        if g.get("gt") not in ("yes", "no"): continue
        label = 1 if g["gt"] == "yes" else 0
        y.append(label)
        splits.append(split_map.get(r["id"], "test"))

        feat = []
        # 1. One-hot (13维)
        for dim in ["position", "medium", "angle", "state"]:
            val = r.get(f"{dim}_status", "")
            for cat in LABEL_VALUES[dim]:
                feat.append(1.0 if val == cat else 0.0)
        # 2. VLM confidence (4维)
        for dim in ["pos", "med", "ang", "state"]:
            feat.append(min(max(float(r.get(f"{dim}_conf", 0.5)), 0.0), 1.0))
        # 3. Interval scores (4维)
        for dim in ["pos", "med", "ang", "state"]:
            feat.append(float(r.get(f"{dim}_score", 0.0)))
        # 4. Final score (1维)
        feat.append(float(r.get("final_score", 0.0)))
        # 5. CV geometry (3维)
        feat.append(float(r.get("iou_parking", 0.0)))
        feat.append(float(r.get("overlap_parking", 0.0)))
        feat.append(float(r.get("overlap_tactile", 0.0)))
        # 6. Depth contact (2维)
        feat.append(int(r.get("contact_tactile", 0)))
        feat.append(int(r.get("contact_parking", 0)))
        # 7. YOLO detection flags (3维)
        feat.append(int(r.get("tactile_detected", 0)))
        feat.append(int(r.get("curb_detected", 0)))
        feat.append(int(r.get("parking_lane_detected", 0)))
        # 8. Main bike confidence (1维)
        feat.append(float(r.get("main_bike_conf", 0.0)))
        # 9. 交互特征 (关键)
        one_hot_offset = 0
        med_blind_idx = one_hot_offset + 4 + 1  # medium[盲道] is 5th one-hot
        pos_noref_idx = one_hot_offset + 3      # position[无参照] is 4th one-hot
        pos_over_idx = one_hot_offset + 2       # position[超界] is 3rd one-hot
        if len(feat) > max(med_blind_idx, pos_noref_idx, pos_over_idx):
            med_blind = feat[med_blind_idx]
            pos_noref = feat[pos_noref_idx]
            pos_over = feat[pos_over_idx]
            contact_tactile = int(r.get("contact_tactile", 0))
            contact_parking = int(r.get("contact_parking", 0))
            curb_detected = int(r.get("curb_detected", 0))
            # medium_盲道 x contact_tactile
            feat.append(med_blind * contact_tactile)
            # medium_盲道 x (1-contact_tactile)
            feat.append(med_blind * (1 - contact_tactile))
            # position_超界 x (1-iou_parking)
            feat.append(pos_over * (1 - float(r.get("iou_parking", 0.0))))
            # position_无参照 x curb_detected
            feat.append(pos_noref * curb_detected)
        else:
            feat.extend([0.0] * 4)

        X.append(feat)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    n_feat = X.shape[1]
    feature_names = [
        "pos_合规","pos_压线","pos_超界","pos_无参照",
        "med_合规","med_盲道","med_绿化","med_禁停区",
        "ang_合规","ang_斜停","ang_NA",
        "state_正立","state_倒伏",
        "pos_conf","med_conf","ang_conf","state_conf",
        "pos_score","med_score","ang_score","state_score",
        "final_score",
        "iou_parking","overlap_parking","overlap_tactile",
        "contact_tactile","contact_parking",
        "tactile_detected","curb_detected","parking_lane_detected",
        "main_bike_conf",
        "inter_med_blind_x_contact", "inter_med_blind_x_nocontact",
        "inter_pos_over_x_noiou", "inter_pos_noref_x_curb",
    ]

    print(f"Total: {len(y)}, Features: {n_feat}")
    train_i = [i for i,s in enumerate(splits) if s=="train"]
    val_i = [i for i,s in enumerate(splits) if s=="val"]
    test_i = [i for i,s in enumerate(splits) if s=="test"]
    print(f"train={len(train_i)} val={len(val_i)} test={len(test_i)}")

    X_tr, y_tr = X[train_i], y[train_i]
    X_val, y_val = X[val_i], y[val_i]
    X_te, y_te = X[test_i], y[test_i]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    best_model, best_C, best_ba = None, None, -1
    for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=42, solver="liblinear")
        lr.fit(X_tr_s, y_tr)
        y_vp = lr.predict(X_val_s)
        ba = (recall_score(y_val, y_vp, pos_label=1) + recall_score(y_val, y_vp, pos_label=0)) / 2
        vr = recall_score(y_val, y_vp, pos_label=0)
        print(f"  C={C:<6} val_ba={ba:.4f} val_vr={vr:.4f}")
        if ba > best_ba and vr >= 0.6:
            best_ba, best_model, best_C = ba, lr, C

    if best_model is None:
        best_model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=42, solver="liblinear")
        best_model.fit(X_tr_s, y_tr)
        best_C = 0.1

    # Test
    y_tp = best_model.predict(X_te_s)
    y_tproba = best_model.predict_proba(X_te_s)[:, 1]
    cm = confusion_matrix(y_te, y_tp)
    te_acc = accuracy_score(y_te, y_tp)
    te_pre = precision_score(y_te, y_tp, pos_label=1)
    te_rec = recall_score(y_te, y_tp, pos_label=1)
    te_spec = recall_score(y_te, y_tp, pos_label=0)
    te_f1 = f1_score(y_te, y_tp, pos_label=1)
    te_ba = (te_rec + te_spec) / 2
    print(f"\n=== Test ===")
    print(f"Acc={te_acc:.4f} Prec={te_pre:.4f} Rec={te_rec:.4f} F1={te_f1:.4f} BA={te_ba:.4f} VR={te_spec:.4f}")
    print(f"TP={cm[1,1]} FN={cm[1,0]} FP={cm[0,1]} TN={cm[0,0]}")

    # Feature importance
    print("\n=== Top 15 Features ===")
    coef = best_model.coef_[0]
    for i in np.argsort(np.abs(coef))[::-1][:15]:
        print(f"  {feature_names[i]:<35} coef={coef[i]:+.4f}")

    # Save
    joblib.dump(best_model, os.path.join(out_dir, "lr_model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    result = {
        "config": {"C": best_C, "features": n_feat},
        "test": {"accuracy": round(te_acc,4), "precision": round(te_pre,4), "recall": round(te_rec,4),
                 "f1": round(te_f1,4), "balanced_accuracy": round(te_ba,4), "violation_recall": round(te_spec,4),
                 "tp": int(cm[1,1]), "fn": int(cm[1,0]), "fp": int(cm[0,1]), "tn": int(cm[0,0])},
        "feature_importance": {feature_names[i]: round(float(coef[i]),4) for i in np.argsort(np.abs(coef))[::-1][:20]},
    }
    json.dump(result, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {os.path.join(out_dir, 'results.json')}")

if __name__ == "__main__":
    main()