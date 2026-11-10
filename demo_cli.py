"""
Backup demo that needs no browser and no Streamlit.

    python demo_cli.py            # metrics for both models + live terminal feed + PNG report
    python demo_cli.py --no-live  # skip the live feed

It writes  results_report.png  (confusion matrix, ROC curve, score histogram, model comparison).
"""
import argparse
import time

import numpy as np

from anomaly_core import ATTACK_TYPES, Detector, evaluate, generate_flows, roc_points, split_data


def fmt(m):
    return (f"precision {m['precision']:.3f} | recall {m['recall']:.3f} | F1 {m['f1']:.3f} | "
            f"FPR {m['fpr']:.3f} | AUC {m['auc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()

    df = generate_flows(8000, 0.12, 42)
    train, test = split_data(df, 0.3, 42)
    print(f"Dataset: {len(df)} flows, {df['label'].mean():.1%} attacks "
          f"({len(train)} train / {len(test)} test)\n")

    results, scores = {}, {}
    for kind, setting in [(Detector.ISO, 0.97), (Detector.RF, 0.5)]:
        det = Detector(kind).fit(train, train["label"])
        s = det.score(test)
        p = (s > det.threshold(setting)).astype(int)
        results[kind], scores[kind] = evaluate(test["label"], s, p), (det, s, p)
        print(f"{kind:17s} {fmt(results[kind])}")
        print(f"{'':17s} confusion matrix  TN={results[kind]['tn']} FP={results[kind]['fp']} "
              f"FN={results[kind]['fn']} TP={results[kind]['tp']}")

    det, s, p = scores[Detector.ISO]
    print("\nIsolation Forest detection rate by attack type:")
    for a in ATTACK_TYPES:
        mask = (test["attack_type"] == a).values
        print(f"  {a:11s} {p[mask].mean():.1%}")

    top = test.assign(score=s)[p == 1].sort_values("score", ascending=False).head(5)
    print("\nTop alerts and why they were flagged:")
    for row, why in zip(top.itertuples(), det.reasons(top)):
        print(f"  {row.src_ip:16s} score {row.score:.3f}  {why}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        m = results[Detector.ISO]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax[0, 0].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax[0, 0].text(j, i, str(v), ha="center", va="center", fontsize=14)
        ax[0, 0].set_xticks([0, 1], ["Pred normal", "Pred attack"])
        ax[0, 0].set_yticks([0, 1], ["Actual normal", "Actual attack"])
        ax[0, 0].set_title("Isolation Forest: confusion matrix")
        for kind in (Detector.ISO, Detector.RF):
            f, t = roc_points(test["label"], scores[kind][1])
            ax[0, 1].plot(f, t, label=f"{kind} (AUC {results[kind]['auc']:.2f})")
        ax[0, 1].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0, 1].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
        ax[0, 1].legend()
        ax[1, 0].hist(s[test["label"].values == 0], bins=40, alpha=0.7, label="Normal")
        ax[1, 0].hist(s[test["label"].values == 1], bins=40, alpha=0.7, label="Attack")
        ax[1, 0].axvline(det.threshold(0.97), color="k", ls="--", label="alert threshold")
        ax[1, 0].set(title="Isolation Forest anomaly scores", xlabel="score")
        ax[1, 0].legend()
        names = ["precision", "recall", "f1", "fpr"]
        w = 0.35
        for k, kind in enumerate((Detector.ISO, Detector.RF)):
            ax[1, 1].bar(np.arange(4) + k * w, [results[kind][n] for n in names], w, label=kind)
        ax[1, 1].set_xticks(np.arange(4) + w / 2, ["Precision", "Recall", "F1", "FPR"])
        ax[1, 1].set_title("Model comparison")
        ax[1, 1].legend()
        fig.tight_layout()
        fig.savefig("results_report.png", dpi=140)
        print("\nSaved results_report.png")
    except ImportError:
        print("\n(matplotlib not installed, skipping the PNG report)")

    if args.no_live:
        return
    print("\n--- Live monitor (attack burst between seconds 15 and 24) ---")
    total = 0
    for t in range(1, 41):
        burst = 15 <= t < 25
        b = generate_flows(60, 0.65 if burst else 0.02, 1000 + t, ["DoS"] if burst else None)
        sc = det.score(b)
        n_alert = int((sc > det.threshold(0.97)).sum())
        total += n_alert
        status = "ATTACK SUSPECTED" if n_alert > 9 else "normal"
        bar = "#" * min(n_alert, 50)
        print(f"t={t:2d}s  alerts={n_alert:2d}  total={total:3d}  {status:16s} {bar}")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
