# Network Anomaly Detection using Machine Learning (live demo)

Arjun Saproo (2024a1r058) and Parth Kohli (2024a1r052), Model Institute of Engineering and Technology, CSE.

A working prototype of the pipeline in the project proposal:

    flow records -> pre-processing -> ML model -> anomaly score -> alert -> analyst review

It has an interactive dashboard (Streamlit) and a command-line backup demo.

## 1. Setup (once, needs internet)

Install Python 3.10 or newer, then in this folder:

    python -m venv venv
    venv\Scripts\activate            (Windows)     source venv/bin/activate   (Mac/Linux)
    pip install -r requirements.txt

## 2. Run

    streamlit run app.py             # dashboard opens at http://localhost:8501
    python demo_cli.py               # backup: no browser needed, also writes results_report.png
    python -m unittest -v            # quick self-checks

**Do a full dry run on the presentation laptop the day before.** If the browser demo misbehaves on the day, `demo_cli.py` shows the same results in the terminal.

## 3. Five-minute demo script

1. **Dataset tab** (30 s): "These are network flows. Attacks look different from normal traffic: DoS has a very high connection rate, port scans touch many ports." Change the *Feature* dropdown to `unique_dst_ports`.
2. **Detection results tab** (90 s): sidebar model = Isolation Forest. Point at precision, recall, F1, false-positive rate and the confusion matrix. Explain: the model never saw an attack label; it learned only what normal traffic looks like.
3. **Move the threshold slider** (30 s): raise it and show fewer false alarms but more missed attacks. This is the core trade-off.
4. **Alerts tab** (45 s): show the "why flagged" column (explainable alerts) and set a few analyst verdicts (human in the loop). Download the CSV report.
5. **Live monitor tab** (90 s): press *Start live monitoring*. Normal traffic first, then the attack starts at second 15; the status turns red and alerts spike, then it recovers.
6. **Model comparison tab** (30 s): Random Forest (supervised) scores higher than Isolation Forest (unsupervised); say why (it sees labels).

## 4. Be honest about the results

* The default data is **synthetic**, generated in `anomaly_core.py` from typical flow behaviour. It is easy, so Random Forest scores almost perfectly. Say this during the presentation. Do not present these numbers as real-world performance.
* For real results, download CIC-IDS2017 or UNSW-NB15 and use *Upload CSV*. Tick the CIC-IDS2017 box for CIC files. That converter is best effort: CIC files lack per-source port and connection counters, so `unique_dst_ports` is set to 1 and `conn_rate` reuses the packet rate. Expect lower scores than on the synthetic data, and report whatever you actually measure.
* Alerts are leads for an analyst, not proof of an attack. The system never blocks traffic by itself.

## 5. Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: dataset, detection results, alerts, live monitor, model comparison |
| `anomaly_core.py` | Data generator, CSV loaders, Isolation Forest / Random Forest wrapper, alert explanations, metrics |
| `demo_cli.py` | Terminal version of the demo plus `results_report.png` |
| `test_core.py` | Unit tests |
| `sample_flows.csv` | Example file showing the required CSV columns |
| `assets/miet_logo.png` | Logo shown in the sidebar |

## 6. How it works (for viva questions)

* **Features (10):** duration, forward/backward packets, source/destination bytes, packet rate, bytes per packet, distinct destination ports, connection rate, SYN ratio, plus one-hot protocol. Numeric features are log-transformed and standardised.
* **Isolation Forest:** random trees isolate unusual points in fewer splits. Trained on normal traffic only. The alert threshold is a percentile of normal training scores, so 0.97 means roughly 3 % of normal flows raise an alert.
* **Random Forest:** supervised baseline trained on labelled flows with class weights for the imbalance.
* **Evaluation:** 70/30 stratified split. Precision, recall, F1, false-positive rate, ROC-AUC and a confusion matrix; accuracy is not used alone because attacks are a small share of flows.
* **Explainability:** each alert lists the three features that deviate most from normal traffic (robust z-score).
* **Known weakness:** low-and-slow brute force looks like normal traffic and is the attack Isolation Forest misses most.
