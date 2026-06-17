"""F7 — Análisis de resultados del test con usuarios reales.

Lee los CSV generados por `SignTestLogger.kt` (uno por tester
y por ronda), calcula accuracy, matriz de confusión, latencia
y mejora por calibración, y renderiza un reporte markdown en
`docs/F7_USER_TEST_REPORT.md`.

Uso:
    python scripts/analyze_f7_results.py \\
        --csv-dir reports/f7 \\
        --output docs/F7_USER_TEST_REPORT.md

Si no se pasan flags, usa los defaults de arriba y rellena la
plantilla `docs/F7_USER_TEST_REPORT.template.md`.

El script NO requiere que los CSV se hayan preprocesado: el
schema es el escrito por `SignTestLogger` (ver CSV_SCHEMA).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


CSV_SCHEMA = [
    "tester_id",
    "round",
    "rep",
    "index",
    "prompted_sign",
    "predicted_sign",
    "confidence",
    "latency_ms",
    "timestamp_ms",
    "seed",
    "dominance",
    "note",
]

TARGET_NO_CALIB = 0.90
TARGET_WITH_CALIB = 0.95
REFERENCE_LOSO_TRANSFORMER = 0.8739


@dataclass
class Trial:
    tester_id: str
    round: int
    rep: int
    index: int
    prompted: str
    predicted: str
    confidence: float
    latency_ms: int
    timestamp_ms: int
    seed: int
    dominance: str
    note: str

    @property
    def correct(self) -> bool:
        return self.prompted == self.predicted


@dataclass
class Aggregates:
    trials: List[Trial] = field(default_factory=list)

    def per_user_accuracy(self, round_: int) -> Dict[str, Tuple[int, int, float]]:
        out: Dict[str, Tuple[int, int, float]] = {}
        grouped: Dict[str, List[Trial]] = defaultdict(list)
        for t in self.trials:
            if t.round == round_:
                grouped[t.tester_id].append(t)
        for uid, ts in grouped.items():
            n = len(ts)
            c = sum(1 for t in ts if t.correct)
            out[uid] = (c, n, c / n if n else 0.0)
        return out

    def per_sign_accuracy(self, round_: int) -> Dict[str, Tuple[int, int, float]]:
        out: Dict[str, Tuple[int, int, float]] = {}
        grouped: Dict[str, List[Trial]] = defaultdict(list)
        for t in self.trials:
            if t.round == round_:
                grouped[t.prompted].append(t)
        for sign, ts in grouped.items():
            n = len(ts)
            c = sum(1 for t in ts if t.correct)
            out[sign] = (c, n, c / n if n else 0.0)
        return out

    def confusion_matrix(self, round_: int, labels: List[str]) -> List[List[int]]:
        idx = {l: i for i, l in enumerate(labels)}
        m = [[0] * len(labels) for _ in labels]
        for t in self.trials:
            if t.round != round_:
                continue
            r = idx.get(t.prompted)
            c = idx.get(t.predicted)
            if r is None or c is None:
                continue
            m[r][c] += 1
        return m

    def latencies(self, round_: int) -> List[int]:
        return [t.latency_ms for t in self.trials if t.round == round_]

    def overall_accuracy(self, round_: int) -> Tuple[int, int, float]:
        ts = [t for t in self.trials if t.round == round_]
        n = len(ts)
        c = sum(1 for t in ts if t.correct)
        return c, n, c / n if n else 0.0

    def per_user_dominance(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for t in self.trials:
            if t.tester_id not in out:
                out[t.tester_id] = t.dominance
        return out


def _load_csv(path: Path) -> List[Trial]:
    trials: List[Trial] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CSV_SCHEMA if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        for row in reader:
            try:
                trials.append(
                    Trial(
                        tester_id=row["tester_id"],
                        round=int(row["round"]),
                        rep=int(row["rep"]),
                        index=int(row["index"]),
                        prompted=row["prompted_sign"],
                        predicted=row["predicted_sign"],
                        confidence=float(row["confidence"]),
                        latency_ms=int(row["latency_ms"]),
                        timestamp_ms=int(row["timestamp_ms"]),
                        seed=int(row["seed"]),
                        dominance=row["dominance"],
                        note=row.get("note", "") or "",
                    )
                )
            except (ValueError, KeyError) as e:
                print(f"  skip row in {path.name}: {e}", file=sys.stderr)
    return trials


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f_ = int(k)
    c_ = min(f_ + 1, len(s) - 1)
    if f_ == c_:
        return float(s[f_])
    return s[f_] + (s[c_] - s[f_]) * (k - f_)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _md_confusion(cm: List[List[int]], labels: List[str]) -> str:
    header = ["true \\ pred"] + labels
    rows = []
    for i, row in enumerate(cm):
        rows.append([labels[i]] + [str(v) for v in row])
    return _md_table(header, rows)


def _vocabulary() -> List[str]:
    path = PROJECT_ROOT / "data" / "sequences" / "vocabulary.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return list(d.keys())
    return []


def _verdict(value: float, target: float) -> str:
    if value >= target:
        return f"GO ({value:.2%} ≥ {target:.0%})"
    return f"NO-GO ({value:.2%} < {target:.0%})"


def render_report(
    agg: Aggregates,
    template_text: str,
    *,
    n_users: int,
) -> str:
    vocab = _vocabulary()
    if not vocab:
        vocab = sorted({t.prompted for t in agg.trials} | {t.predicted for t in agg.trials})

    r1_c, r1_n, r1_acc = agg.overall_accuracy(1)
    r2_c, r2_n, r2_acc = agg.overall_accuracy(2)
    rel = (r2_acc - r1_acc) / r1_acc if r1_acc > 0 else 0.0

    per_user_r1 = agg.per_user_accuracy(1)
    per_user_r2 = agg.per_user_accuracy(2)
    per_sign_r1 = agg.per_sign_accuracy(1)
    per_sign_r2 = agg.per_sign_accuracy(2)
    dominance = agg.per_user_dominance()

    lat_r1 = agg.latencies(1)
    lat_r2 = agg.latencies(2)
    med_r1 = statistics.median(lat_r1) if lat_r1 else 0.0
    med_r2 = statistics.median(lat_r2) if lat_r2 else 0.0
    p95_r1 = _percentile(lat_r1, 0.95)
    p95_r2 = _percentile(lat_r2, 0.95)

    cm_r1 = agg.confusion_matrix(1, vocab)
    cm_r2 = agg.confusion_matrix(2, vocab)

    per_user_rows: List[List[str]] = []
    for uid in sorted(set(list(per_user_r1.keys()) + list(per_user_r2.keys()))):
        c1, n1, a1 = per_user_r1.get(uid, (0, 0, 0.0))
        c2, n2, a2 = per_user_r2.get(uid, (0, 0, 0.0))
        delta = a2 - a1
        per_user_rows.append(
            [uid, dominance.get(uid, "?"), f"{c1}/{n1}", f"{a1:.2%}", f"{c2}/{n2}", f"{a2:.2%}", f"{delta:+.2%}"]
        )

    per_sign_rows: List[List[str]] = []
    for sign in vocab:
        c1, n1, a1 = per_sign_r1.get(sign, (0, 0, 0.0))
        c2, n2, a2 = per_sign_r2.get(sign, (0, 0, 0.0))
        delta = a2 - a1
        per_sign_rows.append(
            [sign, f"{c1}/{n1}", f"{a1:.2%}", f"{c2}/{n2}", f"{a2:.2%}", f"{delta:+.2%}"]
        )

    summary_table = _md_table(
        ["Métrica", "Ronda 1 (sin calib)", "Ronda 2 (con calib)", "Δ"],
        [
            ["Accuracy global", f"{r1_acc:.2%} ({r1_c}/{r1_n})", f"{r2_acc:.2%} ({r2_c}/{r2_n})", f"{r2_acc - r1_acc:+.2%}"],
            ["Latencia mediana (ms)", f"{med_r1:.0f}", f"{med_r2:.0f}", f"{med_r2 - med_r1:+.0f}"],
            ["Latencia p95 (ms)", f"{p95_r1:.0f}", f"{p95_r2:.0f}", f"{p95_r2 - p95_r1:+.0f}"],
        ],
    )

    user_table = _md_table(
        ["Tester", "Mano dom.", "R1 (correctos/n)", "R1 acc", "R2 (correctos/n)", "R2 acc", "Δ"],
        per_user_rows,
    )
    sign_table = _md_table(
        ["Seña", "R1 (c/n)", "R1 acc", "R2 (c/n)", "R2 acc", "Δ"],
        per_sign_rows,
    )

    cm_r1_md = _md_confusion(cm_r1, vocab)
    cm_r2_md = _md_confusion(cm_r2, vocab)

    veredict_r1 = _verdict(r1_acc, TARGET_NO_CALIB)
    veredict_r2 = _verdict(r2_acc, TARGET_WITH_CALIB)

    placeholders = {
        "{{N_USERS}}": str(n_users),
        "{{N_TRIALS_R1}}": str(r1_n),
        "{{N_TRIALS_R2}}": str(r2_n),
        "{{ACC_R1}}": f"{r1_acc:.2%}",
        "{{ACC_R2}}": f"{r2_acc:.2%}",
        "{{REL_IMPROVEMENT}}": f"{rel:+.2%}",
        "{{MED_LAT_R1}}": f"{med_r1:.0f}",
        "{{MED_LAT_R2}}": f"{med_r2:.0f}",
        "{{P95_LAT_R1}}": f"{p95_r1:.0f}",
        "{{P95_LAT_R2}}": f"{p95_r2:.0f}",
        "{{VEREDICT_R1}}": veredict_r1,
        "{{VEREDICT_R2}}": veredict_r2,
        "{{TARGET_NO_CALIB}}": f"{TARGET_NO_CALIB:.0%}",
        "{{TARGET_WITH_CALIB}}": f"{TARGET_WITH_CALIB:.0%}",
        "{{REFERENCE_LOSO}}": f"{REFERENCE_LOSO_TRANSFORMER:.2%}",
        "{{USER_TABLE}}": user_table,
        "{{SIGN_TABLE}}": sign_table,
        "{{CM_R1}}": cm_r1_md,
        "{{CM_R2}}": cm_r2_md,
        "{{SUMMARY_TABLE}}": summary_table,
    }

    out = template_text
    for k, v in placeholders.items():
        out = out.replace(k, v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default="reports/f7", help="directorio con los CSV P0X_round{1,2}.csv")
    parser.add_argument("--output", default="docs/F7_USER_TEST_REPORT.md", help="ruta del reporte markdown")
    parser.add_argument(
        "--template",
        default="docs/F7_USER_TEST_REPORT.template.md",
        help="plantilla markdown con placeholders {{...}}",
    )
    args = parser.parse_args()

    csv_dir = (PROJECT_ROOT / args.csv_dir).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    template = (PROJECT_ROOT / args.template).resolve()

    if not csv_dir.exists():
        print(f"ERROR: {csv_dir} no existe", file=sys.stderr)
        return 2

    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        print(f"ERROR: no hay CSVs en {csv_dir}", file=sys.stderr)
        return 2

    agg = Aggregates()
    testers: set = set()
    for p in csvs:
        ts = _load_csv(p)
        agg.trials.extend(ts)
        testers.update(t.tester_id for t in ts)
        print(f"  loaded {p.name}: {len(ts)} trials")

    if not template.exists():
        print(f"ERROR: plantilla {template} no existe", file=sys.stderr)
        return 2

    rendered = render_report(agg, template.read_text(encoding="utf-8"), n_users=len(testers))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"\nWrote {output} ({len(rendered)} chars, {len(testers)} testers, {len(agg.trials)} trials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
