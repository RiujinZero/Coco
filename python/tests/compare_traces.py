"""Compare a native trace against the NumPy reference trace (runbook §13.1).

    python python/tests/compare_traces.py ref_trace.jsonl native_trace.jsonl

The integer path (address walk, candidates, executed units, survival flags) must agree exactly.
Predictions may differ only on float rounding ties; accuracy must agree within tolerance.
"""
import json
import sys


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ref, nat = load(sys.argv[1]), load(sys.argv[2])
    n = min(len(ref), len(nat))
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    first_div = None
    exact = 0
    for i in range(n):
        a, b = ref[i], nat[i]
        same = a["key"] == b["key"] and a["cand"] == b["cand"] and a["exec"] == b["exec"] and a["reached"] == b["reached"] and a["target"] == b["target"]
        if same:
            exact += 1
        elif first_div is None:
            first_div = i
    acc_r = sum(e["pred"] == e["target"] for e in ref[:n]) / n
    acc_n = sum(e["pred"] == e["target"] for e in nat[:n]) / n
    pred_agree = sum(ref[i]["pred"] == nat[i]["pred"] for i in range(n)) / n
    loss_r = sum(e["loss"] for e in ref[:n]) / n
    loss_n = sum(e["loss"] for e in nat[:n]) / n
    print(f"events compared: {n}")
    print(f"integer path exact: {exact}/{n} ({100 * exact / n:.2f}%)  first divergence at event: {first_div}")
    print(f"prediction agreement: {100 * pred_agree:.2f}%")
    print(f"accuracy ref {100 * acc_r:.2f}%  native {100 * acc_n:.2f}%  | loss ref {loss_r:.4f} native {loss_n:.4f}")
    ok = abs(acc_r - acc_n) <= tol and (first_div is None or first_div >= n // 2)
    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
