"""Exploratory, coverage-relaxed tests from CACHED raw sweeps. No re-running of any policy."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_budget_curves import compute_metrics, run_family, PRIMARY_TESTS

RAW = {
    "decoupled": ROOT/"results"/"budget_curves_mixed_decoupled_s4000_4029_confirmatory_raw.csv",
    "mixed":     ROOT/"results"/"budget_curves_mixed_s3000_3029_confirmatory_raw.csv",
}

def main():
    for name, path in RAW.items():
        m = compute_metrics(pd.read_csv(path))
        out = []
        for metric, mc in (("auc", 0.5), ("acc@4", 0.6)):
            t, _ = run_family(m, metric, "primary_relaxed", PRIMARY_TESTS, min_cov=mc)
            out.append(t)
        res = pd.concat(out, ignore_index=True)
        res.to_csv(ROOT/"results"/f"{name}_RELAXED_exploratory_tests.csv", index=False)
        print(name)
        print(res[["ref","vs","metric","diff_mean","ci_95_lo","ci_95_hi","p_val","p_holm"]].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
