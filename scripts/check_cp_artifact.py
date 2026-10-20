import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scripts.random_streams import generate_random_drift_stream

for mode in ["mixed", "mixed_decoupled"]:
    print(f"\n--- MODE: {mode} ---")
    for s in range(3000, 3005):
        df, cps = generate_random_drift_stream(s, drift_mode=mode)
        inf = df.attrs["informative"]
        feats = [f"feature_{j}" for j in range(15)]
        wins = {w: g[feats].mean().values for w, g in df.groupby((df.TransactionDT // 2.592e6).astype(int))}
        for c in cps:
            old, new = inf[c-1], inf[c]
            before = np.abs(wins[c] - wins[c-1])        # shift entering the CP window
            after = np.abs(wins[c+1] - wins[c-1])
            print(s, c, "old-informative shift:", after[old].mean().round(3),
                  "| other shift:", np.delete(after, old).mean().round(3))
