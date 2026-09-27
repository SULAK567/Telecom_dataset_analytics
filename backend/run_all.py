# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

"""
Run the entire pipeline end to end.

    python3 backend/run_all.py            # everything
    python3 backend/run_all.py --from 06  # resume from a given step

Each step is a standalone script, so any step can also be run on its own.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATASET_SCRIPTS_DIR = REPO_ROOT / "datasets" / "scripts"

STEPS = [
    ("01", "01_build_tower_table.py",        "Merge raw 404/405, decode carriers, join states"),
    ("02", "02_quality_and_clean.py",        "Data-quality audit, coordinate filter, dedupe"),
    ("03", "03_aggregate_state.py",          "Phase 2 — state-level aggregation"),
    ("04", "04_fetch_districts.py",          "Phase 3a — download district boundaries"),
    ("05", "05_aggregate_district.py",       "Phase 3b — district spatial join + aggregation"),
    ("06", "06_feature_engineering.py",      "Phase 4 — features + rule-based tier"),
    ("07", "07_eda.py",                      "Phase 5 — exploratory figures"),
    ("08", "08_clustering.py",               "Phase 6 — K-Means + DBSCAN"),
    ("09", "09_supervised.py",               "Phase 7 — Random Forest + SHAP"),
    ("10", "10_prepare_dashboard_assets.py", "Phase 8a — simplify boundaries"),
]

# Step 01 is the original build script; it took ~10 min on the full 2.5M
# rows. It is skipped by default because its output (tower_table_clean.csv)
# is already present, and its script is no longer kept in this repo.
DEFAULT_SKIP = {"01"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="02",
                    help="step number to start from (default 02)")
    ap.add_argument("--include-01", action="store_true",
                    help="also re-run the slow raw-data build step")
    args = ap.parse_args()

    skip = set() if args.include_01 else DEFAULT_SKIP
    started = False
    results = []

    print("=" * 78)
    print("TELECOM COVERAGE ANALYTICS — FULL PIPELINE")
    print("=" * 78)

    for num, script, desc in STEPS:
        if num == args.start:
            started = True
        if not started or num in skip:
            print(f"\n[{num}] SKIPPED — {desc}")
            results.append((num, script, "skipped", 0.0))
            continue

        path = HERE / script if (HERE / script).exists() else DATASET_SCRIPTS_DIR / script
        if not path.exists():
            print(f"\n[{num}] SKIPPED — script not found ({script})")
            results.append((num, script, "skipped", 0.0))
            continue

        print(f"\n{'=' * 78}\n[{num}] {desc}\n     {path.name}\n{'=' * 78}")

        t0 = time.time()
        proc = subprocess.run([sys.executable, "-u", str(path)], cwd=str(HERE))
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"\n[{num}] FAILED after {dt:.1f}s (exit {proc.returncode})")
            results.append((num, script, "FAILED", dt))
            summarise(results)
            sys.exit(proc.returncode)

        print(f"\n[{num}] done in {dt:.1f}s")
        results.append((num, script, "ok", dt))

    summarise(results)
    print("\nLaunch the dashboard with:\n    streamlit run frontend/dashboard.py")


def summarise(results):
    print("\n" + "=" * 78)
    print("PIPELINE SUMMARY")
    print("=" * 78)
    total = 0.0
    for num, script, status, dt in results:
        total += dt
        print(f"  [{num}] {script:<34s} {status:<8s} {dt:>7.1f}s")
    print(f"  {'TOTAL':<39s} {total:>16.1f}s")


if __name__ == "__main__":
    main()
