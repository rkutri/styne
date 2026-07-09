"""Scaling benchmark: DNA spectral operators vs a dense-Cholesky reference.

Log-determinant is not timed: DNA's is structurally 0 (whitened prior), the dense
one is O(N) after the factorisation already counted in setup. Both run at matched
N; DNA continues past the largest N a dense factor reaches.
"""

#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import platform
import socket
import statistics
import subprocess
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from styne.gp.direct import DirectGPEngine
from styne.gp.dna import DNAFourierEngine
from styne.statistics.stationary import MaternCovariance1D
from styne.utility.grid import Grid

SETUP, APPLY = "operator_setup", "operator_application"
DNA, DENSE = "dna_spectral", "dense_cholesky_reference"


# --- provenance -------------------------------------------------------------

def _git(field: str) -> str:
    cmd = {"commit": ["git", "rev-parse", "--short", "HEAD"],
           "dirty": ["git", "status", "--porcelain"]}[field]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
    if field == "commit":
        return out.strip() or "unknown"
    return "dirty" if out.strip() else "clean"


def effective_blas_threads() -> str:
    try:
        import threadpoolctl
        counts = [i["num_threads"] for i in threadpoolctl.threadpool_info()
                  if "num_threads" in i]
        return str(max(counts)) if counts else "unrecorded"
    except Exception:
        return "unrecorded"


def build_provenance(configName: str) -> Dict[str, str]:
    import scipy
    import styne
    from importlib.metadata import PackageNotFoundError, version
    try:
        styneVersion = version("styne")
    except PackageNotFoundError:
        styneVersion = getattr(styne, "__version__", "unknown")
    return {
        "script": "dna_scaling.py", "config": configName,
        "git_commit": _git("commit"), "git_state": _git("dirty"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname().split(".")[0],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "styne_version": styneVersion,
        "blas_threads_requested": os.environ.get("OMP_NUM_THREADS", "1"),
        "blas_threads_effective": effective_blas_threads(),
    }


# --- measurement ------------------------------------------------------------

def time_median(action: Callable[[], Any], repeats: int) -> float:
    """Median wall-clock over repeats, tracing off."""
    ds = []
    for _ in range(repeats):
        t = time.perf_counter()
        action()
        ds.append(time.perf_counter() - t)
    return statistics.median(ds)


def peak_heap(action: Callable[[], Any]) -> int:
    """Python-heap peak for one call (separate pass; tracemalloc misses native
    BLAS/LAPACK workspace, so treat as a lower bound)."""
    tracemalloc.start()
    try:
        action()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def measure_dna(N: int, repeats: int) -> Dict[str, Tuple[float, int, int]]:
    interior = N - 2
    spectral = 2 * interior + 1
    cov = MaternCovariance1D(0.1, 2.5, 1.0)

    # Fresh engine per call
    def setup():
        DNAFourierEngine(interior, d=1, alpha=1.0).build_covariance(cov)

    setupTime, setupMem = time_median(setup, repeats), peak_heap(setup)

    engine = DNAFourierEngine(interior, d=1, alpha=1.0)
    engine.build_covariance(cov)
    real = engine.build_realisation()
    real.spectralWeights = engine.spectralWeights
    real.coefficient = np.ones(spectral, dtype=float)

    field = real.evaluate_native()
    assert field.shape[0] == N and np.isfinite(field).all() and np.std(field) > 0, \
        f"evaluate_native did not populate: shape={field.shape}, std={np.std(field)}"

    apply_ = real.evaluate_native
    applyTime, applyMem = time_median(apply_, repeats), peak_heap(apply_)
    storage = 8 * spectral
    return {SETUP: (setupTime, setupMem, storage),
            APPLY: (applyTime, applyMem, storage)}


def measure_dense(N: int, repeats: int) -> Dict[str, Tuple[float, int, int]]:
    grid = Grid(np.linspace(0.0, 1.0, N))
    cov = MaternCovariance1D(0.1, 2.5, 1.0)
    nuggets = [0.0, 1e-8, 1e-6]

    for nugget in nuggets:
        engine = DirectGPEngine(grid, nugget=nugget)
        try:
            engine.build_covariance(cov)
            break
        except np.linalg.LinAlgError:
            if nugget == nuggets[-1]:
                raise
            print(f"  nugget {nugget:.0e} failed at N={N}, retrying")

    def setup():
        DirectGPEngine(grid, nugget=nugget).build_covariance(cov)

    setupTime, setupMem = time_median(setup, repeats), peak_heap(setup)

    v = np.ones(N, dtype=float)
    apply_ = lambda: engine._shapeCovariance.apply_chol_factor(v)
    applyTime, applyMem = time_median(apply_, repeats), peak_heap(apply_)
    storage = 8 * N * N
    return {SETUP: (setupTime, setupMem, storage),
            APPLY: (applyTime, applyMem, storage)}


def fit_slopes(rows: List[Dict[str, Any]], tailFrom: int) -> List[Dict[str, Any]]:
    """Fit log-log slopes on N >= tailFrom (below it, per-call overhead flattens
    the curve). Reports a shared-range fit (N up to the largest the dense
    reference reached) for both methods, plus a full-range fit for DNA including
    DNA-only sizes."""
    denseMaxOk = max(
        [int(r["N"]) for r in rows
         if r["method"] == DENSE and r["status"] == "ok"] or [tailFrom])
    print(f"\n--- log-log slopes vs N (N>={tailFrom}) ---")
    fits: List[Dict[str, Any]] = []
    for op in (SETUP, APPLY):
        for method in (DNA, DENSE):
            scopes = [("shared", denseMaxOk)]
            if method == DNA:
                scopes.append(("full", 10 ** 12))
            for scope, nCap in scopes:
                pts = [r for r in rows if r["op"] == op and r["method"] == method
                       and r["status"] == "ok"
                       and tailFrom <= int(r["N"]) <= nCap]
                if len(pts) < 2:
                    continue
                nv = [int(r["N"]) for r in pts]
                slope = float(np.polyfit(np.log(nv),
                              np.log([float(r["median_time_s"]) for r in pts]), 1)[0])
                print(f"  {op:<20} {method:<24} {scope:<6} slope={slope:5.2f}"
                      f"  (N {min(nv)}..{max(nv)}, {len(pts)} pts)")
                fits.append({"op": op, "method": method, "scope": scope,
                             "slope": slope, "n_min": min(nv), "n_max": max(nv),
                             "n_points": len(pts)})
    return fits


# --- summary table ----------------------------------------------------------

def write_summary_table(rows: List[Dict[str, Any]], outPath: Path) -> None:
    """One row per N, wide format: DNA vs dense times, speedup ratios, storage."""
    def look(N, op, method):
        for r in rows:
            if (int(r["N"]) == N and r["op"] == op and r["method"] == method):
                return r
        return None

    ns = sorted({int(r["N"]) for r in rows})
    fields = ["N", "d_dna",
              "dna_setup_s", "dense_setup_s", "setup_speedup_x",
              "dna_apply_s", "dense_apply_s", "apply_speedup_x",
              "dense_storage_bytes", "dense_status"]
    table: List[Dict[str, Any]] = []
    for N in ns:
        dSetup = look(N, SETUP, DNA)
        cSetup = look(N, SETUP, DENSE)
        dApply = look(N, APPLY, DNA)
        cApply = look(N, APPLY, DENSE)

        def ratio(dense, dna):
            try:
                dv, nv = float(dense["median_time_s"]), float(dna["median_time_s"])
                return round(dv / nv, 2) if np.isfinite(dv) and nv > 0 else ""
            except (TypeError, ValueError):
                return ""

        table.append({
            "N": N,
            "d_dna": dSetup["d_dna"] if dSetup else "",
            "dna_setup_s": f"{float(dSetup['median_time_s']):.3e}" if dSetup else "",
            "dense_setup_s": (f"{float(cSetup['median_time_s']):.3e}"
                              if cSetup and cSetup["status"] == "ok" else ""),
            "setup_speedup_x": ratio(cSetup, dSetup) if (cSetup and cSetup["status"] == "ok") else "",
            "dna_apply_s": f"{float(dApply['median_time_s']):.3e}" if dApply else "",
            "dense_apply_s": (f"{float(cApply['median_time_s']):.3e}"
                              if cApply and cApply["status"] == "ok" else ""),
            "apply_speedup_x": ratio(cApply, dApply) if (cApply and cApply["status"] == "ok") else "",
            "dense_storage_bytes": cSetup["estimated_core_array_bytes"] if cSetup else "",
            "dense_status": cSetup["status"] if cSetup else "",
        })

    tPath = Path(str(outPath).replace(".csv", "_table.csv"))
    with open(tPath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    print(f"saved: {tPath}")

    # console preview
    print("\n  N        dna_setup   dense_setup  x     dna_apply   dense_apply  x")
    for r in table:
        print(f"  {r['N']:<8} {r['dna_setup_s']:<11} {r['dense_setup_s']:<12} "
              f"{str(r['setup_speedup_x']):<5} {r['dna_apply_s']:<11} "
              f"{r['dense_apply_s']:<12} {r['apply_speedup_x']}")


# --- plot -------------------------------------------------------------------

def plot(rows: List[Dict[str, Any]], outPath: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def series(op, method):
        pts = sorted([r for r in rows if r["op"] == op and r["method"] == method
                      and r["status"] == "ok"], key=lambda r: int(r["N"]))
        return ([int(r["N"]) for r in pts],
                [float(r["median_time_s"]) for r in pts])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, op, title in ((axes[0], SETUP, "operator setup"),
                          (axes[1], APPLY, "operator application")):
        xd, yd = series(op, DNA)
        xc, yc = series(op, DENSE)
        if xd:
            ax.loglog(xd, yd, "o-", label="DNA spectral", color="#1f77b4")
        if xc:
            ax.loglog(xc, yc, "s-", label="dense Cholesky ref.", color="#d62728")
        if xd:
            x0, y0 = xd[0], yd[0]
            xr = np.array([x0, xd[-1]], dtype=float)
            ax.loglog(xr, y0 * (xr / x0), ":", color="gray", lw=1, label="slope 1")
            ax.loglog(xr, y0 * (xr / x0) ** 3, "--", color="gray", lw=1, label="slope 3")
        ax.set_title(title)
        ax.set_xlabel("observation sites N")
        ax.set_ylabel("median time (s)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    imgPath = Path(str(outPath).replace(".csv", ".png"))
    fig.savefig(imgPath, dpi=140)
    plt.close(fig)
    print(f"saved: {imgPath}")


# --- driver -----------------------------------------------------------------

def row(N, dDna, op, method, t, mem, storage, status) -> Dict[str, Any]:
    return {"N": N, "d_dna": dDna, "d_dense": N, "op": op, "method": method,
            "median_time_s": t, "tracemalloc_peak_bytes": mem,
            "estimated_core_array_bytes": storage, "status": status,
            "blas_threads_requested": os.environ.get("OMP_NUM_THREADS", "1"),
            "blas_threads_effective": effective_blas_threads()}


def run(isProduction: bool) -> None:
    configName = "production" if isProduction else "smoke"
    repeats = 10 if isProduction else 5
    if isProduction:
        shared = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
        dnaOnly = [25600, 51200, 102400, 204800]
    else:
        shared, dnaOnly = [12, 22, 52, 102, 202, 502], []

    # Below tailFrom, fixed per-call overhead dominates and flattens the slope.
    tailFrom = 200

    rows: List[Dict[str, Any]] = []
    denseFailed = False
    print(f"=== scaling_reference ({configName}) ===")

    for N in shared + dnaOnly:
        spectral = 2 * (N - 2) + 1
        print(f"N={N} (d_dna={spectral}) ...")
        dna = measure_dna(N, repeats)
        for op in (SETUP, APPLY):
            t, mem, st = dna[op]
            rows.append(row(N, spectral, op, DNA, t, mem, st, "ok"))

        if N in dnaOnly or denseFailed:
            status = "not_attempted" if N in dnaOnly else "skipped_prior_failure"
            for op in (SETUP, APPLY):
                rows.append(row(N, spectral, op, DENSE, np.nan, 0, 8 * N * N, status))
            continue
        try:
            dense = measure_dense(N, repeats)
            for op in (SETUP, APPLY):
                t, mem, st = dense[op]
                rows.append(row(N, spectral, op, DENSE, t, mem, st, "ok"))
        except (MemoryError, np.linalg.LinAlgError) as exc:
            denseFailed = True
            print(f"  dense failed at N={N}: {type(exc).__name__}")
            for op in (SETUP, APPLY):
                rows.append(row(N, spectral, op, DENSE, np.nan, 0, 8 * N * N,
                                f"failed:{type(exc).__name__}"))

    fits = fit_slopes(rows, tailFrom)

    fields = ["N", "d_dna", "d_dense", "op", "method", "median_time_s",
              "tracemalloc_peak_bytes", "estimated_core_array_bytes", "status",
              "blas_threads_requested", "blas_threads_effective"]
    outDir = Path(__file__).parent / "results"
    outDir.mkdir(parents=True, exist_ok=True)
    prov = build_provenance(configName)
    outPath = outDir / f"scaling_{configName}_{prov['git_commit']}_{prov['hostname']}.csv"
    with open(outPath, "w", newline="", encoding="utf-8") as fh:
        for k, v in prov.items():
            fh.write(f"# {k}: {v}\n")
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {outPath}")

    if fits:
        sp = Path(str(outPath).replace(".csv", "_slopes.csv"))
        with open(sp, "w", newline="", encoding="utf-8") as fh:
            for k, v in prov.items():
                fh.write(f"# {k}: {v}\n")
            fh.write(f"# tail_from_N: {tailFrom}\n")
            w = csv.DictWriter(fh, fieldnames=["op", "method", "scope", "slope",
                                               "n_min", "n_max", "n_points"])
            w.writeheader()
            w.writerows(fits)
        print(f"saved: {sp}")

    write_summary_table(rows, outPath)
    plot(rows, outPath)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DNA vs dense-Cholesky scaling.")
    p.add_argument("--production", action="store_true",
                   help="Full grid instead of smoke grid.")
    run(p.parse_args().production)
