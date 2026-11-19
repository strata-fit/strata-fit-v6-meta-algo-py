"""Compare local sklearn MICE vs. federated MICE (meta-algo repo, optionally the standalone repo).

Ported from strata-fit-v6-imputation-py test/compare_fed_vs_local_mice.py (Josef Baliko).
Lines marked META-ALGO differ from the original: the meta-algo MICE is now the iterative
version (same algorithm as the standalone repo), so it is the primary federated method here
and the standalone repo is the optional comparison.

PRIVACY: this script never writes row-level data. It only prints / saves aggregate summary
statistics (errors, rates, coefficients, per-column means/SDs). Sites are simulated in-process.
Summary CSVs + report.txt are ALWAYS written to --out-dir (default ./fed_vs_local_out).
The whole report (comparison first, diagnostics last) is printed once, at the very end of all runs.

Masked-truth design: a fraction of OBSERVED values in --cols is hidden (MCAR), imputed by every
method, and scored against the truth.

  meta  : strata-fit-v6-meta-algo-py (this repo)    - global means + chained multi-round MICE
  stand : strata-fit-v6-imputation-py (optional)     - same algorithm (n/a if not importable)

--aux : auxiliary predictor-only columns (used and imputed along the way, never masked or scored)


Example:
  python tests/compare_fed_vs_local_mice.py --csv path/to/data.csv \
      --cols DAS28 CRP ESR --aux SJC28 TJC28 Age_diagnosis --n-sites 3 --max-iter 5 \
      --group-col pat_ID --reps 5 --diagnose
"""
import argparse, contextlib, io, logging, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

warnings.filterwarnings("ignore")

# META-ALGO: this repo is required, the standalone repo is optional (swapped vs. original)
from strata_fit_v6_meta_algo_py.imputation import MiceImputer as MetaMice, fit_mice_model
try:
    from strata_fit_v6_imputation_py import run_local_imputation
    from strata_fit_v6_imputation_py.imputation_strategies.mice import MiceImputer as StandMice
except Exception:  # standalone repo not installed -> standalone rows stay n/a
    StandMice = None

_OUT = []
_TABLES = {}  # name -> DataFrame, written as CSV when --save-dir is given (aggregate only)


def say(text=""):
    """Collect a report line. Nothing is printed until ALL runs are finished (see __main__),
    so framework chatter can never push the report away."""
    _OUT.append(text)


@contextlib.contextmanager
def quiet():
    """Silence the framework's 'INFO: ...' chatter so it cannot bury the report."""
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            yield
        finally:
            logging.disable(logging.NOTSET)


# ----------------------------------------------------------------------------- helpers
def split_sites(df, n, seed, by=None):
    rng = np.random.default_rng(seed)
    if by:
        keys = df[by].unique(); rng.shuffle(keys)
        return [df[df[by].isin(k)].reset_index(drop=True) for k in np.array_split(keys, n)]
    return [df.iloc[i].reset_index(drop=True) for i in np.array_split(rng.permutation(len(df)), n)]


def fit_stand(sites, cols, max_iter):
    with quiet():
        return run_local_imputation(sites, imputation_config={
            "schema_version": 1, "strategy": "mice",
            "parameters": {"columns": cols, "max_iter": max_iter}})


def fed_stand(sites, cols, max_iter):
    cfg, m = fit_stand(sites, cols, max_iter), StandMice()
    return [pd.DataFrame(m.impute(s.copy(), cfg)) for s in sites]


# META-ALGO: iterative fit through fit_mice_model (was a single compute/aggregate round)
def fit_meta(sites, cols, max_iter):
    m = MetaMice()
    with quiet():
        return fit_mice_model(
            lambda state: [m.compute(s, cols, global_state=state) for s in sites],
            columns=cols, max_iter=max_iter)


def fed_meta(sites, cols, max_iter):
    cfg, m = fit_meta(sites, cols, max_iter), MetaMice()
    return [pd.DataFrame(m.impute(s.copy(), cfg)) for s in sites]


def local_mice(df, cols, est, max_iter, seed=0):
    return IterativeImputer(estimator=est, max_iter=max_iter, random_state=seed).fit_transform(
        df[cols].to_numpy(float))


def reassemble(parts, rid, cols):
    return pd.concat(parts).set_index("_rid").loc[rid][cols].to_numpy(float)


def nrmse_cols(pred, truth, mask):
    """Per-column RMSE / SD(truth); NaN for columns with nothing hidden (e.g. aux)."""
    out = np.full(pred.shape[1], np.nan)
    for j in range(pred.shape[1]):
        m = mask[:, j]
        if m.any():
            out[j] = np.sqrt(np.mean((pred[m, j] - truth[m, j]) ** 2)) / np.nanstd(truth[:, j])
    return out


# ----------------------------------------------------------------------------- experiment
def run(df, cols, aux, n_sites, frac, reps, seed, by, max_iter):
    allc = cols + aux
    score_idx = np.arange(len(cols))  # aux columns come last and are never masked
    per_rep, per_col = [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        truth = df[allc].to_numpy(float)
        hide = np.zeros_like(truth, dtype=bool)
        hide[:, score_idx] = ~np.isnan(truth[:, score_idx]) & (rng.random((len(df), len(cols))) < frac)
        X = truth.copy(); X[hide] = np.nan
        d = df.copy(); d[allc] = X
        sites, rid = split_sites(d, n_sites, seed + r, by), d["_rid"]

        preds = {
            "mean": np.where(np.isnan(X), np.nanmean(X, 0), X),
            "local_BayesRidge": local_mice(d, allc, BayesianRidge(), max_iter),
            "local_OLS": local_mice(d, allc, LinearRegression(), max_iter),
            # META-ALGO: standalone optional, meta always available and iterative
            "fed_standalone": (reassemble(fed_stand([s.copy() for s in sites], allc, max_iter), rid, allc)
                               if StandMice else np.full_like(truth, np.nan)),
            "fed_meta": reassemble(fed_meta([s.copy() for s in sites], allc, max_iter), rid, allc),
        }
        alt = reassemble(fed_meta(split_sites(d, n_sites + 1, seed + 99 + r, by), allc, max_iter), rid, allc)
        cols_err = {k: nrmse_cols(v, truth, hide) for k, v in preds.items()}
        row = {k: np.nanmean(v) for k, v in cols_err.items()}
        row["fed_standalone_vs_localOLS"] = np.nanmean(nrmse_cols(preds["fed_standalone"], preds["local_OLS"], hide))
        row["fed_meta_vs_localOLS"] = np.nanmean(nrmse_cols(preds["fed_meta"], preds["local_OLS"], hide))
        row["split_invariance_maxabs"] = float(np.max(np.abs(preds["fed_meta"] - alt)))  # META-ALGO
        per_rep.append(row)
        per_col.append(pd.DataFrame(cols_err, index=allc).loc[cols])
        print(f"  rep {r + 1}/{reps} done", file=sys.stderr, flush=True)
    return pd.DataFrame(per_rep), sum(per_col) / len(per_col)


# ----------------------------------------------------------------------------- reporting
def recovery(base, method, ref):
    """% of the improvement local MICE achieves over the mean baseline that `method` also achieves."""
    gain = base - ref
    return np.nan if not np.isfinite(gain) or gain < 1e-6 else 100 * (base - method) / gain


def verdict(rec):
    if np.isnan(rec): return "n/a"
    if rec >= 95: return "essentially equivalent to local MICE"
    if rec >= 80: return "close to local MICE"
    if rec >= 50: return "usable, clearly weaker than local MICE"
    if rec >= 0:  return "barely better than mean imputation"
    return "WORSE than mean imputation"


def report(res, percol, a):
    m, sd = res.mean(), res.std(ddof=0)
    base, ref = m["mean"], m["local_OLS"]
    say("\n" + "=" * 74)
    say(f"RESULT  ({a.reps} reps, {a.n_sites} simulated sites, hidden={a.frac:.0%}, max_iter={a.max_iter}, "
        f"scored cols={len(a.cols)}, aux cols={len(a.aux)})")
    say("=" * 74)
    say("\n1) Error on hidden values (nRMSE, lower = better; mean-imputation = ~1.0)")
    say(f"   {'method':<22}{'nRMSE':>8}{'± sd':>8}   {'improvement recovered*':>24}")
    for k, label in [("mean", "mean baseline"), ("local_BayesRidge", "local sklearn (BR)"),
                     ("local_OLS", "local sklearn (OLS)"), ("fed_standalone", "federated standalone"),
                     ("fed_meta", "federated meta-algo")]:
        if np.isnan(m[k]):
            say(f"   {label:<22}{'n/a':>8}"); continue
        rec = "" if k == "mean" else f"{recovery(base, m[k], ref):>22.0f} %"
        say(f"   {label:<22}{m[k]:>8.4f}{sd[k]:>8.4f}   {rec}")
    say("   *100% = as good as local OLS MICE, 0% = no better than the column mean")

    rows = []
    for k in ["mean", "local_BayesRidge", "local_OLS", "fed_standalone", "fed_meta"]:
        rows.append({"method": k, "nRMSE": m[k], "nRMSE_sd_over_reps": sd[k],
                     "improvement_recovered_pct": np.nan if k == "mean" else recovery(base, m[k], ref),
                     "gap_to_local_OLS": np.nan if k == "mean" else m[k] - ref})
    _TABLES["summary_methods"] = pd.DataFrame(rows)
    _TABLES["per_rep_results"] = res.reset_index().rename(columns={"index": "rep"})
    _TABLES["per_column_nRMSE"] = percol.reset_index().rename(columns={"index": "column"})
    say("\n2) Quick verdict")
    for k, label in [("fed_standalone", "standalone"), ("fed_meta", "meta-algo ")]:
        if not np.isnan(m[k]):
            rec = recovery(base, m[k], ref)
            say(f"   {label}: {rec:.0f}% recovered, gap to local = {m[k]-ref:+.4f} nRMSE -> {verdict(rec)}")
    if not np.isnan(m["fed_standalone"]):  # META-ALGO: standalone is the optional one
        gap = m["fed_meta"] - m["fed_standalone"]
        say(f"   standalone vs meta-algo: {abs(gap):.4f} nRMSE in favour of "
            f"{'standalone' if gap > 0 else 'meta-algo'}")

    say("\n3) Distance of imputed values to local MICE (nRMSE, 0 = identical)")
    say(f"   meta-algo: {m['fed_meta_vs_localOLS']:.4f}"  # META-ALGO: meta first, standalone optional
        + (f"   standalone: {m['fed_standalone_vs_localOLS']:.4f}" if not np.isnan(m['fed_standalone_vs_localOLS']) else ""))

    say("\n4) Per-column nRMSE (mean over reps; aux columns are not scored)")
    say(percol.rename(columns={"local_BayesRidge": "loc_BR", "local_OLS": "loc_OLS",
                               "fed_standalone": "fed_stand"})
        .round(3).dropna(axis=1, how="all").to_string())

    inv = res["split_invariance_maxabs"].max()
    say("\n5) Sanity checks")
    say(f"   split invariance (N vs N+1 sites), max abs diff: {inv:.2e} -> "
        f"{'OK' if inv < 1e-6 else 'PROBLEM: result depends on site split'}")
    if m["fed_meta"] >= base:  # META-ALGO
        say("   WARNING: federated is not better than mean imputation - check columns/dtypes.")
    say("   Note: hidden cells are MCAR and add to real missingness; compare methods, not absolutes.")
    say("=" * 74)


def diagnose(df, cols, aux, n_sites, frac, seed, by, max_iter):
    """Aggregate-only diagnostics (no individual values such as min/max are reported)."""
    allc = cols + aux
    X = df[allc].astype(float)
    say("\n" + "=" * 74 + "\nDIAGNOSTICS (aggregate summary statistics only)\n" + "=" * 74)
    n_pat = df[by].nunique() if by else None
    say(f"rows={len(df)}" + (f", patients={n_pat}, rows/patient={len(df)/n_pat:.1f}" if by else "")
        + f", simulated sites={n_sites} (~{len(df)//n_sites} rows each)")

    say("\n1) Column profile  (SENTINEL? = one repeated value >30% of observed cells; "
        "OUTLIER = >0.1% of cells further than 10 SD from the median)")
    rows = []
    for c in allc:
        v = X[c].dropna()
        sd = v.std()
        top_share = v.value_counts(normalize=True).iloc[0] if len(v) else np.nan
        ext = 100 * np.mean(np.abs(v - v.median()) > 10 * sd) if sd and sd > 0 else np.nan
        flags = []
        if top_share > 0.30: flags.append("SENTINEL?")
        if ext > 0.1: flags.append("OUTLIER")
        if not sd or np.isnan(sd): flags.append("CONSTANT")
        rows.append({"col": c + (" (aux)" if c in aux else ""), "miss%": 100 * X[c].isna().mean(),
                     "mean": v.mean(), "median": v.median(), "sd": sd,
                     "top_val_share": top_share, "extreme%": ext, "flags": ",".join(flags)})
    prof = pd.DataFrame(rows).set_index("col").round(3)
    _TABLES["diag_column_profile"] = prof.reset_index()
    say(prof.to_string())

    Z = X.fillna(X.mean()); Z = (Z - Z.mean()) / Z.std().replace(0, np.nan)
    Z = Z.dropna(axis=1, how="all").fillna(0)
    C = np.corrcoef(Z.to_numpy().T)
    iu = np.triu_indices_from(C, 1)
    order = np.argsort(-np.abs(C[iu]))[:5]
    say("\n2) Collinearity")
    say("   strongest pairwise correlations:")
    for k in order:
        say(f"     {Z.columns[iu[0][k]]:<12} ~ {Z.columns[iu[1][k]]:<12} r={C[iu][k]:+.3f}")
    ev = np.linalg.eigvalsh(C)
    cond = ev.max() / max(ev.min(), 1e-12)
    vif = np.diag(np.linalg.pinv(C))
    say(f"   condition number of correlation matrix: {cond:.1f}   (>~100 = ill-conditioned)")
    say(f"   max VIF: {vif.max():.1f} ({Z.columns[int(vif.argmax())]})   (>~10 = strongly collinear)")
    _TABLES["diag_collinearity"] = pd.DataFrame(
        [{"metric": "condition_number", "value": cond, "detail": ""},
         {"metric": "max_VIF", "value": vif.max(), "detail": Z.columns[int(vif.argmax())]}] +
        [{"metric": "top_corr", "value": C[iu][k], "detail": f"{Z.columns[iu[0][k]]} ~ {Z.columns[iu[1][k]]}"}
         for k in order])

    sites = split_sites(df, n_sites, seed, by)
    cfg, m = fit_meta(sites, allc, max_iter), MetaMice()  # META-ALGO
    sd = X.std().to_numpy()
    big, coef_rows = [], []
    for g in cfg["state"]["global_estimates"]:
        t, ni = g["feat_idx"], g["neighbor_indices"]
        std_coef = np.abs(np.array(g["coef"])) * sd[ni] / (sd[t] if sd[t] else np.nan)
        big.append((float(np.nanmax(std_coef)), allc[t]))
        coef_rows += [{"target": allc[t], "predictor": allc[j], "std_abs_coef": float(sc)}
                      for j, sc in zip(ni, std_coef)]
    _TABLES["diag_std_coefficients"] = pd.DataFrame(coef_rows)
    big.sort(reverse=True)
    say("\n3) Federated model (real data)")
    say("   largest |standardised coefficient| per target (>~3 suggests unstable fit):")
    say("     " + ", ".join(f"{n}={v:.2f}" for v, n in big[:5]))
    imp = pd.concat([pd.DataFrame(m.impute(s_.copy(), cfg)) for s_ in sites]).set_index("_rid").loc[df["_rid"]]
    say("   imputed cells vs observed cells (OUT = % imputed outside observed range; drift shows as SD blow-up):")
    say(f"     {'column':<12}{'obs mean':>10}{'imp mean':>10}{'obs sd':>9}{'imp sd':>9}{'OUT%':>7}")
    iv_rows = []
    for c in cols:
        miss = X[c].isna().to_numpy()
        if not miss.any(): continue
        iv = imp[c].to_numpy(float)[miss]
        out = 100 * np.mean((iv < X[c].min()) | (iv > X[c].max()))
        say(f"     {c:<12}{X[c].mean():>10.2f}{np.nanmean(iv):>10.2f}{X[c].std():>9.2f}{np.nanstd(iv):>9.2f}{out:>7.0f}")
        iv_rows.append({"column": c, "obs_mean": X[c].mean(), "imp_mean": np.nanmean(iv),
                        "obs_sd": X[c].std(), "imp_sd": np.nanstd(iv), "imputed_outside_observed_range_pct": out})
    _TABLES["diag_imputed_vs_observed"] = pd.DataFrame(iv_rows)

    say("\n4) Iteration sweep (1 masked rep; fed should approach local, NOT get worse with more iterations)")
    rng = np.random.default_rng(seed)
    truth = df[allc].to_numpy(float)
    hide = np.zeros_like(truth, dtype=bool)
    hide[:, :len(cols)] = ~np.isnan(truth[:, :len(cols)]) & (rng.random((len(df), len(cols))) < frac)
    Xm = truth.copy(); Xm[hide] = np.nan
    d = df.copy(); d[allc] = Xm
    ss, rid = split_sites(d, n_sites, seed, by), d["_rid"]
    say(f"   {'max_iter':<10}{'fed nRMSE':>11}{'local OLS':>11}")
    sweep = []
    for it in sorted({1, 2, 3, 5, max_iter}):
        fed = reassemble(fed_meta([s_.copy() for s_ in ss], allc, it), rid, allc)  # META-ALGO
        loc = local_mice(d, allc, LinearRegression(), it)
        fe, le = np.nanmean(nrmse_cols(fed, truth, hide)), np.nanmean(nrmse_cols(loc, truth, hide))
        say(f"   {it:<10}{fe:>11.4f}{le:>11.4f}")
        sweep.append({"max_iter": it, "fed_nRMSE": fe, "local_OLS_nRMSE": le})
    _TABLES["diag_iteration_sweep"] = pd.DataFrame(sweep)
    say("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cols", nargs="+", required=True, help="columns to impute AND score")
    ap.add_argument("--aux", nargs="*", default=[], help="auxiliary predictor-only columns (not scored)")
    ap.add_argument("--n-sites", type=int, default=3); ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=5); ap.add_argument("--group-col", default=None)
    ap.add_argument("--diagnose", action="store_true",
                    help="also print aggregate diagnostics (profile, collinearity, coefficients, iteration sweep)")
    ap.add_argument("--out-dir", "--save-dir", dest="out_dir", default="fed_vs_local_out",
                    help="directory for summary CSVs + report.txt (aggregate statistics only, "
                         "NO patient-level data); default: fed_vs_local_out")
    a = ap.parse_args()
    a.aux = [c for c in a.aux if c not in a.cols]
    df = pd.read_csv(a.csv); df["_rid"] = np.arange(len(df))
    print(f"Running {a.reps} reps ... (report is printed at the end)", flush=True)
    res, percol = run(df, a.cols, a.aux, a.n_sites, a.frac, a.reps, a.seed, a.group_col, a.max_iter)
    report(res, percol, a)
    if a.diagnose:
        diagnose(df, a.cols, a.aux, a.n_sites, a.frac, a.seed, a.group_col, a.max_iter)

    os.makedirs(a.out_dir, exist_ok=True)
    for name, tbl in _TABLES.items():
        tbl.to_csv(os.path.join(a.out_dir, f"{name}.csv"), index=False)
    say(f"\nSummary files written to {os.path.abspath(a.out_dir)}: "
        + ", ".join(f"{n}.csv" for n in _TABLES) + ", report.txt  (aggregate statistics only)")
    with open(os.path.join(a.out_dir, "report.txt"), "w") as f:
        f.write("\n".join(_OUT) + "\n")
    print("\n".join(_OUT), flush=True)  # single print, very last output of the script