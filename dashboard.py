from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def read_csv(name: str) -> pd.DataFrame:
    path = MODELS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_repo_csv(relative_path: str) -> pd.DataFrame:
    path = BASE_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def format_float(value: object, digits: int = 4, percent: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "missing"
    if pd.isna(numeric):
        return "missing"
    if percent:
        return f"{numeric:.1%}"
    return f"{numeric:.{digits}f}"


def count_status(df: pd.DataFrame, status_column: str, status: str) -> int:
    if df.empty or status_column not in df.columns:
        return 0
    return int((df[status_column].astype(str).str.lower() == status.lower()).sum())


def render_current_research_status(
    st,
    phase43b_claims: pd.DataFrame,
    phase43b_primary_comparison: pd.DataFrame,
    phase43b_experiment_results: pd.DataFrame,
    phase46_submission_gates: pd.DataFrame,
    phase46_claim_audit: pd.DataFrame,
    research_grade_report: pd.DataFrame,
) -> None:
    st.header("Current Research Status: Phase 46")
    st.caption(
        "Paper-facing summary of the repaired validation path, the one-shot locked "
        "external holdout, and the final submission gate package."
    )
    st.warning(
        "Important: older dashboard panels below are retained as audit history. "
        "The current paper claim must come from Phase 43B/46 artifacts, not from "
        "the invalidated early positive-looking experiments."
    )

    if phase43b_claims.empty or phase43b_primary_comparison.empty:
        st.info(
            "Run the Phase 43B locked external evaluation and Phase 46 completion "
            "package to populate the current research summary."
        )
        return

    global_reference = phase43b_primary_comparison[
        phase43b_primary_comparison["reference_method"] == "global_lgbm"
    ]
    hmm_reference = phase43b_primary_comparison[
        phase43b_primary_comparison["reference_method"] == "regime_lgbm_hmm"
    ]
    primary_row = (
        global_reference.iloc[0]
        if not global_reference.empty
        else phase43b_primary_comparison.iloc[0]
    )
    candidate = str(primary_row.get("final_candidate", "missing"))

    candidate_rows = phase43b_experiment_results[
        phase43b_experiment_results["method"] == candidate
    ] if "method" in phase43b_experiment_results.columns else pd.DataFrame()
    candidate_row = candidate_rows.iloc[0] if not candidate_rows.empty else primary_row

    st.info(
        "Current safe claim: the frozen guided-HMM candidate satisfies the "
        "prewritten locked relative IC/Sharpe rule versus the two primary references. "
        "It does not support a profitable or deployable trading strategy."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frozen candidate", candidate)
    c2.metric(
        "Δ mean asset IC vs global",
        format_float(primary_row.get("delta_mean_asset_IC"), digits=4),
    )
    c3.metric("Locked Sharpe", format_float(candidate_row.get("Sharpe"), digits=3))
    c4.metric(
        "Locked total return",
        format_float(candidate_row.get("total_return"), percent=True),
    )

    if not hmm_reference.empty:
        h1, h2, h3 = st.columns(3)
        hmm_row = hmm_reference.iloc[0]
        h1.metric(
            "Δ mean asset IC vs raw HMM",
            format_float(hmm_row.get("delta_mean_asset_IC"), digits=4),
        )
        h2.metric(
            "Δ Sharpe vs raw HMM",
            format_float(hmm_row.get("delta_Sharpe"), digits=3),
        )
        h3.metric(
            "Equal locked coverage",
            str(bool(hmm_row.get("coverage_equal", False))),
        )

    st.subheader("Phase 43B Locked Claims")
    st.dataframe(phase43b_claims, width="stretch")

    if not phase46_submission_gates.empty:
        st.subheader("Phase 46 Submission Gates")
        passed = count_status(phase46_submission_gates, "status", "pass")
        conditional = count_status(phase46_submission_gates, "status", "conditional_pass")
        failed = count_status(phase46_submission_gates, "status", "fail")
        g1, g2, g3 = st.columns(3)
        g1.metric("Passed Gates", passed)
        g2.metric("Conditional Gates", conditional)
        g3.metric("Failed Gates", failed)
        st.dataframe(phase46_submission_gates, width="stretch")

    if not phase46_claim_audit.empty:
        st.subheader("Phase 46 Claim Audit")
        st.dataframe(phase46_claim_audit, width="stretch")

    if not research_grade_report.empty:
        st.subheader("Research-Grade Regression Gate")
        pass_count = count_status(research_grade_report, "status", "PASS")
        fail_count = count_status(research_grade_report, "status", "FAIL")
        warn_count = count_status(research_grade_report, "status", "WARN")
        r1, r2, r3 = st.columns(3)
        r1.metric("PASS", pass_count)
        r2.metric("WARN", warn_count)
        r3.metric("FAIL", fail_count)


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Run: pip install -r requirements.txt")
        return

    st.set_page_config(page_title="Adaptive Alpha Lab", layout="wide")
    st.title("Adaptive Alpha Lab")
    st.caption(
        "Research-grade regime-aware quant ML benchmark with repaired validation, "
        "locked external adjudication, claim control, and paper-facing evidence gates."
    )
    st.info(
        "Current result: Phase 43B/46 supports only a narrow locked relative "
        "improvement claim for the frozen guided-HMM candidate. Negative locked "
        "Sharpe and total return block any profitable-strategy claim."
    )

    phase43b_claims = read_csv("phase43b_locked_external_claims.csv")
    phase43b_primary_comparison = read_csv("phase43b_locked_external_primary_comparison.csv")
    phase43b_experiment_results = read_csv("phase43b_locked_external_experiment_results.csv")
    phase46_submission_gates = read_csv("phase46_submission_gate_matrix.csv")
    phase46_claim_audit = read_csv("phase46_final_claim_audit.csv")
    research_grade_report = read_csv("research_grade_check_report.csv")
    results = read_csv("experiment_results.csv")
    walkforward_results = read_csv("walkforward_experiment_results.csv")
    walkforward_comparison = read_csv("walkforward_comparison.csv")
    walkforward_regime_summary = read_csv("walkforward_regime_summary.csv")
    guided_alpha_comparison = read_csv("guided_alpha_comparison.csv")
    robustness_results = read_csv("robustness_results.csv")
    robustness_summary = read_csv("robustness_summary.csv")
    robustness_wins = read_csv("robustness_wins.csv")
    stress_results = read_csv("robustness_stress_results.csv")
    stress_summary = read_csv("robustness_stress_summary.csv")
    stress_wins = read_csv("robustness_stress_wins.csv")
    statistical_summary = read_csv("statistical_method_summary.csv")
    statistical_pairwise = read_csv("statistical_pairwise_tests.csv")
    statistical_compact = read_csv("statistical_test_summary.csv")
    statistical_corrections = read_csv("statistical_multiple_testing.csv")
    statistical_claims = read_csv("statistical_claims.csv")
    statistical_psr = read_csv("statistical_sharpe_diagnostics.csv")
    paper_claim_tests = read_csv("paper_claim_tests.csv")
    paper_statistical_summary = read_csv("paper_statistical_summary.csv")
    regime_summary = read_csv("regime_benchmark_summary.csv")
    regime_stability = read_csv("regime_stability_summary.csv")
    regime_quality = read_csv("regime_quality_summary.csv")
    regime_agreement = read_csv("regime_agreement_matrix.csv")
    compute_profile = read_csv("compute_profile.csv")
    ablation_budget = read_csv("ablation_budget.csv")
    ablation_results = read_csv("ablation_results.csv")
    ablation_summary = read_csv("ablation_summary.csv")
    compute_budget_summary = read_csv("compute_budget_summary.csv")
    guided_encoder_summary = read_csv("guided_encoder_summary.csv")
    guided_encoder_loss = read_csv("guided_encoder_loss.csv")
    guided_encoder_comparison = read_csv("guided_encoder_comparison.csv")
    time_frequency_summary = read_csv("time_frequency_encoder_summary.csv")
    time_frequency_loss = read_csv("time_frequency_encoder_loss.csv")
    time_frequency_comparison = read_csv("time_frequency_encoder_comparison.csv")
    feature_importance_global = read_csv("feature_importance_global.csv")
    feature_importance_by_regime = read_csv("feature_importance_by_regime.csv")
    feature_family_summary = read_csv("feature_family_summary.csv")
    per_regime = read_csv("per_regime_stats.csv")
    validation_audit = read_csv("validation_audit.csv")
    fold_audit = read_csv("fold_audit.csv")
    target_dist = read_csv("target_distribution.csv")
    target_quality = read_csv("target_quality.csv")
    run_index = read_repo_csv("runs/run_index.csv")
    literature_matrix = read_repo_csv("reports/literature_matrix.csv")

    render_current_research_status(
        st,
        phase43b_claims,
        phase43b_primary_comparison,
        phase43b_experiment_results,
        phase46_submission_gates,
        phase46_claim_audit,
        research_grade_report,
    )

    st.header("Experiment Results")
    if results.empty:
        st.info("Run python src/alpha_models.py to generate experiment_results.csv.")
    else:
        st.dataframe(results, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        best_ic = results.sort_values("IC", ascending=False).iloc[0]
        best_dd = results.sort_values("drawdown", ascending=False).iloc[0]
        c1.metric("Best IC", f"{best_ic['IC']:.4f}", best_ic["method"])
        c2.metric("Best Sharpe", f"{results['Sharpe'].max():.3f}")
        c3.metric("Lowest Drawdown", f"{best_dd['drawdown']:.2%}", best_dd["method"])
        c4.metric("Methods Tested", len(results))

    st.header("Fold-Local Regime Refit")
    if walkforward_results.empty:
        st.info("Run python src/walkforward_regimes.py to generate fold-local regime results.")
    else:
        st.caption(
            "Strict benchmark where regime assignment models are refit inside each "
            "walk-forward fold using training history only."
        )
        st.dataframe(walkforward_results, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        best_ic = walkforward_results.sort_values("IC", ascending=False).iloc[0]
        best_sharpe = walkforward_results.sort_values("Sharpe", ascending=False).iloc[0]
        c1.metric("Best Fold-Local IC", f"{best_ic['IC']:.4f}", best_ic["method"])
        c2.metric("Best Fold-Local Sharpe", f"{best_sharpe['Sharpe']:.3f}", best_sharpe["method"])
        c3.metric("Fold-Local Methods", len(walkforward_results))
        c4.metric("Rows / Method", int(walkforward_results["n_test_rows"].max()))
        guided_rows = walkforward_results[
            walkforward_results["regime_method"].isin(["hmm_guided_gmm", "hmm_guided_hmm"])
        ]
        if not guided_rows.empty:
            st.subheader("Phase 20 Guided Alpha Retest")
            st.caption(
                "Guided embeddings are frozen from Phase 19B, but their GMM/HMM "
                "assignment layers are refit inside each walk-forward fold."
            )
            g1, g2, g3, g4 = st.columns(4)
            best_guided_ic = guided_rows.sort_values("IC", ascending=False).iloc[0]
            best_guided_sharpe = guided_rows.sort_values("Sharpe", ascending=False).iloc[0]
            hmm_reference = walkforward_results[walkforward_results["method"] == "regime_lgbm_hmm"]
            if hmm_reference.empty:
                delta_vs_hmm = 0.0
            else:
                delta_vs_hmm = float(best_guided_ic["IC"] - hmm_reference.iloc[0]["IC"])
            g1.metric("Best Guided IC", f"{best_guided_ic['IC']:.4f}", best_guided_ic["method"])
            g2.metric("Best Guided Sharpe", f"{best_guided_sharpe['Sharpe']:.3f}", best_guided_sharpe["method"])
            g3.metric("Guided IC vs HMM", f"{delta_vs_hmm:+.4f}")
            g4.metric("Guided Methods", len(guided_rows))
            if not guided_alpha_comparison.empty:
                st.dataframe(guided_alpha_comparison, width="stretch")
        if not walkforward_comparison.empty:
            st.subheader("Offline vs Fold-Local")
            st.dataframe(walkforward_comparison, width="stretch")
        if not walkforward_regime_summary.empty:
            st.subheader("Fold-Local Regime Summary")
            st.dataframe(walkforward_regime_summary, width="stretch")
        walkforward_img = MODELS_DIR / "walkforward_equity_curve.png"
        if walkforward_img.exists():
            st.image(str(walkforward_img), width="stretch")

    st.header("Robustness Matrix")
    if robustness_summary.empty:
        st.info("Run python src/robustness.py to generate Phase 14A robustness artifacts.")
    else:
        st.caption(
            "Phase 14A repeats the fold-local benchmark across BTC-only, ETH-only, "
            "and BTC+ETH scopes at 4h, 8h, and 24h horizons."
        )
        st.dataframe(robustness_summary, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Grid Cells", len(robustness_summary))
        if robustness_wins.empty:
            c2.metric("Most IC Wins", "missing")
            c3.metric("Most Sharpe Wins", "missing")
            c4.metric("Most Drawdown Wins", "missing")
        else:
            ic_leader = robustness_wins[robustness_wins["metric"] == "IC"].sort_values("wins", ascending=False).iloc[0]
            sharpe_leader = robustness_wins[robustness_wins["metric"] == "Sharpe"].sort_values("wins", ascending=False).iloc[0]
            drawdown_leader = robustness_wins[robustness_wins["metric"] == "drawdown"].sort_values("wins", ascending=False).iloc[0]
            c2.metric("Most IC Wins", int(ic_leader["wins"]), ic_leader["method"])
            c3.metric("Most Sharpe Wins", int(sharpe_leader["wins"]), sharpe_leader["method"])
            c4.metric("Most Drawdown Wins", int(drawdown_leader["wins"]), drawdown_leader["method"])
        if not robustness_wins.empty:
            st.subheader("Robustness Win Counts")
            st.dataframe(robustness_wins, width="stretch")
        if not robustness_results.empty:
            st.subheader("Full Robustness Results")
            st.dataframe(robustness_results, width="stretch")
        robustness_img = MODELS_DIR / "robustness_heatmap.png"
        if robustness_img.exists():
            st.image(str(robustness_img), width="stretch")

    st.header("Stress Robustness")
    if stress_summary.empty:
        st.info("Run python src/robustness_stress.py to generate Phase 14B stress artifacts.")
    else:
        st.caption(
            "Phase 14B re-scores fold-local predictions across transaction costs, "
            "signal thresholds, and bull/sideways/bear market periods."
        )
        st.dataframe(stress_summary, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stress Cells", len(stress_summary))
        if stress_wins.empty:
            c2.metric("Signal IC Leader", "missing")
            c3.metric("Sharpe Leader", "missing")
            c4.metric("Return Leader", "missing")
        else:
            signal_leader = stress_wins[stress_wins["metric"] == "signal_IC"].sort_values("wins", ascending=False).iloc[0]
            sharpe_leader = stress_wins[stress_wins["metric"] == "Sharpe"].sort_values("wins", ascending=False).iloc[0]
            return_leader = stress_wins[stress_wins["metric"] == "total_return"].sort_values("wins", ascending=False).iloc[0]
            c2.metric("Signal IC Leader", int(signal_leader["wins"]), signal_leader["method"])
            c3.metric("Sharpe Leader", int(sharpe_leader["wins"]), sharpe_leader["method"])
            c4.metric("Return Leader", int(return_leader["wins"]), return_leader["method"])
        if not stress_wins.empty:
            st.subheader("Stress Win Counts")
            st.dataframe(stress_wins, width="stretch")
        if not stress_results.empty:
            st.subheader("Full Stress Results")
            st.dataframe(stress_results, width="stretch")
        stress_img = MODELS_DIR / "robustness_stress_heatmap.png"
        if stress_img.exists():
            st.image(str(stress_img), width="stretch")

    st.header("Statistical Significance")
    if statistical_summary.empty:
        st.info("Run python src/statistical_tests.py to generate Phase 15A significance artifacts.")
    else:
        st.caption(
            "Phase 15A turns point estimates into fold-level confidence intervals, "
            "paired tests, and row-level forecast-loss checks."
        )
        st.dataframe(statistical_summary, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        best_ic = statistical_summary.sort_values("mean_fold_IC", ascending=False).iloc[0]
        c1.metric("Best Mean Fold IC", f"{best_ic['mean_fold_IC']:.4f}", best_ic["method"])
        c2.metric("IC CI Low", f"{best_ic['IC_ci_low']:.4f}")
        c3.metric("IC CI High", f"{best_ic['IC_ci_high']:.4f}")
        c4.metric("Positive IC Folds", int(best_ic["positive_ic_folds"]))
        stats_img = MODELS_DIR / "statistical_ic_confidence_intervals.png"
        if stats_img.exists():
            st.image(str(stats_img), width="stretch")
        correction_img = MODELS_DIR / "statistical_multiple_testing.png"
        if correction_img.exists():
            st.subheader("Multiple-Testing Correction")
            st.image(str(correction_img), width="stretch")
        psr_img = MODELS_DIR / "statistical_sharpe_diagnostics.png"
        if psr_img.exists():
            st.subheader("Probabilistic Sharpe Diagnostics")
            st.image(str(psr_img), width="stretch")
        if not statistical_claims.empty:
            st.subheader("Corrected Claim Summary")
            st.dataframe(statistical_claims, width="stretch")
        if not statistical_psr.empty:
            st.subheader("PSR Table")
            st.dataframe(statistical_psr, width="stretch")
        if not statistical_corrections.empty:
            st.subheader("Multiple-Testing Table")
            st.dataframe(statistical_corrections, width="stretch")
        if not statistical_compact.empty:
            st.subheader("Focused Test Summary")
            st.dataframe(statistical_compact, width="stretch")
        if not statistical_pairwise.empty:
            st.subheader("All Pairwise Tests")
            st.dataframe(statistical_pairwise, width="stretch")

    st.header("Paper Claim Tests")
    if paper_statistical_summary.empty:
        st.info("Run python src/paper_claim_tests.py to generate Phase 26 paper claim artifacts.")
    else:
        st.caption(
            "Phase 26 maps the Phase 25 ablation suite onto fold-level statistical evidence "
            "and assigns paper-safe claim language."
        )
        st.dataframe(paper_statistical_summary, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Paper Claims", len(paper_statistical_summary))
        directional = paper_statistical_summary[
            paper_statistical_summary["paper_status"].isin(
                ["directionally_supported", "raw_suggestive", "statistically_supported", "metric_family_supported"]
            )
        ]
        mechanism = paper_statistical_summary[
            paper_statistical_summary["paper_status"] == "mechanism_supported_no_fold_p_value"
        ]
        c2.metric("Directional+", len(directional))
        c3.metric("Mechanism Supported", len(mechanism))
        if "guided_hmm_alpha_vs_raw_feature_hmm_alpha" in set(paper_statistical_summary["comparison"]):
            main_claim = paper_statistical_summary[
                paper_statistical_summary["comparison"] == "guided_hmm_alpha_vs_raw_feature_hmm_alpha"
            ].iloc[0]
            c4.metric("Main Claim", main_claim["paper_status"])
        else:
            c4.metric("Main Claim", "missing")
        claim_img = MODELS_DIR / "paper_claim_tests.png"
        if claim_img.exists():
            st.image(str(claim_img), width="stretch")
        if not paper_claim_tests.empty:
            st.subheader("Metric-Level Paper Tests")
            st.dataframe(paper_claim_tests, width="stretch")

    st.header("Run Registry")
    if run_index.empty:
        st.info("Run python src/archive_run.py to freeze a versioned research snapshot.")
    else:
        st.caption("Versioned snapshots keep baseline artifacts from being overwritten by later experiment runs.")
        latest_run = run_index.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registered Runs", len(run_index))
        c2.metric("Latest Run", latest_run["run_id"])
        c3.metric("Artifacts", int(latest_run["artifact_count"]))
        c4.metric("Missing", int(latest_run["missing_artifact_count"]))
        st.dataframe(run_index, width="stretch")

    st.header("Compute Plan")
    if compute_profile.empty:
        st.info("Run python src/compute_plan.py to generate Phase 17 compute-planning artifacts.")
    else:
        st.caption(
            "Phase 17 estimates the local cost of encoder retraining before launching "
            "HMM-guided, time-frequency, or ablation experiments."
        )
        row = compute_profile.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Device", row["device"])
        c2.metric("One Encoder Run", f"{float(row['estimated_full_train_minutes']):.1f} min")
        c3.metric("12-Run Grid", f"{float(row['estimated_ablation_hours']):.1f} h")
        c4.metric("Budget Status", row["budget_status"])
        if not compute_budget_summary.empty:
            st.subheader("Budget Summary")
            st.dataframe(compute_budget_summary, width="stretch")
        if not ablation_budget.empty:
            st.subheader("Initial Ablation Queue")
            st.dataframe(ablation_budget, width="stretch")
        compute_img = MODELS_DIR / "compute_budget_plan.png"
        if compute_img.exists():
            st.image(str(compute_img), width="stretch")

    st.header("Minimal Ablation Suite")
    if ablation_summary.empty:
        st.info("Run python src/ablation_suite.py to generate Phase 25 ablation artifacts.")
    else:
        st.caption(
            "Phase 25 compares only the mechanisms needed for the paper: objective guidance, "
            "HMM versus GMM assignment, time-frequency augmentation, and the classical HMM reference."
        )
        st.dataframe(ablation_summary, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        supported = ablation_summary[ablation_summary["phase25_decision"] == "supported"]
        win_rate_col = first_existing_column(ablation_summary, ["metric_win_rate", "candidate_win_rate"])
        c1.metric("Comparisons", len(ablation_summary))
        c2.metric("Supported", len(supported))
        if win_rate_col:
            strongest = ablation_summary.sort_values(win_rate_col, ascending=False).iloc[0]
            c3.metric("Best Win Rate", f"{strongest[win_rate_col]:.2f}", strongest["comparison"])
        else:
            c3.metric("Best Win Rate", "missing")
        c4.metric("Families", ablation_summary["ablation_family"].nunique())
        ablation_img = MODELS_DIR / "ablation_heatmap.png"
        if ablation_img.exists():
            st.image(str(ablation_img), width="stretch")
        if not ablation_results.empty:
            st.subheader("Metric-Level Ablation Rows")
            st.dataframe(ablation_results, width="stretch")

    st.header("HMM-Guided Encoder")
    if guided_encoder_summary.empty:
        st.info("Run python src/guided_encoder.py to generate Phase 18 HMM-guided encoder artifacts.")
    else:
        st.caption(
            "Phase 18 trains a separate guided encoder using HMM states as weak supervision. "
            "The existing production encoder and canonical regime files are not overwritten."
        )
        c1, c2, c3, c4 = st.columns(4)
        best_nmi = guided_encoder_summary.sort_values("hmm_reference_nmi", ascending=False).iloc[0]
        best_sil = guided_encoder_summary.sort_values("silhouette", ascending=False).iloc[0]
        c1.metric("Best HMM NMI", f"{best_nmi['hmm_reference_nmi']:.3f}", best_nmi["method"])
        c2.metric("Best Silhouette", f"{best_sil['silhouette']:.3f}", best_sil["method"])
        c3.metric("Epochs", int(guided_encoder_summary["epochs"].max()))
        c4.metric("Final Loss", f"{guided_encoder_summary['final_loss'].min():.3f}")
        st.dataframe(guided_encoder_summary, width="stretch")
        if not guided_encoder_comparison.empty:
            st.subheader("Guided vs Baseline Regime Structure")
            st.dataframe(guided_encoder_comparison, width="stretch")
        if not guided_encoder_loss.empty:
            st.subheader("Training Loss")
            st.dataframe(guided_encoder_loss, width="stretch")
        loss_img = MODELS_DIR / "guided_encoder_loss_curve.png"
        if loss_img.exists():
            st.image(str(loss_img), width="stretch")
        matrix_cols = st.columns(2)
        for col, method in zip(matrix_cols, ["hmm_guided_gmm", "hmm_guided_hmm"]):
            path = MODELS_DIR / f"guided_encoder_transition_{method}.png"
            with col:
                st.caption(method)
                if path.exists():
                    st.image(str(path))
                else:
                    st.info("missing")

    st.header("Time-Frequency Encoder Prototype")
    if time_frequency_summary.empty:
        st.info(
            "Run python src/guided_encoder.py --augmentation time_frequency "
            "to generate Phase 22 time-frequency encoder artifacts."
        )
    else:
        st.caption(
            "Phase 22A appends FFT magnitude bands to each guided-encoder window. "
            "This is a structural prototype, not a downstream alpha claim yet."
        )
        c1, c2, c3, c4 = st.columns(4)
        best_tf_nmi = time_frequency_summary.sort_values("hmm_reference_nmi", ascending=False).iloc[0]
        best_tf_purity = time_frequency_summary.sort_values("hmm_reference_purity", ascending=False).iloc[0]
        c1.metric("Best TF HMM NMI", f"{best_tf_nmi['hmm_reference_nmi']:.3f}", best_tf_nmi["method"])
        c2.metric("Best TF Purity", f"{best_tf_purity['hmm_reference_purity']:.3f}", best_tf_purity["method"])
        c3.metric("Epochs", int(time_frequency_summary["epochs"].max()))
        c4.metric("Input Features", int(time_frequency_summary["input_features"].max()))
        st.dataframe(time_frequency_summary, width="stretch")
        if not time_frequency_comparison.empty:
            st.subheader("Time-Frequency vs Baseline Regime Structure")
            st.dataframe(time_frequency_comparison, width="stretch")
        if not time_frequency_loss.empty:
            st.subheader("Time-Frequency Training Loss")
            st.dataframe(time_frequency_loss, width="stretch")
        loss_img = MODELS_DIR / "time_frequency_encoder_loss_curve.png"
        if loss_img.exists():
            st.image(str(loss_img), width="stretch")
        matrix_cols = st.columns(2)
        for col, method in zip(matrix_cols, ["tf_hmm_guided_gmm", "tf_hmm_guided_hmm"]):
            path = MODELS_DIR / f"time_frequency_encoder_transition_{method}.png"
            with col:
                st.caption(method)
                if path.exists():
                    st.image(str(path))
                else:
                    st.info("missing")

    st.header("Interpretability")
    if feature_importance_global.empty or feature_importance_by_regime.empty:
        st.info("Run python src/interpretability.py to generate Phase 23 interpretability artifacts.")
    else:
        st.caption(
            "Phase 23 aggregates fold-local LightGBM feature importance for the global, raw-HMM, "
            "and guided-HMM models. This explains which market features drive each regime-conditioned alpha model."
        )
        guided = feature_importance_by_regime[
            feature_importance_by_regime["method"] == "regime_lgbm_hmm_guided_hmm"
        ].copy()
        metric = (
            "mean_shap_share"
            if "mean_shap_share" in guided.columns and guided["mean_shap_share"].notna().any()
            else "mean_gain_share"
        )
        c1, c2, c3, c4 = st.columns(4)
        if not guided.empty:
            top_guided = guided.sort_values(metric, ascending=False).iloc[0]
            c1.metric("Top Guided Feature", top_guided["feature"], f"regime {top_guided['regime']}")
            c2.metric("Feature Family", top_guided["feature_family"])
            c3.metric("Explained Regimes", guided["regime"].nunique())
            c4.metric("Features Tracked", guided["feature"].nunique())
        else:
            c1.metric("Guided Rows", 0)
            c2.metric("Feature Family", "missing")
            c3.metric("Explained Regimes", 0)
            c4.metric("Features Tracked", 0)
        st.subheader("Global Feature Importance")
        st.dataframe(feature_importance_global.sort_values("rank_within_model_regime").head(12), width="stretch")
        st.subheader("Guided-HMM Top Features by Regime")
        top_guided = guided.sort_values(["regime", "rank_within_model_regime"]).groupby("regime").head(5)
        st.dataframe(top_guided, width="stretch")
        if not feature_family_summary.empty:
            st.subheader("Feature Family Summary")
            st.dataframe(feature_family_summary, width="stretch")
        importance_img = MODELS_DIR / "feature_importance_by_regime.png"
        if importance_img.exists():
            st.image(str(importance_img), width="stretch")
        family_img = MODELS_DIR / "feature_family_importance.png"
        if family_img.exists():
            st.image(str(family_img), width="stretch")

    st.header("Research Positioning")
    st.caption(
        "Phase 19A maps the project against time-series contrastive learning, "
        "financial regime switching, financial ML validation, and regime-conditioned alpha modeling."
    )
    if literature_matrix.empty:
        st.info("Add reports/literature_matrix.csv to show the related-work source matrix.")
    else:
        cluster_counts = literature_matrix["cluster"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Literature Clusters", literature_matrix["cluster"].nunique())
        c2.metric("Mapped Sources", len(literature_matrix))
        c3.metric("Time-Series CL", int(cluster_counts.get("contrastive_time_series", 0)))
        c4.metric("Finance/Validation", int(
            cluster_counts.get("financial_regimes", 0)
            + cluster_counts.get("financial_ml_validation", 0)
        ))
        st.dataframe(literature_matrix, width="stretch")

    st.header("Validation Audit")
    if validation_audit.empty:
        st.info("Run python src/validation_audit.py to generate validation_audit.csv.")
    else:
        pass_count = int((validation_audit["status"] == "PASS").sum())
        warn_count = int((validation_audit["status"] == "WARN").sum())
        fail_count = int((validation_audit["status"] == "FAIL").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed Checks", pass_count)
        c2.metric("Warnings", warn_count)
        c3.metric("Failures", fail_count)
        st.dataframe(validation_audit, width="stretch")
        if not fold_audit.empty:
            st.subheader("Fold Audit")
            st.dataframe(fold_audit, width="stretch")

    st.header("Targets")
    if target_dist.empty:
        st.info("Run python src/targets.py to generate target_distribution.csv.")
    else:
        st.dataframe(target_dist, width="stretch")
        if not target_quality.empty:
            st.subheader("Target Quality")
            st.dataframe(target_quality, width="stretch")
        img = MODELS_DIR / "target_distribution.png"
        if img.exists():
            st.image(str(img), caption="Target label distribution")

    st.header("Regime Quality")
    if regime_quality.empty:
        st.info("Run python src/regime_quality.py to generate Phase 16 regime-quality artifacts.")
    else:
        st.caption(
            "Phase 16 evaluates regime structure independently of alpha performance. "
            "HMM is used as a classical reference proxy, not as ground truth."
        )
        all_scope_quality = regime_quality[regime_quality["symbol_scope"] == "ALL"]
        c1, c2, c3, c4 = st.columns(4)
        if not all_scope_quality.empty:
            most_balanced = all_scope_quality.sort_values("regime_balance_entropy", ascending=False).iloc[0]
            most_persistent = all_scope_quality.sort_values("avg_regime_duration", ascending=False).iloc[0]
            best_reference = all_scope_quality[
                all_scope_quality["method"] != "hmm"
            ].sort_values("hmm_reference_nmi", ascending=False).iloc[0]
            best_purity = all_scope_quality[
                all_scope_quality["method"] != "hmm"
            ].sort_values("hmm_reference_purity", ascending=False).iloc[0]
            c1.metric("Most Balanced", most_balanced["method"], f"{most_balanced['regime_balance_entropy']:.3f}")
            c2.metric("Most Persistent", most_persistent["method"], f"{most_persistent['avg_regime_duration']:.1f} bars")
            c3.metric("Best HMM NMI", best_reference["method"], f"{best_reference['hmm_reference_nmi']:.3f}")
            c4.metric("Best HMM Purity", best_purity["method"], f"{best_purity['hmm_reference_purity']:.3f}")
        st.dataframe(regime_quality, width="stretch")
        quality_img = MODELS_DIR / "regime_quality_heatmap.png"
        agreement_img = MODELS_DIR / "regime_agreement_heatmap.png"
        if quality_img.exists():
            st.subheader("Quality Heatmap")
            st.image(str(quality_img), width="stretch")
        if agreement_img.exists():
            st.subheader("Pairwise Agreement")
            st.image(str(agreement_img), width="stretch")
        if not regime_agreement.empty:
            st.subheader("Agreement Table")
            st.dataframe(regime_agreement, width="stretch")

    st.header("Regime Benchmarking")
    if regime_summary.empty:
        st.info("Run python src/baselines.py to generate regime benchmark artifacts.")
    else:
        st.dataframe(regime_summary, width="stretch")
        if not regime_stability.empty:
            st.subheader("Regime Stability Diagnostics")
            st.dataframe(regime_stability, width="stretch")
            stability_img = MODELS_DIR / "regime_stability.png"
            if stability_img.exists():
                st.image(str(stability_img), width="stretch")
        st.subheader("Per-Regime Statistics")
        st.dataframe(per_regime, width="stretch")
        st.subheader("Transition Matrices")
        matrix_methods = ["contrastive", "contrastive_hmm", "hmm", "kmeans", "vol_bucket"]
        matrix_cols = st.columns(len(matrix_methods))
        for col, method in zip(matrix_cols, matrix_methods):
            path = MODELS_DIR / f"transition_matrix_{method}.png"
            with col:
                st.caption(method)
                if path.exists():
                    st.image(str(path))
                else:
                    st.info("missing")

    st.header("Visual Artifacts")
    images = [
        ("UMAP Regime Structure", "umap_improved.png"),
        ("Regime Timeline", "regime_timeline.png"),
        ("Backtest Dashboard", "phase4_dashboard.png"),
        ("Equity Curve", "equity_curve.png"),
    ]
    for title, name in images:
        path = MODELS_DIR / name
        if path.exists():
            st.subheader(title)
            st.image(str(path), width="stretch")


if __name__ == "__main__":
    main()
