# Python 3.12
from __future__ import annotations

from typing import Callable, Optional

from controller.state import AppState
from core.fake_mm_logs import generate_mm_logs
from core.clean_module import clean_data
from core.detect_pairs import (
    detect_pairs_level1, apply_level1_to_master,
    detect_pairs_level2, apply_level2_to_master,
    detect_pairs_level3, apply_level3_to_master,
)

ProgressFn = Optional[Callable[[str], None]]


def load_data(state: AppState, *, progress: ProgressFn = None) -> None:
    if progress:
        progress("Generating logs...")

    df = generate_mm_logs(
        start_date=state.filters.start_date,
        end_date=state.filters.end_date,
        trades_per_day=30_000,
    )

    if progress:
        progress("Cleaning logs...")

    dfc, meta = clean_data(df, add_keys=True, strict=False)

    state.df_raw = df
    state.df_clean = dfc
    state.meta = meta
    state.clear_results()

    if progress:
        progress("Loaded.")


def run_pairing(state: AppState, *, progress: ProgressFn = None) -> None:
    if state.df_clean is None:
        raise RuntimeError("No data loaded. Please load data first.")

    dfc = state.df_clean
    df_master = dfc.copy()

    # ---------- L1 ----------
    if state.filters.enable_L1:
        if progress:
            progress(f"Running L1 pairing (window={state.filters.l1_window})...")
        res1 = detect_pairs_level1(
            dfc,
            time_window=state.filters.l1_window,
            debug_checks=False,
        )
        df_master = apply_level1_to_master(df_master, res1, prefix="L1")
        state.pairs_L1 = res1["pairs_df"]

    # ---------- L2 ----------
    if state.filters.enable_L2:
        if progress:
            progress(f"Running L2 grouping (window={state.filters.l2_window})...")
        res2 = detect_pairs_level2(
            dfc,
            time_window=state.filters.l2_window,
            debug_checks=False,
        )
        df_master = apply_level2_to_master(df_master, res2, prefix="L2")
        state.groups_L2 = res2["groups_df"]

    # ---------- L3 (HEAVY) ----------
    if state.filters.enable_L3:
        if progress:
            progress(
                f"Running L3 pairing "
                f"(window={state.filters.l3_window}, delta_qty={state.filters.delta_qty})..."
            )

        res3 = detect_pairs_level3(
            dfc,
            time_window=state.filters.l3_window,
            delta_qty=state.filters.delta_qty,
            debug_checks=False,
        )

        df_master = apply_level3_to_master(df_master, res3, prefix="L3")
        state.pairs_L3 = res3["pairs_df"]

    else:
        if progress:
            progress("Skipping L3 pairing (disabled).")
        state.pairs_L3 = None

    state.df_master = df_master

    if progress:
        progress("Pairing done.")


def summarize(state: AppState) -> dict:
    out = {
        "loaded": state.df_clean is not None,
        "n_trades": int(len(state.df_clean)) if state.df_clean is not None else 0,
        "warnings": state.meta.get("warnings", []) if state.meta else [],
    }

    if state.df_master is not None:
        out["has_results"] = True
        out["n_pairs_L1"] = int(len(state.pairs_L1)) if state.pairs_L1 is not None else 0
        out["n_groups_L2"] = int(len(state.groups_L2)) if state.groups_L2 is not None else 0
        out["n_pairs_L3"] = int(len(state.pairs_L3)) if state.pairs_L3 is not None else 0
    else:
        out["has_results"] = False

    return out
