# detect_pairs.py
# Python 3.12
from __future__ import annotations

import numpy as np
import pandas as pd


def _time_window_to_ns(time_window: str | int | float) -> int:
    if isinstance(time_window, str):
        return int(pd.Timedelta(time_window).value)
    return int(float(time_window) * 1_000_000_000)


def _require_cols(dfc: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in dfc.columns]
    if missing:
        raise KeyError(f"dfc missing required columns: {missing}")


# =========================
# L1 (LIFO) exact quantity
# =========================
def detect_pairs_level1(
    dfc: pd.DataFrame,
    *,
    time_window: str | int | float = "60s",
    remaining_mask: np.ndarray | None = None,
    pair_id_start: int = 1,
    pair_type: str = "L1_exact",
    debug_checks: bool = True,
    strict_checks: bool = False,
) -> dict:
    """
    L1 (LIFO):
      - same (YearMonthDay, ISIN_id, CP_id, Quantity)
      - sell -> buy
      - dt <= time_window
      - LIFO (closest previous sell)
      - PnL_per_leg = (Quantity*(Price_sell-Price_buy)) / 2
      - DeltaRef = Ref_sell - Ref_buy
      - No reuse
    """

    _require_cols(dfc, ["TradeTime_ns", "YearMonthDay", "ISIN_id", "CP_id", "Quantity", "Side", "Price", "Ref"])

    def _ok(name: str, extra: str = ""):
        if debug_checks:
            print(f"[L1 CHECK] {name} OK ✓{('  ' + extra) if extra else ''}")

    def _fail(name: str, extra: str = ""):
        msg = f"[L1 CHECK] {name} FAILED ❌{('  ' + extra) if extra else ''}"
        if strict_checks:
            raise AssertionError(msg)
        if debug_checks:
            print(msg)

    n = len(dfc)
    if remaining_mask is None:
        remaining_mask = np.ones(n, dtype=bool)
    else:
        remaining_mask = np.asarray(remaining_mask, dtype=bool)
        if remaining_mask.shape != (n,):
            raise ValueError("remaining_mask must have shape (len(dfc),)")

    tw_ns = _time_window_to_ns(time_window)

    t_ns = dfc["TradeTime_ns"].to_numpy(np.int64, copy=False)
    day = dfc["YearMonthDay"].to_numpy(np.int32, copy=False)
    isin = dfc["ISIN_id"].to_numpy(np.int32, copy=False)
    cp = dfc["CP_id"].to_numpy(np.int32, copy=False)
    qty = dfc["Quantity"].to_numpy(np.int32, copy=False)
    side = dfc["Side"].to_numpy(np.int8, copy=False)
    price = dfc["Price"].to_numpy(np.float64, copy=False)
    ref = dfc["Ref"].to_numpy(np.float64, copy=False)

    idx0 = np.flatnonzero(remaining_mask)

    pair_id = np.full(n, -1, dtype=np.int32)
    pair_pnl = np.full(n, np.nan, dtype=np.float64)          # per leg
    pair_dt_ns = np.full(n, -1, dtype=np.int64)
    pair_delta_ref = np.full(n, np.nan, dtype=np.float64)
    paired_mask = np.zeros(n, dtype=bool)

    if idx0.size == 0:
        return {
            "pair_id": pair_id,
            "pair_pnl": pair_pnl,
            "pair_dt_ns": pair_dt_ns,
            "pair_delta_ref": pair_delta_ref,
            "paired_mask": paired_mask,
            "pairs_df": pd.DataFrame(columns=["sell_idx", "buy_idx", "pnl_per_leg", "dt_ns", "delta_ref", "pair_id", "pair_type"]),
            "next_pair_id": int(pair_id_start),
            "meta": {"n_pairs": 0, "n_paired_trades": 0, "time_window_ns": int(tw_ns)},
        }

    # group by (day, isin, cp, qty), sort by time
    order = np.lexsort((t_ns[idx0], qty[idx0], cp[idx0], isin[idx0], day[idx0]))
    idx = idx0[order]

    pairs_sell, pairs_buy, pairs_pnlL, pairs_dt, pairs_dref, pairs_pid = [], [], [], [], [], []
    pid = int(pair_id_start)

    d = day[idx]; i = isin[idx]; c = cp[idx]; q = qty[idx]
    run_break = np.empty(idx.size, dtype=bool)
    run_break[0] = True
    run_break[1:] = (d[1:] != d[:-1]) | (i[1:] != i[:-1]) | (c[1:] != c[:-1]) | (q[1:] != q[:-1])
    run_starts = np.flatnonzero(run_break)
    run_ends = np.r_[run_starts[1:], idx.size]

    for rs, re in zip(run_starts, run_ends):
        g_idx = idx[rs:re]  # time ordered

        sell_idx_stack = np.empty(g_idx.size, dtype=np.int64)
        sell_t_stack = np.empty(g_idx.size, dtype=np.int64)
        head = 0
        tail = 0

        for j in g_idx:
            sj = side[j]
            tj = t_ns[j]

            if sj == -1:
                sell_idx_stack[tail] = j
                sell_t_stack[tail] = tj
                tail += 1
                continue

            if sj == 1:
                while head < tail and (tj - sell_t_stack[head] > tw_ns):
                    head += 1

                if head < tail:
                    sell_pos = tail - 1  # LIFO
                    s_idx = int(sell_idx_stack[sell_pos])
                    dt = int(tj - sell_t_stack[sell_pos])

                    if 0 < dt <= tw_ns:
                        pnl_total = float(qty[s_idx]) * (float(price[s_idx]) - float(price[j]))
                        pnl_per_leg = pnl_total / 2.0
                        dref = float(ref[s_idx]) - float(ref[j])

                        pair_id[s_idx] = pid
                        pair_id[j] = pid
                        pair_pnl[s_idx] = pnl_per_leg
                        pair_pnl[j] = pnl_per_leg
                        pair_dt_ns[s_idx] = dt
                        pair_dt_ns[j] = dt
                        pair_delta_ref[s_idx] = dref
                        pair_delta_ref[j] = dref
                        paired_mask[s_idx] = True
                        paired_mask[j] = True

                        pairs_sell.append(s_idx)
                        pairs_buy.append(int(j))
                        pairs_pnlL.append(pnl_per_leg)
                        pairs_dt.append(dt)
                        pairs_dref.append(dref)
                        pairs_pid.append(pid)

                        pid += 1
                        tail -= 1  # consume sell

    pairs_df = pd.DataFrame({
        "sell_idx": np.array(pairs_sell, dtype=np.int64),
        "buy_idx": np.array(pairs_buy, dtype=np.int64),
        "pnl_per_leg": np.array(pairs_pnlL, dtype=np.float64),
        "dt_ns": np.array(pairs_dt, dtype=np.int64),
        "delta_ref": np.array(pairs_dref, dtype=np.float64),
        "pair_id": np.array(pairs_pid, dtype=np.int32),
        "pair_type": pair_type,
    })

    meta = {
        "n_pairs": int(len(pairs_df)),
        "n_paired_trades": int(paired_mask.sum()),
        "time_window_ns": int(tw_ns),
        "pair_id_start": int(pair_id_start),
        "pair_id_end": int(pid - 1) if pid > pair_id_start else int(pair_id_start - 1),
    }

    if debug_checks or strict_checks:
        if debug_checks:
            print(f"[L1 CHECK] Summary: pairs={meta['n_pairs']}, paired_trades={meta['n_paired_trades']}, window_ns={tw_ns}")

        used = pair_id[pair_id >= 0]
        if used.size:
            _, cnt = np.unique(used, return_counts=True)
            if np.all(cnt == 2):
                _ok("Non-reuse")
            else:
                _fail("Non-reuse", f"bad_counts={cnt[cnt!=2][:10]}")
        else:
            _ok("Non-reuse", "no pairs")

        if np.all(~paired_mask | remaining_mask):
            _ok("Remaining-mask safety")
        else:
            _fail("Remaining-mask safety")

    return {
        "pair_id": pair_id,
        "pair_pnl": pair_pnl,  # per leg
        "pair_dt_ns": pair_dt_ns,
        "pair_delta_ref": pair_delta_ref,
        "paired_mask": paired_mask,
        "pairs_df": pairs_df,
        "next_pair_id": pid,
        "meta": meta,
    }


def apply_level1_to_master(df_master: pd.DataFrame, res: dict, *, prefix: str = "L1") -> pd.DataFrame:
    out = df_master.copy()
    out[f"PairID_{prefix}"] = res["pair_id"]
    out[f"PairPnL_{prefix}"] = res["pair_pnl"]  # per leg
    out[f"TimeWindow_{prefix}_ns"] = res["pair_dt_ns"]
    out[f"TimeWindow_{prefix}_s"] = np.where(
        out[f"TimeWindow_{prefix}_ns"].to_numpy() >= 0,
        out[f"TimeWindow_{prefix}_ns"].to_numpy() / 1e9,
        np.nan,
    )
    out[f"DeltaRef_{prefix}"] = res["pair_delta_ref"]
    return out


# =========================
# L2 (LIFO-style sells) split buys
# =========================
def detect_pairs_level2(
    dfc: pd.DataFrame,
    *,
    time_window: str | int | float = "600s",
    remaining_mask: np.ndarray | None = None,
    group_id_start: int = 1,
    group_type: str = "L2_split_buys",
    max_buys_in_window: int = 10,
    min_buys: int = 2,
    debug_checks: bool = True,
    strict_checks: bool = False,
) -> dict:
    """
    L2:
      - same (YearMonthDay, ISIN_id, CP_id)
      - 1 sell qty=X, multiple buys sum qty=X
      - last buy within time_window of sell
      - sells processed newest->oldest (LIFO-style)
      - PnL_per_leg = PnL_total / N_legs
      - DeltaRef = Ref_sell - Ref_last_buy
      - No reuse (consume legs)
    """

    _require_cols(dfc, ["TradeTime_ns", "YearMonthDay", "ISIN_id", "CP_id", "Quantity", "Side", "Price", "Ref"])

    def _ok(name: str, extra: str = ""):
        if debug_checks:
            print(f"[L2 CHECK] {name} OK ✓{('  ' + extra) if extra else ''}")

    def _fail(name: str, extra: str = ""):
        msg = f"[L2 CHECK] {name} FAILED ❌{('  ' + extra) if extra else ''}"
        if strict_checks:
            raise AssertionError(msg)
        if debug_checks:
            print(msg)

    n = len(dfc)
    if remaining_mask is None:
        remaining_mask = np.ones(n, dtype=bool)
    else:
        remaining_mask = np.asarray(remaining_mask, dtype=bool)
        if remaining_mask.shape != (n,):
            raise ValueError("remaining_mask must have shape (len(dfc),)")

    tw_ns = _time_window_to_ns(time_window)

    t = dfc["TradeTime_ns"].to_numpy(np.int64, copy=False)
    day = dfc["YearMonthDay"].to_numpy(np.int32, copy=False)
    isin = dfc["ISIN_id"].to_numpy(np.int32, copy=False)
    cp = dfc["CP_id"].to_numpy(np.int32, copy=False)
    qty = dfc["Quantity"].to_numpy(np.int32, copy=False)
    side = dfc["Side"].to_numpy(np.int8, copy=False)
    price = dfc["Price"].to_numpy(np.float64, copy=False)
    ref = dfc["Ref"].to_numpy(np.float64, copy=False)

    eligible = remaining_mask.copy()

    group_id = np.full(n, -1, dtype=np.int32)
    group_pnl = np.full(n, np.nan, dtype=np.float64)          # per leg
    group_dt_ns = np.full(n, -1, dtype=np.int64)
    group_delta_ref = np.full(n, np.nan, dtype=np.float64)
    group_nlegs = np.zeros(n, dtype=np.int16)
    used_mask = np.zeros(n, dtype=bool)

    idx0 = np.flatnonzero(eligible)
    if idx0.size == 0:
        return {
            "group_id": group_id,
            "group_pnl": group_pnl,
            "group_dt_ns": group_dt_ns,
            "group_delta_ref": group_delta_ref,
            "group_nlegs": group_nlegs,
            "used_mask": used_mask,
            "groups_df": pd.DataFrame(columns=["sell_idx", "buy_idx_list", "pnl_per_leg", "dt_ns", "delta_ref", "nlegs", "group_id", "group_type"]),
            "next_group_id": int(group_id_start),
            "meta": {"n_groups": 0, "n_used_trades": 0, "time_window_ns": int(tw_ns)},
        }

    order = np.lexsort((t[idx0], cp[idx0], isin[idx0], day[idx0]))
    idx = idx0[order]

    groups_sell, groups_buys, groups_pnlL, groups_dt, groups_dref, groups_nlegs, groups_gid = [], [], [], [], [], [], []
    gid = int(group_id_start)

    d = day[idx]; i = isin[idx]; c = cp[idx]
    run_break = np.empty(idx.size, dtype=bool)
    run_break[0] = True
    run_break[1:] = (d[1:] != d[:-1]) | (i[1:] != i[:-1]) | (c[1:] != c[:-1])
    run_starts = np.flatnonzero(run_break)
    run_ends = np.r_[run_starts[1:], idx.size]

    for rs, re in zip(run_starts, run_ends):
        g_idx = idx[rs:re]  # time ordered within (day, isin, cp)

        g_sells = g_idx[side[g_idx] == -1]
        g_buys = g_idx[side[g_idx] == 1]
        if g_sells.size == 0 or g_buys.size == 0:
            continue

        buy_t = t[g_buys]

        # LIFO-style: newest sells first
        for s_idx in g_sells[::-1]:
            if not eligible[s_idx]:
                continue

            qs = int(qty[s_idx])
            if qs <= 0:
                continue

            ts = int(t[s_idx])
            win_end = ts + tw_ns

            left = int(np.searchsorted(buy_t, ts + 1, side="left"))
            right = int(np.searchsorted(buy_t, win_end, side="right"))
            if right <= left:
                continue

            cand = g_buys[left:right]
            cand = cand[eligible[cand]]
            if cand.size == 0:
                continue

            if cand.size > max_buys_in_window:
                cand = cand[:max_buys_in_window]

            cand_qty = qty[cand].astype(np.int32, copy=False)
            if int(cand_qty.sum()) < qs:
                continue

            # subset sum (dict DP) first hit
            prev = {0: (-1, -1)}
            hit = None
            for pos in range(cand.size):
                qv = int(cand_qty[pos])
                if qv <= 0:
                    continue
                keys = list(prev.keys())
                for ssum in keys:
                    nsum = ssum + qv
                    if nsum > qs:
                        continue
                    if nsum not in prev:
                        prev[nsum] = (ssum, pos)
                        if nsum == qs:
                            hit = pos
                            break
                if hit is not None:
                    break

            if hit is None or qs not in prev:
                continue

            chosen_pos = []
            cur = qs
            while cur != 0:
                p_sum, p_pos = prev[cur]
                if p_pos < 0:
                    break
                chosen_pos.append(p_pos)
                cur = p_sum
            if cur != 0:
                continue

            chosen_pos.reverse()
            chosen_buys = cand[np.array(chosen_pos, dtype=np.int64)]
            if chosen_buys.size < min_buys:
                continue

            last_buy_idx = int(chosen_buys[-1])
            dt_ns = int(t[last_buy_idx] - ts)
            if not (0 < dt_ns <= tw_ns):
                continue

            pnl_total = float(qs) * float(price[s_idx]) - float(np.sum(qty[chosen_buys].astype(np.float64) * price[chosen_buys]))
            nlegs = int(1 + chosen_buys.size)
            pnl_per_leg = pnl_total / float(nlegs)
            dref = float(ref[s_idx]) - float(ref[last_buy_idx])

            legs = np.concatenate(([s_idx], chosen_buys))
            eligible[legs] = False
            used_mask[legs] = True

            group_id[legs] = gid
            group_pnl[legs] = pnl_per_leg
            group_dt_ns[legs] = dt_ns
            group_delta_ref[legs] = dref
            group_nlegs[legs] = np.int16(nlegs)

            groups_sell.append(int(s_idx))
            groups_buys.append(chosen_buys.astype(np.int64).tolist())
            groups_pnlL.append(pnl_per_leg)
            groups_dt.append(dt_ns)
            groups_dref.append(dref)
            groups_nlegs.append(nlegs)
            groups_gid.append(gid)

            gid += 1

    groups_df = pd.DataFrame({
        "sell_idx": np.array(groups_sell, dtype=np.int64),
        "buy_idx_list": groups_buys,
        "pnl_per_leg": np.array(groups_pnlL, dtype=np.float64),
        "dt_ns": np.array(groups_dt, dtype=np.int64),
        "delta_ref": np.array(groups_dref, dtype=np.float64),
        "nlegs": np.array(groups_nlegs, dtype=np.int16),
        "group_id": np.array(groups_gid, dtype=np.int32),
        "group_type": group_type,
    })

    meta = {
        "n_groups": int(len(groups_df)),
        "n_used_trades": int(used_mask.sum()),
        "time_window_ns": int(tw_ns),
        "group_id_start": int(group_id_start),
        "group_id_end": int(gid - 1) if gid > group_id_start else int(group_id_start - 1),
        "max_buys_in_window": int(max_buys_in_window),
        "min_buys": int(min_buys),
    }

    if debug_checks or strict_checks:
        if debug_checks:
            print(f"[L2 CHECK] Summary: groups={meta['n_groups']}, used_trades={meta['n_used_trades']}, window_ns={tw_ns}")

        if np.array_equal((group_id >= 0), used_mask):
            _ok("used_mask matches group_id")
        else:
            _fail("used_mask matches group_id")

        if np.all(~used_mask | remaining_mask):
            _ok("Remaining-mask safety")
        else:
            _fail("Remaining-mask safety")

    return {
        "group_id": group_id,
        "group_pnl": group_pnl,  # per leg
        "group_dt_ns": group_dt_ns,
        "group_delta_ref": group_delta_ref,
        "group_nlegs": group_nlegs,
        "used_mask": used_mask,
        "groups_df": groups_df,
        "next_group_id": gid,
        "meta": meta,
    }


def apply_level2_to_master(df_master: pd.DataFrame, res: dict, *, prefix: str = "L2") -> pd.DataFrame:
    out = df_master.copy()
    out[f"GroupID_{prefix}"] = res["group_id"]
    out[f"GroupPnL_{prefix}"] = res["group_pnl"]  # per leg
    out[f"TimeWindow_{prefix}_ns"] = res["group_dt_ns"]
    out[f"TimeWindow_{prefix}_s"] = np.where(
        out[f"TimeWindow_{prefix}_ns"].to_numpy() >= 0,
        out[f"TimeWindow_{prefix}_ns"].to_numpy() / 1e9,
        np.nan,
    )
    out[f"NLegs_{prefix}"] = res["group_nlegs"]
    out[f"DeltaRef_{prefix}"] = res["group_delta_ref"]
    return out


# =========================
# L3 (LIFO) qty tolerance (buy<=sell)
# =========================
def detect_pairs_level3(
    dfc: pd.DataFrame,
    *,
    time_window: str | int | float = "3600s",
    delta_qty: int = 1,
    remaining_mask: np.ndarray | None = None,
    pair_id_start: int = 1,
    pair_type: str = "L3_qty_tolerant",
    debug_checks: bool = True,
    strict_checks: bool = False,
) -> dict:
    """
    L3 (LIFO):
      - same (YearMonthDay, ISIN_id, CP_id)
      - sell -> buy
      - constraint: buy_qty <= sell_qty and (sell_qty - buy_qty) <= delta_qty
      - LIFO scan for eligible sell
      - PnL_per_leg = (min(qsell,qbuy)*(Price_sell-Price_buy)) / 2
      - DeltaRef = Ref_sell - Ref_buy
      - No reuse
    """

    _require_cols(dfc, ["TradeTime_ns", "YearMonthDay", "ISIN_id", "CP_id", "Quantity", "Side", "Price", "Ref"])

    def _ok(name: str, extra: str = ""):
        if debug_checks:
            print(f"[L3 CHECK] {name} OK ✓{('  ' + extra) if extra else ''}")

    def _fail(name: str, extra: str = ""):
        msg = f"[L3 CHECK] {name} FAILED ❌{('  ' + extra) if extra else ''}"
        if strict_checks:
            raise AssertionError(msg)
        if debug_checks:
            print(msg)

    n = len(dfc)
    if remaining_mask is None:
        remaining_mask = np.ones(n, dtype=bool)
    else:
        remaining_mask = np.asarray(remaining_mask, dtype=bool)
        if remaining_mask.shape != (n,):
            raise ValueError("remaining_mask must have shape (len(dfc),)")

    tw_ns = _time_window_to_ns(time_window)
    delta_qty = int(delta_qty)
    if delta_qty < 0:
        raise ValueError("delta_qty must be >= 0")

    t = dfc["TradeTime_ns"].to_numpy(np.int64, copy=False)
    day = dfc["YearMonthDay"].to_numpy(np.int32, copy=False)
    isin = dfc["ISIN_id"].to_numpy(np.int32, copy=False)
    cp = dfc["CP_id"].to_numpy(np.int32, copy=False)
    qty = dfc["Quantity"].to_numpy(np.int32, copy=False)
    side = dfc["Side"].to_numpy(np.int8, copy=False)
    price = dfc["Price"].to_numpy(np.float64, copy=False)
    ref = dfc["Ref"].to_numpy(np.float64, copy=False)

    idx0 = np.flatnonzero(remaining_mask)

    pair_id = np.full(n, -1, dtype=np.int32)
    pair_pnl = np.full(n, np.nan, dtype=np.float64)          # per leg
    pair_dt_ns = np.full(n, -1, dtype=np.int64)
    pair_delta_ref = np.full(n, np.nan, dtype=np.float64)
    paired_mask = np.zeros(n, dtype=bool)

    if idx0.size == 0:
        return {
            "pair_id": pair_id,
            "pair_pnl": pair_pnl,
            "pair_dt_ns": pair_dt_ns,
            "pair_delta_ref": pair_delta_ref,
            "paired_mask": paired_mask,
            "pairs_df": pd.DataFrame(columns=["sell_idx", "buy_idx", "pnl_per_leg", "dt_ns", "delta_ref", "pair_id", "pair_type"]),
            "next_pair_id": int(pair_id_start),
            "meta": {"n_pairs": 0, "n_paired_trades": 0, "time_window_ns": int(tw_ns), "delta_qty": int(delta_qty)},
        }

    # group by (day, isin, cp), sort by time
    order = np.lexsort((t[idx0], cp[idx0], isin[idx0], day[idx0]))
    idx = idx0[order]

    pairs_sell, pairs_buy, pairs_pnlL, pairs_dt, pairs_dref, pairs_pid = [], [], [], [], [], []
    pid = int(pair_id_start)

    d = day[idx]; i = isin[idx]; c = cp[idx]
    run_break = np.empty(idx.size, dtype=bool)
    run_break[0] = True
    run_break[1:] = (d[1:] != d[:-1]) | (i[1:] != i[:-1]) | (c[1:] != c[:-1])
    run_starts = np.flatnonzero(run_break)
    run_ends = np.r_[run_starts[1:], idx.size]

    for rs, re in zip(run_starts, run_ends):
        g_idx = idx[rs:re]  # time ordered

        sell_idx_stack = np.empty(g_idx.size, dtype=np.int64)
        sell_t_stack = np.empty(g_idx.size, dtype=np.int64)
        sell_q_stack = np.empty(g_idx.size, dtype=np.int32)
        head = 0
        tail = 0

        for j in g_idx:
            sj = side[j]
            tj = int(t[j])

            if sj == -1:
                sell_idx_stack[tail] = j
                sell_t_stack[tail] = tj
                sell_q_stack[tail] = qty[j]
                tail += 1
                continue

            if sj == 1:
                while head < tail and (tj - sell_t_stack[head] > tw_ns):
                    head += 1
                if head >= tail:
                    continue

                qbuy = int(qty[j])

                found_pos = -1
                for pos in range(tail - 1, head - 1, -1):  # LIFO scan
                    qsell = int(sell_q_stack[pos])
                    if (qbuy <= qsell) and ((qsell - qbuy) <= delta_qty):
                        found_pos = pos
                        break
                if found_pos == -1:
                    continue

                s_idx = int(sell_idx_stack[found_pos])
                dt = int(tj - int(sell_t_stack[found_pos]))
                if not (0 < dt <= tw_ns):
                    continue

                qmin = min(int(qty[s_idx]), qbuy)
                pnl_total = float(qmin) * (float(price[s_idx]) - float(price[j]))
                pnl_per_leg = pnl_total / 2.0
                dref = float(ref[s_idx]) - float(ref[j])

                pair_id[s_idx] = pid
                pair_id[j] = pid
                pair_pnl[s_idx] = pnl_per_leg
                pair_pnl[j] = pnl_per_leg
                pair_dt_ns[s_idx] = dt
                pair_dt_ns[j] = dt
                pair_delta_ref[s_idx] = dref
                pair_delta_ref[j] = dref
                paired_mask[s_idx] = True
                paired_mask[j] = True

                pairs_sell.append(s_idx)
                pairs_buy.append(int(j))
                pairs_pnlL.append(pnl_per_leg)
                pairs_dt.append(dt)
                pairs_dref.append(dref)
                pairs_pid.append(pid)
                pid += 1

                # consume sell by swap-remove
                last = tail - 1
                if found_pos != last:
                    sell_idx_stack[found_pos] = sell_idx_stack[last]
                    sell_t_stack[found_pos] = sell_t_stack[last]
                    sell_q_stack[found_pos] = sell_q_stack[last]
                tail -= 1

    pairs_df = pd.DataFrame({
        "sell_idx": np.array(pairs_sell, dtype=np.int64),
        "buy_idx": np.array(pairs_buy, dtype=np.int64),
        "pnl_per_leg": np.array(pairs_pnlL, dtype=np.float64),
        "dt_ns": np.array(pairs_dt, dtype=np.int64),
        "delta_ref": np.array(pairs_dref, dtype=np.float64),
        "pair_id": np.array(pairs_pid, dtype=np.int32),
        "pair_type": pair_type,
    })

    meta = {
        "n_pairs": int(len(pairs_df)),
        "n_paired_trades": int(paired_mask.sum()),
        "time_window_ns": int(tw_ns),
        "delta_qty": int(delta_qty),
        "pair_id_start": int(pair_id_start),
        "pair_id_end": int(pid - 1) if pid > pair_id_start else int(pair_id_start - 1),
    }

    if debug_checks or strict_checks:
        if debug_checks:
            print(f"[L3 CHECK] Summary: pairs={meta['n_pairs']}, paired_trades={meta['n_paired_trades']}, delta={delta_qty}, window_ns={tw_ns}")

        used = pair_id[pair_id >= 0]
        if used.size:
            _, cnt = np.unique(used, return_counts=True)
            if np.all(cnt == 2):
                _ok("Non-reuse")
            else:
                _fail("Non-reuse", f"bad_counts={cnt[cnt!=2][:10]}")
        else:
            _ok("Non-reuse", "no pairs")

        if np.all(~paired_mask | remaining_mask):
            _ok("Remaining-mask safety")
        else:
            _fail("Remaining-mask safety")

    return {
        "pair_id": pair_id,
        "pair_pnl": pair_pnl,  # per leg
        "pair_dt_ns": pair_dt_ns,
        "pair_delta_ref": pair_delta_ref,
        "paired_mask": paired_mask,
        "pairs_df": pairs_df,
        "next_pair_id": pid,
        "meta": meta,
    }


def apply_level3_to_master(df_master: pd.DataFrame, res: dict, *, prefix: str = "L3") -> pd.DataFrame:
    out = df_master.copy()
    out[f"PairID_{prefix}"] = res["pair_id"]
    out[f"PairPnL_{prefix}"] = res["pair_pnl"]  # per leg
    out[f"TimeWindow_{prefix}_ns"] = res["pair_dt_ns"]
    out[f"TimeWindow_{prefix}_s"] = np.where(
        out[f"TimeWindow_{prefix}_ns"].to_numpy() >= 0,
        out[f"TimeWindow_{prefix}_ns"].to_numpy() / 1e9,
        np.nan,
    )
    out[f"DeltaRef_{prefix}"] = res["pair_delta_ref"]
    return out
