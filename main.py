#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at the case target.

Reads one case (schema makefun/gta6-trophies-case/v1) from the input dir and
writes five quantiles (5/25/50/75/95%) of each trophy's earn rate at the
target to the output dir. Python 3.13 stdlib only, offline, deterministic.

Model, per trophy (rates are E/N: earners over players tracked):
  * The page Exophase shows is only refreshed every few hours, so
    consecutive reads are often identical.  Identical reads are collapsed
    to one distinct page each; only distinct pages carry information.
  * Frozen history (a single distinct page, the launch-night practice
    shape): the page is stale and the next refresh dilutes every non-100%
    rate, because the player base grows much faster than the earners do
    while the page is frozen.  The median is shifted down on the logit
    scale by 0.321 per hidden hour (measured from the real launch-night
    refresh: earners x1.34 against players x6.06 over ~4.7 h), capped at
    5 hidden hours, with a wide spread.
  * Live history: logit(rate) is fitted against x = ln(1 + hours since
    launch) with a quadratic (least squares over the last 48 h of distinct
    pages).  Growth is back-loaded -- new cohorts keep arriving while the
    player base's growth decelerates -- so the curve is convex in log
    time; the quadratic term is shrunk toward zero and clipped to >= 0 so
    a noisy fit cannot run away, and the extrapolated median is clamped
    to a bounded move from the last read.  With too few distinct pages
    the fit falls back to a line, then to the last read.
  * Spread: the fit's residual sd combined in quadrature with a
    random-walk term that grows with the horizon and a fixed model
    uncertainty, clipped to a sane band.  The 100% trophy stays 100%.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Tunables (chosen on the synthetic validation harness, see README).
WINDOW_HOURS = 24.0        # fit window for the live-history fit: growth is
                           # back-loaded, so a longer window drags the slope
                           # down (a 48 h window scored 0.59 on finals)
DILUTION_PER_HOUR = 0.321  # logit fall per hidden hour while frozen
DILUTION_CAP_HOURS = 5.0   # hidden hours counted at most
FROZEN_SPREAD = 0.75       # logit spread when the history is frozen
WALK = 0.35                # random-walk coefficient, per sqrt(horizon days):
                           # narrower than the house 0.5, but kept honest --
                           # the simulator rewards even narrower spreads, and
                           # that would be overfitting to one world model
MODEL_SD = 0.15            # fixed model uncertainty, logit
SPREAD_MIN = 0.30
SPREAD_MAX = 2.0
QUAD_SHRINK = 4.0          # c *= n / (n + QUAD_SHRINK)
QUAD_MAX = 0.8             # quadratic coefficient clipped to [0, QUAD_MAX]
BLEND_QUAD = 0.30          # weight of the quadratic extrapolation vs the line:
                           # the curve is convex in log time, but a raw
                           # quadratic overshoots the target badly (median
                           # +0.61 logit on synthetic finals); blending
                           # centres the median error (see README)
MOVE_CLAMP = 4.0           # |median logit - last logit| at most this


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp_pct(pct)
    return math.log(c / (100.0 - c))


def invlogit_pct(lg):
    if lg >= 20.0:
        return 100.0
    if lg <= -20.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals half-up, JS Math.round style."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def solve3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination with pivoting."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        s = m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))
        x[r] = s / m[r][r]
    return x


def fit_poly(xs, ys, degree):
    """Least-squares polynomial (degree 1 or 2) in centred x.

    Returns (coeffs, resid_sd, n) with coeffs in powers of (x - xm), or
    None when the fit is degenerate.  A tiny ridge keeps it stable.
    """
    n = len(xs)
    if n < degree + 1:
        return None
    xm = sum(xs) / n
    xc = [x - xm for x in xs]
    k = degree + 1
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for xi, yi in zip(xc, ys):
        pw = [1.0]
        for _ in range(degree):
            pw.append(pw[-1] * xi)
        for i in range(k):
            atb[i] += pw[i] * yi
            for j in range(k):
                ata[i][j] += pw[i] * pw[j]
    for i in range(k):
        ata[i][i] += 1e-9
    if degree == 1:
        det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
        if abs(det) < 1e-12:
            return None
        a0 = (atb[0] * ata[1][1] - atb[1] * ata[0][1]) / det
        a1 = (ata[0][0] * atb[1] - ata[1][0] * atb[0]) / det
        coeffs = [a0, a1]
    else:
        coeffs = solve3(ata, atb)
        if coeffs is None:
            return None
    resid = []
    for xi, yi in zip(xc, ys):
        pred = sum(c * (xi ** p) for p, c in enumerate(coeffs))
        resid.append(yi - pred)
    dof = n - k
    sd = math.sqrt(sum(r * r for r in resid) / dof) if dof > 0 else 0.0
    return coeffs, sd, n, xm


def distinct_pages(case):
    """Collapse consecutive identical reads to one page each.

    Returns [(slot_dt, players_tracked, rates_tuple)] in time order.
    """
    pages = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev:
            continue
        prev = key
        pages.append((parse_iso(read["slot"]), read["players_tracked"],
                      tuple(read["rates"])))
    return pages


def forecast_case(case):
    at = parse_iso(case["at"])
    target = parse_iso(case["target"])
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0
    pages = distinct_pages(case)
    trophies = case["trophies"]
    n_t = len(trophies)
    last_rates = list(pages[-1][2]) if pages else [None] * n_t
    # Fall back to earlier pages for trophies missing from the last page.
    for i in range(n_t):
        if last_rates[i] is None:
            for _, _, rates in reversed(pages):
                if rates[i] is not None:
                    last_rates[i] = rates[i]
                    break
    rows = []
    if len(pages) <= 1:
        # Frozen history: the page is stale; the next refresh dilutes.
        page_time = pages[0][0] if pages else at
        hidden = (target - page_time).total_seconds() / 3600.0
        hidden = min(max(hidden, 0.0), DILUTION_CAP_HOURS)
        shift = -DILUTION_PER_HOUR * hidden
        for i, trophy in enumerate(trophies):
            rate = last_rates[i]
            if rate is None:
                rate = 0.0
            if rate >= 99.99:
                rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
                continue
            med = logit(rate) + shift
            qs = [round4(min(100.0, max(0.0, invlogit_pct(med + z * FROZEN_SPREAD))))
                  for z in Z]
            for j in range(1, 5):
                qs[j] = max(qs[j], qs[j - 1])
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    window_start = at.timestamp() - WINDOW_HOURS * 3600.0
    x_target = math.log(1.0 + (target - LAUNCH).total_seconds() / 3600.0)
    base_spread = WALK * math.sqrt(horizon_days)
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            rate = 0.0
        if rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        last_lg = logit(rate)
        xs, ys = [], []
        for slot, _n, rates in pages:
            if slot.timestamp() < window_start:
                continue
            r = rates[i]
            if r is None:
                continue
            hours = (slot - LAUNCH).total_seconds() / 3600.0
            xs.append(math.log(1.0 + hours))
            ys.append(logit(r))
        med = None
        fit_sd = 0.0
        lin = None
        if len(xs) >= 2:
            fit = fit_poly(xs, ys, 1)
            if fit is not None:
                (a0, a1), fit_sd, n, xm = fit
                lin = a0 + a1 * (x_target - xm)
        if len(xs) >= 4:
            fit = fit_poly(xs, ys, 2)
            if fit is not None:
                (a0, a1, a2), fit_sd, n, xm = fit
                a2 = min(max(a2, 0.0), QUAD_MAX) * (n / (n + QUAD_SHRINK))
                d = x_target - xm
                quad = a0 + a1 * d + a2 * d * d
                if lin is not None:
                    med = lin + BLEND_QUAD * (quad - lin)
                else:
                    med = quad
        if med is None:
            med = lin if lin is not None else last_lg
        med = min(max(med, last_lg - MOVE_CLAMP), last_lg + MOVE_CLAMP)
        med = min(max(med, logit(0.01)), logit(99.99))
        spread = math.sqrt(fit_sd * fit_sd + base_spread * base_spread
                           + MODEL_SD * MODEL_SD)
        spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
        qs = [round4(min(100.0, max(0.0, invlogit_pct(med + z * spread))))
              for z in Z]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
