# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/natakakakara-droid/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/natakakakara-droid/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at the case target — five quantiles (5%, 25%, 50%, 75%,
95%) per trophy. Built for the [make.fun](https://make.fun) bounty
"gta6-trophy-rarity-forecast-01".

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `*.in` case in
the input dir and writes one `*.out` forecast per case. Offline, no
network, deterministic — the same case always produces the same bytes.
Runs in ~0.05 s per case.

## The model

A trophy's earn rate is earners over players tracked, and both sides move:
players keep arriving while existing players keep earning. The program
works per trophy, on the logit scale.

- **The page is only refreshed every few hours**, so consecutive reads are
  often identical. Identical reads are collapsed to one distinct page each;
  only distinct pages carry information.
- **Frozen history** (a single distinct page, the launch-night shape): the
  page is stale and the next refresh dilutes every non-100% rate, because
  the player base grows much faster than the earners do while the page is
  frozen. The median is shifted down on the logit scale by 0.321 per
  hidden hour — measured from the real launch-night refresh, where earners
  rose x1.34 against players x6.06 over ~4.7 h — capped at 5 hidden hours,
  with a wide spread. The 100% trophy stays 100%.
- **Live history**: logit(rate) is fitted against `ln(1 + hours since
  launch)` over the last 24 h of distinct pages, as a line blended with a
  quadratic (weight 0.3). Growth is back-loaded — new cohorts keep arriving
  while the player base's growth decelerates — so the curve is convex in
  log time; but a raw quadratic overshoots the target badly, and a window
  longer than 24 h drags the slope down, so both are kept in check. With
  too few distinct pages the fit falls back to a line, then to the last
  read.
- **Spread**: the fit's residual sd combined in quadrature with a
  random-walk term that grows with the horizon and a fixed model
  uncertainty. Deliberately kept honest rather than tuned to the
  simulator's optimum: a narrower spread scores better on synthetic
  worlds, but the real final is not a synthetic world.

## Validation (synthetic harness, 50 cases)

Mean pinball loss on the logit scale (lower is better):

| subset | flat | trend | this entry |
|---|---|---|---|
| finals (40) | 1.593 | 0.281 | **0.211** |
| practice-like (10) | 1.212 | 1.212 | **0.282** |
| all (50) | 1.517 | 0.467 | **0.225** |

The finals margin holds in every regime (fast/slow player growth, fast/slow
progression, high/low plateaus: 0.204–0.218). Output is byte-identical
across repeated runs, and every case is valid.

The synthetic worlds are one model of how launch week could play out, not
a measurement of it — so the spread is wider than the simulator's optimum
and the model itself is a generic log-time fit, not a fitted copy of any
one world's mechanics.
