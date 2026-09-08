# 08 — Validation: face validity, xG, and the reference implementation

Status: done, with one check blocked
Blocked by: 07

Three outside checks, in descending strength.

## Face validity vs tactical consensus
Pep's Barcelona is the most documented team in football and the consensus is
specific: extreme retention, short passing, high connectivity, low directness,
possession sustained in the opposition half. The pipeline was never tuned for any
of it, so independently recovering it is genuine validation. Same shape as 3b
recovering Leicester's counter-attack in coefficients.

**Pre-register the expected direction before looking**, as 3d did with its gate.
If the anchors call Pep's Barcelona direct and long, something is broken.

## Independent xG
Understat runs its own xG model, 2014/15 onward. Per-match correlation against
the chain-level xG. Genuinely independent, unlike FBref.

## Reference implementation
FBref on pass completion, possession share, passes per match, average pass length.

⚠️ **FBref's advanced stats are StatsBomb-derived.** Agreement validates the
aggregation math against a reference implementation, NOT the underlying
measurement. Report it under that label; do not present it as independent
confirmation.

## Acceptance
- Pre-registered directions recorded before the numbers are computed.
- Each check reports agreement AND is labelled independent or not.

## Resolution

**2026-09-06 — 3 tests in `tests/test_external_validation.py`. Suite 173 -> 176.**
Two of three checks done. One is blocked by the source, not by the work.

### 1. Face validity — PASSED, and this one is INDEPENDENT

Directions were sealed in `prereg-08-face-validity.md` and committed in
`3b12613`, **one commit before any number was computed**. That ordering is the
evidence; a direction chosen after seeing the number is a description.

Guardiola's Barcelona, margin over the opponents it actually faced:

| feature | predicted | 2008/09 | 2009/10 | 2010/11 | 2011/12 | verdict |
|---|---|---|---|---|---|---|
| `pass_completion_pct` | > 0 | +13.91 | +15.47 | +17.65 | +16.39 | **4/4** |
| `avg_pass_length_m` | < 0 | −3.68 | −3.36 | −5.07 | −4.86 | **4/4** |
| `pi_connectivity` | > 0 | +0.160 | +0.173 | +0.178 | +0.177 | **4/4** |
| `poss_mean_directness` | < 0 | −0.156 | −0.170 | −0.187 | −0.174 | **4/4** |
| `poss_mean_start_x` *(exploratory)* | > 0 | +9.62 | +2.87 | +6.80 | +8.29 | 4/4 |

16 of 16 required sign predictions hold, and the held-out fifth holds too. The
magnitudes are not marginal: Barcelona completed **14 to 18 percentage points
more** of its passes than the sides it played, while passing 3.4–5.1 m shorter.

**Control passed.** Leicester 2015/16 reads `poss_mean_directness` **+0.082**
against Barcelona's **−0.187**, on the same feature and the same instrument.
Opposite sides of zero, so the instrument separates styles rather than
flattering good teams.

### 2. Aggregation cross-check — PASSED, and it is NOT independent

The FBref comparison was **replaced by a stronger check of the same kind.**
StatsBomb ships xG per shot; the pipeline sums it per (match, team).

| season | rows | max abs difference | exact |
|---|---|---|---|
| PL 2015/16 | 760 | 8.9e-16 | 760/760 |
| Barcelona 2010/11 | 66 | 4.4e-16 | 66/66 |
| Ligue 1 2015/16 | 754 | 1.8e-15 | 754/754 |

**1580 of 1580 rows exact to float precision.** This beats the planned FBref
check on its own terms: FBref's advanced stats are StatsBomb-derived anyway, and
comparing against a rounded published table is approximate where this is exact.
It validates summation, joins and team attribution. It says nothing about
whether StatsBomb's xG model is good.

### 3. Independent xG (Understat) — BLOCKED, not done

Three attempts, then stopped rather than kept hammering:

- `fbref.com` returns **HTTP 403** to automated fetches.
- `understat.com/league/EPL/2015` returns an **18 KB shell**; the season data is
  injected by JavaScript and no `JSON.parse` payload survives in the served
  HTML.

⚠️ **What is actually lost.** This was the only planned check with a genuinely
*independent measurement model*. Every remaining check either compares the
pipeline to itself (2) or to public match results (ticket 07). Nothing in the
phase now tests our xG against a second opinion, and that gap should be stated
in the audit notebook rather than papered over.

**Options, for the user to choose:**
1. Supply Understat data manually (their season CSVs download fine in a browser).
2. Add a dependency (`understatapi` / `soccerdata`) — may hit the same blocking.
3. Accept the gap and record it. Defensible: ticket 07 already provides a fully
   independent check on the *results*, and xG is not load-bearing for Phase 4,
   which reads style anchors rather than xG.
