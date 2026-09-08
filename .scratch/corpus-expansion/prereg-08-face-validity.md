# Pre-registration — ticket 08 face validity

Status: SEALED 2026-09-06, before any number was computed.
Committed ahead of the results on purpose. If it lands in the same commit as
the outcome, it is not a pre-registration.

## Instrument

`src/representation/era.py :: team_contrast(style_df, team, features)` — a
club's season-mean margin over the opponents it actually faced. Chosen because
the Barcelona seasons are single-club, so there is no league field to place them
in, and because the margin is drift-resistant, which matters across a 2008–2016
span of two collection passes.

## Claim under test

Pep's Barcelona is the most documented side in football and the tactical
consensus is specific. The pipeline was never tuned toward any of it, so
independently recovering it is real validation rather than a restatement.

## Predictions — Guardiola's Barcelona, 2008/09 through 2011/12

Direction of Barcelona's margin over its own opponents. **Must hold in all four
seasons** to pass.

| # | feature | predicted | consensus being tested |
|---|---|---|---|
| 1 | `pass_completion_pct` | **> 0** | extreme retention |
| 2 | `avg_pass_length_m` | **< 0** | short passing |
| 3 | `pi_connectivity` | **> 0** | the whole side involved in circulation |
| 4 | `poss_mean_directness` | **< 0** | patient, not direct |

## Exploratory — reported, not asserted

| # | feature | expected | why it is held out |
|---|---|---|---|
| 5 | `poss_mean_start_x` | > 0 | Pep's side pressed high and sustained possession in the opponent half, so this *should* be positive. Held out because 3d-i found Arsenal 2003/04 starting possessions **deeper** than its opponents (0th percentile) while being possession-dominant. Possession dominance evidently does not imply a high start position, so this one could fail for a reason that says nothing about the pipeline. |

## Control — Leicester 2015/16

The instrument has to discriminate, not flatter every good team. Phase 3b found
Leicester **opponent-driven and counter-attacking**, the opposite philosophy.

| # | feature | predicted |
|---|---|---|
| 6 | `poss_mean_directness` | **> 0** — more direct than the sides it beat, opposite in sign to Pep |

If 1–4 hold for Barcelona and 6 holds for Leicester, the same instrument
separated two opposite styles on a feature nobody tuned.

## Failure means

If the anchors call Pep's Barcelona long-passing, low-retention or direct,
something in the pipeline is wrong and Phase 4 must not proceed on it.
