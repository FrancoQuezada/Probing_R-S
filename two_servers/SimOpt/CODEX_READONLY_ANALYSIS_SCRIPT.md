# Codex Read-Only Analysis Script for the Probing and R&S Julia Codebase

Use this as the instruction prompt for Codex. The goal is to understand the codebase. Do not implement changes.

## Role

You are acting as a code analyst for a research prototype in simulation optimization, probing, and ranking-and-selection. Your job is to understand and document the code, not to modify it.

## Hard constraint

Do not edit, rewrite, patch, rename, format, or delete any source file. Do not create replacement code. Do not apply refactors. This is a read-only analysis pass.

## Files to inspect

Inspect these files:

```text
SimOpt009.jl
SimOpt009_2D_clean.jl
CODEX_CONTEXT_Probing_RS.md
```

Use the PDFs only as conceptual background if available:

```text
Probing_R_S-1.pdf
Probing_R_S-2.pdf
```

## Main questions to answer

1. What is the purpose of each Julia file?
2. What functions are defined in each file?
3. What is the execution flow of each experiment?
4. How is a policy represented?
5. How is the policy grid generated?
6. How does probing affect the simulation?
7. How does signal generation depend on `z`?
8. How does disruption probability depend on `z`?
10. What objective is estimated by `SimOpt009.jl`?
11. Why are these objectives not identical?
12. How does `SimOpt009_2D_clean.jl` reduce the 4D problem to 2D?
13. Which files auto-run and which files are safe to include?
14. What numerical safeguards are present?
15. What statistical or conceptual caveats should be preserved for future work?

## Required output format

Produce a report with the following sections:

```text
1. Executive summary
2. File-by-file map
3. Function dependency map
5. Simulation flow in SimOpt009
6. 2D wrapper logic
7. Theoretical objective alignment
9. Computational complexity notes
10. Known caveats and risks
11. Questions for the researcher before any code changes
12. Recommended next steps, without implementing them
```

## Specific issues to check

Check and report on the following, but do not modify anything:

- `SimOpt009.jl` ends with `#run()` and therefore is safe to include.
- `SimOpt009_2D_clean.jl` includes `SimOpt009.jl` and ends with `main()`.
- `SimOpt009_2D_clean.jl` fixes `z[2] = 1.0` and `z[3] = 1.0`.
- `SimOpt009_2D_clean.jl` optimizes only gradient components 1 and 4.
- `SimOpt009_2D_clean.jl` defines `SEED` but does not appear to use it.
- `SimOpt009.jl` clamps probabilities away from 0 and 1.
- `SimOpt009.jl` uses a likelihood-ratio / score-function gradient estimator.
- `SimOpt009.jl` uses a pathwise maximum over policies.

## Important conceptual warning

`SimOpt009.jl` evaluates every policy on each generated path and takes the maximum pathwise value. This is related to the likelihood-ratio gradient notes but is not the same estimator as the two-stage train/evaluate procedure.

This distinction must be preserved in all future documentation and code comments.

## Allowed actions

You may:

- read files,
- summarize code,
- produce diagrams in text form,
- produce pseudocode summaries,
- identify possible future modifications,
- ask clarifying questions,
- point out risks or inconsistencies.

## Forbidden actions

You may not:

- modify `.jl` files,
- produce a patch,
- create a pull request,
- rename functions,
- change formulas,
- change seeds,
- change experiment settings,
- add package files,
- change auto-run behavior,
- optimize performance.

## Final instruction

End your report with a concise list of proposed future work items, clearly labelled as proposals only. Do not implement them.
