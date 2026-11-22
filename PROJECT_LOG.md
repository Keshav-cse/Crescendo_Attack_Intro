# Project log: Crescendo_Attack_Intro
<!-- last-updated: 2026-11-22T09:46:19Z (maintained by /updaterepo) -->

## Current state
- `Defence/` (history buffer, escalation scorer, tokenizer check) and `Defense2/` (DistilBERT classifier `claim.py`,
  guard `core.py`, Llama wrapper, CLI `main.py`, `train.sft.py`, `checker.py`, `pex.py`), data in `data/{train,val,test}.csv`.
- Trained classifier `model/` (~257 MB) and `llama-3.2-3b-instruct/` are **not in git** (too big). Copy them by hand.
- README has the run instructions, a memory warning (lock file `/tmp/defense2-main.lock`) and troubleshooting.

## Tried: don't repeat
| Approach | Last tried | Result | Why it failed / note |
|---|---|---|---|
| Starting several `python3 -m Defense2.main` at once | before 2026-11 | ❌ runs out of memory | Lock file prevents duplicates; `pkill -f 'python.*Defense2.main'` to clear |
| Trusting a perfect score on the tiny test set | before 2026-11 | ⚠️ misleading | Test on harder, held-out prompts not generated from the same templates (README note) |
| Decoding the full prompt in `llm_wrapper.py` | before 2026-11 | ❌ printed system/user headers | Decode only the newly generated tokens |

## What worked
- Merging the old-laptop commit `df83dc3` (updated check.py + Defense2 runtime + README) with GitHub's history: no conflicts (READMEs identical)

## Decisions
- 2026-11-22: Recovered `Defense2/checker.py` (from VS Code history) added to the repo

## Open questions / next steps
- [ ] Put `model/` on a private Hugging Face repo instead of copying by USB?
- [ ] Evaluate on held-out multi-turn attacks and record numbers here

## History
### 2026-11-22: laptop merge
- Did: merged old-laptop changes into the GitHub repo; added recovered checker.py; pushed; remote now github.com/keshav-talwar
- Result: in sync; no experiments run today
