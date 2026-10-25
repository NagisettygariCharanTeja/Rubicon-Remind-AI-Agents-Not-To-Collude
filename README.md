# Rubicon: Remind AI Agents Not To Collude

Code, data and paper for an entry to the [Apart Research AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25) (23–25 October 2026).

**Author:** Charan Teja Nagisettygari (Independent)

## What this is

Two Gemma 3 27B agents run competing farms that sell to one buyer for 20 days. On a random day, a hidden private channel to the rival appears. In the rule arm, the prompt says *"Communicating with Farm X is not allowed."* Every message sent under that rule counts as a violation.

The study runs in three stages, after a preliminary instrument check:

| Stage | Question | Answer |
|---|---|---|
| 0. Instrument check | Does the Jacobian lens (J-lens) see collusion the agent never wrote? | Yes, at matched actions: 58.7% vs 4.8% (instructed), 21.7% vs 0% (peer agreement vs buyer rule) |
| 1. Behaviour | Do agents use the channel against the rule? | 39 of 40 agents broke the rule; numeric quotas fell from 28% to 2%, so the rule changed the words, not the behaviour |
| 2. Detection | Do internals show the intent before the act? | Collusion concepts appear on 89% of first channel days (0% at the same state without a channel) and before 87% of first violations, but they mark the opportunity, not who will act |
| 3. Detection + intervention | What stops the act at the point of no return? | A reminder: violations 88% → 3% at flagged decisions, 68% → 0.5% over whole games (always-on). Steering out the lens concepts: 88% → 84% (n.s.), 23% of replies broken. Concepts stay in ~55% of scratchpads |

The paper is in [`paper/`](paper/). Every number in it comes from the files in [`results/`](results/); [`results/headline_numbers.csv`](results/headline_numbers.csv) holds the main estimates with 95% bootstrap intervals over games.

## Repository layout

```
paper/                 the report (PDF)
docs/                  research_log.md (every result and decision, dated), environment_spec.md
env/                   the farm market (env.py, v0.4 with sell decisions in env_v04.py), prompts, benchmarks
stage0_lens_check/     scripted-state tests: go/no-go, full history, instructed collusion (Option A),
                       J-lens extraction and the concept-family echo filter (strict_concept_filter.py)
stage1_behaviour/      hidden-channel game (channel_game_v2.py) and message summaries
stage2_detection/      J-lens replay of 2,400 agent-days, analysis, Nimble labelling, agreement durability
stage3_intervention/   monitor, steering hooks, reminders (stage3.py), follow-ups and whole games (stage3x.py),
                       general-reminder test (general_reminder_x7.py), summaries
figures/               figure code; figures/paper/ holds the figures used in the report
tests/                 environment unit tests and mock runs (fake agents, no GPU)
results/               logged outputs (see below)
archive/               superseded versions: v1 channel game (leaked reasoning into messages), day-2 pilot
notebooks/             the Colab/Modal notebook used to run everything
```

## How the code was run

All GPU work ran on one A100-80GB through Modal, driven from Colab. The scripts were written as notebook cells: each expects `CODE_DIR` (where these files live) and `RES_DIR` (where results are written) to be defined, and some local test files still contain the author's sandbox paths (`/home/claude/...`). Change those paths before running.

1. **Environment check (CPU):** `cd env && PYTHONPATH=. python ../tests/test_env.py` (10 tests), then `env/bench.py` and `env/bench_v04.py` reproduce the benchmarks (compete 5,400, collude 8,067, cheat 9,988 vs 3,995).
2. **Stage 0:** run the cells in `stage0_lens_check/` in order (go/no-go → history → Option A → internals P2/P3), then `strict_concept_filter.py` on the saved meta files.
3. **Stage 1:** `stage1_behaviour/channel_game_v2.py` (`run_batch`) plays 20 seeds × 3 arms in vLLM with constrained decoding.
4. **Stage 2:** `stage2_detection/s2_extract.py` replays every agent-day through the J-lens (HF transformers + `jlens`), then `s2_analyze.py`, `s2_checks.py`, `durability.py`. `nimble_label.py` labels messages in a separate environment.
5. **Stage 3:** `stage3_intervention/stage3.py` (`run`) for steering, control and reminder on the flagged decisions; `stage3x.py` for the reminder factorial, side-effect check and whole games; `general_reminder_x7.py` for the general reminder; `summary_x.py` and `analyze3.py` print the tests.
6. **Figures:** `figures/figs_volume.py` (`make_all(RES_DIR)`) recomputes all figures and `headline_numbers.csv` from the raw results; `figures/make_paper_figs.py` redraws the paper figures from `headline_numbers.csv` and `loop_games.csv`; `figures/fig1.svg` is the setup diagram (`render.js` exports it with Playwright).

Models: `google/gemma-3-27b-it` (accept the licence on Hugging Face and provide `HF_TOKEN` as a secret, never in code); J-lens weights from `camilablank/workspace-lenses`; message labels from `bespokelabs/Bespoke-Nimble-9B`.

## Results files

| Path | Contents |
|---|---|
| `results/headline_numbers.csv` | Main estimates with 95% bootstrap CIs (resampling whole games) |
| `results/stage0/` | Stage 0 lens readouts (Option A and full-history tests) |
| `results/stage1v2_full/` | Stage 1 games, all 756 channel messages, Nimble labels, blind human label sheet |
| `results/stage2_lens/meta.jsonl` | J-lens hits for all 2,400 agent-days, by reply segment and layer |
| `results/stage3a/` | Flagged decisions and the none / steer / control / reminder continuations |
| `results/stage3x/` | Reminder factorial, general reminder, side-effect check, whole games (`loop_games.csv`), scratchpad concepts after reminders |

## Limitations

One model and one environment; a community-fitted lens; a reply format that offers a message slot every day; weak message labels (human vs Nimble κ = 0.11–0.39); ten whole games per arm. Reminders regenerate the decision rather than acting mid-reply, and tacit coordination through supply was not measured. See Section 5 and Appendix A of the paper.

## Disclosure and LLM use

Literature review, environment design and code, and pilot games were done before the sprint (18–22 October 2026); all reported results were produced during the sprint. Claude (Anthropic) was used to search literature, write and test code, run analyses, make figures and draft text; the author ran every experiment and checked every result.

## Citation

```
Nagisettygari, C. T. (2026). Rubicon: Remind AI agents not to collude. Apart Research AI Collusion Research Sprint.
```
