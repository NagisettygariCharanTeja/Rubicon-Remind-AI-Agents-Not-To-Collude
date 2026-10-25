"""Paper figures rebuilt from headline_numbers.csv (bootstrap CIs already computed on the Volume) and loop_games.csv."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = pd.read_csv("/home/claude/paper/headline_numbers.csv").set_index("result")
G = pd.read_csv("/home/claude/paper/loop_games.csv")
OUT = "/home/claude/paper/figs"
C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "ink": "#0b0b0b", "ink2": "#52514e",
     "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False, "font.family": "DejaVu Sans"})
pct = matplotlib.ticker.PercentFormatter(1.0)
def e(k): r = H.loc[k]; return r.estimate, r.ci_low, r.ci_high
def save(fig, n):
    for x in ("png", "pdf"): fig.savefig(f"{OUT}/{n}.{x}", dpi=300, bbox_inches="tight")
    plt.close(fig)
def bars(ax, labels, est, cols, ylabel, ymax=1.12):
    x = np.arange(len(labels)); p = [v[0] for v in est]
    ax.bar(x, p, 0.62, color=cols, edgecolor="white", linewidth=1.5, zorder=2)
    lo = [v[0] - v[1] if not np.isnan(v[1]) else 0 for v in est]; hi = [v[2] - v[0] if not np.isnan(v[2]) else 0 for v in est]
    ax.errorbar(x, p, yerr=[lo, hi], fmt="none", ecolor=C["ink2"], capsize=3, elinewidth=1, zorder=3)
    for xi, v in zip(x, est): ax.text(xi, (v[2] if not np.isnan(v[2]) else v[0]) + 0.025, f"{v[0]*100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x, labels); ax.set_ylim(0, ymax); ax.yaxis.set_major_formatter(pct); ax.set_ylabel(ylabel)

# Fig 2
fig, ax = plt.subplots(figsize=(6.4, 2.6)); w = 0.36; x = np.arange(3)
names = ["possible messages sent", "messages with explicit coordination", "messages with numeric quotas"]
for j, (arm, lab, col) in enumerate([("chan", "Neutral channel", C["s1"]), ("chan_comply", "Rule: communicating not allowed", C["s2"])]):
    est = [e(f"Stage 1 {arm}: {n}") for n in names]; p = [v[0] for v in est]
    ax.bar(x + (j - .5) * w, p, w, color=col, edgecolor="white", linewidth=1.5, label=lab, zorder=2)
    ax.errorbar(x + (j - .5) * w, p, yerr=[[v[0]-v[1] for v in est], [v[2]-v[0] for v in est]], fmt="none", ecolor=C["ink2"], capsize=2.5, elinewidth=1, zorder=3)
    for xi, v in zip(x + (j - .5) * w, est): ax.text(xi, v[2] + .02, f"{v[0]*100:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x, ["Possible messages\nsent", "Messages with explicit\ncoordination", "Messages with\nnumeric quotas"]); ax.set_ylim(0, 1.1); ax.yaxis.set_major_formatter(pct)
ax.legend(loc="upper right", fontsize=8); save(fig, "fig2_channel_use")

# Fig 3
labs = ["No channel", "Channel arms,\nbefore it opens", "First day the\nchannel exists", "Later\nopen days"]
keys = ["No channel", "Channel arms, before it opens", "First day the channel exists", "Later open days"]
fig, ax = plt.subplots(figsize=(6.0, 2.5))
bars(ax, labs, [e(f"Stage 2 lens flag: {k}") for k in keys], [C["muted"], C["muted"], C["s1"], C["s1"]], "Agent-days flagged", 1.08)
save(fig, "fig3_lens_flags")

# Fig 4
fig, ax = plt.subplots(figsize=(6.4, 2.6)); w = .26
for j, (n, col) in enumerate(zip(["Lens on/off", "Lens strength", "Own plan text"], [C["muted"], C["s1"], C["s2"]])):
    v = [H.loc[f"Stage 2 predict {t} (held-out games 10-19): {n}"].estimate for t in ["restraint", "coord", "action"]]
    ax.bar(x + (j - 1) * w, v, w, color=col, edgecolor="white", linewidth=1.5, label=n, zorder=2)
    for xi, vv in zip(x + (j - 1) * w, v): ax.text(xi, vv + .01, f"{vv:.2f}", ha="center", fontsize=7.5)
ax.axhline(.5, color=C["ink2"], lw=1, ls=(0, (3, 3))); ax.set_xlim(-.8, 2.5); ax.text(-.78, .505, "chance", fontsize=7.5, color=C["ink2"], va="bottom")
ax.set_xticks(x, ["Restraint\n(sells ≤ half cap)", "Coordination\nmessage", "Either"]); ax.set_ylim(.4, .95); ax.set_ylabel("AUC, held-out games")
ax.legend(fontsize=8, loc="upper left", ncol=3); save(fig, "fig4_prediction")

# Fig 5
arms = [("No intervention", "No\nintervention", C["muted"]), ("Steer: remove lens concept", "Steer: remove\nlens concept", C["s1"]),
        ("Matched control dirs", "Matched\ncontrol dirs", C["axis"]), ("Strictness sentence only", "Strictness\nonly", C["s4"]),
        ("Rule repeated near output", "Rule repeated\nnear output", C["s3"]), ("Rule + strictness", "Rule +\nstrictness", "#e87ba4"),
        ("General rule (no loophole)", "General rule\n(no loophole)", "#f2b48f"), ("General rule + strictness", "General rule\n+ strictness", "#b8461c"),
        ("Name the loophole only", "Name the\nloophole only", "#6250d6"), ("Full reminder (loophole + strict)", "Full reminder\n(loophole + strict)", C["s2"])]
fig, axs = plt.subplots(2, 1, figsize=(10.5, 4.8), sharex=True)
bars(axs[0], [a[1] for a in arms], [e(f"Stage 3 rule violation (message sent), {a[0]}") for a in arms], [a[2] for a in arms], "Message sent\n(rule violation)")
bars(axs[1], [a[1] for a in arms], [e(f"Stage 3 valid reply, {a[0]}") for a in arms], [a[2] for a in arms], "Valid reply")
axs[0].axvline(2.5, color=C["axis"], lw=1, ls=(0, (2, 2))); axs[1].axvline(2.5, color=C["axis"], lw=1, ls=(0, (2, 2)))
axs[0].text(1, 1.06, "representation steering", ha="center", fontsize=8.5, color=C["ink2"]); axs[0].text(6, 1.06, "reminder in the prompt (decision regenerated)", ha="center", fontsize=8.5, color=C["ink2"])
axs[1].tick_params(axis="x", labelsize=7.5); fig.tight_layout(); save(fig, "fig5_interventions")

# Fig 6 (bootstrap over games, as on the Volume)
def boot(v, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]; return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
order = [("none", "No guardrail", C["muted"]), ("monitor", "Lens-triggered\nreminder", C["s1"]), ("always", "Always-on\nreminder", C["s2"])]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, col, yl, isp in [(axs[0], "violation_rate", "Violations per game\n(messages / opportunities)", True), (axs[1], "profit_total", "Joint profit per game\n(coins, both farms)", False)]:
    for i, (a, n, c) in enumerate(order):
        v = G[G.arm == a][col].values
        ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, s=20, color=c, edgecolor="white", linewidth=.8, zorder=3)
        m, lo, hi = boot(v); ax.errorbar(i + .25, m, yerr=[[m - lo], [hi - m]], fmt="o", color=C["ink"], ms=4, capsize=3, elinewidth=1, zorder=4)
    ax.set_xticks(range(3), [n for _, n, _ in order]); ax.set_ylabel(yl)
    if isp: ax.yaxis.set_major_formatter(pct)
axs[0].text(-0.15, 1.04, "A", transform=axs[0].transAxes, fontweight="bold", fontsize=11); axs[1].text(-0.2, 1.04, "B", transform=axs[1].transAxes, fontweight="bold", fontsize=11)
fig.tight_layout(); save(fig, "fig6_closed_loop")

# Fig 7
o7 = [("No reminder*", "No\nreminder*"), ("Strictness only", "Strictness\nonly"), ("Rule repeated near output", "Rule repeated\nnear output"),
      ("Rule + strictness", "Rule +\nstrictness"), ("General rule (no loophole)", "General rule\n(no loophole)"), ("General rule + strictness", "General rule\n+ strictness"),
      ("Name the loophole only", "Name the\nloophole only"), ("Full reminder", "Full reminder\n(loophole + strict)")]
fig, ax = plt.subplots(figsize=(10.0, 2.9)); x = np.arange(len(o7)); w = .38
for j, (k, lab, col) in enumerate([("collusion concept in scratchpad (lens)", "Collusion concept in scratchpad (J-lens)", C["s1"]), ("message sent", "Message sent (rule violation)", C["s2"])]):
    est = [e(f"Stage 3 after reminder, {a}: {k}") for a, _ in o7]; p = [v[0] for v in est]
    ax.bar(x + (j - .5) * w, p, w, color=col, edgecolor="white", linewidth=1.5, label=lab, zorder=2)
    ax.errorbar(x + (j - .5) * w, p, yerr=[[v[0]-v[1] for v in est], [v[2]-v[0] for v in est]], fmt="none", ecolor=C["ink2"], capsize=2.5, elinewidth=1, zorder=3)
    for xi, v in zip(x + (j - .5) * w, est): ax.text(xi, v[2] + .02, f"{v[0]*100:.0f}%", ha="center", fontsize=7.5)
ax.set_xticks(x, [b for _, b in o7], fontsize=7.5); ax.set_ylim(0, 1.15); ax.yaxis.set_major_formatter(pct)
ax.legend(fontsize=8, loc="upper right", ncol=2); save(fig, "fig7_thoughts_vs_acts")
print("ok")

# Fig 5 top panel only (main text); the two-panel version stays for the appendix
fig, ax = plt.subplots(figsize=(10.5, 2.9))
bars(ax, [a[1] for a in arms], [e(f"Stage 3 rule violation (message sent), {a[0]}") for a in arms], [a[2] for a in arms], "Message sent\n(rule violation)")
ax.axvline(2.5, color=C["axis"], lw=1, ls=(0, (2, 2)))
ax.text(1, 1.06, "representation steering", ha="center", fontsize=8.5, color=C["ink2"]); ax.text(6, 1.06, "reminder in the prompt (decision regenerated)", ha="center", fontsize=8.5, color=C["ink2"])
ax.tick_params(axis="x", labelsize=7.5); fig.tight_layout(); save(fig, "fig5_violations")
print("ok2")
