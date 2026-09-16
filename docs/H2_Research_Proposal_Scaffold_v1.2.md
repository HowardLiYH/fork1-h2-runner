# Does Family-Tagged Procedural Memory Survive Cyclic Dormancy?

**A Falsifiable Proposal for Restoration Failure under Busy Interference**  
(H2 / DIC Fork 1 — Research Proposal · Lab-final scaffold v1.2)

Build & Write Lab — Stack ALIGN CLEAR · Supervisor-gated

> **Status:** Lab-final scaffold under Memory Path Stack ALIGN CLEAR (Fork 1 / H2 only).  
> Estimate \(c_{\text{never}}\) **once** from never-learned under a frozen pilot ceiling, then lock \(B = 0.5\,c_{\text{never}}\) for all \(\Pr[\text{restore within } B]\), reacquisition, and \(S\) (no per-cell or post-hoc re-fit).  
> **§6 Results intentionally blank** — no H2 outcome claimed.  
> NicheMem/RISP = schedule/harness only, not evidence. Scope ≠ full DIC residual; Fork 2/3 not in this draft.  
> Open knobs only: PIT cut dates, single frozen open-weight, batch-1.

---

## Abstract

Long-horizon agents that store procedural playbooks for rare financial regimes face a dormancy problem: knowledge useful under one market regime may sit unused while the agent is busy on other families, then must be restored when the regime returns. Prior systems emphasize consolidation efficiency (RecMem), offline rewriting (Auto-Dreamer), retention optimization (OSL-MR), layered half-life decay for financial news (FinMem), or reversible active/dormant/retired states (Reversible Forgetting). None establish whether a *family-tagged* procedural store with capacity pressure (\(\kappa\)) and LRU eviction produces a measurable, dormancy-depth-dependent restoration failure specifically under *busy* interference—versus idle waiting or never-learned baselines.

We propose a falsifiable H2 study. Under a family-tagged store + \(\kappa\) + LRU, for at least three procedural families (earnings, crisis, filings) where busy intervals never reuse a dormant family, restoration probability on busy arms must worsen with dormancy depth \(d \in \{0, 50, 200\}\) at \(\kappa \in \{0.25, 0.50\}\), and remain flat on idle arms and at \(\kappa = 1.00\). The **primary** falsifier is Stack gate G-D: \(\Pr(d=200) \le \Pr(d=0) - 0.25\) at \(\kappa \in \{0.25, 0.50\}\) with \(\kappa=1\) flat. A Kaplan–Meier secondary on reacquisition cost ratio \(c(200)/c(0) \ge 2\) (episodes; \(\varepsilon=0.05\) absolute vs pre-dormancy mean) is *never* the kill. \(c_{\text{never}}\) is estimated once from never-learned under a frozen pilot ceiling; then \(B = 0.5\,c_{\text{never}}\) is locked for \(\Pr[\text{restore within } B]\), cost measurement, and advantage \(S\). A third rung uses \(S\) versus never-learned under a frozen withheld-era partition only when H2 is flat or killed. Empirical results are withheld pending the Fork 1 runner.

**Keywords:** continual learning, procedural memory, dormancy, reacquisition cost, financial regimes, agent memory systems, falsifiable evaluation

## 1. Introduction

Finance agents repeatedly encounter *rare regimes*: earnings seasons, crisis windows, and filing-driven event clusters. Procedural playbooks acquired in one regime often go dormant while the agent continues working on other families. When the dormant regime returns, the agent must restore competence. The scientific question is whether restoration under capacity pressure fails in a predictable, busy-specific way.

Hypothesis H2 (DIC Fork 1) isolates one mechanism: a family-tagged store with capacity fraction \(\kappa\) and LRU eviction. If busy interference over dormancy depth \(d\) causes restoration probability to drop by at least 0.25 from \(d=0\) to \(d=200\) at intermediate \(\kappa\), while idle and \(\kappa=1\) controls stay flat, then H2 survives the primary gate. If that drop fails while \(\kappa=1\) remains flat, H2 dies and the packet returns to Memory Path Stack. NicheMem/RISP supply schedule patterns only—not H2 evidence. Full DIC residual (G-F, Reconcile, dormancy ladder, \(\rho(d)\), keyed paging, fleet R3, baseline bake-offs) and Fork 2/3 are out of this draft.

## 2. Related Work

### 2.1 Consolidation and retention under budget

RecMem [1] is a *polarity foil*: it optimizes when to consolidate, not whether busy interference during dormancy raises procedural restoration failure under family tags. Auto-Dreamer [2] and OSL-MR [3] are **non-kills**: they motivate cost-aware memory but do not instantiate cyclic dormancy of rare-regime finance playbooks with the H2 arm grid and G-D gate.

### 2.2 Financial memory and half-life decay

FinMem [4] layered half-life decay is **not** equivalent to procedural reacquisition cost \(c(d)\). H2 measures episodes to recover pre-dormancy family success under a shared protocol.

### 2.3 Reversible forgetting and dormancy language

Reversible Forgetting [5] supplies dormancy language (active/dormant/retired). H2's system under test remains only family-tagged store + \(\kappa\) + LRU; dormancy ladders and \(\rho(d)\) stay out of v1.

### 2.4 NicheMem / RISP / PopAgent

NicheMem and RISP may contribute schedule scaffolding; their results are not H2 evidence. PopAgent is out of scope. KV-cache idle is not fleet R3.

## 3. Problem Statement and Falsifiable Claim

Families \(F = \{\text{earnings}, \text{crisis}, \text{filings}\}\). After dormancy depth \(d\), restoration probability is measured under checker \(\tau\) on locked eval sets (≥70 instances/family).

**Primary (G-D).** Busy arms: \(\Pr(d=200) \le \Pr(d=0) - 0.25\) at \(\kappa \in \{0.25, 0.50\}\), with \(\kappa=1.00\) flat. Idle predicted flat in \(d\). Failure of the busy drop at both \(\kappa \in \{0.25, 0.50\}\) while \(\kappa=1\) is flat → **H2 dies**, bounce to Stack.

**Secondary (never the kill).** KM \(c(200)/c(0) \ge 2\) in episodes; \(\varepsilon=0.05\) abs vs pre-dormancy mean; \(\theta=0.80\) train-only; censored=max; high censor → KM inconclusive.

**Fail ladder.** (1) G-D holds + \(\kappa=1\) flat → proceed. (2) G-D fails → H2 kill. (3) H2 flat/kill and \(S<0\) under drift → harmfulness path. (4) Rise at \(\kappa=1\) → harness bug.  
\(S = \Pr[\text{restore within } B \mid \text{store}] - \Pr[\text{restore within } B \mid \text{never-learned}]\) (identical probes); cost form same sign. Drift partition: withheld-era / regime-shift, frozen before first \(S\) measure, at \(\kappa \in \{0.25, 0.50\}\) busy.

## 4. Method (System Under Test)

### 4.1 Memory mechanism under test

Under test: **family-tagged store + \(\kappa\) + LRU** only; procedural playbooks; \(\kappa \in \{0.25, 0.50, 1.00\}\). Not built: dormancy ladder, \(\rho(d)\), keyed paging, \(\hat{c}\), fleet R3, bake-off, extractive foils, RecMem/half-life/consolidate arms, G-F/Reconcile.

### 4.2 Experimental factors and frozen \(B\)

Grid: arm ∈ {busy, idle, never-learned, deletion} × \(d \in \{0, 50, 200\}\) × \(\kappa \in \{0.25, 0.50, 1.00\}\) over earnings/crisis/filings. Busy never reuses a dormant family. Frozen: \(\theta=0.80\), \(w=2\).

**\(B\) protocol:** estimate \(c_{\text{never}}\) **once** from the never-learned reference under a frozen pilot ceiling; then lock \(B = 0.5\,c_{\text{never}}\) for every \(\Pr[\text{restore within } B]\), reacquisition budget, and \(S\) cell — **no per-cell or post-hoc re-fit of \(B\)**. Open knobs only: PIT cut dates, one frozen open-weight, batch-1.

### 4.3 Metrics and protocols

- **Pre-dormancy success:** mean on locked eval (≥70/family) immediately before dormancy under \(\tau\).
- **\(\varepsilon\):** within 0.05 abs of that mean (\(\theta\) train-only).
- **\(c\) (episodes):** shared reacq protocol; stop at \(\varepsilon\) or at locked \(B\); censored=max + per-cell censor rate.
- **Primary Pr:** G-D on restoration Pr including \(\Pr[\text{restore within } B]\); 8 seeds; determinism 100/100.

### Table 1. Locked evaluation grid (v1.2)

| Axis | Values | Role |
|------|--------|------|
| Families | earnings, crisis, filings | ≥3 procedural; busy ≠ dormant reuse |
| Arms | busy, idle, never-learned, deletion | busy = H2 focus |
| \(d\) | {0, 50, 200} | cyclic dormancy depth |
| \(\kappa\) | {0.25, 0.50, 1.00} | 1.00 = flat control |
| \(c_{\text{never}}\) / \(B\) | estimate once; \(B=0.5\,c_{\text{never}}\) | same \(B\) for Pr, \(c\), \(S\); no per-cell re-fit |
| Primary | G-D on Pr | only H2 kill |
| Secondary | KM \(c(200)/c(0)\ge 2\) | never the kill |
| \(S\) rung | vs never-learned | harmfulness iff H2 flat/kill + \(S<0\) under drift |
| Open knobs | PIT; open-weight; batch-1 | \(B\) not open |

## 5. Planned Evaluation

After Fork 1 runner exists and Supervisor gates clear: (i) G-D on busy Pr at \(\kappa \in \{0.25, 0.50\}\) with \(\kappa=1\) flatness; (ii) KM secondary under locked \(B\); (iii) \(S\) only if H2 flat/kill, on frozen withheld-era partition. Work Checker remains NicheMem/RISP inheritance contact. §6 filled only from Test artifacts.

## 6. Results

**[RESULTS WITHHELD]**  
6.1 G-D — TBD · 6.2 KM — TBD · 6.3 \(S\) — TBD (rung-3 only).

## 7. Limitations and Non-Claims

H2-only; ≠ full DIC residual; NicheMem/RISP ≠ evidence; FinMem half-life ≠ \(c(d)\); RecMem ≠ H2 support. \(B\) locked after one-shot \(c_{\text{never}}\); open knobs only PIT / open-weight / batch-1. Fork 2/3 out.

## 8. Conclusion

Falsifiable H2 proposal aligned to Stack: busy-specific restoration-Pr drop under \(\kappa\)+LRU, killed by G-D; KM and \(S\) secondary/conditional; \(B\) from one-shot \(c_{\text{never}}\). Results empty until Fork 1 runner + Supervisor clear.

## References

1. RecMem. ACL Findings, 2026. https://aclanthology.org/2026.findings-acl.1619.pdf  
2. Auto-Dreamer. arXiv:2605.20616.  
3. OSL-MR. arXiv:2606.10616.  
4. FinMem. arXiv:2311.13743.  
5. Reversible Forgetting. arXiv:2608.18177.  
6. Memory Path Stack ALIGN CLEAR + Build & Write Lab locks (G-D; one-shot \(c_{\text{never}}\) then \(B=0.5\,c_{\text{never}}\); NicheMem/RISP harness-only).

---

*Document control: H2_Research_Proposal_Scaffold_v1.2 · draft section · G-D primary kill · one-shot \(c_{\text{never}}\) then \(B=0.5\,c_{\text{never}}\) · §6 blank · for Supervisor for Build & Write review.*
