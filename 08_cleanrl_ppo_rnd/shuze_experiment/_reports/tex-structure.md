# Report: structures to imitate for a new Section 9 of `RND_development_document.tex`

File: `/p/rlprojects/RND/07_reconstruction/development_document/RND_development_document.tex` (8040 lines, 448,965 bytes).
Float numbers resolved from `RND_development_document.aux`: `tab:pm-am-envspec` → **Table 64** (p. 109), `tab:pm-am-trainrun1-algos` → **Table 66** (p. 112), `fig:trainrun5-pipeline` → **Figure 16** (p. 90), `tab:trainrun5-trainrun3-rnd` → **Table 53** (p. 88), `tab:trainrun5-hparams` → Table 54, `tab:pm-am-trainrun12-algos` → Table 71.

**Where a new Section 9 goes:** the document has 8 numbered sections plus one appendix section. `\section{point maze + ant maze}` (line 6741) is §8 and runs to line 7832. Line 7834–7836 is `\clearpage / \bibliographystyle{plainnat} / \bibliography{bibliography}`, line 7839 is `\appendix`, and `\section{Distance to ground truth metrics}` (line 7841) is Appendix A. **A new §9 must be spliced between line 7832 and line 7834** (after Figure `fig:pm-am-trainrun12-curves`, before the `\clearpage`+bibliography). Anything after `\appendix` becomes a lettered appendix, not §9.

---

## 1. The three floats, verbatim, with surrounding context

### 1a. Table 64 — `tab:pm-am-envspec` (tex lines 6792–6875)

**Context before (lines 6772–6791)** — the `\paragraph{...}` that leads into it:

```latex
\paragraph{Goal dynamics (code-verified).} The maze environment keeps exactly one active goal at a
time, sampled at reset. A ``diverse'' variant samples that one goal from several candidate cells but
still exposes a single two-dimensional goal, so the agent is never asked to reach more than one goal
at once. Two constructor flags govern what happens when the agent reaches the goal. With the library
defaults (\texttt{continuing\_task=True}, \texttt{reset\_target=False}) the episode never terminates
on success and the goal never moves: the agent keeps collecting the per-step reward for every step
it stays inside the success radius, until the time limit truncates the episode --- this is exactly
the project's previous PointMaze configuration, whose mean extrinsic reward therefore counts steps
spent at the goal. The environment configurations in this section instead set
\texttt{continuing\_task=False}, so the episode terminates the moment the agent is within $0.45$~m of
the goal.\footnote{The installed docstring and the online documentation describe
\texttt{reset\_target} the opposite way from the code: they state that \texttt{True} keeps the goal
in place, whereas the code regenerates the goal after success when \texttt{True} and leaves it fixed
when \texttt{False} (\texttt{point\_maze.py}, \texttt{maze\_v4.py}). The code is authoritative; this
section states the code behavior.} The only place a goal moves mid-episode is the
\texttt{update\_goal} branch, which fires only when \texttt{continuing\_task} and
\texttt{reset\_target} are both true and there is more than one candidate goal cell; under this
section's settings that branch is unreachable, so the goal is fixed at one exact cell center for the
whole run.
```

**The float itself (lines 6792–6875), verbatim:**

```latex
\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.0cm}
>{\raggedright\arraybackslash}p{6.0cm}
>{\raggedright\arraybackslash}p{6.0cm}@{}}
\toprule
\textbf{Parameter} &
\textbf{PointMaze} &
\textbf{AntMaze} \\
& \small(v3: UMaze / Open / Medium / Large) & \small(v5: UMaze / Open / Medium / Large) \\
\midrule
\multicolumn{3}{@{}l}{\textit{Identity and spaces}} \\
agent body & 2-D ball, force-actuated & quadruped Ant (torso $+$ 4 two-segment legs) \\
state passed to the network & $4$-d: $x$, $y$, $v_x$, $v_y$ & $29$-d: $x, y$ $+$ $z$ $+$ torso
  orientation (4) $+$ joint angles (8) $+$ velocities (14); contact forces off \\
goal keys & \texttt{desired\_\allowbreak goal} and \texttt{achieved\_\allowbreak goal} discarded
  (goal never observed) & both goal keys discarded (goal never observed), but $x, y$ are first
  re-attached from \texttt{achieved\_\allowbreak goal}, because AntMaze strips them out of
  \texttt{observation} \\
action & $2$-d force in $[-1,1]$ & $8$-d joint torques in $[-1,1]$ \\
\midrule
\multicolumn{3}{@{}l}{\textit{Maze}} \\
map (rows $\times$ cols) & UMaze $5\times5$ \newline Open $5\times7$ \newline Medium $8\times8$
  \newline Large $9\times12$ & identical maps: \newline UMaze $5\times5$ \newline Open $5\times7$
  \newline Medium $8\times8$ \newline Large $9\times12$ \\
cell size & $1$~m & $4$~m \\
\midrule
\multicolumn{3}{@{}l}{\textit{Episode}} \\
max episode steps (registered default, not overridden) & UMaze $300$ \newline Open $300$ \newline
  Medium $600$ \newline Large $800$ & UMaze $700$ \newline Open $700$ \newline Medium $1000$
  \newline Large $1000$ \\
reward & $-1$ per step; $0$ on the goal-reaching step (sparse reward $-1$, ExPLORe convention) &
  same \\
reaching the goal & \texttt{terminated=True} at distance $\le 0.45$~m
  (\texttt{continuing\_task=False}) & same \\
at the time limit & \texttt{truncated=True}; SAC bootstraps the value (timeout handling) & same \\
\midrule
\multicolumn{3}{@{}l}{\textit{Start and goal}} \\
goal reset & never --- one fixed goal cell, exact center, whole run; the goal cannot move
  mid-episode & same \\
set (1): start $\to$ goal & UMaze: bottom-left $(3,1) \to$ top-left $(1,1)$ \newline Open:
  bottom-left $(3,1) \to$ top-right $(1,5)$ \newline Medium: $(6,1) \to (1,6)$ \newline Large:
  $(7,1) \to (1,10)$ & same cells (identical maps) \\
set (1) config name & \texttt{PointMaze\_\allowbreak UMaze-v3\_\allowbreak start\_\allowbreak
  bottom\_\allowbreak left} \newline \texttt{PointMaze\_\allowbreak Open-v3\_\allowbreak
  start\_\allowbreak bottom\_\allowbreak left} \newline \texttt{PointMaze\_\allowbreak
  Medium-v3\_\allowbreak start\_\allowbreak bottom\_\allowbreak left} \newline
  \texttt{PointMaze\_\allowbreak Large-v3\_\allowbreak start\_\allowbreak bottom\_\allowbreak left} &
  \texttt{AntMaze\_\allowbreak UMaze-v5\_\allowbreak start\_\allowbreak bottom\_\allowbreak left}
  \newline \texttt{AntMaze\_\allowbreak Open-v5\_\allowbreak start\_\allowbreak bottom\_\allowbreak
  left} \newline \texttt{AntMaze\_\allowbreak Medium-v5\_\allowbreak start\_\allowbreak
  bottom\_\allowbreak left} \newline \texttt{AntMaze\_\allowbreak Large-v5\_\allowbreak
  start\_\allowbreak bottom\_\allowbreak left} \\
set (2): start $\to$ goal & UMaze: top-left $(1,1) \to$ bottom-left $(3,1)$ \newline Open: top-left
  $(1,1) \to$ bottom-right $(3,5)$ \newline Medium: $(1,1) \to (6,6)$ \newline Large:
  $(1,1) \to (7,10)$ & same cells (identical maps) \\
set (2) config name & \texttt{PointMaze\_\allowbreak UMaze-v3\_\allowbreak start\_\allowbreak
  top\_\allowbreak left} \newline \texttt{PointMaze\_\allowbreak Open-v3\_\allowbreak
  start\_\allowbreak top\_\allowbreak left} \newline \texttt{PointMaze\_\allowbreak
  Medium-v3\_\allowbreak start\_\allowbreak top\_\allowbreak left} \newline
  \texttt{PointMaze\_\allowbreak Large-v3\_\allowbreak start\_\allowbreak top\_\allowbreak left} &
  \texttt{AntMaze\_\allowbreak UMaze-v5\_\allowbreak start\_\allowbreak top\_\allowbreak left}
  \newline \texttt{AntMaze\_\allowbreak Open-v5\_\allowbreak start\_\allowbreak top\_\allowbreak
  left} \newline \texttt{AntMaze\_\allowbreak Medium-v5\_\allowbreak start\_\allowbreak
  top\_\allowbreak left} \newline \texttt{AntMaze\_\allowbreak Large-v5\_\allowbreak
  start\_\allowbreak top\_\allowbreak left} \\
\bottomrule
\end{tabular}%
}
\caption{Environment specification for the $16$ named configurations ($8$ base environments $\times$
$2$ start/goal sets). PointMaze and AntMaze share identical maze maps per size class and differ only
in the cell size ($1$~m vs $4$~m) and the agent body. Each start/goal set is its own environment
configuration, named by its start corner. UMaze uses the two ends of the U corridor (top-left /
bottom-left); the other maps use the two corner-to-corner diagonals. Cells are 0-based
$(\text{row}, \text{col})$ with row $0$ at the top; the ball or ant is placed at the exact cell
center every episode --- \texttt{position\_\allowbreak noise\_\allowbreak range} is set to $0$ on the
built environment, since it cannot be passed through \texttt{gym.make}. ``same''/``same cells''
means the AntMaze value equals the PointMaze value in that row.}
\label{tab:pm-am-envspec}
\end{table}
```

**Context after (lines 6877–6900)** — the metrics list that follows the table:

```latex
\paragraph{Metrics.} Every train run in this section logs the following five metrics for each
environment configuration it uses. Define $r_t$ as the (shifted) extrinsic reward at env step $t$,
$R = \sum_t r_t$ as the episode return, $k$ as the number of env steps the episode takes with the
goal-reaching step included, and $T_{\max}$ as the episode time limit (the registered default of
Table~\ref{tab:pm-am-envspec}).
\begin{enumerate}[label=\arabic*.,leftmargin=*]
\item \textbf{Episode return} $R = \sum_t r_t$. A successful episode earns $-1$ on each of its first
  $k-1$ steps and $0$ on the terminating step, so $R = -(k-1)$; a failed episode earns $-1$ on all
  $T_{\max}$ steps, so $R = -T_{\max}$.
\item \textbf{Success rate} $= N_{\mathrm{succ}} / N$, where $N_{\mathrm{succ}}$ is the number of
  episodes that end with \texttt{terminated=True} (the agent reached the goal) and $N$ is the total
  number of episodes.
\item \textbf{Steps to reach the goal} $= k$ (the terminating step included), averaged over
  successful episodes only; on a success $R = -(k-1)$, so steps-to-goal $= 1 - R$.
\item \textbf{Maze-cell coverage} $= C_{\mathrm{cell}} / C_{\mathrm{cell}}^{\mathrm{open}}$, where
  $C_{\mathrm{cell}}$ is the number of distinct open maze cells the agent has visited and
  $C_{\mathrm{cell}}^{\mathrm{open}}$ is the total number of open maze cells; one cell is $1$~m for
  PointMaze and $4$~m for AntMaze.
\item \textbf{$1$~m $\times$ $1$~m coverage} $= C_{1\mathrm{m}} / C_{1\mathrm{m}}^{\mathrm{open}}$,
  where $C_{1\mathrm{m}}$ is the number of distinct open $1$~m squares visited and
  $C_{1\mathrm{m}}^{\mathrm{open}}$ is the total number of open $1$~m squares. This equals metric~4
  for PointMaze (its cells are already $1$~m); for AntMaze each open $4$~m cell contains exactly
  $16$ such squares.
\end{enumerate}
```

Note: there is a **frozen copy** of this table for the run that used it — `tab:pm-am-trainrun1-envs` (lines 6952–7017), byte-identical except the set-(2) rows are dropped and the category header reads `\textit{Start and goal (only set (1) is used in this run)}`. Its caption states the freezing rule explicitly. Imitate that pattern if §9 uses a subset of environments.

---

### 1b. Table 66 — `tab:pm-am-trainrun1-algos` (tex lines 7026–7109)

**Context before (lines 7011–7025)** — end of the frozen env table plus the sentence that introduces Table 66:

```latex
\caption{Environments used by train run~1.1: the $8$ set-(1) \texttt{start\_\allowbreak
bottom\_\allowbreak left} configurations. This is Table~\ref{tab:pm-am-envspec} with the set-(2) rows
removed, kept as a separate frozen table so it stays fixed even if
subsubsection~\ref{sec:pm-am-envspec} is extended later. All rows except the start-and-goal block are
identical to Table~\ref{tab:pm-am-envspec}.}
\label{tab:pm-am-trainrun1-envs}
\end{table}

Table~\ref{tab:pm-am-trainrun1-algos} lists the five algorithm columns. Plain SAC and SAC $+$ RND
run on all $8$ configurations; each ground-truth (oracle visit-count) bonus runs on the family it is
defined for --- \texttt{gt\_\allowbreak position\_\allowbreak velocity} on the $4$ PointMaze
configurations, \texttt{gt\_\allowbreak position\_\allowbreak maze\_\allowbreak cell} and
\texttt{gt\_\allowbreak position\_\allowbreak 1m} on the $4$ AntMaze configurations. The SAC $+$ RND
column reuses the \runref{run-5 original-small} RND stack unchanged.
```

**The float itself (lines 7026–7109), verbatim:**

```latex
\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.6cm}
>{\raggedright\arraybackslash}p{2.1cm}
>{\raggedright\arraybackslash}p{3.7cm}
>{\raggedright\arraybackslash}p{3.0cm}
>{\raggedright\arraybackslash}p{2.7cm}
>{\raggedright\arraybackslash}p{2.7cm}@{}}
\toprule
\textbf{Hyperparameter} &
\textbf{SAC} &
\textbf{SAC $+$ RND} &
\textbf{SAC $+$ \texttt{gt\_\allowbreak position\_\allowbreak velocity}} &
\textbf{SAC $+$ \texttt{gt\_\allowbreak position\_\allowbreak maze\_\allowbreak cell}} &
\textbf{SAC $+$ \texttt{gt\_\allowbreak position\_\allowbreak 1m}} \\
& \small(SB3) & \small(run 5 stack) & \small(PointMaze) & \small(AntMaze) & \small(AntMaze) \\
\midrule
\multicolumn{6}{@{}l}{\textit{RL algorithm \& environments}} \\
base RL algorithm & SB3 SAC \texttt{MlpPolicy} & \multicolumn{4}{c}{same} \\
runs on & the $8$ \texttt{start\_\allowbreak bottom\_\allowbreak left} configs &
  the $8$ \texttt{start\_\allowbreak bottom\_\allowbreak left} configs &
  the $4$ PointMaze \texttt{start\_\allowbreak bottom\_\allowbreak left} configs &
  the $4$ AntMaze \texttt{start\_\allowbreak bottom\_\allowbreak left} configs &
  the $4$ AntMaze \texttt{start\_\allowbreak bottom\_\allowbreak left} configs \\
total env steps & $10^{6}$ & \multicolumn{4}{c}{same} \\
\midrule
\multicolumn{6}{@{}l}{\textit{SAC}} \\
discount & $0.99$ (knob; the previous PointMaze-Large runs used $0.999$) &
  \multicolumn{4}{c}{same} \\
optimizer, learning rate & Adam $3{\times}10^{-4}$ (SB3 default) & \multicolumn{4}{c}{same} \\
batch size / tau / buffer & $256$ / $0.005$ / $10^{6}$ (SB3 defaults) & \multicolumn{4}{c}{same} \\
networks & $2\times256$ ReLU (SB3 default) & \multicolumn{4}{c}{same} \\
entropy & auto temperature, target $-\dim(A)$ (SB3 default) & \multicolumn{4}{c}{same} \\
train freq / gradient steps / learning starts & $1$ / $1$ / $100$ (SB3 defaults) &
  \multicolumn{4}{c}{same} \\
\midrule
\multicolumn{6}{@{}l}{\textit{Intrinsic bonus}} \\
bonus input & N/A & next state ($4$-d PointMaze / $29$-d AntMaze) &
  (position $1$~m cell, velocity bin) of the next state; velocity binned $10\times10$ over
  $[-5,5]^2$ & $4$~m maze cell of the next $(x, y)$ & $1$~m square of the next $(x, y)$ \\
bonus form & N/A & RND distillation error, \runref{run-5 original-small} stack (next row) &
  count bonus $\min(1, 1/\sqrt{n})$ (decay rate $-0.5$) & $\min(1, 1/\sqrt{n})$ &
  $\min(1, 1/\sqrt{n})$ \\
RND stack & N/A & target $256 \to$ LeakyReLU($0.2$) $\to 128$; predictor one $128$ block deeper;
  output $128$; orthogonal $\sqrt2$ init, zero bias; obs norm RMS clip $\pm5$ with $6400$-step
  fresh-env warm-up; readout mean squared error over the $128$ dims; reward norm by running std
  (discount $0.99$, non-episodic); Adam $10^{-4}$, $\epsilon\,10^{-8}$; trained on each SAC replay
  batch (proportion $1.0$) --- identical to train run~5 & N/A & N/A & N/A \\
count table shape & N/A & N/A & (rows $\times$ cols) position $\times\,10\times10$ velocity &
  (rows $\times$ cols) of the map & ($4\cdot$rows $\times\,4\cdot$cols) $1$~m squares \\
where the bonus enters & N/A & replay buffer stores extrinsic $+\ \beta \times$ intrinsic (existing
  pipeline) & \multicolumn{3}{c}{same} \\
\midrule
\multicolumn{6}{@{}l}{\textit{Bonus weight}} \\
beta sweep & N/A & \multicolumn{4}{>{\raggedright\arraybackslash}p{12.0cm}}{$\{10^{-3},
  3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}, 10^{-1}, 3{\times}10^{-1}, 1, 3, 10, 30, 10^{2},
  3{\times}10^{2}, 10^{3}, 3{\times}10^{3}, 10^{4}\}$ --- $15$ values (the same for the RND and the
  three ground-truth columns)} \\
\midrule
\multicolumn{6}{@{}l}{\textit{Run bookkeeping (sweep config counts)}} \\
configs in this column & $8$(env configs) $= 8$ & $8$(env configs) $\times$ $15$(bonus weight)
  $= 120$ & $4$(PointMaze env configs) $\times$ $15$(bonus weight) $= 60$ &
  $4$(AntMaze env configs) $\times$ $15$(bonus weight) $= 60$ &
  $4$(AntMaze env configs) $\times$ $15$(bonus weight) $= 60$ \\
total configs in this table & \multicolumn{5}{c}{$8 + 120 + 60 + 60 + 60 = \mathbf{308}$} \\
seeds & \multicolumn{5}{>{\raggedright\arraybackslash}p{14.2cm}}{racing: initial $20$ seeds per
  config (prune floor); survivors race to $100$; then only the best config of each
  (env, algorithm) cell continues to $300$} \\
\bottomrule
\end{tabular}%
}
\caption{Algorithms for train run~1.1. Columns are the five algorithm arms; rows are hyperparameters
grouped by category. ``same'' collapses a value shared with the columns to its left; ``N/A'' marks a
knob that does not apply to a column. The SAC $+$ RND column reuses the \runref{run-5 original-small}
RND stack unchanged (Table~\ref{tab:trainrun5-hparams}). The three ground-truth columns are oracle
visit-count bonuses $\min(1, 1/\sqrt{n})$ over the count table shown, one per environment family. The
config-count rows give each column's contribution as a product annotated with the swept knob, and
the total over the whole table is $308$ configurations, each run at $10^{6}$ env steps.}
\label{tab:pm-am-trainrun1-algos}
\end{table}
```

**Context after (lines 7111–7135)** — the comment block + results paragraph that follows:

```latex
% ==================================================================================================
% Results block added 2026-07-31; the user ended the sweep at this snapshot, so this data is
% final (recolored from the temporary dark brown to black). The three tabulars are
% auto-generated: code/2026-07-31-16-22_pm-am-run1-interim-tables/make_all.py re-reads the run's
% completed per-run records and replaces the blocks between the AUTO-GENERATED TABLE marker
% comments (captions and prose stay hand-edited).
\paragraph{Results (sweep ended at the 2026-07-31 snapshot).}
\label{sec:pm-am-trainrun1-interim}

The sweep was ended at this snapshot, on 2026-07-31, eight days after launch; the numbers below
cover the runs completed by then and are the run's final data. Only records a run wrote at its natural end
(\texttt{completed=true}) enter any aggregate: when a worker is killed mid-run (for example by a
Slurm job reaching its time limit), the run's partial checkpoint record is skipped and the run is
re-queued and re-run in full, so no infrastructure-killed run enters any denominator, and no run
has failed permanently ($0$ in the queue's failed state). The owner's and the collaborator's
workers share one queue, so the numbers pool every submitter's completed runs. As of this
snapshot, $13780$ runs are complete. Of the $308$ configurations, $202$ had been pruned and $106$
were still racing when the sweep ended; no (environment, algorithm) cell reached the winner phase
(which would have started only when every surviving configuration of the cell had $\ge 100$
completed seeds). So the racing stopped at the seed counts shown in the tables: a cell's best
configuration is the best at these counts, least settled in the cells that were still racing many
configurations (named in finding 4 below).
Table~\ref{tab:pm-am-run1-interim-status} gives the racing state per (environment, algorithm)
cell at the stop.
```

---

### 1c. Figure 16 — `fig:trainrun5-pipeline` (tex lines 6031–6042)

**Context before (lines 6008–6030)** — the beta-sweep paragraph and the `\paragraph{Pipeline.}` that introduces the figure:

```latex
\paragraph{Beta sweep and pruning.}
The \runref{run-5 original-small} arm sweeps $\beta\in\{10^{-2},10^{-1},0.5,1,10,10^{2},10^{3},10^{4}\}$. The eight betas
race: let $\bar R_\beta$, $s_\beta$, and $n_\beta$ be the mean, sample standard deviation, and count
of a beta's finished-seed final training-episode rewards, and let the \emph{bar} be the largest
$\bar R_\beta$ over betas with $n_\beta\ge 30$. A beta is stopped once
\begin{align}
n_\beta \ge 30
\qquad\text{and}\qquad
\bar R_\beta + 2.576\,\frac{s_\beta}{\sqrt{n_\beta}} \;<\; \mathrm{bar}
\end{align}
(its one-sided $99\%$ upper confidence limit sits below the running best mean); survivors run to the
full $300$ seeds. The benchmark and reward-norm arms never prune and run all $300$ seeds.

\paragraph{Pipeline.}
Figure~\ref{fig:trainrun5-pipeline} traces the original RND data flow --- environment step, next
state, observation whitening, frozen target and trainable predictor, per-dimension error, readout,
forward-filter reward normalization, $\beta$-scaling, and the sum with the extrinsic reward feeding
the RL update --- with CleanRL's departures marked as callouts: its predictor \emph{update} feeds
the stored current-step observation rather than the next observation; its convolutional LeakyReLU
slope is $0.01$; it clips gradients at global norm $0.5$; and it normalizes advantages per
minibatch. The \runref{run-5 original-small} arm follows the original everywhere the figure's solid path runs (next-state
input throughout, no gradient clip), on SAC's single reward stream.
```

**The float itself (lines 6031–6042), verbatim:**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{code/2026-07-20_run5-writeup/rnd_pipeline.pdf}
\caption{The original RND pipeline (solid path): an environment step yields the next state $s'$,
whitened by a running mean/std and clipped; the frozen target $f(s')$ and the trainable predictor
$\hat f(s')$ give the per-dimension error $e$, read out as the bonus $b$, normalized by the running
std $\sigma_F$ of the forward-filtered intrinsic return, scaled by $\beta$, and added to the
extrinsic reward for the RL update. The predictor-gradient path (blue dashed) terminates only at the
predictor; the RMS statistic updates are dotted. CleanRL's departures from the original are marked as
dashed callouts. The \runref{run-5 original-small} arm follows the solid path on SAC's single reward stream.}
\label{fig:trainrun5-pipeline}
\end{figure}
```

**Context after (lines 6044–6062)** — the comment banner + opening of the big hyperparameter table it precedes:

```latex
% ============================================================================================
% BIG hyperparameter table — columns by source; architecture as net-pipeline strings.
% Table-48 machinery: \footnotesize, tabcolsep 3pt, \resizebox to \textwidth, p{} cols,
% \allowbreak in long \texttt, \multicolumn group-header rows and shared-value collapses, --- for N/A.
% ============================================================================================
\begin{table}[H]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.1cm}
>{\raggedright\arraybackslash}p{3.0cm}
>{\raggedright\arraybackslash}p{3.0cm}
>{\raggedright\arraybackslash}p{3.0cm}
>{\raggedright\arraybackslash}p{2.5cm}
>{\raggedright\arraybackslash}p{2.5cm}
>{\raggedright\arraybackslash}p{3.2cm}@{}}
\toprule
& & & & \multicolumn{2}{c}{\textbf{train run 3 RND}} & \\
\cmidrule(lr){5-6}
```

---

## 2. Preamble (lines 1–110), verbatim

```latex
\documentclass{article}

% Pass the round citation option to natbib BEFORE the NeurIPS style auto-loads it.
% The style loads natbib itself, so a second \usepackage[round]{natbib} would cause
% an "Option clash for package natbib" error; passing the option here avoids it.
\PassOptionsToPackage{round}{natbib}

% Keep the same paper style when the NeurIPS style file is available.
\IfFileExists{neurips_2026.sty}{\usepackage[preprint]{neurips_2026}}{}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}
\usepackage{xcolor}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{enumitem}
\usepackage{longtable}
\usepackage{multirow}
% natbib is loaded by the NeurIPS style above (with the round option passed via
% \PassOptionsToPackage near \documentclass); do not load it again here.
\usepackage[hidelinks,colorlinks=true,citecolor=blue,linkcolor=blue,urlcolor=blue]{hyperref}
\usepackage[capitalize,noabbrev]{cleveref}
\usepackage{subcaption}   % per-panel subcaptions (run-3.2.2 initial-bonus figures)
\usepackage[most]{tcolorbox}
\usepackage[normalem]{ulem}
\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
% Claude-edit review markup (claude-edit-latex skill); ulem + xcolor already loaded above.
\definecolor{claude-brown}{RGB}{133,45,15}
\definecolor{claude-green}{RGB}{0,100,0}
\newcommand{\claudeedit}[1]{{\color{claude-brown}\,#1}}
% run-8.1.2 sweep table (Table \ref{tab:pm-am-trainrun12-algos}): blue marks an arm's critical
% design knobs, incl. inherited "same" cells that are load-bearing for that arm
\definecolor{critical-blue}{RGB}{0,90,200}
\newcommand{\criticalknob}[1]{{\color{critical-blue}#1}}
\newcommand{\claudecomment}[1]{({\color{claude-green} Claude: #1})}
% Provenance-name marker (user request 2026-07-31): every reference to a named run artifact
% (\runref{run-5 original-small}, \runref{run-3.2.2 benchmark}, \runref{run-3.2.3 reward-norm}, \runref{run-3.2.1 winner 1}/2, ...)
% is set bold inside square brackets so it can be found at a glance.
\newcommand{\runref}[1]{\textbf{[#1]}}

% Python code listings (used in the "When statistics are updated" design block).
\usepackage{listings}
\definecolor{code-keyword}{RGB}{0,0,170}
\definecolor{code-comment}{RGB}{110,110,110}
\definecolor{code-string}{RGB}{160,40,40}
\lstdefinestyle{pythoncode}{%
  language=Python,
  basicstyle=\ttfamily\footnotesize,
  keywordstyle=\color{code-keyword},
  commentstyle=\color{code-comment}\itshape,
  stringstyle=\color{code-string},
  showstringspaces=false,
  breaklines=true,
  breakatwhitespace=true,
  columns=fullflexible,
  keepspaces=true,
  frame=single,
  rulecolor=\color{gray!45},
  framesep=4pt,
  xleftmargin=6pt,
  aboveskip=6pt,
  belowskip=6pt,
}

\mathtoolsset{showonlyrefs}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\title{Exploration Bonuses on PointMaze:\\ Method and Environment Development Document}

\author{%
  Anonymous Author(s)\\
  Affiliation\\
  \texttt{email}
}

\begin{document}

\maketitle

\tableofcontents
\clearpage
```

### Preamble facts that matter for a new float

| Item | Value / use |
|---|---|
| Document class | `article` + `neurips_2026.sty` with the `preprint` option; page geometry `textwidth=5.5in` (`neurips_2026.sty:126`). Body font is the NeurIPS 10pt Times. |
| Tables | `booktabs` (`\toprule`/`\midrule`/`\bottomrule`/`\cmidrule(lr){i-j}`), `array` (for `>{\raggedright\arraybackslash}p{..}`), `multirow`, `longtable` (only for the two catalog tables, `tab:notation` and `tab:methods`), `float` (for `[H]`). **No `caption` package and no `\captionsetup` anywhere** — captions use the NeurIPS style's own format. |
| Figures | `graphicx`, `subcaption` (for side-by-side panels). No TikZ in the main document — TikZ figures are compiled standalone and included as PDF. |
| Custom macros used inside floats | `\runref{...}` → bold text in square brackets, marks a named run artifact (e.g. `\runref{run-5 original-small}`). `\criticalknob{...}` → `critical-blue` RGB(0,90,200), marks an arm's load-bearing knobs (used only in Table 71). `\claudeedit{...}` (brown) and `\claudecomment{...}` (dark green) are the review-markup macros — not currently live in §8. `\cmark`/`\xmark` from `pifont`. |
| `\allowbreak` convention | Every long `\texttt{}` snake_case identifier is written `\texttt{a\_\allowbreak b\_\allowbreak c}` so it can wrap inside a `p{}` column. The generator's helper is `tt()` in `make_all.py` (`"\\texttt{" + name.replace("_", "\\_\\allowbreak ") + "}"`). |
| `\shortstack` usage | Only in **column headers** of the auto-generated results tables, always as `\textbf{\shortstack[c]{line one\\line two}}` (lines 7235–7240, 7777–7780) and once in a hand-built results table as `\shortstack{\textbf{episode}\\\textbf{succ.\ rate}}` (lines 5282–5284). Never rotated headers. |
| Math | `mathtools` with `\mathtoolsset{showonlyrefs}` — **an `align` gets a number only if it is `\eqref`'d somewhere**. Equations that are referenced carry a `\label{eq:...}` (e.g. `eq:rms-update`, `eq:obs-norm`, `eq:rwd-norm`). |
| Cross-refs | `hyperref` + `cleveref` are loaded, but the document writes references by hand as `Table~\ref{...}`, `Figure~\ref{...}`, `Section~\ref{...}`, `subsubsection~\ref{...}`, `Eq.~\eqref{...}`. `\cref` is not used. |

---

## 3. The generator of Figure 16

**It is not Python. It is a standalone TikZ document.**

* **Generator path:** `/p/rlprojects/RND/07_reconstruction/development_document/code/2026-07-20_run5-writeup/rnd_pipeline.tex` (3,978 bytes, 68 lines).
* **How it is invoked:** the instruction is in its own line 1 — `% Standalone RND pipeline figure for Train run 5. Compile: latexmk -pdf rnd_pipeline.tex`, run from inside that folder. The build record `_build.log` (10,079 bytes) in the same folder shows the actual command that ran: `pdflatex -interaction=nonstopmode -halt-on-error -recorder "rnd_pipeline.tex"` driven by `Latexmk … Version 4.88`, with pdfTeX from TeX Live 2026 out of the user TinyTeX tree `/u/sl5nw/.TinyTeX/`. One pdflatex pass; output `Output written on rnd_pipeline.pdf (1 page, 140266 bytes)`.
* **Which Python:** none. No Python is involved in this figure. (For contrast, the document's *other* code folders do use Python by full path — `/p/rlprojects/RND/.venvs/exploration/bin/python make_point_set_figure.py` for `code/2026-07-19-03-21_convergence-run1-point-sets/`, and `/p/rlprojects/RND/.venvs/exploration/bin/python make_all.py` for the two interim-table generators.)
* **Where the produced files live:** same folder — `rnd_pipeline.pdf` (the one included in the document), `rnd_pipeline_preview.png` (19,548 bytes, for eyeballing; there is no script in the folder that produces it, so it was made by a one-off rasterizer call), plus `rnd_pipeline.aux/.log/.fls/.fdb_latexmk` and `_build.log`. The folder also holds `section_draft.tex` (19,897 bytes), the reviewable draft of the whole train-run-5 subsection before it was spliced into the main document.
* **How it is included in the tex:** exactly one place, line 6033 — `\includegraphics[width=\linewidth]{code/2026-07-20_run5-writeup/rnd_pipeline.pdf}`. The path is **relative to the document root**, with no `../`. (Run-folder artifacts use the other convention: `\includegraphics[...]{../train_runs/<long-run-slug>/analysis/plots/<name>.pdf}`, e.g. lines 6222 and 7816, and `\input{../train_runs/.../analysis/plots/reward_table.tex}` at lines 6192 and 6209.)

**Style constants** — the whole `tikzpicture` option list (lines 9–22):

| key | value | role |
|---|---|---|
| `font` | `\small` | all node text (standalone class is 10pt, so ≈9pt nominal) |
| `>` | `Stealth[length=2.2mm]` | arrow tip |
| `box` | `draw, rounded corners=1.5pt, align=center, inner sep=3pt, minimum height=8mm, fill=white` | ordinary pipeline stage |
| `frozen` | `box, fill=black!8` | the frozen target network |
| `pred` | `box, fill=green!8` | the trainable predictor |
| `op` | `draw, circle, inner sep=1pt, minimum size=6mm, fill=white` | scalar operator (the `$\times\beta$` node) |
| `flow` | `->, thick` | forward data path (solid) |
| `grad` | `->, dashed, thick, blue!60!black` | predictor-gradient path |
| `stat` | `->, dotted, thick, orange!80!black` | RMS statistic-update path |
| `callout` | `draw, dashed, rounded corners=1.5pt, align=center, font=\scriptsize, inner sep=2.5pt` | annotation box |
| `cleanrl` | `callout, blue!65!black` | a CleanRL-departure annotation |
| `node distance` | `7mm and 9mm` | default `positioning` spacing |

**Full generator body, verbatim** (`rnd_pipeline.tex`, all 68 lines):

```latex
% Standalone RND pipeline figure for Train run 5. Compile: latexmk -pdf rnd_pipeline.tex
% Original RND data flow (solid spine), with CleanRL departures as dashed callouts. The predictor
% gradient path is dashed and terminates ONLY at the predictor; RMS statistic updates are dotted.
\documentclass[tikz,border=6pt]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\begin{document}
\begin{tikzpicture}[
  font=\small,
  >={Stealth[length=2.2mm]},
  box/.style={draw,rounded corners=1.5pt,align=center,inner sep=3pt,minimum height=8mm,fill=white},
  frozen/.style={box,fill=black!8},
  pred/.style={box,fill=green!8},
  op/.style={draw,circle,inner sep=1pt,minimum size=6mm,fill=white},
  flow/.style={->,thick},
  grad/.style={->,dashed,thick,blue!60!black},
  stat/.style={->,dotted,thick,orange!80!black},
  callout/.style={draw,dashed,rounded corners=1.5pt,align=center,font=\scriptsize,inner sep=2.5pt},
  cleanrl/.style={callout,blue!65!black},
  node distance=7mm and 9mm,
]
% ---- main spine (left to right) ----
\node[box] (env) {SAC actor\\$\to$ env step};
\node[box,right=of env] (s) {next state\\$s'$};
\node[box,right=of s] (norm) {obs RMS\\whiten, clip $[-5,5]$};
% target above, predictor below the spine, both fed by norm and feeding the error
\node[frozen,above right=6mm and 12mm of norm] (tgt) {frozen target $f(s')$\\{\scriptsize(no grad)}};
\node[pred,below right=6mm and 12mm of norm] (prd) {trainable\\predictor $\hat f(s')$};
\node[box,right=24mm of norm] (err) {error\\$e=f-\hat f$};
\node[box,right=of err] (read) {readout\\$b=\tfrac1m\sum_j e_j^2$};
\node[box,right=of read] (rnorm) {reward norm\\$F_t=\gamma F_{t-1}+b$;\ \ $b/\sigma_F$};
\node[op,right=of rnorm] (beta) {$\times\beta$};
\node[box,right=of beta] (sum) {$r=r^{\mathrm{ext}}+\beta\,r^{\mathrm{int}}$};
\node[box,right=of sum] (rl) {SAC replay\\$+$ update};
% ---- forward edges ----
\draw[flow] (env) -- (s);
\draw[flow] (s) -- (norm);
\draw[flow] (norm.east) -- (tgt.west);
\draw[flow] (norm.east) -- (prd.west);
\draw[flow] (tgt.east) -- (err.north west);
\draw[flow] (prd.east) -- (err.south west);
\draw[flow] (err) -- (read);
\draw[flow] (read) -- (rnorm);
\draw[flow] (rnorm) -- (beta);
\draw[flow] (beta) -- (sum);
\draw[flow] (sum) -- (rl);
% ---- extrinsic reward bypass (env -> sum): flat orthogonal route below the predictor box ----
\draw[flow] (env.south) |- ($(sum.south)+(0,-1.6)$) node[pos=0.28,below,font=\scriptsize]{$r^{\mathrm{ext}}$} -| (sum.south);
% ---- RL loop back-edge: flat orthogonal route ABOVE all callouts; label at the far right ----
\draw[flow] (rl.north) |- ($(env.north)+(0,4.4)$) node[pos=0.9,above,font=\scriptsize]{next action (RL loop)} -| (env.north);
% ---- predictor gradient path (dashed, terminates at predictor only) ----
\node[below=6mm of read,font=\scriptsize,blue!60!black] (loss) {loss $\tfrac12\lVert e\rVert^2$};
\draw[grad] (read.south) -- (loss.north);
\draw[grad] (loss.west) to[out=180,in=-90] node[below,pos=0.7,font=\scriptsize]{$\nabla_{\theta}$; target frozen} (prd.south east);
% ---- RMS statistic-update paths (dotted) ----
\draw[stat] (s.north) to[out=90,in=90] node[above,pos=0.5,font=\scriptsize]{update $\mu,\sigma$} (norm.north);
\draw[stat] (read.north) to[out=90,in=90] node[above,pos=0.5,font=\scriptsize]{update $\sigma_F$ (per-step, non-episodic)} (rnorm.north);
% ---- CleanRL departure callouts (placed in whitespace, short leaders) ----
\node[cleanrl,above=12mm of tgt] (c1) {CleanRL: conv LeakyReLU slope $0.01$ (vs original $0.2$)};
\draw[dashed,blue!65!black] (c1.south) -- (tgt.north);
\node[cleanrl,below=14mm of prd] (c2) {CleanRL: predictor \emph{update} feeds the stored current obs, not $s'$};
\draw[dashed,blue!65!black] (c2.north) -- (prd.south west);
\node[cleanrl,below=11mm of rl] (c3) {CleanRL: grad clip $0.5$;\\per-minibatch adv.\ norm};
\draw[dashed,blue!65!black] (c3.north) -- (rl.south);
\end{tikzpicture}
\end{document}
```

**Two cautions if §9 needs a diagram.**

1. **Print scale.** `_build.log` records the standalone page size as `papersize=800.50578pt,272.21997pt` — 11.08 in × 3.77 in. Included at `width=\linewidth` (5.5 in) it is scaled to about 0.50, so the figure's `\small` (≈9pt) node text prints at roughly 4.5pt — about half the body font. If the new diagram must satisfy the shared `draw-or-diagrams` rule (every diagram text at least the document's body size at print scale), it has to be designed at 5.5 in wide, not shrunk.
2. **There is a different, skill-sanctioned diagram route.** `/p/rlprojects/.claude/skills/draw-or-diagrams/SKILL.md` specifies matplotlib box-and-arrow figures with numbered step badges plus a matching numbered reading guide under the figure, a reference library at `/p/rlprojects/RLforOR/inventory_management/developement_document/figures/_lib/`, `\begin{figure}[H]`, `\includegraphics[width=\textwidth]{...}`, and a subagent vision check. Figure 16 predates it and does not follow it. Pick one route deliberately.

---

## 4. The project's own RND — the exact stack named "SAC + RND" in Table 66

Table 66's `SAC $+$ RND` column is not defined in Table 66. It says (line 7024) *"The SAC $+$ RND column reuses the `\runref{run-5 original-small}` RND stack unchanged"*, gives a one-cell summary in the `RND stack` row (lines 7073–7077), and points the caption at `Table~\ref{tab:trainrun5-hparams}` (Table 54) for the cell-by-cell values. So the definition is assembled from four places:

* the `RND stack` cell of Table 66, lines **7073–7077**;
* the run-5 hyperparameter table `tab:trainrun5-hparams`, lines **6049–6154** (its last column, `run 5 (this work)` / `\runref{run-5 original-small}`);
* the translation list `\paragraph{From Atari-PPO to PointMaze-SAC: the translation choices.}`, lines **5979–6006**;
* §4 "RND tricks" for the normalization mechanisms, lines **1729–1888** (and the general library definition of the bonus in §2.2.3, lines **266–333**).

Note that "train run 3 RND" (Table 53, lines 5858–5884) is a **different, earlier** RND — the run-5 arm borrows only its widths. Do not confuse the two.

### Complete list of the project's RND design choices (the run-5 original-small stack)

| Knob | Value | Tex line(s) |
|---|---|---|
| **Bonus input** | next state $s'$ — 4-d on PointMaze $(x,y,v_x,v_y)$, 29-d on AntMaze. Not $s$, not $(s,a)$. | 6080–6081 (`RND input … next state $s'$ ($4$-d)`); 7067 (`bonus input … next state ($4$-d PointMaze / $29$-d AntMaze)`); 5984 |
| **Target network** | `state(4) → 256 → LeakyReLU → 128`, frozen, never trained | 6089; 5988–5989; 7073 |
| **Predictor network** | same trunk **one 128-wide LeakyReLU block deeper** than the target: `state(4) → 256 → LReLU → 128 → LReLU → 128`. This asymmetry is the original RND's defining property, kept deliberately. | 6090–6093 (`$\dots128\!\to\!\mathrm{LReLU}\!\to\!128$ (\textbf{1 deeper})`); 5985–5990; 7073 |
| **Output dimension $m$** | $128$ (so the error vector has 128 dims) | 6094; 5986; 7074 (run-1.2 table) |
| **Activation** | LeakyReLU, negative slope **$0.2$** — the TensorFlow `tf.nn.leaky_relu` default of the original paper, explicitly *not* PyTorch's $0.01$ that CleanRL inherits | 6095; 5991–5993 |
| **Weight initialization** | orthogonal, gain $\sqrt2$ | 6096–6097; 7073 |
| **Bias initialization** | **zero**, both nets | 6098; 7073; 5869 (train-run-3 base, same) |
| **Observation (state) normalization** | running mean/std (RMS) per dimension, then clip to $[-5,5]$; the project divides by $\sqrt{\sigma^2+10^{-8}}$ where the original divides by $\sqrt{\sigma^2}$ | 6101–6102 (`RMS, clip $\pm5$ (denom $\mathrm{var}{+}10^{-8}$)`); 5894–5917; 1809–1812 (Eq. `eq:obs-norm`); 272–273 |
| **RMS update rule** | pooled-moments update, $\mu=0,\sigma^2=1,n=10^{-4}$ at start, per dimension; applied to target and predictor input only — SAC still sees raw states | 1783–1801 (Eq. `eq:rms-update`); 1868–1871 |
| **Obs-norm warm-up** | **6400 env steps** of a random agent, rolled on a **fresh throwaway environment** so the training visit counts are untouched | 6103–6104 (`$6400$ env steps (fresh env)`); 5999–6000; 7073 |
| **Bonus readout** | the original's **mean over the 128 dims**, $\tfrac1m\sum_j e_j^2$ — the `mse_mean` readout. Not CleanRL's $\tfrac12\sum_j e_j^2$, and not the project library's own default ($\tfrac12\sum_j e_j^2$, §2.2.3). | 6107–6108; 5994–5995; 5950–5956; 7073; library default at 274–280 |
| **Predictor training loss** | the same squared error, mean over the minibatch (the readout never enters training) | 281–282; 5953; 7474–7475 (`$L_{\mathrm{original}}$` … "is the loss train run~1.1 used") |
| **Intrinsic reward normalization** | **on**. Forward filter $F_t = \gamma_{\mathrm{norm}} F_{t-1} + r^{\mathrm{int}}_t$ with $\gamma_{\mathrm{norm}} = 0.99$, one accumulator per env copy, **non-episodic** (not reset at episode ends); the reward is divided by $\sigma_F$, the running std of the filter values; no mean subtracted, so the bonus stays non-negative. | 6110–6113; 6001; 5922–5937; 1815–1834 (Eq. `eq:rwd-norm`); 7073 |
| **Predictor optimizer** | Adam, learning rate $10^{-4}$, $\epsilon = 10^{-8}$, betas $(0.9,0.999)$, no weight decay, **no gradient clipping**, own optimizer (not shared with SAC) | 6116–6121; 5996; 5958–5961; 7073 |
| **Update proportion** | **$1.0$** — the predictor is trained on the whole SAC replay batch each step, following the original *repository code*, not the paper's Table-5 value of $0.25$ | 6119–6120; 5997–5998; 5962–5965; 7073 |
| **Where the bonus enters** | single reward stream: the replay buffer stores $r^{\mathrm{ext}} + \beta\,r^{\mathrm{int}}$. There is no separate intrinsic value head, so the original's $2:1$ extrinsic-to-intrinsic weighting collapses to the scalar $\beta$ (the faithful point is $\beta = 0.5$). | 6124–6129; 6002–6005; 7080–7081 |
| **Bonus weight $\beta$ (in Table 66)** | swept over the 15-value grid $\{10^{-3}, 3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}, 10^{-1}, 3{\times}10^{-1}, 1, 3, 10, 30, 10^{2}, 3{\times}10^{2}, 10^{3}, 3{\times}10^{3}, 10^{4}\}$, the same grid for the RND and the three oracle columns | 7084–7087 |
| **Bonus weight $\beta$ (in run 5 itself)** | swept over 8 values $\{10^{-2},10^{-1},0.5,1,10,10^{2},10^{3},10^{4}\}$ with a beta race; winner $\beta = 10^{3}$ | 6009–6019; 6178–6183 |
| **Ensemble size** | $N = 1$ (single predictor, single target; the ensemble form exists in the library but is not used) | 325–333; 7687 ("a single predictor and a single environment copy") |
| **Base RL algorithm** | SB3 SAC `MlpPolicy`, $2\times256$ ReLU, Adam $3{\times}10^{-4}$, batch $256$, tau $0.005$, buffer $10^{6}$, auto entropy temperature with target $-\dim(A)$, train_freq 1 / gradient_steps 1 / learning_starts 100, discount $0.99$ in §8 (the run-5 PointMaze runs used $0.999$) | 7047–7064; 6131–6133; 7057 |
| **Steps** | $10^{6}$ env steps per run in Table 66 | 7054 |

**One-sentence definition to reuse:** the project's RND is next-state-input RND with a frozen orthogonal-$\sqrt2$/zero-bias target `4→256→LeakyReLU(0.2)→128`, a predictor one 128-wide LeakyReLU block deeper, observation whitening by running RMS clipped to $\pm5$ after a 6400-step random-agent warm-up on a throwaway env, a mean-over-128-dims squared-error readout, reward normalization by the running std of a non-episodic $\gamma=0.99$ forward filter, an Adam($10^{-4}$, $\epsilon=10^{-8}$) predictor trained on the full replay batch every SAC step, and the bonus folded into SAC's single reward stream as $r^{\mathrm{ext}} + \beta r^{\mathrm{int}}$.

**Contrast table already in the document:** `tab:pm-am-trainrun12-rank1-vs-rnd` (lines 7630–7690) lists knob by knob how the run-3.2.4 rank-1 configuration differs from this RND arm, and its caption ends with the explicit "confirmed identical and therefore not listed" list. That is the model to copy whenever §9 must compare two RND configurations.

---

## 5. Document conventions to follow

### Sectioning and numbering

* Depth used: `\section` → `\subsection` → `\subsubsection` → `\paragraph{...}`. `\paragraph` is the workhorse: it carries the run description, the purpose, each design component, results, and findings. Never a 4th numbered level.
* Every heading is followed immediately by `\label{...}` on its own line. Label vocabulary: `sec:` for headings, `tab:` for tables, `fig:` for figures, `eq:` for equations. §8's labels use the `pm-am-` family prefix (`sec:pm-am`, `sec:pm-am-trainrun1-group`, `sec:pm-am-envspec`, `tab:pm-am-envspec`, `tab:pm-am-trainrun1-algos`, `fig:pm-am-trainrun12-curves`). Pick one prefix for §9 and use it on every label in the section.
* Section titles are sentence case; §8's title is all-lowercase `point maze + ant maze` (an inconsistency in the document, not a rule).
* §8 uses `\setcounter{subsubsection}{-1}` immediately after `\subsection{Train run 1}` (line 6746) so the shared environment specification numbers as **8.1.0** and the two runs as **8.1.1** and **8.1.2**. Copy this if §9 has a shared-spec subsubsection that precedes the numbered runs.
* `\clearpage` is used before a new `\section` (line 6740) and after long catalog tables.
* Wide banner comments (`% ====…` blocks of ~90–100 `=`) mark major spliced blocks and record when/why they were added and which generator owns them — see lines 5825–5829, 6044–6048, 7111–7116.

### Tables

* Boilerplate, in this exact order:
  ```latex
  \begin{table}[H]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3pt}          % 3pt for wide tables, 4–6pt for narrow ones
  \renewcommand{\arraystretch}{1.2}    % 1.0–1.15 on the dense auto-generated tables
  \resizebox{\textwidth}{!}{%          % only when the tabular is wider than \textwidth
  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{Xcm} … @{}}
  ```
  and the matching close is `\end{tabular}%` then `}` on its own line, then `\caption{...}`, then `\label{...}`, then `\end{table}`.
* **`[H]` placement everywhere** (from the `float` package) — no `[t]`/`[htbp]` on tables anywhere in the document.
* **Caption below the tabular, label immediately after the caption.** Never above.
* Column specs are always fixed-width `p{}` with `>{\raggedright\arraybackslash}`, and `@{}` at both ends. Never bare `l`/`c` in a hand-written wide table (the auto-generated results tables do use `l l r r r r r` because their cells are short).
* **Category header rows** are used everywhere: `\multicolumn{n}{@{}l}{\textit{category name}}` on a row of its own, preceded by `\midrule`. Table 66's categories: *RL algorithm & environments*, *SAC*, *Intrinsic bonus*, *Bonus weight*, *Run bookkeeping (sweep config counts)*. In the results tables the block header is a bold `\texttt{}` environment name instead: `\multicolumn{7}{@{}l}{\textbf{\texttt{PointMaze\_\allowbreak UMaze-v3\_\allowbreak start\_\allowbreak bottom\_\allowbreak left}}}`.
* **Column headers carry provenance on a second line**: the header row is `\textbf{Name} & …\\` and the next row is `& \small(provenance) & \small(provenance) …\\` before the first `\midrule`. Table 66: `\small(SB3) & \small(run 5 stack) & \small(PointMaze) & \small(AntMaze) & \small(AntMaze)`. Table 54 uses the same plus a spanning group header with `\cmidrule(lr){5-6}`.
* **Value collapsing**: `\multicolumn{4}{c}{same}` for a value shared by the columns to the right; a wide free-text cell uses `\multicolumn{4}{>{\raggedright\arraybackslash}p{12.0cm}}{…}`.
* **"N/A"** — spelled exactly `N/A` in Tables 66, 71, and the auto-generated status tables, for a knob that does not apply to that column. The caption states it: *"``N/A'' marks a knob that does not apply to a column."* **Do not use `---` or `$-$` for this**; the older Table 54 (line 6082 etc.) uses `---` because it predates the `hyperparameter-table` shared skill, and the skill file explicitly warns not to copy that.
* **Sweep tables get config-count rows** (skill rule 8), as the last category block: one `configs in this column` row per column written as an annotated product (`$8$(env configs) $\times$ $15$(bonus weight) $= 120$`), then a spanning `total configs in this table` row with the total in `\mathbf{}`.
* **Results tables follow the analysis convention** instead: rows ranked by the primary metric, a single `$\uparrow$` on the ranking column only, best `{\boldmath$…$}` and second `\underline{$…$}` per metric column within each block, bookkeeping columns (`$N$`, verdict) never marked. The caption spells out the marking rule and the direction of "better".
* **Auto-generated tables** are spliced between marker comments placed inside `\begin{table}` but outside `\caption`/`\label`:
  ```latex
  % Auto-generated by code/2026-08-04-20-00_pm-am-run12-interim-tables/make_all.py
  % >>> AUTO-GENERATED TABLE START: pm-am-trainrun12-status
  \begin{tabular}{…}…\end{tabular}
  % <<< AUTO-GENERATED TABLE END: pm-am-trainrun12-status
  ```
  The generator (`inject_table` in `make_all.py`) hard-fails unless each marker occurs exactly once. Captions, sizing and prose stay hand-edited. Regeneration command, from the generator folder: `/p/rlprojects/RND/.venvs/exploration/bin/python make_all.py` then `latexmk -cd -pdf ../../RND_development_document.tex`.
* Captions are long and self-contained: they restate what the columns are, what "same"/"N/A"/the marks mean, where the numbers come from, and any caveat (partial snapshot, differing step budgets).
* Cross-references are written `Table~\ref{...}` with a non-breaking tilde. A table that copies or freezes another says so in its caption and `\ref`s the source.

### Figures

* `\begin{figure}[H]` for everything in §8 and the results figures; Figure 16 is the rare `[t]`.
* Order inside the float: `\centering`, `\includegraphics[width=…]{...}` (widths seen: `\linewidth`, `0.98\linewidth`, `0.92\linewidth`, `0.8\linewidth`), then `\caption{...}`, then `\label{...}`.
* Captions name the generator script when there is one, e.g. *"Generated by `\texttt{analysis/code/make\_\allowbreak reward\_\allowbreak curves.py}` in the run folder; regenerated in place as the sweep advances."* (line 7829–7830).

### Side-by-side panels — verbatim example

Two-panel and four-panel figures use `subcaption`'s `subfigure` at `0.49\linewidth` joined by `\hfill`, with rows separated by `\\[3pt]`. Canonical example, lines 5036–5065:

```latex
% ---- Initial RND bonus field, part 1: the zero-velocity study (shared norm + self-norm log) ----
\begin{figure}[H]
\centering
\begin{subfigure}[t]{0.49\linewidth}
  \includegraphics[width=\linewidth]{code/2026-07-09-23-43_rnd-init-bonus-field-surface-heatmap/surface_v0.pdf}
  \caption{Surface, shared normalization.}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
  \includegraphics[width=\linewidth]{code/2026-07-09-23-43_rnd-init-bonus-field-surface-heatmap/heatmap_v0.pdf}
  \caption{Heatmap, shared normalization.}
\end{subfigure}\\[3pt]
\begin{subfigure}[t]{0.49\linewidth}
  \includegraphics[width=\linewidth]{code/2026-07-09-23-43_rnd-init-bonus-field-surface-heatmap/surface_v0_selfnorm_log.pdf}
  \caption{Surface, self-normalized, log \(z\) axis.}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
  \includegraphics[width=\linewidth]{code/2026-07-09-23-43_rnd-init-bonus-field-surface-heatmap/heatmap_v0_selfnorm_log.pdf}
  \caption{Heatmap, self-normalized, log color scale.}
\end{subfigure}
\caption{The \emph{initial} (untrained) RND bonus over the maze at zero
velocity (\(v_x{=}v_y{=}0\)); the network is the step-0 weights of the
\runref{run-3.2.2 benchmark}'s best seed (\texttt{a\_seed} \(111\)).  Top row: values
normalized by the maximum shared with
Figure~\ref{fig:trainrun3-2-2-initfield-vel} (so the three velocity levels are
comparable).  Bottom row: the same field normalized by its \emph{own} maximum
on a log scale, exposing the relief the shared linear scale compresses --- the
bowl spans two orders of magnitude (\(0.008\)--\(1\)).  Orientation matches
Figure~1 (\(x\) right, \(y\) up, start S lower-left, goal G upper-right); maze
walls gray.}
\label{fig:trainrun3-2-2-initfield}
\end{figure}
```

A mixed-size variant (one full-width panel over one 0.66-width panel), lines 5339–5359:

```latex
\begin{figure}[H]
\centering
\begin{subfigure}[t]{\linewidth}
  \includegraphics[width=\linewidth]{code/2026-07-10-21-05_rnd-init-bonus-bias-ablation/heatmaps_by_bias_scheme.pdf}
  \caption{Self-normalized heatmaps at \(v{=}0\) (seed \(111\)), one panel per
  bias scheme; each panel's raw range and max/min ratio are printed below it.}
\end{subfigure}\\[4pt]
\begin{subfigure}[t]{0.66\linewidth}
  \includegraphics[width=\linewidth]{code/2026-07-10-21-05_rnd-init-bonus-bias-ablation/bonus_vs_radius.pdf}
  \caption{Median raw bonus vs.\ distance \(r\) from the whitening center
  (log scale): every nonzero bias raises the floor; the outward growth
  persists.}
\end{subfigure}
\caption{The controlled bias-initialization ablation of
Table~\ref{tab:trainrun3-2-2-bias-ablation}.  In (a) the deep central pit of
the zero-bias field fills in as the bias scale grows, until at
\(\sigma \ge 1\) the field is nearly constant; in (b) the same data as bonus
vs.\ radius shows that the biases add a floor while the slope of the
\(r^2\) growth is unchanged.}
\label{fig:trainrun3-2-2-bias-ablation}
\end{figure}
```

Note: multi-panel *result* figures in §8 are instead one PDF with a matplotlib-drawn panel grid (`fig:pm-am-trainrun12-curves`, line 7814–7832), whose caption spells out the grid ("one column per environment (the tables' order), the SAME data in both rows"). Use `subfigure` only when the panels are genuinely separate files.

### Prose conventions inside a run subsubsection

The §8.1.1 / §8.1.2 template, in order:

1. One paragraph of motivation, then the hypothesis or task list as a real `\begin{enumerate}[label=\arabic*.,leftmargin=*]` with a bold lead-in per item (`\item \textbf{The ground-truth bonus performs better than RND on every PointMaze configuration.} …`). Roman-labelled lists use `[label=(\roman*),leftmargin=*]`. Never inline "(i) … (ii) …".
2. The environment table (a frozen copy if it is a subset of the shared spec).
3. One sentence introducing the algorithm table, then the algorithm table.
4. Definitions of any new quantity, each with "Define $X$ as …" **before** the display equation (see lines 7454–7476, 7478–7489, 7491–7496).
5. The pruning / seed rules as a numbered list.
6. `\paragraph{Provenance.}` naming the run folder, set in a `\begin{quote}\ttfamily\footnotesize\raggedright …\end{quote}` inside `\begin{samepage}`, with `\allowbreak` after every escaped underscore (lines 7692–7714).
7. `\paragraph{Results (…snapshot date…).}` with the truncation/denominator rule stated explicitly ("Only records a run wrote at its natural end (`completed=true`) enter any aggregate"), then `\paragraph{How to read the metrics tables.}` with the metric formulas as a numbered list, then the auto-generated tables, then `\paragraph{Findings.}` as a numbered list, then `\paragraph{Training cost.}`.

---

## Files referenced

* `/p/rlprojects/RND/07_reconstruction/development_document/RND_development_document.tex`
* `/p/rlprojects/RND/07_reconstruction/development_document/RND_development_document.aux` (float numbers), `.toc` (numbering), `.log`
* `/p/rlprojects/RND/07_reconstruction/development_document/neurips_2026.sty` (`textwidth=5.5in`, line 126)
* `/p/rlprojects/RND/07_reconstruction/development_document/code/2026-07-20_run5-writeup/rnd_pipeline.tex` — Figure 16 generator (TikZ standalone)
* `/p/rlprojects/RND/07_reconstruction/development_document/code/2026-07-20_run5-writeup/rnd_pipeline.pdf`, `rnd_pipeline_preview.png`, `_build.log`, `section_draft.tex`
* `/p/rlprojects/RND/07_reconstruction/development_document/code/2026-08-04-20-00_pm-am-run12-interim-tables/make_all.py` and `README.md` — the marker-splice table generator (§8.1.2)
* `/p/rlprojects/RND/07_reconstruction/development_document/code/2026-07-31-16-22_pm-am-run1-interim-tables/make_all.py` — same for §8.1.1
* `/p/rlprojects/RND/07_reconstruction/development_document/code/2026-07-19-03-21_convergence-run1-point-sets/make_point_set_figure.py` — the matplotlib figure-generator pattern
* `/p/rlprojects/.claude/skills/hyperparameter-table/SKILL.md` — the table shape Table 66 obeys
* `/p/rlprojects/.claude/skills/draw-or-diagrams/SKILL.md` — the diagram procedure Figure 16 does *not* obey