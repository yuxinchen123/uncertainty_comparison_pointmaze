# Sources

Every file was downloaded on 2026-11-09 (Pacific time) with `curl` and a generic browser user agent. Each one was
checked with `file` (reported as a PDF document), is larger than 50 KB, and its first-page title was read from text
extracted with Ghostscript and matches the title below. Nothing from the downloads was run.

Each paper has one row in each of the two tables, joined by the number in the first column: the first table gives
the reference, the second the file record. Every file sits at `pdfs/<folder>/<file name>` under this `literature/`
folder. "Version" for an arXiv file is the version stamp printed in the PDF margin.

## References

| # | authors | title | venue, year | version |
|---|---|---|---|---|
| 1 | Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov | Exploration by Random Network Distillation | ICLR 2019 | arXiv v1, 30 Oct 2018 |
| 2 | Sam Lobel, Akhil Bagaria, George Konidaris | Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning | ICML 2023 | arXiv v1, 5 Jun 2023 |
| 3 | Kai Yang, Jian Tao, Jiafei Lyu, Xiu Li | Exploration and Anti-Exploration with Distributional Random Network Distillation | ICML 2024 | arXiv v4, 20 May 2024 |
| 4 | Zhirui Fang, Kai Yang, Jian Tao, Jiafei Lyu, Lusong Li, Li Shen, Xiu Li | Exploration by Random Distribution Distillation | arXiv preprint, 2025 (no venue printed in the file) | arXiv v1, 16 May 2025 |
| 5 | Matej Pecháč, Michal Chovanec, Igor Farkaš | Self-supervised network distillation: an effective approach to exploration in sparse reward environments | arXiv preprint, 2023; v4 says "Preprint submitted to Neurocomputing" | arXiv v4, 10 Jun 2024 |
| 6 | Adrià Puigdomènech Badia, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Bilal Piot, Steven Kapturowski, Olivier Tieleman, Martín Arjovsky, Alexander Pritzel, Andrew Bolt, Charles Blundell | Never Give Up: Learning Directed Exploration Strategies | ICLR 2020 | arXiv v1, 14 Feb 2020 |
| 7 | Tianjun Zhang, Huazhe Xu, Xiaolong Wang, Yi Wu, Kurt Keutzer, Joseph E. Gonzalez, Yuandong Tian | NovelD: A Simple yet Effective Exploration Criterion | NeurIPS 2021 | proceedings version |
| 8 | Mikael Henaff, Roberta Raileanu, Minqi Jiang, Tim Rocktäschel | Exploration via Elliptical Episodic Bonuses | NeurIPS 2022 | arXiv v2, 4 Jan 2023 |
| 9 | Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Rémi Munos | Unifying Count-Based Exploration and Intrinsic Motivation | NeurIPS 2016 | arXiv v2, 7 Nov 2016 |
| 10 | Georg Ostrovski, Marc G. Bellemare, Aäron van den Oord, Rémi Munos | Count-Based Exploration with Neural Density Models | ICML 2017 | arXiv v2, 14 Jun 2017 |
| 11 | Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel | #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning | NeurIPS 2017 | arXiv v3, 5 Dec 2017 |
| 12 | Ian Osband, John Aslanides, Albin Cassirer | Randomized Prior Functions for Deep Reinforcement Learning | NeurIPS 2018 | arXiv v2, 15 Nov 2018 |
| 13 | Jordan T. Ash, Cyril Zhang, Surbhi Goel, Akshay Krishnamurthy, Sham Kakade | Anti-Concentrated Confidence Bonuses for Scalable Exploration | ICLR 2022 | arXiv v2, 11 Apr 2022 |
| 14 | Alexander L. Strehl, Michael L. Littman | An analysis of model-based Interval Estimation for Markov Decision Processes | Journal of Computer and System Sciences 74 (2008) 1309–1331 | publisher's typeset journal version, posted on a UC Irvine course page (not a publisher or arXiv copy) |
| 15 | Alexander L. Strehl, Michael L. Littman | A Theoretical Analysis of Model-Based Interval Estimation | ICML 2005 | proceedings version; contains MBIE but not the MBIE-EB bonus (checked by text search) |
| 16 | Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred A. Hamprecht, Yoshua Bengio, Aaron Courville | On the Spectral Bias of Neural Networks | ICML 2019 | arXiv v3, 31 May 2019 |
| 17 | Arthur Jacot, Franck Gabriel, Clément Hongler | Neural Tangent Kernel: Convergence and Generalization in Neural Networks | NeurIPS 2018 | arXiv v4, 10 Feb 2020 |
| 18 | John Duchi, Elad Hazan, Yoram Singer | Adaptive Subgradient Methods for Online Learning and Stochastic Optimization | JMLR 12 (2011) 2121–2159 | JMLR published version |
| 19 | Lior Fox, Leshem Choshen, Yonatan Loewenstein | DORA The Explorer: Directed Outreaching Reinforcement Action-Selection | ICLR 2018 | arXiv v1, 11 Apr 2018 |
| 20 | Zhaohan Daniel Guo, Shantanu Thakoor, Miruna Pîslar, Bernardo Avila Pires, Florent Altché, Corentin Tallec, Alaa Saade, Daniele Calandriello, Jean-Bastien Grill, Yunhao Tang, Michal Valko, Rémi Munos, Mohammad Gheshlaghi Azar, Bilal Piot | BYOL-Explore: Exploration by Bootstrapped Prediction | NeurIPS 2022 | arXiv v1, 16 Jun 2022 |
| 21 | Moritz A. Zanger, Yijun Wu, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T. J. Spaan | On the Equivalence of Random Network Distillation, Deep Ensembles, and Bayesian Inference | arXiv preprint, 2026 | arXiv v2, 26 Feb 2026 |
| 22 | Marlos C. Machado, Sriram Srinivasan, Michael Bowling | Domain-Independent Optimistic Initialization for Reinforcement Learning | arXiv preprint 1410.4604, 2014 (no venue printed in the file) | no version stamp printed |
| 23 | Hung Guei, Lung-Pin Chen, I-Chen Wu | Optimistic Temporal Difference Learning for 2048 | IEEE Transactions on Games (accepted 2021); arXiv 2111.11090 | accepted manuscript TCIAIG-2020-0093.R2, no arXiv stamp printed |
| 24 | Mohammadamin Davoodabadi, Negin Hashemi Dijujin, Mahdieh Soleymani Baghshah | PreND: Enhancing Intrinsic Motivation in Reinforcement Learning through Pre-trained Network Distillation | arXiv preprint, 2024 | arXiv v1, 2 Oct 2024 |
| 25 | Zhibo Hou, Zhiyu An, Wan Du | Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring | ICLR 2026 | arXiv v2, 3 Apr 2026 |
| 26 | Tabish Rashid, Bei Peng, Wendelin Böhmer, Shimon Whiteson | Optimistic Exploration even with a Pessimistic Initialisation | ICLR 2020 | arXiv v1, 26 Feb 2020 |
| 27 | Eyal Even-Dar, Yishay Mansour | Learning Rates for Q-learning | JMLR 5 (2003) 1–25 | JMLR published version |

Rows 1–18 are the papers named in the task; rows 19 onward were found by the extra search for learned counters,
normalised targets, optimistic starts, learning-rate control of the decay and comparisons with an earlier error.

## Files

| # | downloaded from | date (PT) | folder and file name |
|---|---|---|---|
| 1 | https://arxiv.org/pdf/1810.12894 | 2026-11-09 | `burda-2018-rnd/`<br>`burda-2018-exploration-by-random-network-distillation.pdf` |
| 2 | https://arxiv.org/pdf/2306.03186 | 2026-11-09 | `lobel-2023-coin-flip-network/`<br>`lobel-2023-flipping-coins-pseudocounts.pdf` |
| 3 | https://arxiv.org/pdf/2401.09750 | 2026-11-09 | `yang-2024-drnd/`<br>`yang-2024-distributional-random-network-distillation.pdf` |
| 4 | https://arxiv.org/pdf/2505.11044 | 2026-11-09 | `fang-2025-rdd/`<br>`fang-2025-exploration-by-random-distribution-distillation.pdf` |
| 5 | https://arxiv.org/pdf/2302.11563 | 2026-11-09 | `pechac-2023-snd/`<br>`pechac-2023-self-supervised-network-distillation.pdf` |
| 6 | https://arxiv.org/pdf/2002.06038 | 2026-11-09 | `badia-2020-never-give-up/`<br>`badia-2020-never-give-up.pdf` |
| 7 | https://papers.neurips.cc/paper_files/paper/2021/file/<br>d428d070622e0f4363fceae11f4a3576-Paper.pdf | 2026-11-09 | `zhang-2021-noveld/`<br>`zhang-2021-noveld.pdf` |
| 8 | https://arxiv.org/pdf/2210.05805 | 2026-11-09 | `henaff-2022-e3b/`<br>`henaff-2022-exploration-via-elliptical-episodic-bonuses.pdf` |
| 9 | https://arxiv.org/pdf/1606.01868 | 2026-11-09 | `bellemare-2016-unifying-count-based/`<br>`bellemare-2016-unifying-count-based-exploration.pdf` |
| 10 | https://arxiv.org/pdf/1703.01310 | 2026-11-09 | `ostrovski-2017-neural-density-counts/`<br>`ostrovski-2017-count-based-exploration-neural-density-models.pdf` |
| 11 | https://arxiv.org/pdf/1611.04717 | 2026-11-09 | `tang-2017-hash-exploration/`<br>`tang-2017-hash-exploration.pdf` |
| 12 | https://arxiv.org/pdf/1806.03335 | 2026-11-09 | `osband-2018-randomized-prior-functions/`<br>`osband-2018-randomized-prior-functions.pdf` |
| 13 | https://arxiv.org/pdf/2110.11202 | 2026-11-09 | `ash-2022-anti-concentrated-bonuses/`<br>`ash-2022-anti-concentrated-confidence-bonuses.pdf` |
| 14 | https://www.ics.uci.edu/~dechter/courses/ics-295/<br>winter-2018/papers/2008-littman-aij-main.pdf | 2026-11-09 | `strehl-2008-mbie-eb/`<br>`strehl-2008-analysis-model-based-interval-estimation.pdf` |
| 15 | https://icml.cc/Conferences/2005/proceedings/papers/<br>108_TheoreticalAnalysis_StrehlLittman.pdf | 2026-11-09 | `strehl-2005-mbie-icml/`<br>`strehl-2005-theoretical-analysis-model-based-interval-estimation.pdf` |
| 16 | https://arxiv.org/pdf/1806.08734 | 2026-11-09 | `rahaman-2019-spectral-bias/`<br>`rahaman-2019-spectral-bias.pdf` |
| 17 | https://arxiv.org/pdf/1806.07572 | 2026-11-09 | `jacot-2018-neural-tangent-kernel/`<br>`jacot-2018-neural-tangent-kernel.pdf` |
| 18 | https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf | 2026-11-09 | `duchi-2011-adagrad/`<br>`duchi-2011-adaptive-subgradient-methods.pdf` |
| 19 | https://arxiv.org/pdf/1804.04012 | 2026-11-09 | `fox-2018-dora-e-values/`<br>`fox-2018-dora-the-explorer.pdf` |
| 20 | https://arxiv.org/pdf/2206.08332 | 2026-11-09 | `guo-2022-byol-explore/`<br>`guo-2022-byol-explore.pdf` |
| 21 | https://arxiv.org/pdf/2602.19964 | 2026-11-09 | `zanger-2026-rnd-ensembles-bayes/`<br>`zanger-2026-equivalence-rnd-deep-ensembles-bayesian-inference.pdf` |
| 22 | https://arxiv.org/pdf/1410.4604 | 2026-11-09 | `machado-2014-domain-independent-optimistic-initialization/`<br>`machado-2014-domain-independent-optimistic-initialization.pdf` |
| 23 | https://arxiv.org/pdf/2111.11090 | 2026-11-09 | `guei-2021-optimistic-td-2048/`<br>`guei-2021-optimistic-temporal-difference-learning-2048.pdf` |
| 24 | https://arxiv.org/pdf/2410.01745 | 2026-11-09 | `davoodabadi-2024-prend/`<br>`davoodabadi-2024-prend-pre-trained-network-distillation.pdf` |
| 25 | https://arxiv.org/pdf/2509.25438 | 2026-11-09 | `hou-2025-learning-progress-monitoring/`<br>`hou-2025-beyond-noisy-tvs-learning-progress-monitoring.pdf` |
| 26 | https://arxiv.org/pdf/2002.12174 | 2026-11-09 | `rashid-2020-opiq/`<br>`rashid-2020-optimistic-exploration-pessimistic-initialisation.pdf` |
| 27 | https://jmlr.org/papers/volume5/evendar03a/evendar03a.pdf | 2026-11-09 | `even-dar-2003-learning-rates-q-learning/`<br>`even-dar-2003-learning-rates-for-q-learning.pdf` |

## Not downloaded

**Ciosek, Fortuin, Tomioka, Hofmann, Turner, "Conservative Uncertainty Estimation By Fitting Prior Networks", ICLR
2020** (OpenReview id `BJlahxHYDS`; no arXiv version). Tried `https://openreview.net/pdf?id=BJlahxHYDS`, the
`attachment` form and the `api.openreview.net` / `api2.openreview.net` forms of the same id: every one returned an HTML
page (403, "Challenge verification required"). A Microsoft Research upload address found by search returned "Page not
found". No copy from an allowed host was obtained; `notes.md` summarises it from the August 2026 note.
