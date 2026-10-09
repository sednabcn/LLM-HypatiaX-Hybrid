# Project design -- "Hybrid Methods" paper (HypatiaX case)

## 0. Status / open issues (resolve before running anything)
| # | Issue | Source | Action |
|---|---|---|---|
| I1 | The PDFs describe 5 variants; the package has 7 (+ a "6-pca" row). Confirmed 7 by you. | PDFs vs `experiment_protocol_hybrid.py` | Paper uses V1-V7; V6 = validation-selected residual hybrid, V7 = LLM-prior PySR seeding. |
| I2 | Variant IDs differ between sources: the coverage matrix lists V1-5, V6, V6-pca, V7 (8 rows), while `run_hybrid_all.sh` talks of "methods 1-8" and "method 9". | package | Freeze one ID table in `rsc/db` (`variant`) before ingesting results. |
| I3 | Package says nguyen12 runs only V7; the earlier PDF says Variant 4 also appears in exp3. | package vs PDF | Check `--list-coverage` on the real repo. |
| I4 | Coverage output here used a *fallback catalogue*; case counts are not real. | local run | Re-run with the real `experiment_protocol_*.py`. |
| I5 | Open reliability discrepancy: Portfolio Variance R2=1.000 in-sample vs far-R2=-882.9 (seed 42). | PDF 1, E7 | Becomes experiment A3 + Fig. 6; must not be averaged away. |
| I6 | Pre-v5.0 `use_llm` was a silent no-op (V4/V5 numbers from older builds are really pure PySR). | PDF 1 | Every result row stores `engine_version`; reject rows < v5.0. |
| I7 | No benchmark results are in the uploads, so goals 1-2 are unanswered until experiments run. | -- | Paper has result placeholders only. |

### Status update (iteration 2) -- see `audit/issues/issues.yml` (22 issues) for the live register
- I2 resolved: frozen table in `config/variants.yml` (registry 1=LLM, 2=NN, 3=V1, 4=V2, 5=V4, 6=V5, 7=V3, 8=V6, 9=V7). I3 resolved: only V7 runs on nguyen12 (`--list-coverage`).
- Corrections to this design: `hybrid_exp1b/_pca` are the Portfolio-Variance seed sweeps (= A3 data), `hybrid_exp3b` a multi-seed Nguyen-12 run; default seeds are 42 only (A-017/A-018). Pure-PySR baseline B3 is not in the registry (A-009). E8 (all-30) has no step (A-021).
- Built: ingest, stats, tables T1-T14 (T9/T10 pending), figures F1-F6, F9, F10 (F7/F8 pending), LaTeX paper, audit checks, dry-run flow. Not possible here: real experiments (no PySR/Julia/API key/real repo).

## 1. Paper prototype
**Working title:** Which Hybrid? A Seven-Variant Comparison of LLM-Guided Symbolic Regression for Extrapolation-Reliable Discovery
**Sections:** 1 Introduction - 2 Related work (SR, LLM-for-science, hybrid/ensemble, extrapolation) - 3 The seven variants (taxonomy, Table 1, Fig. 2) - 4 Experimental protocol (4 protocols, splits, baselines, metrics, win-rate definition) - 5 Results: 5.1 win rate (G1), 5.2 extrapolation by domain (G2), 5.3 ablations/noise/sample size - 6 Advantages and limitations (G3) incl. audit findings - 7 Threats to validity (leakage, dead-path, non-determinism) - 8 Conclusion - App. A reproducibility, B per-case tables.
**Style:** LaTeX, JMLR class (config/publishing.yml). Goals G1/G2/G3 map to §5.1/§5.2/§6.

## 2. The compared methods
| ID | Name | Entry symbol | Rule |
|---|---|---|---|
| V1 | Enhanced Hybrid DeFi | `EnhancedHybridSystemDeFi` | scipy refit + R2>=0.85 & margin>0.05 gate |
| V2 | Hybrid All-Domains | `HybridSystemAllDomains` | tie-favours-symbolic, `force_llm` |
| V3 | Ensemble | `ensemble_llm_nn` | weight ∝ R2/σ_resid |
| V4 | LLM-Prior/PySR family | `SymbolicEngineWithLLM` | modes none/seed/hybrid/fallback (0.5/0.90/0.95) |
| V5 | Retry loop + optional physics fallback | `HybridDiscoverySystem._discover_with_retry` | up to 5 PySR attempts (seeds 42+k), early stop R2>=0.95; physics fallback OFF by default (corrects the PDF's 'template fallback'; see A-006, A-022) |
| V6 (+pca) | Validation-selected residual hybrid | `_select_v4_candidate` | 5-candidate pool, edge validation slice, extrapolation guard |
| V7 | LLM-prior PySR seeding | `get_llm_prior` + `PySRRegressor(guesses=)` | seeded vs unseeded run |
| B1-B3 | Baselines | NN, pure LLM, pure PySR | **required for G1; not in the package matrix** |

## 3. Metrics and win-rate definition (G1, G2)
- Primary: **far-extrapolation R2** (`r2_far`, tier from the 5-tier ladder); secondary: in-distribution test R2, near-perfect rate (R2>0.99), catastrophic rate (R2<-1), RMSE/MAE.
- **Win** (per case, per seed-median): variant has the highest `r2_far` among compared methods; ties if |Δ|<0.01 count 0.5. Win rate = wins/n_cases with 10k-bootstrap CI.
- Also report pairwise win rates and Friedman + Nemenyi / Wilcoxon-Holm (Demšar 2006).
- Comparisons only among methods that *ran on the same cases* (coverage matrix); never rank a method across protocols it did not run.
- Domain breakdown (G2): DeFi, physics, chemistry, biology, Nguyen-12, all-30 multi-domain.

## 4. Experiments to develop
| ID | Step | Protocol / split | Methods | Paper § | Goal |
|---|---|---|---|---|---|
| E1 | hybrid_exp1 | DeFi, random | V1-V6 + B | 5.1 | G1 |
| E2 | hybrid_exp1_pca | DeFi, PCA-directed OOD | V1-V6pca + B | 5.2 | G2 |
| E3 | hybrid_exp1b | DeFi (variant b) | V1-V6 + B | 5.1 | G1 |
| E4 | hybrid_exp1b_pca | DeFi PCA (b) | V1-V6pca + B | 5.2 | G2 |
| E5 | hybrid_exp2 | Feynman | V1-V5 + B | 5.1/5.2 | G1,G2 |
| E6 | hybrid_exp3 | Nguyen-12 | V7 + B | 5.1 | G1 |
| E7 | hybrid_exp3b | Nguyen-12 (b) | V7 + B | 5.3 | G1 |
| E8 | cross-variant (all30) | all-30 | all that apply + B | 5.2 | G2 |
| A1 | ablation | seeded vs unseeded, V4 modes | V4, V7 | 5.3 | G3 |
| A2 | noise sweep / sample complexity | DeFi | V1,V2,V3,B | 5.3 | G3 |
| A3 | reliability audit | Portfolio Variance, multi-seed | all | 6, 7 | G3 |
| A4 | leakage & dead-path audit | prompt scan, test-residual check, engine version | all | 7 | G3 |
Seeds 42-46, API-cost smoke mode first (config/experiments.yml).

## 5. List of Tables (contents -> `paper/tables/`)
| T | Caption | Contents / source |
|---|---|---|
| T1 | The seven hybrid variants and baselines | §2 above |
| T2 | Variant x protocol coverage | `--list-coverage` |
| T3 | Experiment inventory | §4 |
| T4 | Win rate per variant and experiment (G1) | `winrate` table, CI |
| T5 | Pairwise win-rate matrix | `result` |
| T6 | Far-extrapolation R2 by domain and variant (G2) | median [IQR] |
| T7 | Near-perfect / catastrophic rates | `result` |
| T8 | Gate decisions: fitted_llm / ensemble / nn shares | `run.decision` |
| T9 | Seeded vs unseeded (V7, V4 modes) | A1 |
| T10 | Noise and sample-size sensitivity | A2 |
| T11 | Advantages and limitations per variant (G3) | PDF 1 §5 + results |
| T12 | Audit: issues, patches, fix-rate | `issue_patch` |
| T13 | Compute/API cost per method | `run.wall_s`, `api_calls` |
| T14 | Hyperparameters and thresholds | config/*.yml |

## 6. List of Figures (-> `paper/figures/`)
| F | Caption | Contents |
|---|---|---|
| F1 | Study pipeline | package -> experiments -> DB -> paper |
| F2 | Decision flow of V1-V7 | 7 flowcharts (from variants report; extend with V6, V7) |
| F3 | Win rate by variant with CI | bars, per protocol |
| F4 | Extrapolation heatmap | domain x variant, median r2_far |
| F5 | Extrapolation ladder | tier attainment per variant |
| F6 | Train vs far R2 | scatter; Portfolio Variance outlier annotated |
| F7 | Noise sweep | R2 vs noise level |
| F8 | Sample complexity | R2 vs n_train |
| F9 | Gate-decision distribution | stacked bars (T8) |
| F10 | Random vs PCA split | paired difference per variant |

## 7. Design mapping: experiment -> section -> figure/table -> reference
| Exp | Section | Table | Figure | Key references (bibkeys) |
|---|---|---|---|---|
| E1,E3 | 5.1 | T4,T5,T7,T8 | F3,F9 | demsar2006statistical, holm1979simple, efron1993bootstrap |
| E2,E4 | 5.2 | T6 | F4,F5,F10 | pearson1901pca, xu2021extrapolate, quinonero2009shift |
| E5 | 5.1,5.2 | T4,T6 | F3,F4 | udrescu2020aifeynman, lacava2021srbench |
| E6,E7 | 5.1,5.3 | T4,T9 | F3 | uy2011nguyen, cranmer2023pysr |
| E8 | 5.2 | T6 | F4 | matsubara2022rethinking |
| A1 | 5.3 | T9 | -- | shojaee2025llmsr, romeraparedes2024funsearch |
| A2 | 5.3 | T10 | F7,F8 | kaplan2020scaling |
| A3 | 6,7 | T12 | F6 | guo2017calibration, balestriero2021extrapolation |
| A4 | 7 | T12 | -- | zhang2017rethinking |
| (method §3) | 3 | T1,T11,T14 | F1,F2 | markowitz1952portfolio, michaelis1913kinetik, lakshminarayanan2017deepensembles, wolpert1992stacking |
Machine-readable copy: build from `rsc/db` (`experiment`, `result`, `reference`) in step 6 of the guide.

## 8. References policy
`paper/references.bib` has 58 *unverified* seed entries (target >100). Metadata was written from memory without volume/pages/DOI. `python scripts/validate_refs.py paper/references.bib --fix --strict` checks each against Crossref/arXiv (title >=0.90, year, first author) and records the outcome in `audit/references_report.csv` and the `reference` table. Only verified entries may be cited; fill the remaining ~45 by searching, not by recall.
