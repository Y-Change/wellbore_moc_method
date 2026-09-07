# Fracture-water-hammer prior-art audit (2026-09-05)

Scope: publicly available papers/proceedings relevant to high-frequency shut-in water-hammer diagnosis in multi-cluster fractured wells. Local extracts are used only for verbatim technical evidence; bibliographic identity and publication type were independently checked against Crossref (and OpenAlex where noted). A conference proceeding is labelled as such and is not silently promoted to a journal article.

## Verified bibliographic records

| Work | DOI | Crossref/OpenAlex year | Type / venue | Verification note |
|---|---|---:|---|---|
| Qiu, Hu, Zhou et al., “Water hammer response characteristics of wellbore-fracture system: Multi-dimensional analysis in time, frequency and quefrency domain” | [10.1016/j.petrol.2022.110425](https://doi.org/10.1016/j.petrol.2022.110425) | 2022 | journal article, *Journal of Petroleum Science and Engineering* 213, 110425 | Crossref exact DOI record |
| Luo, Hu, Zhou et al., “A New Water Hammer Decay Model: Analyzing the Interference of Multiple Fractures and Perforations on Decay Rate” | [10.2118/214658-PA](https://doi.org/10.2118/214658-PA) | 2023 | journal article, *SPE Journal* | Crossref exact DOI record |
| Liu, Liu, Wang, “A Novel MCMC-Based Hydraulic Fracture Diagnostics Approach Using Water Hammer Data” | [10.56952/ARMA-2023-0865](https://doi.org/10.56952/ARMA-2023-0865) | 2023 | proceedings article, 57th US Rock Mechanics/Geomechanics Symposium | Crossref type = proceedings-article |
| Sheludko, Crawford, Oparin, Aleid, “Machine-Learning Assisted Analysis of Frac Water Hammer” | [10.2118/217781-MS](https://doi.org/10.2118/217781-MS) | 2024 | proceedings article, SPE HFTC | Crossref type = proceedings-article |
| Zeng, Wei, Su et al., “Fracture size inversion method based on water hammer signal for shale reservoir” | [10.3389/fenrg.2023.1336148](https://doi.org/10.3389/fenrg.2023.1336148) | 2024 (published 5 Jan) | journal article, *Frontiers in Energy Research* | Crossref exact DOI record; local paper gives publication date |
| Gabry, Ramadan, Soliman, “Estimating Water Hammer Damping Ratios Using Continuous Wavelet Transform for Induced Hydraulic Fracture Complexity Characterization” | [10.2118/225459-PA](https://doi.org/10.2118/225459-PA) | 2025 | journal article, *SPE Journal* | Crossref exact DOI record |
| Iuzofatov, Seleznev, Borisenko, Fedorov, “Next-Gen Fracture Diagnostics: AI-Decoded Water Hammer Physics for Adaptive Design and Field-Wide Optimization” | [10.2118/230381-MS](https://doi.org/10.2118/230381-MS) | 2025 | proceedings article, SPE Caspian Technical Conference | Crossref type = proceedings-article; manuscript itself says contents not peer reviewed by SPE |
| Deng, Yi, Li et al., “A diagnostic model for hydraulic fracture in naturally fractured reservoir utilising water-hammer signal” | [10.1016/j.engfracmech.2025.111347](https://doi.org/10.1016/j.engfracmech.2025.111347) | 2025 | journal article, *Engineering Fracture Mechanics* 325, 111347 | Crossref exact DOI record |
| Sun, He, Li et al., “A Novel Comprehensive Water Hammer Pressure Model for Fracture Geometry Evaluation” | [10.2118/228403-PA](https://doi.org/10.2118/228403-PA) | 2025 | journal article, *SPE Journal* | Crossref exact DOI record |
| Li, Sun, Liu et al., “Quantitative Analysis of Hydraulic Fracture Geometry and Its Relationship with Key Water Hammer Pressure Features” | [10.3390/w17182741](https://doi.org/10.3390/w17182741) | 2025 | journal article, *Water* | Crossref exact DOI record |
| Yang, Li, Jiang et al., “Hydraulic Fracture Geometry Diagnosis Based on High Frequency Water Hammer Signal Analysis” | [10.2118/232521-MS](https://doi.org/10.2118/232521-MS) | 2026 | proceedings article, SPE Oman Petroleum & Energy Show | Crossref exact DOI record; abstract reports field/microseismic comparison |
| Zhang, Sheng, Jiang et al., “Real-Time Diagnostics of Hydraulic Fracture Propagation Using Continuous Wavelet Transform of Transient Pressure Fluctuation” | [10.2118/234685-PA](https://doi.org/10.2118/234685-PA) | 2026 | journal article, *SPE Journal* | Crossref exact DOI record (published 1 Jul 2026) |
| Zhu, Wang, “Automated Water Hammer Analysis for Fracture Parameter Inversion Using High-Frequency Shut-In Pressure Signals During Hydraulic Fracturing” | [10.3390/modelling7030087](https://doi.org/10.3390/modelling7030087) | 2026 | journal article, *Modelling* 7(3), 87 | Crossref + OpenAlex; OpenAlex marks OA published version |

## Technical evidence and limits

### Qiu et al. 2022 (JPSE; 110425)

* The abstract states that the wellbore-fracture system is built by “introducing the branch pipe to the wellbore as the fracture”; MOC is used and validated against FLUENT. Eight cases vary fracture existence, location and length; output is time, frequency and quefrency analysis (local extract, p.1).
* The junction is a three-pipe hydraulic junction: `H^t_{1,n}=H^t_{2,1}=H^t_{3,1}` and `Q^t_{1,n}-Q^t_{2,1}-Q^t_{3,1}=0` (local extract, p.4, around lines 550–554). This is a mass/pressure node, but only one branch-fracture idealisation is studied in each case, not a field-scale set of perforation-cluster internal boundaries.
* MOC uses `Δt=Δx/a` for stability (local extract, p.3, around line 330). Validation parameters include 10-m wellbore, fracture at 7 m, 5-m branch, `a=1490 m/s`, and `Δx=0.01 m`; pump shut-in is 0.005 s (Table 1, local extract p.4). These are laboratory/small-domain conditions.
* The paper itself says prior work had not fully understood propagation in a wellbore-fracture system and that prior analyses often considered only the wellbore; it does not establish a multi-cluster inverse solver. Therefore it supports a *single/branch junction wave-interference* prior-art claim, not “no mathematical boundary treatment exists.”

### Luo et al. 2023 (SPEJ; 10.2118/214658-PA)

* Governing low-Mach equations are the standard MOC pair (local extract p.2): `∂Q/∂t + gA∂H/∂x + fQ|Q|/(2DA)=0`, `∂H/∂t + a²/(gA)∂Q/∂x=0`. Characteristic update imposes `Δx=aΔt` (local extract p.3).
* Bottom boundary is a *lumped RCI* model. Explicit assumptions are: constant-height planar elliptical fractures; constant fracture length; compliance depends only on width change; no leak-off during water hammer; resistance mainly wellbore/near-wellbore (local extract p.3). Near-well pressure adds tortuosity `ΔP_f frac=K_f frac Q^0.5` and perforation `ΔP_f perf=K_f perf Q²`, with `K_f perf=0.807249ρ/(n_p²D_p⁴C_d²)` (local extract p.4).
* Laboratory system: 190-m cast-iron pipe, ID 16 mm, wall 1 mm; pressure monitor 0–5 MPa, 0.01% full-scale accuracy, 10-kHz frequency response; three branch stubs at 15.9, 33.2, 53.6 m; pump 20 Hz (local extract p.5). Field example has 912/484-m horizontal lengths, 3/2-s pump stop, 1470 m/s wave speed, and 3/5 inferred fractures (Table 6, local extract p.10).
* Inversion is matching a scalar decay rate plus waveform features to infer *number* of fractures. It is not a separate unknown for every cluster's position, compliance, conductance and opening state. Hence “multiple-fracture interference and perforation effects” is established prior art, while a fully resolved multi-node operator remains open.

### Liu et al. 2023 ARMA proceedings (10.56952/ARMA-2023-0865)

* Abstract: new multi-cluster boundary includes perforation friction and pressure-dependent leak-off; MOC reproduces signals; MCMC gives posterior statistics for fracture width, length and height; synthetic and field cases are reported (local extract p.1).
* Boundary equations include perforation drop `P_w-P_f=8ρQ²/(π²d⁴N_p²C_D²)` and fracture mass balance `d(2N_fh_fl_fw)/dt=Q-Q_leak`; Carter/Paige forms are used for aperture and leak-off (local extract p.2). The reported synthetic model uses vertical 3180 m + horizontal 2120 m, 5 fractures, 24 opened perforations, half-length 180 m and height 30 m (local extract pp.3–4).
* MCMC table: true half-length 180 m is inferred as 164.43 m with SD 44.52 m; true height 30 m as 32.83 m (SD 6.80); true number 5 as 4 (SD 0.7); true opened perforations 24 as 23 (SD 1.6). Supplying number-of-fractures and opened-perforations as prior information improves half-length to 179.55 m (SD 43.05) and height 30.19 m (SD 4.47) (local extract pp.4–5).
* This is direct published counter-evidence to a claim that water-hammer fracture-geometry inversion has never used Bayesian posterior uncertainty. The unresolved issue is *identifiability of individually located clusters* and computational speed, not existence of MCMC inversion.

### Sheludko et al. 2024 SPE HFTC proceedings (10.2118/217781-MS)

* Uses an automated Python extraction/curve fit to a modified damped sine, then Random Forest classification of productive versus non-productive stages. Abstract reports 8 horizontal wells and 78 stage treatments, with 1-s field data; 70/30 split; test accuracy 0.71 and F1 0.72 (local extract pp.1–3). The longer Methods section reports 210 stages from 17 wells, also at 1-s intervals; this internal dataset-count discrepancy must be flagged.
* Features are amplitude, decay rate, phase, angular frequency, peaks, duration and curve-match parameters. This is a stage-level classification task, not a physics-constrained wavefield surrogate or parameter inversion for cluster impedance.
* The proceeding is a conference paper; the front matter explicitly says contents were not reviewed by SPE and are subject to correction. It supports an early data-science baseline, with limited temporal resolution (1 Hz), rather than a validated high-frequency operator.

### Zeng et al. 2024 (Frontiers in Energy Research; 10.3389/fenrg.2023.1336148)

* Abstract reports transient-flow simulation with fracture effects represented by R, C and I; overall geometry error about 6.28%, with 3.49% away from the well toe and 12.75% near the toe. The authors attribute the near-toe error to complex fracture structure and explicitly call for a more accurate relation between fracture size and R/C/I (local extract p.1). This directly qualifies any claim that current RCI inversion is geometry-complete.

### Gabry et al. 2025 SPEJ (10.2118/225459-PA)

* CWT with complex Morlet wavelet, ridge detection and envelope regression estimates damping coefficients as a proxy for fracture complexity (local extract p.1). The local review states reported correlation with number of propagating fractures, but this is a feature correlation/classification proxy, not cluster-by-cluster inversion.
* The paper discusses Sheludko's 0.71 accuracy/F1 0.72 and works with stage-level damping logs (local extract pp.1–2). It does not impose PDE residuals or jump conditions in a learned operator.

### Iuzofatov et al. 2025 SPE proceedings (10.2118/230381-MS)

* Abstract/full text: HIT acoustic approximation in the Laplace/frequency domain transfers characteristic impedance segment-by-segment; fracture is a parallel lumped impedance. A VAE is trained *unsupervised* on simulated signals; latent nearest-neighbour/projection gives parameter changes (local extract pp.1–4).
* Sampling can reach 200 Hz in practice (local extract p.1). The authors explicitly state that the model “does not account for the spatial characteristics of the fracture but represents it as a parallel segment of the well,” so a single-signal inverse contains significant assumptions (local extract p.3). They frame it as a supplementary initializer, not a definitive solution; this is strong evidence for a spatial/OOD gap, while disproving a blanket claim that no AI-decoded water-hammer work exists.

### Deng et al. 2025 EFM (10.1016/j.engfracmech.2025.111347)

* Abstract/full text: CEEMD enhancement + cepstrum obtains hydraulic-fracture number/location from wellhead high-frequency pressure; MOC then uses an RCI model including leak-off, natural-fracture interaction and multifracture stress shadows (local extract pp.1, 13–15). Signals are ≥200 Hz; useful retained spectrum is 0.1–20 Hz (local extract around lines 372–380).
* Assumptions include constant fluid density/viscosity, one fracture per effectively absorbing perforation cluster, constant-height planar elliptical fractures (local extract around lines 354–365). The multifracture circuit puts each fracture capacitance in parallel and reports an equivalent fracture (E-frac) (local extract around lines 1218–1220). Thus location/number detection and natural-fracture physics are published, but exact non-continuous cluster jump operators are still not supplied.

### Sun et al. 2025 SPEJ (10.2118/228403-PA)

* Abstract reports a compressible-fluid model with tubing friction, perforation friction and fracture filtration; field pressure fit average relative error ≈0.5%; comparison with microseismic gives average half-length and height errors 15.4 m and 4.2 m (local extract p.1).
* Explicit bottom boundary: `p_w-p_f=8ρQ²/(π²N_p²d_p⁴C_D²)` and `d(2N_fh_fl_fw)/dt=Q-Q_leak`; aperture and Carter leak-off are then closed with geometry/stress parameters (local extract p.7). The model acknowledges the planar constant-height assumption is less suitable for complex shallow-reservoir geometry (local extract p.7).
* This is a stronger physics-based forward/inverse baseline than a pure RCI circuit, but it remains a single equivalent boundary with fitted geometry parameters; no neural operator or per-cluster sparse-sensor posterior is demonstrated.

### Li et al. 2025 *Water* (10.3390/w17182741)

* Crossref abstract: multi-cluster staged horizontal-well model includes wellbore, perforation friction and fracture fluid loss; field multi-stage water-hammer data are inverted for fracture geometry. Features are initial amplitude, oscillation count/duration and attenuation; correlations are used for rapid geometry estimation. This is feature-to-geometry correlation, not a PDE-constrained operator.

### Yang et al. 2026 SPE Oman proceedings (10.2118/232521-MS)

* Crossref abstract: RCI hydraulic-electrical analogy, lumped fracture-end boundary, transient wellbore model; periodicity and attenuation iteratively match RCI parameters. Field application reports >85% consistency with microseismic; R correlates with effective volume/conductivity, C with storage/dissipation, and I is most sensitive to width. Wellbore friction and acoustic velocity affect attenuation/stability.
* No sampling rate, sample count or runtime is given in the Crossref abstract. It is a proceedings article. Therefore it confirms continued RCI geometry diagnosis in 2026, but cannot substantiate claims of resolved cluster-level inversion or real-time neural-operator inference.

### Zhang et al. 2026 SPEJ (10.2118/234685-PA)

* Abstract/full text uses 1-Hz treating-pressure data, CWT and wavelet coherence; diagnostic band 0.3–0.5 Hz, with 5-s windows and fuzzy rules classifying four propagation patterns. Evidence is supported by microseismic, distributed fibre-optic and tracer data (local extract pp.1–3).
* It is a propagation-pattern classifier, not a pump-shut-in MOC/PDE inverse for fracture impedance. It should not be used to claim high-frequency cluster parameter inversion has been solved; conversely, it proves recent signal-processing diagnostics are active and must be included in a state-of-art review.

### Zhu & Wang 2026 *Modelling* (10.3390/modelling7030087)

* Crossref abstract: adaptive shape-preserving Kalman denoising, automatic response-segment identification and particle-swarm geometry inversion on field high-frequency shut-in pressure. Reported SNR improves 11.99→25.05 dB, segment extraction runtime 0.84–1.22 s, onset error 0–5 s; one representative stage gives half-length error 6.21% and height error 3.04%.
* The reported runtime is for segment extraction, not end-to-end inversion. No PDE residual, jump-condition loss, operator-learning architecture, per-cluster posterior or OOD test is stated in the abstract. It therefore narrows the automation gap but leaves physics-informed operator inversion open.

## Structural identifiability warning (use in gap analysis)

For the Liu 2023 and Sun 2025 mass-balance closure,
`d(2 N_f h_f l_f w_f)/dt = Q - Q_leak`.
If `h_f`, `w_f` and leak-off closure are held fixed over the short water-hammer interval, the product `N_f l_f` controls storage. Distinct integer `N_f` and compensating `l_f` can therefore produce the same lumped storage contribution, while perforation friction depends on aggregate `N_p` and flow. This is an explicit structural non-uniqueness of the published equivalent-boundary model; it is not evidence that MOC itself is mathematically ill-posed. Liu's posterior SDs and improvement after supplying Nf/Np priors empirically expose this dependence.

## Search log and uncertainty register

* 2026-09-05: Crossref REST exact DOI queries for all records in the table; fields checked were DOI, title, authors, `type`, `published.date-parts`, venue and abstract where available.
* 2026-09-05: OpenAlex REST query `works?search=water hammer fracture` filtered conceptually to 2025–2026; used to identify 2026/2025 candidates and to cross-check `10.3390/modelling7030087` as a published OA journal article. OpenAlex relevance search also returns many unrelated pipeline/rock-water-jet records; these were excluded.
* Local full-text extracts under `research_ob/文献/_extract_tmp/` are the source of equation/page/line quotations. They are not treated as independent publication metadata.
* Crossref abstract for 10.3390/modelling7030087 was available; MDPI XML endpoint returned Access Denied in this environment, so no unquoted full-text details are inferred.
* Qiu 2022 local extract contains a binary/NUL artifact in one search pass; equations and tables were read from surrounding page text and are marked by page/section rather than relying on corrupted line numbers.
* Sheludko 2024 has an internal count inconsistency (abstract: 8 wells/78 stages; Methods: 17 wells/210 stages). Report both, do not silently reconcile.
* Yang 2026 SPE proceedings abstract is bibliographically verified, but sampling/runtime and detailed equations were not available from Crossref; mark these as unknown pending full paper access.
* “2025–2026” OpenAlex discovery results include records with future-dated publication metadata relative to real-world access; only exact Crossref DOI records with published dates are treated as verified here.

