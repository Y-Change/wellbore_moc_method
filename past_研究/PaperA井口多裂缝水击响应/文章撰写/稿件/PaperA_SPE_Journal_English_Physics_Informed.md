# Nonlocal Water-Hammer Cepstral Responses in Multi-Cluster Wellbores: Mechanism Falsification and a Physics-Informed, Non-Circular Reconstruction Framework

## Abstract

Wellhead water-hammer measurements are attractive for diagnosing fluid-taking perforation clusters because a transient pressure record can be acquired without a downhole array. Their interpretation becomes nonlocal when several clusters are connected to the same wellbore: a downstream interface changes an upstream peak, intermediate spacing can attenuate later peaks more strongly than either smaller or larger spacing, and internal reverberation can create peaks at depths where no perforation was designed. We quantify these effects with one-dimensional method-of-characteristics (MOC) simulations and a two-dimensional marginal cepstrum. The benchmark contains 426 independent topological grids (six first-cluster depths, seven values of cluster count from two to eight, and ten spacings from 10 to 100 m; repeated single-cluster rows are excluded from this count). At \(X_1=3000\) m and \(S=20\) m, adding the second cluster reduces the first-cluster apparent peak from 5.157 to 2.705 a.u. (47.5%). For \(n=4\), the third and fourth peaks reach minima of 0.536 and 0.324 a.u. at \(S=30\) m and recover to 1.21 and 0.92 a.u. at \(S=100\) m. At a common total span of 60 m, the terminal-to-first ratio is 0.472, 0.188, 0.158 and 0.072 for \(n=2,3,4,7\), respectively. A stored \(S=30\) m, \(n=4\) profile also contains secondary peaks near 3120 and 3150 m, consistent with the predicted multiple-path depths.

These observations motivate a four-operator reconstruction framework. The nominal perforation depths are the known engineering prior; the set of clusters that actually take fluid is an unknown non-negative state. Ghost peeling is applied only at nominal multiple-path depths that do not coincide with a perforation prior. Transmission compensation uses finite, forward-calibrated factors \([T_{\mathrm{eff}}^{2(k-1)}\eta(S)]^{-1}\), rather than division by a local measured peak. A 0.80-m super-Gaussian focus is applied to fitted active components, and a continuous mask reports evidence and uncertainty in green, yellow and red bands. The zero-preservation property follows directly from non-negative amplitudes and finite gains. Verification on the four requested extreme geometries confirms the raw forward distortions and exposes where the existing oracle implementation is not a valid test of reconstruction. Quantitative blind recovery of inactive clusters and false-peak suppression requires an additional held-out benchmark with complex phase and independent activation states; it is not inferred from the 426 peak table alone.

**Keywords:** water hammer; multi-cluster fracturing; method of characteristics; cepstrum; multipath reverberation; physics-informed inverse problem; uncertainty mask

## 1. Introduction

The effectiveness of a multi-cluster stimulation stage depends on which designed clusters accept fluid. Downhole imaging and distributed sensing can provide direct information; borehole tube-wave studies show that fracture compliance, resonance and reflection can also constrain the downhole response [1,2]. Their cost and operational constraints motivate surface diagnostics. A shut-in or pump-off water-hammer transient is a useful candidate because pressure waves sample impedance changes along the wellbore and can be transformed into a delay or depth representation [3,4]. Recent studies have extended this concept to fluid-entry-depth detection, cluster-level event identification and ultra-high-frequency wellhead monitoring equipment [5--7].

The usual interpretation is local: a delay is assigned to a cluster, and the amplitude of the corresponding cepstral peak is treated as a proxy for that cluster. The quefrency interpretation follows the homomorphic cepstrum framework [8,9], but this interpretation is only conditionally valid. In a multi-cluster wellbore, the measured signal is the output of a connected network. Transmission and reflection at one cluster alter the wavefield incident on every downstream cluster and can also modify the return signal measured at the wellhead. The same nominal total span can therefore produce different terminal responses when the number of interfaces changes. Similarly, a spacing change can modify the waveform, not only translate the arrival time [10--13]. Figure 1 contrasts this rejected local-label reading with the network observation.

The present study asks two separate questions. First, which statements about locality, spacing and total span are rejected by a controlled MOC grid? Second, given a perforation-gun design, how can a reconstruction method remove predictable multiple-path contributions without assuming in advance which designed clusters broke down? The second question is deliberately formulated as a constrained inverse problem. Nominal positions are known; activation is not. A correction that inserts a peak at every nominal position would answer the wrong question and would be circular.

We use the authenticated MOC peak table and the accompanying analyses in Sections 5.1--5.7 of the project. The study is descriptive and mechanistic at the simulation-grid level. The independent unit is a topological MOC grid, not an individual time sample or a cepstral bin. No p-values are reported because the benchmark is a deterministic design-space scan rather than a random sample, and no field noise or repeated experimental replicates are available. Bayesian diagnostics, fracture-size inversion, naturally fractured-reservoir models, wavelet-based damping analysis and automated parameter inversion provide recent context [14--18], but they do not replace the forward evidence used here. The central contribution is consequently a bounded one: a set of falsifiable network observations and a non-circular operator specification that can be evaluated on a future blind benchmark.

## 2. Forward Model and Observation Operator

### 2.1 Governing equations

The wellbore is represented as a one-dimensional, constant-area, weakly compressible conduit. The simulated pressure variable is expressed as hydraulic head \(H(x,t)\), and \(Q(x,t)\) is volumetric flow rate:

\[
\frac{\partial H}{\partial t}+\frac{a^2}{gA}\frac{\partial Q}{\partial x}=0,
\qquad
\frac{\partial Q}{\partial t}+gA\frac{\partial H}{\partial x}+
\frac{f}{2DA}Q|Q|=0 .
\tag{1}
\]

Here \(a\) is acoustic wave speed, \(A=\pi D^2/4\) is the cross-sectional area, \(D\) is the internal diameter, \(g\) is gravitational acceleration and \(f\) is the Darcy--Weisbach friction factor. The main benchmark uses steady friction: \(f\) is set from the initial Reynolds number and is not updated with instantaneous velocity. Non-steady Brunone friction is outside the claim boundary of this paper.

At a cluster node \(x_k\), mass conservation and the lumped fracture/leak-off relation are

\[
Q_{L,k}-Q_{R,k}=Q_{f,k},
\qquad
Q_{f,k}=C_{f,k}\frac{\mathrm d H_{f,k}}{\mathrm dt}+
k_{\mathrm{leak},k}\sqrt{H_{f,k}-H_{\mathrm{ext}}},
\tag{2}
\]

where \(C_{f,k}\) is a lumped compliance, \(k_{\mathrm{leak},k}\) is a leak-off coefficient and \(H_{\mathrm{ext}}\) is the external head. All clusters share the same \(C_f\) and \(k_{\mathrm{leak}}\) in the benchmark, so the data isolate topology and spacing effects rather than fracture-property heterogeneity.

The MOC integrates Eq. (1) along \(\mathrm dx/\mathrm dt=\pm a\) with Courant number one, following standard transient-flow formulations [3,4]. The common parameter set is \(D=0.1397\) m, \(a=1450\) m/s, density \(1000\) kg/m\(^3\), kinematic viscosity \(10^{-6}\) m\(^2\)/s, roughness \(4.5\times10^{-5}\) m, initial velocity \(1.0\) m/s, initial head 300 m, \(\Delta t=10^{-3}\) s, \(C_f=10^{-5}\) m\(^2\), and \(k_{\mathrm{leak}}=10^{-4}\) m\(^{5/2}\)/s. The pump-off time is 1 s. Frequency-dependent and engineering unsteady-friction alternatives are documented in [19--21]; non-steady Brunone friction is outside the claim boundary of this paper.

### 2.2 Design space and independent units

The nominal cluster depths are

\[
x_{\mathrm{perf},k}=X_1+(k-1)S,\qquad k=1,\ldots,n,
\tag{3}
\]

with \(X_1\in\{2000,2500,3000,3500,4000,4500\}\) m, \(S\in\{10,20,\ldots,100\}\) m and \(n\in\{1,\ldots,8\}\). The CSV contains 4,320 cluster-level rows because both steady and Brunone friction tables are stored and the single-cluster case is repeated over spacing labels. For the steady-friction topology analysis, \(n=2\ldots8\) gives \(6\times7\times10=420\) grids and the single-cluster baseline contributes six additional grids, for 426 independent topologies.

### 2.3 Cepstral observation

The wellhead head record is transformed with a 30-s Hamming window and 5-s hop; filtering choices and parameters can affect reflection-period extraction from pump shut-in water-hammer records [22]. Let \(C(x,t)\) denote the resulting time--depth marginal cepstral field. The apparent peak assigned to the \(i\)-th nominal position is extracted conditionally within

\[
P_i=\max_{|x-x_i|\le r}\left[-\sum_t C(x,t)\right],
\qquad r=\min(15\ \mathrm m,0.49S),
\tag{4}
\]

where \(x_i\) is the simulated geometric depth in the benchmark analysis. Thus \(P_i\) is a local apparent response, not a direct measurement of fracture energy; because the extractor searches a simulated geometric neighbourhood, this is benchmark extraction rather than blind localization. We also use \(\alpha_i=P_i/P_1\), \(R_{\mathrm{end}}=P_n/P_1\), \(L_{\mathrm{span}}=(n-1)S\), and the fixed-geometry descriptive fit \(P_i=A\exp[-\gamma(i-1)]\). The fit parameter \(\gamma\) is an empirical envelope parameter; it is not identified with a single-interface transmission coefficient.

## 3. Mechanism Falsification Tests

Wellbore--fracture water-hammer and multi-fracture decay studies provide the relevant physical context [10--18], while tube-wave diagnostics show that uncertainty and frequency-dependent impedance must be treated explicitly [1,2].

### 3.1 Downstream feedback falsifies a local first-peak label

The multiplicity comparison is shown in Figure 6.

At \(X_1=3000\) m and \(S=20\) m, the single-cluster peak is 5.157 a.u. Adding a second cluster reduces it to 2.705 a.u., a 47.5% decrease. The first peak therefore depends on downstream topology even though its nominal depth and local cluster parameters are unchanged. Across multi-cluster grids the first peak varies between approximately 2.45 and 3.55 a.u.; the result is not a second universal constant, but it is sufficient to reject a single-cluster calibration used without network conditioning.

### 3.2 Spacing changes amplitude as well as delay

The continuous profiles and peak vectors are summarized in Figure 5.

For \(n=4\), the steady-friction peak vectors at \(X_1=3000\) m are:

| \(S\) (m) | \(P_1\) | \(P_2\) | \(P_3\) | \(P_4\) |
|---:|---:|---:|---:|---:|
| 10 | 2.738 | 1.421 | 0.964 | 0.634 |
| 20 | 2.844 | 1.317 | 0.614 | 0.450 |
| 30 | 3.122 | 1.442 | 0.536 | 0.324 |
| 40 | 3.050 | 1.514 | 0.596 | 0.327 |
| 50 | 3.063 | 1.541 | 0.609 | 0.436 |
| 60 | 3.088 | 1.464 | 0.592 | 0.437 |
| 70 | 3.040 | 1.527 | 1.064 | 0.693 |
| 80 | 3.090 | 1.581 | 1.158 | 0.878 |
| 100 | 3.159 | 1.705 | 1.209 | 0.920 |

The low response of the third and fourth clusters at 20--60 m and the recovery at larger spacing are not a simple delay shift. The two-cluster control is comparatively flat: \(P_1\approx2.70\)--2.74 a.u. and \(P_2\approx1.24\)--1.30 a.u. The emergence of the trough for \(n\ge3\) therefore identifies a network-level interaction in this forward model. The table does not, by itself, distinguish phase cancellation from other combinations of reflection, wave-packet overlap and the finite analysis window.

### 3.3 Total span is not sufficient

The spacing--multiplicity interaction is shown in Figure 8a--c.

At \(L_{\mathrm{span}}=60\) m, the terminal-to-first ratios are 0.4722 for \(n=2,S=60\) m, 0.1877 for \(n=3,S=30\) m, 0.1582 for \(n=4,S=20\) m and 0.0716 for \(n=7,S=10\) m. Equal span therefore does not imply equal response. The number of interfaces and their spacing must remain explicit variables.

### 3.4 Depth is not a separable scalar gain

The depth main effect, joint depth--spacing map and fixed-geometry envelope are shown in Figures 7, 8d--f and 9.

The joint \(X_1,S\) analysis shows that relative responses, particularly for later clusters, vary with first-cluster depth. At \(X_1=3000\) m the fixed-geometry exponential fits have mean \(R^2=0.9864\) for \(n=3\ldots8\), but \(\gamma\) changes with spacing. For \(n=8\), \(\gamma=0.684\) at \(S=30\) m and \(\gamma=0.294\) at \(S=80\) m. The high \(R^2\) is a compact description within a geometry; it is not evidence for a universal spatial attenuation law.

## 4. Physics-Informed Reconstruction

Figure 3 presents the four operators as forward-derived templates and sensitivity proxies; it does not show a corrected profile or a blind-recovery result.

### 4.1 Non-circular inverse formulation

The reconstruction input is the observed profile \(y(x)=P_{2D}(x)\), together with the engineering prior in Eq. (3). The unknown state is a non-negative activity vector \(z=(z_1,\ldots,z_n)\), where \(z_k=0\) denotes a cluster that contributes no fluid-taking response. Unit-response templates \(h_k(x)\) and multiple-path templates \(g_m(x)\) are generated by the same MOC forward model or by a separately calibrated impulse-response library. We estimate activity and nuisance ghost coefficients with

\[
\min_{z\ge0,\ c\ge0}
\left\|W^{1/2}\left[y-Hz-Gc\right]\right\|_2^2
 +\lambda_z\|z\|_1+\lambda_c\|c\|_2^2,
\tag{5}
\]

where \(H=[h_1,\ldots,h_n]\), \(G=[g_2,\ldots,g_M]\), \(W\) weights reliable depth samples, and \(c\) contains ghost amplitudes. The coefficients \(c_m\) are constrained by forward path bounds, for example \(0\le c_m\le\bar\rho_m\sum_k z_k\), so a completely inactive stage cannot create a ghost-only fracture profile. The regularization parameters are selected on held-out forward simulations and are not tuned from the target profile.

### 4.2 Ghost peeling operator \(G_{\mathrm{peel}}\)

Here \(G_{\min}\in[0,1)\) is a pre-registered lower retention bound calibrated on held-out forward profiles.

Multiple-path candidate depths are

\[
x_{\mathrm{ghost},m}=X_1+mS,\qquad m\ge2,
\tag{6}
\]

after removing positions that coincide with the nominal perforation set. Let \(\kappa(u)=\exp(-u^2)\), \(w_g\) be the calibrated ghost width, and \(\rho_m\in[0,0.95]\) be a path coefficient obtained from forward impulse responses. The notch is

\[
G_{\mathrm{peel}}(x)=
\operatorname{clip}\left[1-
\sum_{m\in\mathcal G}\rho_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right),
\ G_{\min},1\right],
\tag{7}
\]

where \(\mathcal G\) excludes \(x_{\mathrm{ghost},m}\in\{x_{\mathrm{perf},k}\}\). The operator suppresses a predicted path family; it does not declare every sample at that depth to be false. Candidate overlap with a nominal perforation is retained for the activity fit and is not notched.

### 4.3 Transmission and phase compensation \(K_{\mathrm{comp}}\)

The gain assigned to the \(k\)-th nominal cluster is

\[
g_k(S)=\frac{1}{T_{\mathrm{eff}}^{2(k-1)}\eta(S)},
\qquad
K_{\mathrm{comp}}(x)=1+\sum_{k=1}^{n}[g_k(S)-1]
\,\omega_k(x),
\tag{8}
\]

with \(T_{\mathrm{eff}}\in(0,1]\) obtained from a forward transmission calibration and \(\eta(S)\in[\eta_{\min},1]\) representing the calibrated phase-cancellation magnitude. A physically motivated phase model is

\[
\eta(S)=\operatorname{clip}\left(\left|1+R_{\mathrm{frac}}e^{-\mathrm i4\pi S/\lambda_0}\right|,\eta_{\min},1\right),
\tag{9}
\]

where \(R_{\mathrm{frac}}\) is a complex reflection coefficient and \(\lambda_0\) is the reference wavelength. The window \(\omega_k\) is centred on the nominal design depth and normalized to a finite support. It is not computed by dividing by the local observed peak and it does not target a prescribed amplitude. The reconstructed cluster coefficient is \(\tilde z_k=g_kz_k\). Consequently, \(z_k=0\Rightarrow\tilde z_k=0\) for every finite \(g_k\); compensation cannot fabricate a fracture. Figure 3 uses the fixed illustrative value \(T_{\mathrm{eff}}=0.784\) only to visualize a finite cascade-gain template; this value is not identified from the 426-grid peak table and is not a parameter estimate of this study.

### 4.4 Super-Gaussian focusing \(F_{\mathrm{focus}}\)

For each component with \(z_k\) above a pre-registered activity threshold, a centre \(\hat x_k\) is estimated by a local template fit within the nominal window. The focus kernel is

\[
F_{\mathrm{focus}}(x)=
\sum_{k:z_k>\tau_z}q_k
\exp\left[-\left(\frac{x-\hat x_k}{w_{\mathrm{core}}}\right)^4\right],
\qquad w_{\mathrm{core}}=0.80\ \mathrm m,
\tag{10}
\]

where \(q_k\) is the fitted activity confidence. The kernel is applied to fitted active components, not to a true-state-labelled profile. An inactive component has \(q_k=0\), so focusing does not turn a noise maximum into a fracture.

### 4.5 Continuous authenticity mask \(M(x)\)

The scales and weights in Eq. (11), including \(w_t\), \(u_k\) and \(v_m\), are calibrated on held-out forward simulations with prescribed noise perturbations and frozen before test inspection.

Let \(u_k\) combine normalized coefficient signal-to-uncertainty, residual agreement with \(h_k\), and a spacing-overlap penalty. Let \(v_m\) be the predicted ghost evidence. We define

\[
M(x)=\operatorname{clip}\left[
\sum_k u_k\tilde z_k\,\kappa\left(\frac{x-\hat x_k}{w_t}\right)
-\sum_{m\in\mathcal G}v_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right),0,1\right].
\tag{11}
\]

The colour bands are operational labels: green \(M\ge0.70\), yellow \(0.20\le M<0.70\), and red \(M<0.20\). They are confidence bands for this inverse model, not probabilities and not field-wide detection limits. The final profile can be written as

\[
\tilde P(x)=M(x)G_{\mathrm{peel}}(x)K_{\mathrm{comp}}(x)
\left[\sum_k z_kh_k(x)\right]F_{\mathrm{focus}}(x).
\tag{12}
\]

For deployment, Eq. (12) must be evaluated with \(H,G,T_{\mathrm{eff}},\eta\), thresholds and uncertainty calibration fixed before inspecting the test state. The legacy exploratory script in the project is not used as validation because it divides a target amplitude by a local raw peak and constructs \(M\) from supplied true positions.

## 5. Extreme Case Verification

The raw profiles, nominal priors and path candidates for the four requested geometries are summarized in Figure 4. These panels are forward evidence rather than masked fracture-state estimates.

The 426-grid table verifies forward distortions. The operator-specific claims below are stated as verification criteria and boundary checks; they are not presented as blind reconstruction accuracy because the existing operator outputs use oracle geometry.

| Case | Design | Raw forward evidence | Required non-circular check |
|---|---|---|---|
| 1. Phase-cancellation/ghost trap | \(X_1=3000\) m, \(S=30\) m, \(n=4\) | \(P=[3.122,1.442,0.536,0.324]\) a.u.; secondary peaks near 3119.2 and 3149.7 m correspond to nominal ghost depths 3120 and 3150 m | Fit \(z\) using nominal templates; notch only non-perforation ghost positions; report whether \(z_3,z_4\) remain supported and report residuals |
| 2. Dense cluster limit | \(S=10\) m, \(n=4\) | \(P=[2.738,1.421,0.964,0.634]\) a.u.; adjacent nominal windows are close relative to the 30-s analysis response | Report coefficient covariance and use yellow \(M\) where components are not separately resolved; do not claim four independent peaks from design positions alone |
| 3. Deep cascade | \(S=20\) m, \(n=8\) | \(P=[3.420,1.616,0.695,0.554,0.406,0.274,0.231,0.184]\) a.u.; \(R_{\mathrm{end}}=0.0537\) | Apply finite \(g_k\) from an independent transmission calibration and test zero-preservation with inactive synthetic clusters; report uncertainty inflation as gain increases |
| 4. High-spacing raw baseline | \(S=80\) m, \(n=4\) | \(P=[3.090,1.581,1.158,0.878]\) a.u.; \(R_{\mathrm{end}}=0.2841\) | Test whether \(K_{\mathrm{comp}}\) is near unity only where an independent calibration predicts higher transmission, and verify that the mask does not create extra peaks |

The Case 1 profile is particularly useful because the stored continuous MOC profile contains peaks at approximately 3000.29, 3029.29, 3058.29, 3088.75, 3119.20 and 3149.65 m. The latter two are close to the \(m=4\) and \(m=5\) ghost depths after the \(m=2\) and \(m=3\) positions coincide with designed clusters. This supports the existence of the path geometry, while their amplitudes (approximately 0.223 and 0.140 a.u. in the stored profile) do not justify the stronger statement that all ghosts are dominant or that a fixed notch removes them without loss.

The exploratory implementation provides a useful negative result. When it is run with true positions supplied, the apparent equalization is incomplete: in the \(S=30\) m, \(n=4\) case the third and fourth true-position values collapse to zero after its thresholding, and in the \(S=20\) m, \(n=8\) case two late clusters are lost. This is expected from its target-division and oracle-mask design and is why those outputs are excluded from the evidence chain. A publishable operator benchmark should report precision, recall, count error, localization error, ghost suppression ratio and coefficient uncertainty on held-out activation patterns, with all parameters frozen before evaluation.

## 6. Discussion and Downstream Inversion Link

The forward grid changes the object that a wellhead peak represents. \(P_i\) is a network-conditioned apparent response, not a local fracture label. The 47.5% first-peak change, the spacing trough for \(n\ge3\), and the separation of equal-span curves are mutually consistent with a connected multi-interface system. They do not uniquely identify a single reflection coefficient or a single phase-cancellation mechanism. Complex spectra, path ablations and controlled phase perturbations would be needed for that stronger attribution.

The reconstruction formulation preserves this evidence boundary. Equation (5) separates known design geometry from unknown fluid-taking activity. Equation (8) restores only a finite physical gain derived from forward transmission and phase factors. This distinction matters operationally: target equalization can make a visually attractive profile while concealing a missing cluster or amplifying a ghost. In contrast, the non-negative activity state carries the diagnostic decision, and a zero state remains zero under compensation.

The mask \(M\) should be interpreted as a decision aid for subsequent inversion. A green interval can be passed to a hydraulic-opening or inflow-resistance inversion only when the coefficient uncertainty and residual checks meet pre-registered limits. Yellow intervals should be propagated as ambiguous alternatives, for example by sampling the posterior or by reporting a range of \(z_k\). Red intervals should be excluded from quantitative inversion but retained in the audit trail because a red value may reflect either a predicted ghost or insufficient information. The mask is therefore a confidence accounting device, not a probability of fracture breakdown.

The downstream inverse problem can be written generically as \(z_k=\mathcal H(w_{f,k},C_{f,k},k_{\mathrm{leak},k})\), where \(w_{f,k}\) is hydraulic opening and \(\mathcal H\) is a forward-calibrated map. The present work supplies a physics-conditioned \(z_k\) and its uncertainty; it does not identify \(w_{f,k}\) because the benchmark holds fracture properties fixed. Any opening inversion must be evaluated with independent property perturbations, wave-speed error, friction-model uncertainty, field noise and off-design cluster locations.

Several limitations are material. The main table contains deterministic simulations, not field replicates. The peak extractor searches a true neighbourhood and therefore cannot establish blind localization accuracy. The 426 grids do not contain explicit activation labels for all possible partial-breakdown states, and the stored ghost examples are available for selected continuous profiles rather than every topology. Finally, the empirical exponential envelope has high within-geometry \(R^2\) but is not a universal transmission law. These limitations define the next validation step rather than weaken the forward falsification results.

## 7. Conclusions

1. A downstream cluster changes the apparent response of an upstream cluster. At \(X_1=3000\) m and \(S=20\) m, \(P_1\) decreases from 5.157 to 2.705 a.u. when \(n\) changes from one to two.

2. For \(n\ge3\), spacing changes later-cluster amplitudes non-monotonically. At \(n=4\), \(P_3/P_4\) are 0.536/0.324 a.u. at \(S=30\) m and 1.209/0.920 a.u. at \(S=100\) m. A two-cluster control is comparatively flat, supporting a multi-interface network interpretation.

3. Total span is not a sufficient descriptor: at 60 m, \(R_{\mathrm{end}}\) is 0.472, 0.188, 0.158 and 0.072 for \(n=2,3,4,7\), respectively. The spacing and interface count must be retained in any calibration.

4. The proposed \(G_{\mathrm{peel}}\)-\(K_{\mathrm{comp}}\)-\(F_{\mathrm{focus}}\)-\(M\) framework is non-circular only when nominal perforation depths are used as a prior and activation is estimated as a non-negative unknown. Finite physics-derived compensation preserves zero activity and cannot fabricate a fracture.

5. The four extreme geometries define a reproducible blind-benchmark protocol. The current 426-grid evidence verifies the forward traps and selected ghost depths, but it does not establish 100% recovery, universal ghost suppression or quantitative hydraulic-opening inversion. Those claims require held-out activation states, independent calibration of \(T_{\mathrm{eff}}\) and \(\eta(S)\), and uncertainty-aware evaluation.

## Figure captions and file mapping

**Figure 1. Local-label reading versus network observation.** The left panel is the naive local interpretation rejected by the forward evidence; the right panel shows the wellhead response after multiple-interface transmission, reflection and cepstral observation.

**Figure 2. Forward model and observation operator.** (a) Lumped cluster-node balance; (b) single-cluster and four-cluster profiles; (c) nominal windows and local peak extraction. Panel (c) uses simulated geometric neighbourhoods and is benchmark extraction, not blind localization.

**Figure 3. Forward-derived de-aliasing operator templates.** Candidate path notch, finite gain, candidate focusing kernels and the \(M_{\mathrm{sens}}\) prior/transmission proxy are shown for the Case 1 geometry. The fixed illustrative value \(T_{\mathrm{eff}}=0.784\) is normalized for display, \(\eta(S)\) is not included, and this value is not inferred from the 426-grid peak table; no corrected profile or fracture probability is shown.

**Figure 4. Forward-evidence matrix for the four extreme geometries.** Green, yellow and grey bands are k-dependent high, ambiguous and low-information sensitivity proxies; burgundy bands are non-perforation path candidates. The profiles are raw and no blind recovery is claimed.

**Figure 5. Spacing main effect.** Continuous profiles, absolute peak amplitudes and relative responses for \(n=4\).

**Figure 6. Multiplicity and terminal decay.** Peak and end-to-first ratios as the number of interfaces changes.

**Figure 7. First-cluster depth effect.** Absolute and relative responses across \(X_1\).

**Figure 8. Spacing--multiplicity and depth--spacing interactions.** (a) \(P_1(S,n)\); (b) \(R_{\mathrm{end}}(S,n)\); (c) \(R_{\mathrm{end}}\) versus \(L_{\mathrm{span}}\), grouped by multiplicity; (d) \(P_1(S,X_1)\); (e) \(\alpha_2(S,X_1)\); and (f) \(\alpha_2(S)\) at the six first-cluster depths. Grey points mark authenticated grid locations; the continuous surfaces are interpolated only for visualization. File: `图表数据/Figure_8_Interaction_Phase_Maps.{png,svg,pdf}`.

**Figure 9. Fixed-geometry decay envelope.** Descriptive \(P_i=A\exp[-\gamma(i-1)]\) fits; \(\gamma\) is not a universal transmission coefficient.

**Figure 10. Operational identifiability envelope.** Threshold-dependent count, apparent signal-to-noise and spacing feasibility.

The main figures are exported as 400-dpi PNG, editable SVG and PDF using the same font, line widths and palette. Legacy 5-by-1 oracle exports and the target-equalization calibration figure are excluded from the evidence chain.

## References

1. Bakku, S. K., Fehler, M. C. & Burns, D. R. Fracture compliance estimation using borehole tube waves. *Geophysics* 78, D249–D260 (2013). DOI: 10.1190/geo2012-0521.1.
2. Liang, C., O’Reilly, O., Dunham, E. M. & Moos, D. Hydraulic fracture diagnostics from Krauklis-wave resonance and tube-wave reflections. *Geophysics* 82, D171–D186 (2017). DOI: 10.1190/geo2016-0480.1.
3. Ghidaoui, M. S., Zhao, M., McInnis, D. A. & Axworthy, D. H. A review of water hammer theory and practice. *Applied Mechanics Reviews* 58, 49–76 (2005). DOI: 10.1115/1.1828050.
4. Chaudhry, M. H. *Applied Hydraulic Transients*, 3rd ed. (Springer, 2014). DOI: 10.1007/978-1-4614-8538-4.
5. Dong, X. et al. Research and application of hydraulic fracturing fluid entry depth detection method based on water-hammer signal. *Geoenergy Science and Engineering* 246, 213556 (2025). DOI: 10.1016/j.geoen.2024.213556.
6. Sun, S. et al. Identification of fluid-entry clusters and diagnosis of downhole events based on high-frequency water hammer pressure. *International Journal of Rock Mechanics and Mining Sciences* 200, 106437 (2026). DOI: 10.1016/j.ijrmms.2026.106437.
7. Cheng, Y. et al. Development and effectiveness verification of ultra-high-frequency water-hammer wave monitoring equipment for large-scale fracturing of unconventional oil and gas. *Flow Measurement and Instrumentation* 111, 103424 (2026). DOI: 10.1016/j.flowmeasinst.2026.103424.
8. Childers, D. G., Skinner, D. P. & Kemerait, R. C. The cepstrum: a guide to processing. *Proceedings of the IEEE* 65, 1428–1443 (1977). DOI: 10.1109/PROC.1977.10747.
9. Oppenheim, A. V. & Schafer, R. W. From frequency to quefrency: a history of the cepstrum. *IEEE Signal Processing Magazine* 21, 95–106 (2004). DOI: 10.1109/MSP.2004.1328092.
10. Qiu, Y. et al. Water hammer response characteristics of wellbore-fracture system: multi-dimensional analysis in time, frequency and quefrency domain. *Journal of Petroleum Science and Engineering* 213, 110425 (2022). DOI: 10.1016/j.petrol.2022.110425.
11. Luo, Y. et al. A new water hammer decay model: analyzing the interference of multiple fractures and perforations on decay rate. *SPE Journal* 28, 1973–1985 (2023). DOI: 10.2118/214658-PA.
12. Hu, X. et al. Evaluation of multi-fractures geometry based on water hammer signals: a new comprehensive model and field application. *Journal of Hydrology* 612, 128240 (2022). DOI: 10.1016/j.jhydrol.2022.128240.
13. Sun, S. et al. A novel comprehensive water hammer pressure model for fracture geometry evaluation. *SPE Journal* 30, 5350–5366 (2025). DOI: 10.2118/228403-PA.
14. Liu, Lijun, Liu, Yongzan & Wang, Xiaoguang. A novel MCMC-based hydraulic fracture diagnostics approach using water hammer data. In *57th US Rock Mechanics/Geomechanics Symposium* (2023). DOI: 10.56952/ARMA-2023-0865.
15. Zeng, B. et al. Fracture size inversion method based on water hammer signal for shale reservoir. *Frontiers in Energy Research* 11, 1336148 (2024). DOI: 10.3389/fenrg.2023.1336148.
16. Deng, S. et al. A diagnostic model for hydraulic fracture in naturally fractured reservoir utilising water-hammer signal. *Engineering Fracture Mechanics* 325, 111347 (2025). DOI: 10.1016/j.engfracmech.2025.111347.
17. Gabry, M. A., Ramadan, A. & Soliman, M. Y. Estimating water hammer damping ratios using continuous wavelet transform for induced hydraulic fracture complexity characterization. *SPE Journal* 30, 3587–3611 (2025). DOI: 10.2118/225459-PA.
18. Zhu, M. & Wang, H. Automated water hammer analysis for fracture parameter inversion using high-frequency shut-in pressure signals during hydraulic fracturing. *Modelling* 7, 87 (2026). DOI: 10.3390/modelling7030087.
19. Zielke, W. Frequency-dependent friction in transient pipe flow. *Journal of Basic Engineering* 90, 109–115 (1968). DOI: 10.1115/1.3605049.
20. Vardy, A. E. & Brown, J. M. B. Transient, turbulent, smooth pipe friction. *Journal of Hydraulic Research* 33, 435–456 (1995). DOI: 10.1080/00221689509498654.
21. Bergant, A., Simpson, A. R. & Vitkovsky, J. Developments in unsteady pipe flow friction modelling. *Journal of Hydraulic Research* 39, 249–257 (2001). DOI: 10.1080/00221680109499828.
22. Dong, X. et al. The influence of filtering methods and parameters on reflection period from pump shut-in water hammer signals: a comprehensive study. *Geoenergy Science and Engineering* 257, 214278 (2026). DOI: 10.1016/j.geoen.2025.214278.

### Reproducibility statement

The forward evidence is stored in `01_几何网格/峰值表/decay_table.csv`; the analysis directories are `5.1_间距S主效应分析`, `5.2_裂缝总数n主效应分析`, `5.3_首缝深度X1主效应分析`, `5.4_间距与缝数交互相图分析`, `5.5_深度与间距交互相图分析`, `5.6_空间衰减包络与模型拟合` and `5.7_可辨识度边界与可行域评估`. The exact MOC and cepstrum settings are given in Section 2. The exploratory oracle implementation is retained for audit but is not a validation implementation for this manuscript. A release version should include the constrained solver for Eq. (5), the forward-derived \(H\) and \(G\) libraries, a held-out activation benchmark, and a machine-readable table of all mask and uncertainty metrics.
