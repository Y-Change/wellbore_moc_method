# kernel_loading_audit

run_id: `r4_P0_kernel_20260906_231434_0002042f`

- YAML: `E:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion\configs\friction_zielke.yaml`
- sha256: `ab0ed3bbc84e891e1e6e59fdfd124da25315db30af8bc7223d7c867426fe211b`
- M=12, domain_tau=[1e-08, 100.0]
- 加载方法: `yaml.safe_load` + `kernel_fits_from_config`；**未**调用 `get_kernel_fits`
- used_online_refit: False
- Stage-1 CONFIG_DIR: `E:\water_hammer_research\wellbore_moc_method\PaperC_CJNO_Wellbore_Inversion\configs`
- 分段耗时 (s): kernel_load=0.011, warmup_none=0.196, steady_zvb_setup=0.002, short_zvb=0.003
- 短窗 ZVB 每步 8.69e-05 s；未先把超时放大 10 倍
- 首案 Nx512 实测 Cr0=1, dt_reduced=False
