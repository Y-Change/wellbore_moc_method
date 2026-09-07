# Stage 1 — MOC high-fidelity benchmark

Run everything from `code/`:

```text
python test_units.py
python validate.py                 # §3.3 gates; writes goldens/, Fig_2, table_moc_gate.tex
python generate_dataset.py --pilot --workers 8
python generate_dataset.py --formal   # refused until every gate is PASS
```

From the PaperC root:

```text
python run_all_reproducible.py --stage 1 --seed 42 --device cpu --smoke
```

Outputs live only under this folder (`data/`, `figures/`, `tables/`, `01_stage_conclusion.md`).
