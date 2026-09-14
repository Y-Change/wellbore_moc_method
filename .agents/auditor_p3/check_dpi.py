from PIL import Image
from pathlib import Path

fig_dir = Path("PaperC_CJNO_Wellbore_Inversion/output/figures")
for p in sorted(fig_dir.glob("*.png")):
    if p.name.startswith("fig"):
        with Image.open(p) as img:
            dpi = img.info.get("dpi")
            print(f"{p.name:<35}: size={img.size}, dpi={dpi}")
