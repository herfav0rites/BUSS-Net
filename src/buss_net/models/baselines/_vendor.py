"""Resolve upstream vendor source trees under experiments/comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# model.name -> (experiment stem, vendor subdirectory)
VENDOR_LAYOUT: dict[str, tuple[str, str]] = {
    "unet": ("C10_unet", "Pytorch-UNet"),
    "nnunet": ("C01_nnunet", "nnUNet"),
    "unetpp": ("C02_unetpp", "pytorch-nested-unet"),
    "mseg": ("C03_mseg", "HarDNet-MSEG"),
    "dcrnet": ("C04_dcrnet", "DCRNet"),
    "acsnet": ("C06_acsnet", "ACSNet"),
    "pranet": ("C07_pranet", "PraNet"),
    "eunet": ("_retired_eunet", "Enhanced-U-Net"),
    "sanet": ("C08_sanet", "SANet"),
}


def vendor_root(model_name: str, config: dict | None = None) -> Path:
    """Return vendor checkout root for a comparison baseline."""

    model_name = model_name.lower()
    layout_stem, layout_vendor = VENDOR_LAYOUT[model_name]
    layout_path = PROJECT_ROOT / "experiments" / "comparisons" / layout_stem / "vendor" / layout_vendor

    if config:
        vendor_rel = config.get("model", {}).get("vendor_dir")
        if vendor_rel:
            path = PROJECT_ROOT / vendor_rel
            if path.is_dir():
                return path
        exp_name = str(config.get("experiment", {}).get("name", ""))
        if exp_name:
            vendor_base = PROJECT_ROOT / "experiments" / "comparisons" / exp_name / "vendor"
            preferred = vendor_base / layout_vendor
            if preferred.is_dir():
                return preferred
            if vendor_base.is_dir():
                children = [p for p in vendor_base.iterdir() if p.is_dir()]
                if len(children) == 1:
                    return children[0]
                for child in children:
                    if (child / "README.md").is_file() or any(child.rglob("*.py")):
                        return child
    return layout_path


def ensure_vendor_on_path(model_name: str, config: dict | None = None) -> Path:
    root = vendor_root(model_name, config)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Vendor source for '{model_name}' not found at {root}. "
            "Restore the declared upstream source tree under the experiment's vendor directory."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
