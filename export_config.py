"""Export generation settings from configs/base.yaml into the Flutter asset bundle.

The app must generate with the SAME settings used during evaluation. If the
phone runs at a different temperature or top_p than the SCORE harness did, the
reported quality numbers describe a different system than the one demonstrated
— the same failure mode the guard parity test exists to prevent, one layer up.

    python mobile/export_config.py
"""
import json, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    inf = base["inference"]

    cfg = {
        "_comment": "GENERATED from configs/base.yaml by mobile/export_config.py. "
                    "Do not edit. Regenerate after any change to the inference block, "
                    "or the app will generate under different settings than those used "
                    "to produce the reported results.",
        "_source": "configs/base.yaml -> inference",
        "system_prompt": base["system_prompt"].strip(),
        "generation": {
            "temperature": inf["temperature"],
            "max_new_tokens": inf["max_new_tokens"],
            "top_p": inf["top_p"],
            "top_k": inf["top_k"],
            "do_sample": inf["do_sample"],
        },
        "quantization_target": base["quantization"]["target_format"],
        "context_length": 1024,
    }

    out = ROOT / "mobile" / "assets" / "model_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  temperature {cfg['generation']['temperature']}  "
          f"top_p {cfg['generation']['top_p']}  top_k {cfg['generation']['top_k']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
