"""
Preflight for the hackathon: checks the Terrarium/CoLLAB install, the inference key
(OPENROUTER_API_KEY or NEBIUS_API_KEY), and (optionally) one live model call through
the shared endpoint.

    uv run python -m experiments.preflight                      # checks only
    uv run python -m experiments.preflight --ping --model <id>  # + one tiny chat call (--model required)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon_compute import llm_config


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'✅' if ok else '❌'} {label}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ping", action="store_true", help="make one tiny model call through the shared endpoint")
    ap.add_argument("--model", default=None, help="model for --ping (default: openai/gpt-4o-mini)")
    args = ap.parse_args()

    ok = True

    ok &= check("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    try:
        import llm_server  # noqa: F401
        import terrarium

        ok &= check("terrarium-agents importable", True, getattr(terrarium, "__version__", "0.1.1"))
    except Exception as e:  # noqa: BLE001
        ok &= check("terrarium-agents importable", False, f"{e} - run `uv sync`")

    collab = os.getenv("TERRARIUM_COLLAB_PATH") or str(ROOT / "external" / "CoLLAB")
    has_collab = (Path(collab) / "problem_layer").is_dir()
    ok &= check("CoLLAB checkout", has_collab, collab)
    if has_collab and not os.getenv("TERRARIUM_COLLAB_PATH"):
        os.environ["TERRARIUM_COLLAB_PATH"] = collab
    try:
        from terrarium.utils import get_implemented_environment_names

        names = get_implemented_environment_names()
        ok &= check("environments load (needs CoLLAB)", bool(names), ", ".join(names))
    except Exception as e:  # noqa: BLE001
        ok &= check("environments load (needs CoLLAB)", False, str(e)[:200])

    has_key = llm_config.config()["has_key"]
    ok &= check(
        "inference key set (OPENROUTER_API_KEY or NEBIUS_API_KEY)",
        has_key,
        ""
        if has_key
        else "copy compute.env.example to compute.env at the repo root and set OPENROUTER_API_KEY (or NEBIUS_API_KEY)",
    )

    if args.ping and ok:
        from experiments.common.providers import DEFAULT_MODEL, make_openrouter_client
        from poseidon_compute import resolve_model

        model = args.model or DEFAULT_MODEL
        resolved_model = resolve_model(model)
        client = make_openrouter_client({"provider": "openrouter", "openrouter": {"model": model}})
        try:
            _, text = client.generate_response(
                input=client.init_context("Reply with one word.", "Say ok."),
                params={"model": resolved_model, "max_completion_tokens": 5, "temperature": 0.0},
            )
            ok &= check(f"model call ({resolved_model})", True, repr(text.strip()[:40]))
        except Exception as e:  # noqa: BLE001
            ok &= check(f"model call ({resolved_model})", False, str(e)[:300])

    print(
        "\nAll good - try:\n  uv run python -m experiments.collusion.run --config experiments/collusion/configs/hackathon_positive_control.yaml"
        if ok
        else "\nFix the ❌ items above, then re-run."
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
