"""Import-only shim for `flash_attn`, used ONLY while the real wheel builds.

Why this exists
---------------
`lingbot-va/wan_va/modules/model.py` does an unconditional module-level import:

    try:
        from flash_attn_interface import flash_attn_func
    except:
        from flash_attn import flash_attn_func

so the module cannot be imported at all unless *some* `flash_attn` is importable --
even though the RoboTwin serving path never uses it. Both the checkpoint
(`transformer/config.json`: `"attn_mode": "torch"`) and the server
(`wan_va_server.py`, `load_transformer(..., attn_mode="torch")`) select
`custom_sdpa`, i.e. `torch.nn.functional.scaled_dot_product_attention`.
`flash_attn_func` is only ever bound when `attn_mode == 'flashattn'`
(`model.py`, WanAttention.__init__).

This shim therefore cannot change any numerics: it raises if it is ever *called*.
That turns "flash-attn is unused" from an assumption into an enforced invariant --
if a code path ever does reach it, the run dies loudly instead of silently
producing a number under a different attention kernel.

Activated automatically, and only when needed. `instinctwm.runtime.lingbot_install`
adds this directory to `sys.path` if and only if `flash_attn` is not already
importable, so a real wheel always wins and nothing is shadowed.

This used to live outside the package and required the caller to set PYTHONPATH --
which meant serving LingBot-VA depended on knowing about a file that no document
mentioned. It is a real package on disk rather than a synthesised module because
`transformers` calls `importlib.util.find_spec("flash_attn")`, which raises
`ValueError: flash_attn.__spec__ is None` for a hand-built module object.
"""

__version__ = "0.0.0+instinctwm-import-shim"
__is_instinctwm_shim__ = True


def flash_attn_func(*args, **kwargs):
    raise RuntimeError(
        "flash_attn_func was CALLED, but only the InstinctWM import-only shim is "
        "installed. The RoboTwin serving path is supposed to run attn_mode='torch' "
        "(custom_sdpa). Something selected attn_mode='flashattn'. Refusing to run: "
        "a benchmark number produced under an unintended attention kernel is not a "
        "valid number. Install the real flash-attn, or fix the attn_mode."
    )


def __getattr__(name):
    raise AttributeError(
        f"flash_attn.{name} requested from the InstinctWM import-only shim "
        f"(only `flash_attn_func` is stubbed). Install the real flash-attn."
    )
