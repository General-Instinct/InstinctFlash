"""Network serving — the openpi-wire websocket policy server over `Runtime`.

One import deeper than the public API on purpose: `Runtime` stays transport-free
(tests/test_runtime_facade.py pins that), and serving is the optional `[serve]` extra
(`websockets` + `msgpack`; numpy rides with the runtime extra).

    instinctflash serve <model-id> --serve.port=8000        # the CLI verb
    WebsocketPolicyServer(runtime).serve_forever()          # the library form
"""

from instinctflash.serving.ws_server import WebsocketPolicyServer, default_metadata

__all__ = ["WebsocketPolicyServer", "default_metadata"]
