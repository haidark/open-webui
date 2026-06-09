import sys
import types

# `open_webui.utils.payload` pulls in `open_webui.utils.task`, which imports a
# single constant from the heavyweight `open_webui.config` module (redis, ML
# deps, etc.). The cache-control helper under test is pure and needs none of
# that, so we stub the config module before it is imported.
if "open_webui.config" not in sys.modules:
    _config_stub = types.ModuleType("open_webui.config")
    _config_stub.DEFAULT_RAG_TEMPLATE = ""
    sys.modules["open_webui.config"] = _config_stub
