import asyncio
import http
import json
import logging
import pathlib
import re
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        timing_log_path: str | pathlib.Path | None = None,
        timing_log_dir: str | pathlib.Path | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._timing_log_path = pathlib.Path(timing_log_path) if timing_log_path else None
        self._timing_log_dir = pathlib.Path(timing_log_dir) if timing_log_dir else None
        self._request_index = 0
        self._prompt_to_task_index: dict[str, int] = {}
        if self._timing_log_path is not None:
            self._timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._timing_log_dir is not None:
            self._timing_log_dir.mkdir(parents=True, exist_ok=True)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.perf_counter()
                action = self._policy.infer(obs)
                infer_ms = (time.perf_counter() - infer_time) * 1000

                action["server_timing"] = {
                    "infer_ms": infer_ms,
                }

                self._write_timing_record(obs, action)
                await websocket.send(packer.pack(action))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _write_timing_record(self, obs: dict, action: dict) -> None:
        log_path = self._get_timing_log_path(obs)
        if log_path is None:
            return

        record = {
            "request_index": self._request_index,
            "server_timing": {
                "infer_ms": action.get("server_timing", {}).get("infer_ms"),
            },
            "policy_timing": {
                "infer_ms": action.get("policy_timing", {}).get("infer_ms"),
            },
        }

        with log_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._request_index += 1

    def _get_timing_log_path(self, obs: dict) -> pathlib.Path | None:
        if self._timing_log_dir is None:
            return self._timing_log_path

        prompt = str(obs.get("prompt", "unknown_task"))
        if prompt not in self._prompt_to_task_index:
            self._prompt_to_task_index[prompt] = len(self._prompt_to_task_index)

        task_index = self._prompt_to_task_index[prompt]
        task_name = _sanitize_filename(prompt)
        return self._timing_log_dir / f"{task_index:02d}_{task_name}.jsonl"


def _sanitize_filename(value: str, *, max_len: int = 96) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return (value or "unknown_task")[:max_len]


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
