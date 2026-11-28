"""Python end of ``matlab/runGuidanceBridge.m``.

The runner listens on localhost; a MATLAB process (Windows, started from WSL
through interop, or started by hand) connects and answers guidance requests.
One MATLAB session serves many runs, so its start-up and the checkpoint export
happen once per campaign.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import time as wallclock
from pathlib import Path

import numpy as np

from .paths import PIPELINE_ROOT

DEFAULT_MATLAB = os.environ.get("ARES_MATLAB", "/mnt/d/bin/matlab.exe")


class BridgeError(RuntimeError):
    pass


def windows_path(path: Path | str) -> str:
    """WSL path -> Windows path (identity when already on Windows)."""
    text = str(path)
    if os.name == "nt" or not text.startswith("/"):
        return text
    if shutil.which("wslpath"):
        return subprocess.run(["wslpath", "-w", text], check=True, capture_output=True, text=True).stdout.strip()
    raise BridgeError(f"Cannot convert {text} to a Windows path (wslpath missing).")


class MatlabBridge:
    def __init__(self, port: int = 0, matlab: str = DEFAULT_MATLAB, launch: bool = True,
                 log_file: Path | None = None, connect_timeout_s: float = 300.0, reply_timeout_s: float = 3600.0):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Windows reaches WSL services through localhost forwarding (WSL2 default).
        self.server.bind(("0.0.0.0", port))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.matlab = matlab
        self.process: subprocess.Popen | None = None
        self.connection: socket.socket | None = None
        self.reply_timeout_s = reply_timeout_s
        self.log_file = log_file or (PIPELINE_ROOT / "results" / "matlab_bridge.log")
        if launch:
            self._launch()
        else:
            print(f"Waiting for MATLAB: runGuidanceBridge(\"localhost\", {self.port})")
        self.server.settimeout(connect_timeout_s)
        try:
            self.connection, _ = self.server.accept()
        except socket.timeout as exc:
            raise BridgeError(f"MATLAB did not connect within {connect_timeout_s:.0f} s. "
                              f"See {self.log_file}") from exc
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.settimeout(reply_timeout_s)

    def _launch(self) -> None:
        if not Path(self.matlab).exists() and shutil.which(self.matlab) is None:
            raise BridgeError(f"MATLAB not found at {self.matlab}. Set ARES_MATLAB to matlab.exe.")
        # WSL interop drops double quotes and splits on spaces, so keep paths out of the
        # command: MATLAB starts in the bridge folder and derives the ARES root itself.
        command = f"runGuidanceBridge('localhost', {self.port})"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_file, "w")
        self.process = subprocess.Popen([self.matlab, "-batch", command], stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, cwd=str(PIPELINE_ROOT / "matlab"))
        print(f"Started MATLAB guidance bridge (port {self.port}); log: {self.log_file}")

    # ---- protocol --------------------------------------------------------------
    def _send(self, header: dict, payload: bytes = b"") -> None:
        data = json.dumps(header).encode("utf-8")
        self.connection.sendall(struct.pack("<I", len(data)) + data + struct.pack("<I", len(payload)) + payload)

    def _receive_exact(self, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            chunk = self.connection.recv(count - len(chunks))
            if not chunk:
                raise BridgeError(f"MATLAB closed the connection. See {self.log_file}")
            chunks.extend(chunk)
        return bytes(chunks)

    def request(self, header: dict, payload: bytes = b"") -> dict:
        self._send(header, payload)
        size = struct.unpack("<I", self._receive_exact(4))[0]
        reply = json.loads(self._receive_exact(size).decode("utf-8"))
        extra = struct.unpack("<I", self._receive_exact(4))[0]
        if extra:
            self._receive_exact(extra)
        if not reply.get("ok", False):
            raise BridgeError(f"MATLAB error: {reply.get('error')}")
        return reply

    # ---- API ---------------------------------------------------------------------
    def init(self, options: dict, model_path: str, export_python: str = "", guidance: str = "planner") -> dict:
        return self.request({"type": "init", "options": options, "model_path": model_path,
                             "export_python": export_python, "guidance": guidance})

    def step(self, time_s: float, position_ned, velocity_ned, euler_zyx, body_rates, available: bool,
             points_matlab_order: np.ndarray | None, velocity_request=None) -> dict:
        payload = b""
        if available:
            if points_matlab_order is None:
                raise BridgeError("Lidar marked available but no scan supplied.")
            payload = np.ascontiguousarray(points_matlab_order, dtype="<f4").reshape(-1, 3).tobytes()
        header = {"type": "step", "time": float(time_s),
                  "position": [float(x) for x in position_ned], "velocity": [float(x) for x in velocity_ned],
                  "euler_zyx": [float(x) for x in euler_zyx], "body_rates": [float(x) for x in body_rates],
                  "available": bool(available)}
        if velocity_request is not None:
            header["velocity_request"] = [float(x) for x in velocity_request]
        return self.request(header, payload)

    def retarget(self, goal_ned, position_ned, velocity_ned, euler_zyx) -> dict:
        """Swarm: new goal mid-flight (``retarget`` in runGuidanceBridge.m)."""
        return self.request({"type": "retarget", "goal": [float(x) for x in goal_ned],
                             "position": [float(x) for x in position_ned],
                             "velocity": [float(x) for x in velocity_ned],
                             "euler_zyx": [float(x) for x in euler_zyx]})

    def end_run(self) -> None:
        self.request({"type": "end"})

    def close(self) -> None:
        try:
            if self.connection is not None:
                self.request({"type": "quit"})
        except Exception:
            pass
        for item in (self.connection, self.server):
            try:
                if item is not None:
                    item.close()
            except Exception:
                pass
        self.connection = None
        if self.process is not None:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def __enter__(self) -> "MatlabBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
