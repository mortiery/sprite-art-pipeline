"""Minimal ComfyUI HTTP client + API-format graph builder (stdlib only).

Usage:
    from pipeline.comfy import Comfy, Graph
    g = Graph()
    ckpt = g.add("CheckpointLoaderSimple", ckpt_name="sd_xl_base_1.0.safetensors")
    ...
    g.add("SaveImage", images=img, filename_prefix="ccr/test")
    files = Comfy().run(g, "experiments/out")   # -> list of downloaded file paths
"""
from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    pass


class Graph:
    """Builds a ComfyUI API-format prompt: {node_id: {class_type, inputs}}.

    `add()` returns a NodeRef; index it (`ref[1]`) for outputs other than the first.
    NodeRefs passed as inputs are serialised to ComfyUI's ["node_id", output_index] links.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self._n = 0

    def add(self, class_type: str, _title: str | None = None, **inputs) -> "NodeRef":
        self._n += 1
        nid = str(self._n)
        node = {"class_type": class_type, "inputs": {k: _ser(v) for k, v in inputs.items()}}
        if _title:
            node["_meta"] = {"title": _title}
        self.nodes[nid] = node
        return NodeRef(nid, 0)

    def to_json(self) -> dict:
        return self.nodes

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.nodes, indent=2))


class NodeRef:
    def __init__(self, nid: str, idx: int = 0) -> None:
        self.nid, self.idx = nid, idx

    def __getitem__(self, idx: int) -> "NodeRef":
        return NodeRef(self.nid, idx)


def _ser(v):
    if isinstance(v, NodeRef):
        return [v.nid, v.idx]
    return v


class Comfy:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex

    # ---- low level -------------------------------------------------------
    def _req(self, path: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 30):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(f"HTTP {e.code} on {path}: {e.read().decode(errors='replace')[:4000]}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise ComfyError(f"ComfyUI not reachable at {self.base} ({e}). Is the server running / did it crash?") from None

    def _json(self, path: str, payload: dict | None = None, timeout: float = 30):
        data = json.dumps(payload).encode() if payload is not None else None
        hdr = {"Content-Type": "application/json"} if payload is not None else {}
        for attempt in range(4):            # GET polls occasionally return an empty body while the server is busy
            raw = self._req(path, data, hdr, timeout)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if payload is not None or attempt == 3:      # never blindly repeat a POST
                    raise ComfyError(f"non-JSON response from {path}: {raw[:200]!r}") from None
                time.sleep(1.0)

    # ---- API -------------------------------------------------------------
    def alive(self) -> bool:
        try:
            self._json("/system_stats", timeout=5)
            return True
        except ComfyError:
            return False

    def upload_image(self, path: str | Path, name: str | None = None, overwrite: bool = True) -> str:
        """Upload a local image into ComfyUI's input folder; returns the name to use in LoadImage."""
        path = Path(path)
        name = name or path.name
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n{'true' if overwrite else 'false'}".encode(),
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        res = json.loads(self._req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, 60))
        sub = res.get("subfolder") or ""
        return f"{sub}/{res['name']}" if sub else res["name"]

    def queue(self, graph: Graph | dict) -> str:
        prompt = graph.to_json() if isinstance(graph, Graph) else graph
        res = self._json("/prompt", {"prompt": prompt, "client_id": self.client_id})
        if res.get("node_errors"):
            raise ComfyError(f"node errors: {json.dumps(res['node_errors'])[:4000]}")
        return res["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 1800, poll: float = 1.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = self._json(f"/history/{prompt_id}")
            if prompt_id in hist:
                h = hist[prompt_id]
                status = h.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise ComfyError(f"execution error: {json.dumps(msgs)[:4000]}")
                if status.get("completed", True):
                    return h
            time.sleep(poll)
        raise ComfyError(f"timeout after {timeout}s waiting for {prompt_id}")

    def download(self, history: dict, out_dir: str | Path, stem: str | None = None) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for nid, out in sorted(history.get("outputs", {}).items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
            for img in out.get("images", []):
                q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
                data = self._req(f"/view?{q}", timeout=120)
                name = img["filename"]
                if stem:
                    name = f"{stem}{'' if len(files) == 0 else f'_{len(files)}'}{Path(name).suffix}"
                p = out_dir / name
                p.write_bytes(data)
                files.append(p)
        return files

    def run(self, graph: Graph | dict, out_dir: str | Path, stem: str | None = None, timeout: float = 1800,
            oom_retries: int = 2, server_wait: float = 900) -> list[Path]:
        """Queue, wait, download all SaveImage outputs. Returns local file paths.
        On a GPU out-of-memory error the models are unloaded (POST /free) and the job is retried.
        If the server disappears (crash on a model swap), wait up to `server_wait` seconds for it to be
        restarted and submit the job again."""
        t0 = time.time()
        ooms = crashes = 0
        while True:
            try:
                if getattr(self, "_prefree", False):            # see below: start every job with empty VRAM
                    self.free()
                    time.sleep(2)
                hist = self.wait(self.queue(graph), timeout)
                break
            except ComfyError as e:
                msg = str(e).lower()
                if "out of memory" in msg and ooms < oom_retries:
                    # Seen on this machine: with models of the previous job still cached, the VAE decode at the
                    # end does not fit and the whole job is lost. From now on unload before every job - a
                    # reload from the page cache costs seconds, a failed job a minute.
                    ooms += 1
                    self._prefree = True
                    time.sleep(1)
                elif "not reachable" in msg and crashes < 3:
                    crashes += 1
                    self._wait_for_server(server_wait)
                else:
                    raise
        self.last_history = hist
        files = self.download(hist, out_dir, stem)
        self.last_seconds = time.time() - t0
        return files

    def _wait_for_server(self, server_wait: float) -> None:
        """The server is gone (on this machine: ROCm page fault while a model is moved out of VRAM).
        If the environment variable COMFY_RESTART_CMD is set, run it (detached) to bring the server back;
        otherwise wait for somebody to restart it."""
        cmd = os.environ.get("COMFY_RESTART_CMD")
        time.sleep(15)                                          # let a crashing process finish dying
        if cmd and not self.alive():
            print(f"[comfy] server not reachable - starting it with COMFY_RESTART_CMD: {cmd}", flush=True)
            log = open(os.environ.get("COMFY_RESTART_LOG", "/tmp/comfyui-restart.log"), "ab")
            subprocess.Popen(cmd, shell=True, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        else:
            print(f"[comfy] server not reachable - waiting up to {server_wait:.0f}s for a restart", flush=True)
        t1 = time.time()
        while not self.alive():
            if time.time() - t1 > server_wait:
                raise ComfyError(f"ComfyUI did not come back within {server_wait:.0f}s")
            time.sleep(5)
        time.sleep(5)

    def run_video(self, sample_graph: Graph, decode_graph_for, out_dir: str | Path, timeout: float = 3600) -> list[Path]:
        """Two-step video job: run `sample_graph` (ends in SaveLatent), unload all models, then run
        `decode_graph_for(latent_name)` and download its frames. Keeps the VAE decode from running out of
        memory next to a large diffusion model."""
        t0 = time.time()
        self.run(sample_graph, out_dir, timeout=timeout)
        lat = [l for out in self.last_history.get("outputs", {}).values() for l in out.get("latents", [])]
        if not lat:
            raise ComfyError("the sampling graph saved no latent")
        name = (lat[0].get("subfolder", "") + "/" if lat[0].get("subfolder") else "") + lat[0]["filename"]
        self.free()
        time.sleep(2)                                           # the queue worker applies the unload when idle
        files = self.run(decode_graph_for(f"{name} [{lat[0].get('type', 'output')}]"), out_dir, timeout=timeout)
        self.last_seconds = time.time() - t0
        return files

    def free(self, unload_models: bool = True) -> None:
        # /free answers with an empty body
        self._req("/free", json.dumps({"unload_models": unload_models, "free_memory": True}).encode(),
                  {"Content-Type": "application/json"}, 30)
