#!/usr/bin/env bash
# SGLang process management for all three judge metrics.

JUDGE_SERVER_PID=""

stop_judge_server() {
    [[ -n "${JUDGE_SERVER_PID}" ]] || return 0
    echo "[eval] Stopping the SGLang service started by this evaluation"
    kill -TERM -- "-${JUDGE_SERVER_PID}" 2>/dev/null || true
    local attempt
    for ((attempt=0; attempt<30; attempt++)); do
        if ! kill -0 -- "-${JUDGE_SERVER_PID}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    # The process group belongs exclusively to the service launched below.
    if kill -0 -- "-${JUDGE_SERVER_PID}" 2>/dev/null; then
        kill -KILL -- "-${JUDGE_SERVER_PID}" 2>/dev/null || true
    fi
    wait "${JUDGE_SERVER_PID}" 2>/dev/null || true
    JUDGE_SERVER_PID=""
}

check_judge_api() {
    "${3:-python3}" - "$1" "$2" <<'PY'
import json
import sys
import urllib.request
from urllib.parse import urlsplit, urlunsplit

endpoint, model = sys.argv[1:]
try:
    # /v1/models responds before SGLang finishes its internal warmup. Waiting
    # for /health prevents evaluation requests from starving that warmup.
    parsed = urlsplit(endpoint)
    health_url = urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
    with urllib.request.urlopen(health_url, timeout=2) as response:
        if response.status != 200:
            raise ValueError(f"SGLang health check returned HTTP {response.status}")
    url = endpoint.rsplit("/chat/completions", 1)[0] + "/models"
    with urllib.request.urlopen(url, timeout=2) as response:
        names = {row["id"] for row in json.load(response)["data"]}
    if model not in names:
        raise ValueError(f"Requested model {model!r} is not served; available: {sorted(names)}")
except Exception as error:
    print(f"[eval] API unavailable: {error}", file=sys.stderr)
    sys.exit(1)
PY
}

start_judge_server() {
    # All configuration is passed explicitly by run_eval.sh, using the active
    # environment's interpreter. No JUDGE_*/SGLANG_* exports are required.
    local python="$1" model_path="$2" model="$3" endpoint="$4" log="$5"
    local gpus="$6" tp_size="$7" startup_timeout="$8"
    local server_python server_bin server_cxx library_paths server_port
    server_python="$(command -v "${python}")" || {
        echo "[eval] Python is not executable: ${python}" >&2
        return 1
    }
    server_bin="$(cd "$(dirname "${server_python}")" && pwd)"
    # SGLang's TVM-FFI kernels require C++20. Use the server environment's
    # compiler for both host extensions and nvcc's host compilation.
    server_cxx="$(PATH="${server_bin}:${PATH}" command -v "${CXX:-c++}")" || {
        echo "[eval] A C++20 compiler must be available in the active environment." >&2
        return 1
    }
    server_port="$("${server_python}" - "${endpoint}" <<'PY'
import socket
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}
        or url.path != "/v1/chat/completions" or not url.port):
    sys.exit("Managed SGLang requires http://127.0.0.1:PORT/v1/chat/completions")
with socket.socket() as probe:
    # Match uvicorn's bind behavior so a previous judge's TIME_WAIT sockets
    # do not prevent the next judge from reusing the same port.
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", url.port))
    except OSError as error:
        sys.exit(f"SGLang port {url.port} unavailable: {error}; select another port with --port")
print(url.port)
PY
)" || return 1
    # Discover existing NVIDIA wheel libraries without modifying either environment.
    library_paths="$("${server_python}" - "${server_cxx}" <<'PY'
import site
import subprocess
import sys
from pathlib import Path

paths = []
# Compiled kernels must load the runtime supplied by the selected compiler.
# A newer compiler can otherwise compile successfully and fail at dlopen on
# older systems because their libstdc++ lacks the required GLIBCXX symbols.
for name in ("libstdc++.so.6", "libgcc_s.so.1"):
    path = Path(subprocess.check_output(
        [sys.argv[1], f"-print-file-name={name}"], text=True
    ).strip())
    if path.is_file():
        paths.append(str(path.resolve().parent))
paths.extend(str(path) for root in site.getsitepackages()
             for path in sorted((Path(root) / "nvidia").glob("*/lib")))
print(":".join(dict.fromkeys(paths)))
PY
)" || return 1
    echo "[eval] Starting ${model} with SGLang: GPUs=${gpus}, TP=${tp_size}; log=${log}"
    # Original successful deployment: Triton attention, no CPU weight offload,
    # CUDA graph disabled, custom all-reduce disabled, overlap disabled.
    setsid env \
        PATH="${server_bin}:${PATH}" \
        CUDA_VISIBLE_DEVICES="${gpus}" \
        LD_LIBRARY_PATH="${library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        NCCL_P2P_DISABLE=1 \
        CXX="${server_cxx}" \
        NVCC_PREPEND_FLAGS="-ccbin=${server_cxx}${NVCC_PREPEND_FLAGS:+ ${NVCC_PREPEND_FLAGS}}" \
        PYTHONUNBUFFERED=1 \
        "${server_python}" -m sglang.launch_server \
        --model-path "${model_path}" \
        --served-model-name "${model}" \
        --host 127.0.0.1 --port "${server_port}" \
        --tp-size "${tp_size}" \
        --context-length 8192 --mem-fraction-static 0.85 \
        --chunked-prefill-size 4096 --cpu-offload-gb 0 \
        --attention-backend triton \
        --disable-cuda-graph --disable-custom-all-reduce --disable-overlap-schedule \
        >"${log}" 2>&1 &
    JUDGE_SERVER_PID=$!
    local deadline=$((SECONDS + startup_timeout))
    local next_update=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${JUDGE_SERVER_PID}" 2>/dev/null; then
            echo "[eval] SGLang exited during startup; see ${log}" >&2
            tail -n 25 "${log}" >&2
            stop_judge_server
            return 1
        fi
        if check_judge_api "${endpoint}" "${model}" "${server_python}" 2>/dev/null; then
            echo "[eval] ${model} SGLang is ready (warmup complete, requested model verified)"
            return 0
        fi
        if (( SECONDS >= next_update )); then
            echo "[eval] Waiting for ${model} SGLang; see ${log}"
            next_update=$((SECONDS + 30))
        fi
        sleep 2
    done
    echo "[eval] SGLang startup timed out; see ${log}" >&2
    stop_judge_server
    return 1
}
