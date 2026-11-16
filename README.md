# Qwen GGUF on two V100 32 GB GPUs with llama.cpp

Two independent `llama-server` replicas, one on each GPU, plus an async Python client and an RPS benchmark. Responses are non-streaming; reasoning is disabled using `--reasoning off` where supported by the chat template.

## Requirements

- Linux, two NVIDIA V100 GPUs (32 GB per GPU), CUDA-enabled `llama-server` installed.
- GGUF model file. By default, `start.sh` expects `/models/Qwen3.6-27B-Q4_K_M.gguf`.
- Python 3.10+.
- Check whether your locally installed `llama-server --help` supports the flags (`--reasoning`, `-fa`, `-ngl`, `--split-mode`, etc.). CLI options may vary with llama.cpp version. In particular, the precise model architecture requires a compatible recent llama.cpp build.

## Run

Set the paths for your inference host in `config.env`. `MODEL_PATH` points to your GGUF file. Set `LLAMA_CPP_DIR` to the root of your llama.cpp checkout, such as `/opt/llama.cpp`. The launcher uses `build/bin/llama-server` inside that directory. Leave `LLAMA_CPP_DIR` empty to find `llama-server` on `PATH`.

You can also set `MODEL_PATH` and `LLAMA_CPP_DIR` in the environment. To use a different build location, set `LLAMA_SERVER` to the full executable path.

```bash
pip install -r requirements.txt
chmod +x start.sh
./start.sh
```

In another terminal:

```bash
python infer.py
python bench.py --total 100 --concurrency 2
python bench.py --total 100 --concurrency 4
python bench.py --total 100 --concurrency 8
python bench.py --total 100 --concurrency 16
```

The endpoints are `http://127.0.0.1:8081` and `http://127.0.0.1:8082`.

## Memory and concurrency

Defaults: `SLOTS=2`, `CTX_PER_SLOT=32768`, so each replica uses `-np 2 -c 65536`. With the typical partitioned-slot context allocation this targets a **32K maximum per active slot**, including prompt and completion. This is a configured limit, **not** a guarantee that the model and its runtime buffers fit in a single 32 GB V100. Try `SLOTS=1` if the server runs out of memory. You can override `SLOTS` and `CTX_PER_SLOT` as environment variables.

A higher number of concurrent HTTP requests does not imply that many active inference slots; extra requests queue. The benchmark measures completed requests/sec (RPS), not merely requests accepted by HTTP.

## Notes

- These scripts assume separate whole-model replicas; no distributed computation is performed between GPUs.
- Round-robin balancing is simple; real production traffic may benefit from an in-flight-aware load balancer and retries.
- The benchmark uses short synthetic prompts; replace these with representative real workloads for meaningful estimates.
- To expose a remote API, put an authenticated reverse proxy in front of the localhost services instead of binding them directly to 0.0.0.0.
