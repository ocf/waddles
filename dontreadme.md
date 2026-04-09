# Waddles

## SGLang (Old)

```bash
cd ~/waddles
virtualenv venv
source venv/bin/activate
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'
CUDA_VISIBLE_DEVICES=1,2 python -m sglang.launch_server --model-path Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 30000 --tp-size 2 --mem-fraction-static 0.4 --context-length 32768 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --quantization moe_wna16
```

## vLLM

```bash
cd ~/waddles
virtualenv venv
source venv/bin/activate
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu129 --extra-index-url https://download.pytorch.org/whl/cu129
pip install git+https://github.com/huggingface/transformers.git
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,2 vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --port 30000 --tensor-parallel-size 2 --gpu-memory-utilization 0.4 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --disable-custom-all-reduce
```

## Environment

.env
```
DISCORD_TOKEN=OTI3MzkyMDQ4MTI3Mzg0NTAx.YdI2Yw.b8G9f2R_jK3L5mN8Pq1RsTh1sI5pR0b4b1YfAk3uVwXyZ
```

.watchtower.env
```
WATCHTOWER_HTTP_API_UPDATE=true
WATCHTOWER_HTTP_API_TOKEN=4f9a7d2b8c1e6f3a0d9b5c7e4a2f4ketoo8d1b6c0e9a3f7d
WATCHTOWER_CLEANUP=true
WATCHTOWER_POLL_INTERVAL=86400
WATCHTOWER_INCLUDE_RESTARTING=true
```
