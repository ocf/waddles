# Waddles

## SGLang

```bash
cd ~/waddles
virtualenv venv
source venv/bin/activate
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'
CUDA_VISIBLE_DEVICES=1,2 python -m sglang.launch_server --model-path Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --port 30000 --tp-size 2 --mem-fraction-static 0.4 --context-length 32768 --reasoning-parser qwen3 --quantization moe_wna16
```

## Environment

.env
```
DISCORD_TOKEN=OTI3MzkyMDQ4MTI3Mzg0NTAx.YdI2Yw.b8G9f2R_jK3L5mN8Pq1RsTh1sI5pR0b4b1YfAk3uVwXyZ
OWNER_USERS=446290930723717120,1023113941624295434
```

.watchtower.env
```
WATCHTOWER_HTTP_API_UPDATE=true
WATCHTOWER_HTTP_API_TOKEN=4f9a7d2b8c1e6f3a0d9b5c7e4a2f4ketoo8d1b6c0e9a3f7d
WATCHTOWER_CLEANUP=true
WATCHTOWER_POLL_INTERVAL=86400
WATCHTOWER_INCLUDE_RESTARTING=true
```
