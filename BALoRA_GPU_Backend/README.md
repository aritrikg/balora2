# BALoRA GPU inference service

Use this on the GPU computer or server. Keep the backend directory, trained adapters and token on that host.

## Linux setup

Use an NVIDIA GPU with a sufficiently recent driver for the CUDA 12.8 PyTorch wheel, Python 3.12 and Git. Run these commands from the folder containing `server.py`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python setup_models.py --results "/path/to/BALoRA_Full_Run_Results(1).zip"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Python accepts either OS's absolute path to the archive. Setup checks the archive and adapter hashes, fetches the original SAM2 commit, and downloads the official backbone. Add `--backbone "/path/to/sam2.1_hiera_base_plus.pt"` if you already have a local copy of the backbone.

## Serve inference

Generate an access token on the GPU host and save it securely. The output from the first command is the token.

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
read -rsp "BALoRA API token: " BALORA_API_TOKEN
export BALORA_API_TOKEN
python -m uvicorn server:api --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 8 --no-access-log
```

On Windows PowerShell, use `$env:BALORA_API_TOKEN = Read-Host "BALoRA API token"` in place of the two shell token commands. Keep one worker so the GPU holds one active model. On a different model selection, the backend releases the previous model and loads the requested one.

Point your HTTPS reverse proxy or tunnel to `http://127.0.0.1:8000`. If a managed GPU host provides HTTPS in front of your container, bind Uvicorn to the interface and port required by that host. Configure the proxy to allow a 16 MiB request and allow enough time for first inference, including load time. Use the public HTTPS origin or service base URL in Streamlit settings.

`GET /healthz` confirms the server is up. `POST /segment` accepts an authenticated slice request from the Streamlit frontend and returns a binary PNG. The uploaded volume remains in Streamlit session memory. The GPU service receives only one selected slice and does not write it to disk. This service is for research demonstration. Use de-identified data you are authorized to process on the chosen hosts.

[PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/) · [HTTPS proxy documentation](https://fastapi.tiangolo.com/deployment/https/)
