# Add CT uploads to baloraapp.streamlit.app

The existing Streamlit app can accept and preview CT uploads. Segmentation uses the study's CUDA inference code on a separate NVIDIA GPU computer or server. Uploading the dashboard files enables preview as soon as Streamlit rebuilds. Connecting the GPU server enables **Generate mask**.

## 1. Update the existing Streamlit app

1. Open your `balora` GitHub repository on `main`.
2. Copy the four files inside this package's `BALoRA_Streamlit` folder into the existing `BALoRA_Streamlit/` directory. Replace `app.py` and `requirements.txt`. Add `cloud_upload.py` and `upload_scans.py`. The app uses your existing `data/` and `study_data.py`.
3. Commit the changes. The entry point remains `BALoRA_Streamlit/app.py`. Reopen your current Streamlit URL and select **Upload CT**.

You can now upload a de-identified research CT in NIfTI, PNG, or JPEG and preview it. The uploader accepts files up to 32 MB, NIfTI expands up to 128 MiB, and a selected slice can be up to 2048 pixels per side. If a volume is too large, export the target slice to a 2D NIfTI or an already-windowed 8-bit PNG.

## 2. Start the GPU service

Set up the separate `BALoRA_GPU_Backend` directory on an NVIDIA GPU computer or server, following its README. You need your completed `BALoRA_Full_Run_Results(1).zip`, a reachable HTTPS address for the service, and a shared access token. A laptop with an NVIDIA GPU can serve as the backend if you provide an HTTPS tunnel or reverse proxy reachable from Streamlit Cloud. Keep this backend running whenever mask generation is needed.

Confirm the backend is reachable by opening its HTTPS URL with `/healthz` appended. The response `{"status":"ok"}` verifies that the web server is reachable. Testing the first mask verifies that the model files and GPU are ready.

## 3. Connect Streamlit Cloud to the GPU

In the existing app's **Settings → Secrets**, add this TOML with your real HTTPS address and the same token used on the GPU computer.

```toml
[balora_gpu]
url = "https://your-gpu-service.example"
token = "paste-the-token-you-generated"
```

Keep the token in Streamlit settings and on the GPU host. The browser upload is processed by Streamlit Cloud. **Generate mask** sends only the selected, windowed slice and its prompt coordinates to the backend. Keep scans and checkpoints out of GitHub. If the dashboard is public, any visitor can use its configured backend. Set visibility to private when you want to restrict that access.

Reopen **Upload CT**, select a slice and choose a point inside the target organ using X and Y. Select **Generate mask** and download the original-grid 2D PNG.

[Streamlit Cloud secrets settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
