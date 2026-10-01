import time
import requests
from huggingface_hub import snapshot_download

attempt = 0
while True:
    attempt += 1
    print(f"START attempt {attempt}", flush=True)
    try:
        path = snapshot_download(
            repo_id="robotflow/vr-folding",
            repo_type="dataset",
            allow_patterns=["data/folding/*"],
            local_dir="/home/agilex/ljm/dyf/vr_folding_download",
            max_workers=2,
        )
        print(f"DOWNLOAD COMPLETE: {path}", flush=True)
        break
    except requests.exceptions.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None and status < 500 and status not in (408, 429):
            raise
        delay = min(30 * attempt, 300)
        print(f"NETWORK ERROR: {type(exc).__name__}: {exc}", flush=True)
        print(f"Retrying in {delay} seconds", flush=True)
        time.sleep(delay)
