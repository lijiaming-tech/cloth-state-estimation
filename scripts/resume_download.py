"""下载 robotflow/vr-folding 的 data/folding/*（分卷压缩包，共 269 个文件）。

代理说明（重要）
----------------
本机同时存在两套代理变量，而 Python 的 requests/urllib **优先采用小写变量**：

    http_proxy / https_proxy / all_proxy   = 127.0.0.1:7897   <- Clash Verge, 已失效
    HTTP_PROXY / HTTPS_PROXY / ALL_PROXY   = 127.0.0.1:7890   <- iKuuu, 可用

旧版本因此实际走的是失效的 7897，连续 4 天只有 SSL 错误。
本版本在进程内**显式固定**为 7890，并清空 all_proxy（SOCKS 通道同样不可用），
不依赖调用方传入的环境变量。
"""

import os
import sys
import time
from pathlib import Path

# ---- 代理必须在 import huggingface_hub 之前设定 ----
PROXY = "http://127.0.0.1:7890"
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ[key] = PROXY
# all_proxy 指向失效的 SOCKS 端口，必须清掉，否则 requests 可能选中它
for key in ("all_proxy", "ALL_PROXY"):
    os.environ.pop(key, None)
# 不要把 huggingface.co 放进 NO_PROXY，否则会退化成直连（本网络直连不通）
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import requests  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

REPO_ID = "robotflow/vr-folding"
LOCAL_DIR = "/home/agilex/ljm/dyf/vr_folding_download"
FOLDING_DIR = Path(LOCAL_DIR) / "data" / "folding"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def progress():
    """返回 (文件数, 总字节)。用于在两次重试之间显示是否真的在增长。"""
    if not FOLDING_DIR.is_dir():
        return 0, 0
    files = [p for p in FOLDING_DIR.iterdir() if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main():
    log(f"PROXY={PROXY}  (all_proxy 已清除)")
    log(f"repo={REPO_ID}  local_dir={LOCAL_DIR}")
    n, b = progress()
    log(f"启动前进度: {n} 个文件, {b / 2**30:.3f} GiB")

    attempt = 0
    while True:
        attempt += 1
        log(f"START attempt {attempt}")
        try:
            path = snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["data/folding/*"],
                local_dir=LOCAL_DIR,
                max_workers=4,
            )
            n, b = progress()
            log(f"DOWNLOAD COMPLETE: {path}")
            log(f"结束时进度: {n} 个文件, {b / 2**30:.3f} GiB")
            break
        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status not in (408, 429):
                log(f"FATAL HTTP {status}, 不再重试")
                raise
            n, b = progress()
            delay = min(30 * attempt, 300)
            log(f"NETWORK ERROR: {type(exc).__name__}: {str(exc)[:200]}")
            log(f"当前进度: {n} 个文件, {b / 2**30:.3f} GiB;  {delay}s 后重试")
            time.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            n, b = progress()
            delay = min(30 * attempt, 300)
            log(f"ERROR: {type(exc).__name__}: {str(exc)[:200]}")
            log(f"当前进度: {n} 个文件, {b / 2**30:.3f} GiB;  {delay}s 后重试")
            time.sleep(delay)


if __name__ == "__main__":
    main()
