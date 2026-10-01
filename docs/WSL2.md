# WSL2 backend (recommended for FP8 on Windows)

The upstream SparkDiffusion FP8 W8A8 path needs **row-wise** `torch._scaled_mm`.
Linux PyTorch CUDA builds have it on RTX 50-series (SM120); Windows PyTorch CUDA
builds do not (`Rowwise scaling is not currently supported on your device`).
Running the runtime inside WSL2 gives a Windows ComfyUI the exact upstream
Linux code path, while ComfyUI itself keeps running natively on Windows.

```
ComfyUI (Windows) ──> ComfyUI-SparkDiffusion ──> wsl.exe --exec <venv python> launcher.py
                                                   └─ official SparkDiffusion entry point (Linux)
```

Only `wsl.exe` is invoked from Windows. The prompt and every option travel in a
UTF-8 job file, Windows paths are converted to `/mnt/<drive>/...`
(any drive letter, spaces and non-ASCII names are fine), and the video is
written straight into the ComfyUI output folder.

## 1. Requirements

* Windows 11 with WSL2 (`wsl --version`) and a recent NVIDIA Windows driver.
  Do **not** install an NVIDIA driver inside the distro; WSL exposes the
  Windows driver (`/usr/lib/wsl/lib`).
* A distro, e.g. `wsl --install -d Ubuntu-24.04`.
  To keep it off the system drive you can import it instead:

  ```powershell
  wsl --import SparkDiffusion-Ubuntu D:\WSL\SparkDiffusion-Ubuntu ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz --version 2
  ```

  (rootfs images: <https://cloud-images.ubuntu.com/wsl/releases/24.04/current/>).
  A dedicated distro can be removed cleanly later with `wsl --unregister <name>`.

## 2. Runtime inside the distro

```bash
# inside WSL (as a user with sudo, or as root in a dedicated distro)
sudo apt-get update
sudo apt-get install -y gcc g++ git ffmpeg python3-dev ca-certificates curl
curl -LsSf https://astral.sh/uv/install.sh | sh          # or use python3 -m venv
sudo mkdir -p /opt/sparkdiffusion && sudo chown "$USER" /opt/sparkdiffusion
cd /opt/sparkdiffusion
uv venv venv --python 3.12
uv pip install --python venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
# requirements-runtime.txt ships with the node (adjust the path to your ComfyUI; the folder is
# custom_nodes/sparkdiffusion for Registry/Manager installs, custom_nodes/ComfyUI-SparkDiffusion for git clones)
uv pip install --python venv/bin/python -r "/mnt/c/ComfyUI/custom_nodes/sparkdiffusion/requirements-runtime.txt"
git clone https://github.com/AlibabaResearch/SparkDiffusion
```

`gcc` is required: Triton compiles a small launcher stub at runtime.

Check the GPU from inside the distro:

```bash
nvidia-smi
/opt/sparkdiffusion/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## 3. Point ComfyUI at it

On the **SparkDiffusion Runtime** node (or in `sparkdiffusion_config.json`):

| field | example |
|---|---|
| backend | `wsl2` |
| wsl_distro | `Ubuntu-24.04` |
| sparkdiffusion_repo | `/opt/sparkdiffusion/SparkDiffusion` (Linux path) |
| python | `/opt/sparkdiffusion/venv/bin/python` (Linux path) |
| model_root | `D:\AI\models\sparkdiffusion` (Windows path) or `/home/user/models` (Linux path) |

Then run `python scripts/doctor.py --backend wsl2 --wsl-distro Ubuntu-24.04 ...`
(or the **SparkDiffusion Doctor** node). It should report
`FP8 W8A8 (row-wise) ... PASS`.

## 4. Where to keep the models

* **Windows drive** (`D:\...` -> `/mnt/d/...`): simplest, shared with Windows tools.
  Reads go through WSL's 9P file sharing. In our test the 28.6 GB 14B
  checkpoint was loaded and FP8-converted in about 3 minutes from `/mnt/e`.
* **Inside the distro** (`/home/user/models`): faster first load, but the files
  live in the distro's virtual disk. The host resolves such paths via
  `\\wsl.localhost\<distro>\...` for validation.

## 5. Memory

WSL2 gives the Linux VM at most 50 % of host RAM by default. The 14B
checkpoints are memory-mapped and converted to FP8 block by block (upstream's
staged quantization), which worked with a 31 GB VM on a 64 GB host. If you see
the runtime killed by the OOM killer, raise the limit in `%UserProfile%\.wslconfig`
(this file affects all distros; edit it deliberately):

```ini
[wsl2]
memory=48GB
```

On Linux runtimes the launcher sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (unless you set it yourself)
to reduce reserved-but-unused VRAM, which matters on 32 GB cards where the
Windows desktop also uses VRAM.

## 6. Cancellation

Pressing *Cancel* in ComfyUI writes a cancel file that the launcher watches,
then kills the launcher's process group inside WSL (`wsl.exe --exec kill`) and
finally the Windows `wsl.exe` process tree. Killing `wsl.exe` alone does not
reliably stop the Linux process, which is why all three steps exist.
