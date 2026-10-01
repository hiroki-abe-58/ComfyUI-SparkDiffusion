"""Helpers shared by the nodes. ComfyUI modules are imported lazily so the
package stays importable (and testable) outside ComfyUI."""

from __future__ import annotations

import importlib
import os
from typing import Callable, List, Optional, Tuple

from ..spark_runtime.command import GenerationRequest
from ..spark_runtime.process import SparkCancelled
from ..spark_runtime.profiles import Profile
from ..spark_runtime.runner import RunResult, generate

CATEGORY = "SparkDiffusion"
RUNTIME_TYPE = "SPARKDIFFUSION_RUNTIME"
PROFILE_TYPE = "SPARKDIFFUSION_PROFILE"
MAX_SEED = 0xFFFFFFFF


def output_directory() -> str:
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except ImportError:
        return os.path.abspath("output")


def temp_directory() -> Optional[str]:
    try:
        import folder_paths

        path = folder_paths.get_temp_directory()
        os.makedirs(path, exist_ok=True)
        return path
    except ImportError:
        return None


def split_prefix(filename_prefix: str, base_dir: str) -> Tuple[str, str, str]:
    """Return ``(full_output_dir, subfolder, name)``; refuse paths escaping ``base_dir``."""
    prefix = (filename_prefix or "sparkdiffusion").replace("\\", "/").strip("/")
    subfolder, _, name = prefix.rpartition("/")
    full = os.path.abspath(os.path.join(base_dir, subfolder))
    base = os.path.abspath(base_dir)
    if os.path.commonpath([full, base]) != base:
        raise ValueError(f"filename_prefix {filename_prefix!r} points outside the output directory")
    return full, subfolder, name or "sparkdiffusion"


def interrupted() -> bool:
    try:
        import comfy.model_management as mm

        return bool(mm.processing_interrupted())
    except ImportError:
        return False


def make_progress(total: int) -> Callable[[dict], None]:
    try:
        import comfy.utils

        bar = comfy.utils.ProgressBar(total)
    except ImportError:
        bar = None
    state = {"done": 0}

    def on_event(ev: dict) -> None:
        if ev.get("event") != "progress" or bar is None:
            return
        state["done"] = min(total, state["done"] + 1)
        bar.update_absolute(state["done"], total)

    return on_event


def run_generation(runtime, profile: Profile, req: GenerationRequest, filename_prefix: str) -> RunResult:
    base = output_directory()
    out_dir, _, name = split_prefix(filename_prefix, base)
    try:
        return generate(
            runtime,
            profile,
            req,
            out_dir,
            filename_prefix=name,
            work_root=temp_directory(),
            cancel_check=interrupted,
            on_event=make_progress(profile.num_steps * req.num_samples),
        )
    except SparkCancelled:
        try:
            import comfy.model_management as mm

            raise mm.InterruptProcessingException()
        except ImportError:
            raise


def ui_for(paths: List[str]) -> dict:
    base = os.path.abspath(output_directory())
    items = []
    for p in paths:
        rel = os.path.relpath(os.path.dirname(os.path.abspath(p)), base)
        items.append(
            {
                "filename": os.path.basename(p),
                "subfolder": "" if rel == "." else rel.replace("\\", "/"),
                "type": "output",
            }
        )
    return {"images": items, "animated": (True,)}


def video_output(path: str):
    """A ComfyUI ``VIDEO`` object when the running ComfyUI supports it, else None."""
    for mod, attr in (("comfy_api.latest", "InputImpl"), ("comfy_api.input_impl", None)):
        try:
            module = importlib.import_module(mod)
            impl = getattr(module, attr) if attr else module
            return impl.VideoFromFile(path)
        except (ImportError, AttributeError):
            continue
    return None


def load_frames(path: str, all_frames: bool):
    """Decode the MP4 into an ``IMAGE`` tensor ``[N, H, W, 3]`` in ``[0, 1]``."""
    import numpy as np
    import torch

    frames = []
    try:
        import av

        with av.open(path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if not all_frames:
                    break
    except ImportError:
        import imageio.v3 as iio  # type: ignore[import-not-found]

        arr = iio.imread(path)
        frames = list(arr if all_frames else arr[:1])
    stacked = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)
