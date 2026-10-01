"""SparkDiffusion model profiles.

Each profile pins one released SparkDiffusion checkpoint to the exact upstream
settings it was distilled for: entry point, ``--model_size`` config, step
count, RoLA top-k ratio and resolution. Values come from the upstream README
model table and the Hugging Face model cards of the
``alibabagroup/SparkWan*`` repositories.

``topk_ratio`` is the RoLA *keep* ratio; sparsity is ``1 - topk_ratio``
(0.1 -> 90 %, 0.05 -> 95 %, 0.03 -> 97 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..schedules import crossdistill

# Validation status shown in the README compatibility table and in node output.
STATUS_TESTED = "tested"
STATUS_IMPLEMENTED = "implemented-untested"
STATUS_EXPERIMENTAL = "experimental"
STATUS_PLANNED = "planned"

# Native Wan asset file names (identical across the Wan 2.1 / 2.2 repositories).
VAE_FILE = "Wan2.1_VAE.pth"
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_DIR = "google/umt5-xxl"
CLIP_FILE = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

ENTRY_WAN21_T2V = "sparkdiffusion/inference/wan2pt1_t2v_distilled_infer.py"
ENTRY_WAN21_I2V = "sparkdiffusion/inference/wan2pt1_i2v_distilled_infer.py"
ENTRY_WAN22_T2V = "sparkdiffusion/inference/wan2pt2_t2v_distilled_infer.py"

ASPECT_RATIOS: Tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4")

# Upstream ``VIDEO_RES_SIZE_INFO`` ("width:height") for the resolutions the
# released checkpoints use; mirrored here so the UI can report output size.
RESOLUTION_SIZES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "480p": {"1:1": (640, 640), "4:3": (640, 480), "3:4": (480, 640), "16:9": (832, 480), "9:16": (480, 832)},
    "720p": {"1:1": (960, 960), "4:3": (960, 720), "3:4": (720, 960), "16:9": (1280, 720), "9:16": (720, 1280)},
}

# Shared Wan assets (VAE, umT5, tokenizer) can come from any native Wan repo.
SHARED_ASSET_DIRS: Tuple[str, ...] = (
    "Wan2.1-T2V-14B",
    "Wan2.1-T2V-1.3B",
    "Wan2.1-I2V-14B-720P",
    "Wan2.1-I2V-14B-480P",
    "Wan2.2-T2V-A14B",
)
CLIP_ASSET_DIRS: Tuple[str, ...] = ("Wan2.1-I2V-14B-720P", "Wan2.1-I2V-14B-480P")


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    family: str  # "wan2.1" | "wan2.2"
    task: str  # "t2v" | "i2v"
    model_size: str  # "1.3B" | "14B" | "A14B"
    upstream_model_size: str  # --model_size passed to the entry point
    resolution: str  # "480p" | "720p"
    num_steps: int
    topk_ratio: float
    topk_range: Tuple[float, float]  # (most sparse, least sparse) supported by the checkpoint
    hf_repo: str
    checkpoint_files: Tuple[str, ...]
    base_repo: str  # native Wan repo the shared assets are downloaded from
    entrypoint: str
    status: str
    notes: str = ""
    extra_assets: Tuple[str, ...] = field(default_factory=tuple)
    hf_revision: str = ""  # Hugging Face commit of the checkpoint repo validated by this project

    @property
    def sparsity(self) -> float:
        return round(1.0 - self.topk_ratio, 6)

    @property
    def scheduler(self) -> str:
        return f"crossdistill-{self.num_steps}step"

    @property
    def repo_dirname(self) -> str:
        return self.hf_repo.split("/", 1)[1]

    @property
    def needs_clip(self) -> bool:
        return CLIP_FILE in self.extra_assets

    def timesteps(self, sigma_max: float = crossdistill.SIGMA_MAX_DEFAULT) -> Tuple[float, ...]:
        """Full RF schedule. For Wan 2.2 this is the concatenation high + low."""
        if self.family == "wan2.2":
            high_n, low_n = crossdistill.wan22_split_steps(self.num_steps)
            high, low = crossdistill.wan22_timesteps(high_n, low_n, sigma_max)
            return high + low[1:]
        return crossdistill.crossdistill_timesteps(self.num_steps, sigma_max)

    def size(self, aspect_ratio: str) -> Tuple[int, int]:
        """Output ``(width, height)`` for T2V (I2V follows the input image aspect)."""
        return RESOLUTION_SIZES[self.resolution][aspect_ratio]

    def validate_topk(self, topk_ratio: float) -> float:
        lo, hi = self.topk_range
        if not (lo - 1e-9 <= topk_ratio <= hi + 1e-9):
            raise ValueError(
                f"top-k ratio {topk_ratio} is outside the range supported by {self.label} "
                f"({lo}..{hi}, i.e. {100 * (1 - hi):.0f}%..{100 * (1 - lo):.0f}% sparsity)"
            )
        return topk_ratio

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "task": self.task,
            "model_size": self.model_size,
            "upstream_model_size": self.upstream_model_size,
            "resolution": self.resolution,
            "num_steps": self.num_steps,
            "topk_ratio": self.topk_ratio,
            "sparsity": self.sparsity,
            "scheduler": self.scheduler,
            "hf_repo": self.hf_repo,
            "hf_revision": self.hf_revision,
            "checkpoint_files": list(self.checkpoint_files),
            "base_repo": self.base_repo,
            "entrypoint": self.entrypoint,
            "status": self.status,
            "required_assets": required_assets(self),
        }


# Hugging Face commits of the shared-asset repositories validated by this project.
ASSET_REVISIONS: Dict[str, str] = {
    "Wan-AI/Wan2.1-T2V-1.3B": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
    "Wan-AI/Wan2.1-I2V-14B-720P": "8823af45fcc58a8aa999a54b04be9abc7d2aac98",
}

PROFILES: Dict[str, Profile] = {}


def _register(profile: Profile) -> None:
    if profile.key in PROFILES:
        raise ValueError(f"duplicate profile {profile.key}")
    profile.validate_topk(profile.topk_ratio)
    crossdistill.crossdistill_timesteps(profile.num_steps)  # validates step count
    PROFILES[profile.key] = profile


_register(
    Profile(
        key="wan2.1-t2v-1.3b-480p-s90-4step",
        label="Wan2.1 T2V 1.3B 480P 90% 4-step",
        family="wan2.1",
        task="t2v",
        model_size="1.3B",
        upstream_model_size="1.3B_rola",
        resolution="480p",
        num_steps=4,
        topk_ratio=0.1,
        topk_range=(0.1, 0.1),
        hf_repo="alibabagroup/SparkWan2.1-T2V-1.3B-480P-0.90Sparsity",
        hf_revision="26342a92ed1e0744172eddfd71da236cb73c412f",
        checkpoint_files=("SparkWan2.1-T2V-1.3B-480P-0.90Sparsity.pth",),
        base_repo="Wan-AI/Wan2.1-T2V-1.3B",
        entrypoint=ENTRY_WAN21_T2V,
        status=STATUS_TESTED,
        notes="Smoke-test / correctness profile.",
    )
)
_register(
    Profile(
        key="wan2.1-t2v-14b-480p-s90-4step",
        label="Wan2.1 T2V 14B 480P 90% 4-step",
        family="wan2.1",
        task="t2v",
        model_size="14B",
        upstream_model_size="14B_rola",
        resolution="480p",
        num_steps=4,
        topk_ratio=0.1,
        topk_range=(0.1, 0.1),
        hf_repo="alibabagroup/SparkWan2.1-T2V-14B-480P-0.90Sparsity",
        hf_revision="1bc44b3a0193121c9fe6d57b0e7a2d475dda3a02",
        checkpoint_files=("SparkWan2.1-T2V-14B-480P-0.90Sparsity.pth",),
        base_repo="Wan-AI/Wan2.1-T2V-14B",
        entrypoint=ENTRY_WAN21_T2V,
        status=STATUS_TESTED,
    )
)
_register(
    Profile(
        key="wan2.1-t2v-14b-720p-s95-3step",
        label="Wan2.1 T2V 14B 720P 95% 3-step",
        family="wan2.1",
        task="t2v",
        model_size="14B",
        upstream_model_size="14B_rola",
        resolution="720p",
        num_steps=3,
        topk_ratio=0.05,
        topk_range=(0.05, 0.1),
        hf_repo="alibabagroup/SparkWan2.1-T2V-14B-720P-0.95Sparsity-3Step",
        hf_revision="8fd67ee46dc680d7bf111340555eadd12aed17dd",
        checkpoint_files=("SparkWan2.1-T2V-14B-720P-0.95Sparsity-3Step.pth",),
        base_repo="Wan-AI/Wan2.1-T2V-14B",
        entrypoint=ENTRY_WAN21_T2V,
        status=STATUS_TESTED,
        notes="Headline benchmark profile.",
    )
)
_register(
    Profile(
        key="wan2.1-t2v-14b-720p-s97-4step",
        label="Wan2.1 T2V 14B 720P 97% 4-step",
        family="wan2.1",
        task="t2v",
        model_size="14B",
        upstream_model_size="14B_rola",
        resolution="720p",
        num_steps=4,
        topk_ratio=0.03,
        topk_range=(0.03, 0.1),
        hf_repo="alibabagroup/SparkWan2.1-T2V-14B-720P-0.97Sparsity",
        hf_revision="1f0a6754be58650116effc2bd33db3e396026a78",
        checkpoint_files=("SparkWan2.1-T2V-14B-720P-0.97Sparsity.pth",),
        base_repo="Wan-AI/Wan2.1-T2V-14B",
        entrypoint=ENTRY_WAN21_T2V,
        status=STATUS_TESTED,
    )
)
_register(
    Profile(
        key="wan2.1-i2v-14b-720p-s97-4step",
        label="Wan2.1 I2V 14B 720P 97% 4-step",
        family="wan2.1",
        task="i2v",
        model_size="14B",
        upstream_model_size="14B_rola",
        resolution="720p",
        num_steps=4,
        topk_ratio=0.03,
        topk_range=(0.03, 0.1),
        hf_repo="alibabagroup/SparkWan2.1-I2V-14B-720P-0.97Sparsity",
        hf_revision="eb224f5db6f5fe2c2f07ab63ad0859589294e478",
        checkpoint_files=("SparkWan2.1-I2V-14B-720P-0.97Sparsity.pth",),
        base_repo="Wan-AI/Wan2.1-I2V-14B-720P",
        entrypoint=ENTRY_WAN21_I2V,
        status=STATUS_TESTED,
        extra_assets=(CLIP_FILE,),
    )
)
_register(
    Profile(
        key="wan2.2-t2v-a14b-480p-s95-4step",
        label="Wan2.2 T2V A14B 480P 95% 4-step (2+2 experts)",
        family="wan2.2",
        task="t2v",
        model_size="A14B",
        upstream_model_size="A14B_rola",
        resolution="480p",
        num_steps=4,
        topk_ratio=0.05,
        topk_range=(0.05, 0.1),
        hf_repo="alibabagroup/SparkWan2.2-T2V-14B-480P-0.95Sparsity",
        hf_revision="9f6a3fad94fa9153bcf6f46bce372961351127f0",
        checkpoint_files=(
            "SparkWan2.2-T2V-14B-480P-0.95Sparsity-High.pth",
            "SparkWan2.2-T2V-14B-480P-0.95Sparsity-Low.pth",
        ),
        base_repo="Wan-AI/Wan2.2-T2V-A14B",
        entrypoint=ENTRY_WAN22_T2V,
        status=STATUS_TESTED,
        notes="Two 14B experts swapped through CPU memory (upstream); validated with 64 GB system RAM.",
    )
)

DEFAULT_PROFILE_KEY = "wan2.1-t2v-1.3b-480p-s90-4step"


def get_profile(key: str) -> Profile:
    try:
        return PROFILES[key]
    except KeyError:
        raise KeyError(f"Unknown SparkDiffusion profile {key!r}; known: {sorted(PROFILES)}") from None


def profile_labels(task: Optional[str] = None) -> List[str]:
    return [p.key for p in PROFILES.values() if task is None or p.task == task]


def required_assets(profile: Profile) -> List[str]:
    """Relative asset names the profile needs besides its checkpoint(s)."""
    return [VAE_FILE, T5_FILE, TOKENIZER_DIR, *profile.extra_assets]


def sparsity_to_topk(sparsity: float) -> float:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    return round(1.0 - sparsity, 6)


def topk_to_sparsity(topk_ratio: float) -> float:
    if not 0.0 < topk_ratio <= 1.0:
        raise ValueError(f"top-k ratio must be in (0, 1], got {topk_ratio}")
    return round(1.0 - topk_ratio, 6)


def validate_num_frames(num_frames: int) -> int:
    """Wan VAE compresses time by 4: valid frame counts are ``4k + 1``."""
    if num_frames < 5 or (num_frames - 1) % 4 != 0:
        raise ValueError(f"num_frames must be 4k+1 and >= 5 (e.g. 33, 49, 81), got {num_frames}")
    return num_frames
