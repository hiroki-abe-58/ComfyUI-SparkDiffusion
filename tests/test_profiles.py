import dataclasses

import pytest

from conftest import load

P = load("spark_runtime.profiles")

# key -> (model_size, upstream --model_size, task, resolution, steps, topk, sparsity, checkpoint count)
EXPECTED = {
    "wan2.1-t2v-1.3b-480p-s90-4step": ("1.3B", "1.3B_rola", "t2v", "480p", 4, 0.10, 0.90, 1),
    "wan2.1-t2v-14b-480p-s90-4step": ("14B", "14B_rola", "t2v", "480p", 4, 0.10, 0.90, 1),
    "wan2.1-t2v-14b-720p-s95-3step": ("14B", "14B_rola", "t2v", "720p", 3, 0.05, 0.95, 1),
    "wan2.1-t2v-14b-720p-s97-4step": ("14B", "14B_rola", "t2v", "720p", 4, 0.03, 0.97, 1),
    "wan2.1-i2v-14b-720p-s97-4step": ("14B", "14B_rola", "i2v", "720p", 4, 0.03, 0.97, 1),
    "wan2.2-t2v-a14b-480p-s95-4step": ("A14B", "A14B_rola", "t2v", "480p", 4, 0.05, 0.95, 2),
}


def test_profile_set_is_complete():
    assert set(P.PROFILES) == set(EXPECTED)
    assert P.DEFAULT_PROFILE_KEY in P.PROFILES


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_profile_mappings(key):
    size, upstream, task, res, steps, topk, sparsity, n_ckpt = EXPECTED[key]
    p = P.get_profile(key)
    assert p.model_size == size
    assert p.upstream_model_size == upstream
    assert p.task == task
    assert p.resolution == res
    assert p.num_steps == steps
    assert p.topk_ratio == pytest.approx(topk)
    assert p.sparsity == pytest.approx(sparsity)
    assert len(p.checkpoint_files) == n_ckpt
    assert p.scheduler == f"crossdistill-{steps}step"
    assert p.hf_repo.startswith("alibabagroup/SparkWan")
    for f in p.checkpoint_files:
        assert f.endswith(".pth")
    # the checkpoint file name encodes resolution and sparsity
    assert res.upper() in p.checkpoint_files[0]
    assert f"{sparsity:.2f}Sparsity" in p.checkpoint_files[0]


def test_model_size_matches_entrypoint():
    for p in P.PROFILES.values():
        if p.family == "wan2.2":
            assert p.entrypoint.endswith("wan2pt2_t2v_distilled_infer.py")
        elif p.task == "i2v":
            assert p.entrypoint.endswith("wan2pt1_i2v_distilled_infer.py")
            assert p.needs_clip
        else:
            assert p.entrypoint.endswith("wan2pt1_t2v_distilled_infer.py")
            assert not p.needs_clip


def test_required_assets():
    t2v = P.get_profile("wan2.1-t2v-1.3b-480p-s90-4step")
    i2v = P.get_profile("wan2.1-i2v-14b-720p-s97-4step")
    assert P.required_assets(t2v) == [P.VAE_FILE, P.T5_FILE, P.TOKENIZER_DIR]
    assert P.CLIP_FILE in P.required_assets(i2v)


def test_topk_range_validation():
    p = P.get_profile("wan2.1-t2v-14b-720p-s97-4step")
    assert p.validate_topk(0.05) == 0.05
    assert p.validate_topk(0.1) == 0.1
    with pytest.raises(ValueError):
        p.validate_topk(0.02)
    with pytest.raises(ValueError):
        p.validate_topk(0.2)
    one = P.get_profile("wan2.1-t2v-1.3b-480p-s90-4step")
    with pytest.raises(ValueError):
        one.validate_topk(0.05)


def test_sparsity_topk_conversion():
    assert P.sparsity_to_topk(0.97) == pytest.approx(0.03)
    assert P.topk_to_sparsity(0.05) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        P.sparsity_to_topk(1.0)
    with pytest.raises(ValueError):
        P.topk_to_sparsity(0.0)


@pytest.mark.parametrize("n,ok", [(81, True), (5, True), (33, True), (80, False), (4, False), (1, False)])
def test_num_frames_validation(n, ok):
    if ok:
        assert P.validate_num_frames(n) == n
    else:
        with pytest.raises(ValueError):
            P.validate_num_frames(n)


def test_sizes():
    assert P.get_profile("wan2.1-t2v-1.3b-480p-s90-4step").size("16:9") == (832, 480)
    assert P.get_profile("wan2.1-t2v-14b-720p-s95-3step").size("16:9") == (1280, 720)
    assert P.get_profile("wan2.1-t2v-14b-720p-s95-3step").size("9:16") == (720, 1280)


def test_wan22_timesteps_are_concatenated():
    p = P.get_profile("wan2.2-t2v-a14b-480p-s95-4step")
    ts = p.timesteps()
    assert ts[2] == pytest.approx(0.875)
    assert len(ts) == 5 and ts[-1] == 0.0


def test_override_keeps_profile_valid():
    p = dataclasses.replace(P.get_profile("wan2.1-t2v-14b-720p-s95-3step"), topk_ratio=0.1)
    assert p.sparsity == pytest.approx(0.9)
    assert p.as_dict()["sparsity"] == pytest.approx(0.9)


def test_unknown_profile():
    with pytest.raises(KeyError):
        P.get_profile("nope")


def test_statuses_are_known():
    allowed = {P.STATUS_TESTED, P.STATUS_IMPLEMENTED, P.STATUS_EXPERIMENTAL, P.STATUS_PLANNED}
    assert all(p.status in allowed for p in P.PROFILES.values())
