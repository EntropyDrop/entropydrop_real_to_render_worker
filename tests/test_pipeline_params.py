from types import SimpleNamespace

import pytest

import tasks


PIPELINE_PAYLOAD = {
    "prompt_file": "prompt.txt",
    "template_files": ["t1.png", "t2.png", "t3.png"],
    "provider_model": "provider-model",
    "image_size": "1K",
    "aspect_ratio": "1:1",
    "dense_uv_checkpoint_file": "model.pt",
    "DMR_mappings_dir": "mappings",
}


def test_pipeline_params_are_built_only_from_task_payload():
    pipeline = tasks.SkinPipelineParams.from_payload(PIPELINE_PAYLOAD)

    assert pipeline.prompt_file == "prompt.txt"
    assert pipeline.template_files == ("t1.png", "t2.png", "t3.png")
    assert pipeline.to_payload() == PIPELINE_PAYLOAD


def test_stage1_assets_use_worker_root_directories(monkeypatch, tmp_path):
    prompts_root = tmp_path / "custom-prompts"
    templates_root = tmp_path / "custom-templates"
    prompts_root.mkdir()
    templates_root.mkdir()
    (prompts_root / "prompt.txt").write_text("render", encoding="utf-8")
    for filename in ("t1.png", "t2.png", "t3.png"):
        (templates_root / filename).write_bytes(b"template")
    monkeypatch.setattr(
        tasks,
        "get_settings",
        lambda: SimpleNamespace(
            prompts_root_dir=str(prompts_root),
            templates_root_dir=str(templates_root),
        ),
    )

    prompt, template_paths = tasks.load_stage1_assets(
        tasks.SkinPipelineParams.from_payload(PIPELINE_PAYLOAD)
    )

    assert prompt == "render"
    assert [path.parent for path in template_paths] == [templates_root] * 3


def test_stage1_rejects_resource_path_traversal(monkeypatch, tmp_path):
    payload = {**PIPELINE_PAYLOAD, "prompt_file": "../prompt.txt"}
    monkeypatch.setattr(
        tasks,
        "get_settings",
        lambda: SimpleNamespace(
            prompts_root_dir=str(tmp_path),
            templates_root_dir=str(tmp_path),
        ),
    )

    with pytest.raises(ValueError, match="bare filename"):
        tasks.load_stage1_assets(tasks.SkinPipelineParams.from_payload(payload))
