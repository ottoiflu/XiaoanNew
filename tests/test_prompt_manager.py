"""modules.prompt.manager 单元测试"""

import pytest
import yaml

from modules.prompt.manager import Prompt, PromptManager


class TestPrompt:
    def test_fields(self):
        p = Prompt(name="test", version="1.0", description="d", content="c")
        assert p.name == "test" and p.author == "Unknown"

    def test_str(self):
        p = Prompt(name="p1", version="2.0", description="", content="x")
        assert "p1" in str(p) and "2.0" in str(p)


class TestPromptManager:
    def test_list(self, sample_prompt_dir):
        pm = PromptManager(sample_prompt_dir)
        prompts = pm.list_prompts()
        assert "test_prompt" in prompts

    def test_get_valid(self, sample_prompt_dir):
        pm = PromptManager(sample_prompt_dir)
        p = pm.get("test_prompt")
        assert p.name == "test_prompt" and p.content == "你是一名测试员。"

    def test_get_nonexistent(self, sample_prompt_dir):
        with pytest.raises(FileNotFoundError, match="不存在"):
            PromptManager(sample_prompt_dir).get("nope")

    def test_get_bad_prompt(self, sample_prompt_dir):
        with pytest.raises(ValueError, match="缺少必填字段"):
            PromptManager(sample_prompt_dir).get("bad_prompt")

    def test_cache(self, sample_prompt_dir):
        pm = PromptManager(sample_prompt_dir)
        assert pm.get("test_prompt") is pm.get("test_prompt")

    def test_reload(self, sample_prompt_dir):
        pm = PromptManager(sample_prompt_dir)
        pm.get("test_prompt")
        pm.reload()
        assert pm._cache == {}

    def test_reload_single(self, sample_prompt_dir):
        pm = PromptManager(sample_prompt_dir)
        pm.get("test_prompt")
        pm.reload("test_prompt")
        assert "test_prompt" not in pm._cache

    def test_get_content(self, sample_prompt_dir):
        assert PromptManager(sample_prompt_dir).get_content("test_prompt") == "你是一名测试员。"

    def test_info(self, sample_prompt_dir):
        info = PromptManager(sample_prompt_dir).info("test_prompt")
        assert info["name"] == "test_prompt" and info["content_length"] > 0

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "e"
        d.mkdir()
        assert PromptManager(str(d)).list_prompts() == []

    def test_nonexistent_dir(self, tmp_path):
        assert PromptManager(str(tmp_path / "nope")).list_prompts() == []

    def test_yml_extension(self, tmp_path):
        d = tmp_path / "y"
        d.mkdir()
        with open(str(d / "yt.yml"), "w", encoding="utf-8") as f:
            yaml.dump({"name": "yt", "content": "c"}, f)
        assert PromptManager(str(d)).get("yt").content == "c"

    def test_version_coercion(self, tmp_path):
        d = tmp_path / "v"
        d.mkdir()
        with open(str(d / "ver.yaml"), "w") as f:
            yaml.dump({"name": "ver", "version": 2, "content": "c"}, f)
        assert PromptManager(str(d)).get("ver").version == "2"

    def test_unicode(self, tmp_path):
        d = tmp_path / "zh"
        d.mkdir()
        with open(str(d / "中文.yaml"), "w", encoding="utf-8") as f:
            yaml.dump({"name": "中文", "content": "质检员"}, f, allow_unicode=True)
        assert "质检员" in PromptManager(str(d)).get("中文").content
