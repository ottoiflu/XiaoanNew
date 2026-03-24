"""modules.config.settings 单元测试"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.config.settings import (
    Settings,
    get_env,
    get_env_bool,
    get_env_int,
    get_env_list,
)


class TestGetEnv:
    def test_returns_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_env("NONEXISTENT_KEY_12345", "fallback") == "fallback"

    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"MY_TEST_KEY": "hello"}):
            assert get_env("MY_TEST_KEY") == "hello"

    def test_required_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="必填环境变量"):
                get_env("MISSING_REQUIRED", required=True)

    def test_required_ok_when_present(self):
        with patch.dict(os.environ, {"PRESENT": "val"}):
            assert get_env("PRESENT", required=True) == "val"

    def test_empty_string_treated_as_missing_for_required(self):
        with patch.dict(os.environ, {"EMPTY": ""}):
            with pytest.raises(ValueError):
                get_env("EMPTY", required=True)

    def test_default_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_env("NOPE") is None


class TestGetEnvList:
    def test_basic_comma_separated(self):
        with patch.dict(os.environ, {"KEYS": "a,b,c"}):
            assert get_env_list("KEYS") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"KEYS": " a , b , c "}):
            assert get_env_list("KEYS") == ["a", "b", "c"]

    def test_empty_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_list("NOPE") == []
            assert get_env_list("NOPE", ["x"]) == ["x"]

    def test_custom_separator(self):
        with patch.dict(os.environ, {"P": "/a:/b:/c"}):
            assert get_env_list("P", sep=":") == ["/a", "/b", "/c"]

    def test_empty_items_filtered(self):
        with patch.dict(os.environ, {"KEYS": "a,,b,,,c"}):
            assert get_env_list("KEYS") == ["a", "b", "c"]

    def test_single_value(self):
        with patch.dict(os.environ, {"KEYS": "only_one"}):
            assert get_env_list("KEYS") == ["only_one"]


class TestGetEnvBool:
    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_truthy_values(self, val):
        with patch.dict(os.environ, {"FLAG": val}):
            assert get_env_bool("FLAG") is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "off", "random", ""])
    def test_falsy_values(self, val):
        with patch.dict(os.environ, {"FLAG": val}):
            assert get_env_bool("FLAG") is False

    def test_default_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_bool("NOPE") is False

    def test_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_bool("NOPE", default=True) is True


class TestGetEnvInt:
    def test_valid_int(self):
        with patch.dict(os.environ, {"PORT": "8080"}):
            assert get_env_int("PORT") == 8080

    def test_default_on_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_int("NOPE", 42) == 42

    def test_invalid_returns_default(self):
        with patch.dict(os.environ, {"PORT": "not_a_number"}):
            assert get_env_int("PORT", 5000) == 5000

    def test_negative_int(self):
        with patch.dict(os.environ, {"VAL": "-10"}):
            assert get_env_int("VAL") == -10

    def test_zero(self):
        with patch.dict(os.environ, {"VAL": "0"}):
            assert get_env_int("VAL", 99) == 0


class TestSettings:
    def _make_env(self, **overrides):
        base = {
            "ENVIRONMENT": "testing",
            "VLM_API_KEYS": "key1,key2",
            "OCR_API_KEY": "ocr_key",
            "API_BASE_URL": "https://example.com",
            "VLM_MODEL": "test-model",
            "OCR_MODEL": "ocr-model",
            "YOLO_WEIGHTS": "/tmp/fake_weights.pt",
            "INFERENCE_DEVICE": "cpu",
            "FLASK_PORT": "8080",
            "DEBUG_MODE": "false",
            "MAX_WORKERS": "4",
        }
        base.update(overrides)
        return base

    def test_load_basic(self):
        with patch.dict(os.environ, self._make_env(), clear=True):
            s = Settings.load()
            assert s.ENVIRONMENT == "testing"
            assert s.VLM_API_KEYS == ["key1", "key2"]
            assert s.FLASK_PORT == 8080
            assert s.DEBUG_MODE is False
            assert s.MAX_WORKERS == 4
            assert isinstance(s.PROJECT_ROOT, Path)

    def test_vlm_api_key_property(self):
        with patch.dict(os.environ, self._make_env(), clear=True):
            s = Settings.load()
            assert s.VLM_API_KEY == "key1"

    def test_vlm_api_key_empty(self):
        with patch.dict(os.environ, self._make_env(VLM_API_KEYS=""), clear=True):
            s = Settings.load()
            assert s.VLM_API_KEY == ""

    def test_is_development(self):
        with patch.dict(os.environ, self._make_env(ENVIRONMENT="development"), clear=True):
            s = Settings.load()
            assert s.is_development is True
            assert s.is_production is False

    def test_is_production(self):
        with patch.dict(os.environ, self._make_env(ENVIRONMENT="production"), clear=True):
            s = Settings.load()
            assert s.is_production is True

    def test_invalid_environment_falls_back(self):
        with patch.dict(os.environ, self._make_env(ENVIRONMENT="staging"), clear=True):
            s = Settings.load()
            assert s.ENVIRONMENT == "development"

    def test_validate_missing_keys(self):
        with patch.dict(os.environ, self._make_env(VLM_API_KEYS="", OCR_API_KEY=""), clear=True):
            s = Settings.load()
            warnings = s.validate()
            assert any("VLM_API_KEYS" in w for w in warnings)
            assert any("OCR_API_KEY" in w for w in warnings)

    def test_validate_missing_weights(self):
        with patch.dict(os.environ, self._make_env(YOLO_WEIGHTS="/nonexistent/path.pt"), clear=True):
            s = Settings.load()
            warnings = s.validate()
            assert any("YOLO" in w for w in warnings)

    def test_validate_all_ok(self, tmp_path):
        weights = tmp_path / "w.pt"
        weights.touch()
        with patch.dict(os.environ, self._make_env(YOLO_WEIGHTS=str(weights)), clear=True):
            s = Settings.load()
            assert len(s.validate()) == 0

    def test_print_status_no_crash(self, capsys):
        with patch.dict(os.environ, self._make_env(), clear=True):
            s = Settings.load()
            s.print_status()
            out = capsys.readouterr().out
            assert "配置状态" in out
