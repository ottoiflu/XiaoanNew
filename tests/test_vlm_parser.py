"""modules.vlm.parser 单元测试"""

import json

import pytest

from modules.vlm.parser import VLMResult, normalize_label, parse_vlm_response


class TestNormalizeLabel:
    @pytest.mark.parametrize(
        "inp,exp",
        [
            ("yes", "yes"),
            ("YES", "yes"),
            ("true", "yes"),
            ("1", "yes"),
            ("合格", "yes"),
            ("合规", "yes"),
            ("是", "yes"),
            ("positive", "yes"),
            ("[合规]", "yes"),
        ],
    )
    def test_yes_labels(self, inp, exp):
        assert normalize_label(inp) == exp

    @pytest.mark.parametrize(
        "inp,exp",
        [
            ("no", "no"),
            ("NO", "no"),
            ("false", "no"),
            ("0", "no"),
            ("不合格", "no"),
            ("违规", "no"),
            ("否", "no"),
            ("negative", "no"),
            ("不合规", "no"),
            ("[不合格]", "no"),
        ],
    )
    def test_no_labels(self, inp, exp):
        assert normalize_label(inp) == exp

    def test_no_overrides_yes_substring(self):
        assert normalize_label("不合规") == "no"
        assert normalize_label("不合格") == "no"

    def test_empty_string(self):
        assert normalize_label("") == ""

    def test_unrecognized(self):
        assert normalize_label("maybe") == ""
        assert normalize_label("xyz") == ""
        # "unknown_label" contains "no" substring, so it maps to "no"
        assert normalize_label("unknown_label") == "no"

    def test_whitespace(self):
        assert normalize_label("  yes  ") == "yes"
        assert normalize_label("\tno\n") == "no"

    def test_mixed_case_chinese(self):
        assert normalize_label("合规停放") == "yes"
        assert normalize_label("违规停放") == "no"


class TestVLMResult:
    def test_defaults(self):
        r = VLMResult()
        assert r.composition == ""
        assert r.is_valid is True  # no parse_error

    def test_is_valid_false(self):
        r = VLMResult(parse_error="err")
        assert r.is_valid is False

    def test_statuses(self):
        r = VLMResult(composition="a", angle="b", distance="c", context="d")
        assert r.statuses == ("a", "b", "c", "d")


class TestParseVlmResponse:
    def _resp(self, scores, analysis="ok"):
        return json.dumps({"scores": scores, "step_by_step_analysis": analysis}, ensure_ascii=False)

    def test_valid_json(self):
        resp = self._resp(
            {
                "composition_status": "[合规]",
                "angle_status": "[合规]",
                "distance_status": "[完全合规]",
                "context_status": "[合规]",
            }
        )
        r = parse_vlm_response(resp)
        assert r.is_valid
        assert r.composition == "[合规]"
        assert r.raw_json is not None

    def test_json_in_text(self):
        text = "前缀" + self._resp({"composition_status": "[合规]"}) + "后缀"
        assert parse_vlm_response(text).is_valid

    def test_no_json(self):
        r = parse_vlm_response("普通文本")
        assert not r.is_valid
        assert "未匹配" in r.parse_error

    def test_empty(self):
        assert not parse_vlm_response("").is_valid

    def test_invalid_json(self):
        r = parse_vlm_response("{bad json}")
        assert not r.is_valid
        assert "解析异常" in r.parse_error

    def test_missing_scores(self):
        r = parse_vlm_response(json.dumps({"other": 1}))
        assert r.is_valid
        assert r.composition == ""

    def test_partial_scores(self):
        r = parse_vlm_response(self._resp({"composition_status": "[合规]"}))
        assert r.composition == "[合规]"
        assert r.angle == ""

    def test_numeric_status(self):
        r = parse_vlm_response(self._resp({"composition_status": 1}))
        assert r.composition == "1"

    def test_very_long(self):
        long = "a" * 100000 + self._resp({"composition_status": "[合规]"})
        assert parse_vlm_response(long).is_valid
