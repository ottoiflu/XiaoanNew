"""VLM 响应解析与标签标准化

新四维：position / medium / angle / state
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────── 标签标准化 ────────────────────────────

_YES_KEYWORDS = ("yes", "true", "1", "合格", "合规", "是", "positive", "正", "[合规]")
_NO_KEYWORDS = ("no", "false", "0", "不合格", "违规", "否", "negative", "负", "不合规", "[不合格]")


def normalize_label(label: str) -> str:
    """将各种标签格式统一为 'yes' / 'no' / ''"""
    if not label:
        return ""
    s = str(label).strip().lower()
    if any(k in s for k in _NO_KEYWORDS):
        return "no"
    if any(k in s for k in _YES_KEYWORDS):
        return "yes"
    return ""


# ──────────────────────────── VLM 解析结果 ────────────────────────────


@dataclass
class VLMResult:
    """VLM 新四维解析结果：position / medium / angle / state + 置信度"""

    position: str = ""
    medium: str = ""
    angle: str = ""
    state: str = ""
    position_confidence: float = 0.5
    medium_confidence: float = 0.5
    angle_confidence: float = 0.5
    state_confidence: float = 0.5
    reason: str = ""
    raw_json: Optional[dict] = field(default=None, repr=False)
    parse_error: str = ""

    @property
    def is_valid(self) -> bool:
        return not self.parse_error

    @property
    def statuses(self) -> tuple[str, str, str, str]:
        return (self.position, self.medium, self.angle, self.state)

    @property
    def confidences(self) -> dict[str, float]:
        return {
            "position": self.position_confidence,
            "medium": self.medium_confidence,
            "angle": self.angle_confidence,
            "state": self.state_confidence,
        }


# ──────────────────────────── 解析函数 ────────────────────────────


def parse_vlm_response(response_text: str) -> VLMResult:
    """从 VLM 文本响应中提取新四维状态

    新四维：position / medium / angle / state
    对应 JSON 字段：position_status / medium_status / angle_status / state_status
    """
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return VLMResult(parse_error="未匹配到JSON结构")

        data = json.loads(json_match.group())
        scores = data.get("scores", {})

        return VLMResult(
            position=str(scores.get("position_status", "")).strip(),
            medium=str(scores.get("medium_status", "")).strip(),
            angle=str(scores.get("angle_status", "")).strip(),
            state=str(scores.get("state_status", "")).strip(),
            position_confidence=float(scores.get("position_confidence", 0.5)),
            medium_confidence=float(scores.get("medium_confidence", 0.5)),
            angle_confidence=float(scores.get("angle_confidence", 0.5)),
            state_confidence=float(scores.get("state_confidence", 0.5)),
            reason=str(data.get("step_by_step_analysis", "")),
            raw_json=data,
        )
    except Exception as e:
        return VLMResult(parse_error=f"解析异常: {e}")