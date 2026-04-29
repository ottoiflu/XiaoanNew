"""VLM 响应解析与标签标准化

统一各脚本中分散的 VLM JSON 解析和标签归一化逻辑。
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
    # 先匹配否定关键词（更具体），避免 '不合规' 被 '合规' 误匹配
    if any(k in s for k in _NO_KEYWORDS):
        return "no"
    if any(k in s for k in _YES_KEYWORDS):
        return "yes"
    return ""


# ──────────────────────────── VLM 解析结果 ────────────────────────────


@dataclass
class VLMResult:
    """VLM 四维度解析结果"""

    composition: str = ""
    angle: str = ""
    distance: str = ""
    context: str = ""
    reason: str = ""
    suggestion: str = ""
    raw_json: Optional[dict] = field(default=None, repr=False)
    parse_error: str = ""

    @property
    def is_valid(self) -> bool:
        return not self.parse_error

    @property
    def statuses(self) -> tuple[str, str, str, str]:
        return (self.composition, self.angle, self.distance, self.context)


# ──────────────────────────── 解析函数 ────────────────────────────


def parse_vlm_response(response_text: str) -> VLMResult:
    """从 VLM 文本响应中提取结构化四维度状态

    仅负责提取，不做合规判定（判定逻辑由 ScoringEngine 处理）。
    """
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return VLMResult(parse_error="未匹配到JSON结构")

        data = json.loads(json_match.group())
        scores = data.get("scores", {})

        return VLMResult(
            composition=str(scores.get("composition_status", "")).strip(),
            angle=str(scores.get("angle_status", "")).strip(),
            distance=str(scores.get("distance_status", "")).strip(),
            context=str(scores.get("context_status", "")).strip(),
            reason=str(data.get("step_by_step_analysis", "")),
            suggestion=str(scores.get("suggestion", "")).strip(),
            raw_json=data,
        )
    except Exception as e:
        return VLMResult(parse_error=f"解析异常: {e}")
