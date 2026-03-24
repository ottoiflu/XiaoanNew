"""VLM/OCR API 调用重试机制

对 OpenAI 兼容 API 的可恢复网络错误进行指数退避重试。
仅重试瞬时网络故障（超时、连接断开、速率限制），
业务逻辑错误和认证错误不在重试范围内。
"""

from __future__ import annotations

import logging

from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (APITimeoutError, APIConnectionError, RateLimitError)

_default_retry = retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@_default_retry
def chat_completion_with_retry(client, **kwargs):
    """带自动重试的 chat completion 调用

    对 APITimeoutError / APIConnectionError / RateLimitError
    执行最多 3 次指数退避重试（2s → 4s → 8s）。
    其他异常立即抛出，不进行重试。

    Args:
        client: OpenAI 兼容客户端实例
        **kwargs: 传递给 client.chat.completions.create() 的全部参数
    Returns:
        API 响应对象
    """
    return client.chat.completions.create(**kwargs)
