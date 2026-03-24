"""VLM/OCR API 重试机制单元测试

验证 chat_completion_with_retry 的指数退避重试行为：
- 可恢复异常自动重试（最多 3 次）
- 不可恢复异常立即抛出
- 重试后成功返回正确结果
- 重试次数耗尽后最终异常正确抛出
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from modules.vlm.retry import RETRYABLE_EXCEPTIONS, chat_completion_with_retry


def _make_response(content="ok"):
    """构造 OpenAI 兼容的 mock 响应对象"""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


# ==================== 基础重试行为 ====================


class TestRetryOnRecoverableErrors:
    """可恢复异常触发重试"""

    @patch("modules.vlm.retry.chat_completion_with_retry.retry.sleep", side_effect=lambda x: None)
    def test_timeout_then_success(self, _wait):
        """超时一次后第二次成功"""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            _make_response("recovered"),
        ]
        result = chat_completion_with_retry(client, model="test")
        assert result.choices[0].message.content == "recovered"
        assert client.chat.completions.create.call_count == 2

    @patch("modules.vlm.retry.chat_completion_with_retry.retry.sleep", side_effect=lambda x: None)
    def test_connection_error_then_success(self, _wait):
        """连接错误一次后恢复"""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            _make_response("ok"),
        ]
        result = chat_completion_with_retry(client, model="test")
        assert result.choices[0].message.content == "ok"
        assert client.chat.completions.create.call_count == 2

    @patch("modules.vlm.retry.chat_completion_with_retry.retry.sleep", side_effect=lambda x: None)
    def test_rate_limit_then_success(self, _wait):
        """速率限制后恢复"""
        client = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}
        client.chat.completions.create.side_effect = [
            RateLimitError(
                message="rate limited",
                response=resp_429,
                body=None,
            ),
            _make_response("ok"),
        ]
        result = chat_completion_with_retry(client, model="test")
        assert result.choices[0].message.content == "ok"
        assert client.chat.completions.create.call_count == 2

    @patch("modules.vlm.retry.chat_completion_with_retry.retry.sleep", side_effect=lambda x: None)
    def test_two_failures_then_success(self, _wait):
        """连续两次失败后第三次成功（刚好在上限内）"""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            _make_response("third_try"),
        ]
        result = chat_completion_with_retry(client, model="test")
        assert result.choices[0].message.content == "third_try"
        assert client.chat.completions.create.call_count == 3


# ==================== 重试耗尽 ====================


class TestRetryExhaustion:
    """重试次数耗尽后正确抛出异常"""

    @patch("modules.vlm.retry.chat_completion_with_retry.retry.sleep", side_effect=lambda x: None)
    def test_three_timeouts_raises(self, _wait):
        """连续 3 次超时后抛出 APITimeoutError"""
        client = MagicMock()
        client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with pytest.raises(APITimeoutError):
            chat_completion_with_retry(client, model="test")
        assert client.chat.completions.create.call_count == 3

    @patch("modules.vlm.retry.chat_completion_with_retry.retry.sleep", side_effect=lambda x: None)
    def test_three_connection_errors_raises(self, _wait):
        """连续 3 次连接错误后抛出"""
        client = MagicMock()
        client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
        with pytest.raises(APIConnectionError):
            chat_completion_with_retry(client, model="test")
        assert client.chat.completions.create.call_count == 3

    @patch("modules.vlm.retry.chat_completion_with_retry.retry.sleep", side_effect=lambda x: None)
    def test_mixed_retryable_errors_exhaust(self, _wait):
        """混合可恢复异常类型耗尽重试"""
        client = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}
        client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            RateLimitError(message="limited", response=resp_429, body=None),
        ]
        with pytest.raises(RateLimitError):
            chat_completion_with_retry(client, model="test")
        assert client.chat.completions.create.call_count == 3


# ==================== 不可恢复异常 ====================


class TestNoRetryOnNonRecoverableErrors:
    """不可恢复异常立即抛出，不重试"""

    def test_auth_error_no_retry(self):
        """认证错误不触发重试"""
        client = MagicMock()
        resp_401 = MagicMock()
        resp_401.status_code = 401
        resp_401.headers = {}
        client.chat.completions.create.side_effect = AuthenticationError(
            message="invalid key",
            response=resp_401,
            body=None,
        )
        with pytest.raises(AuthenticationError):
            chat_completion_with_retry(client, model="test")
        assert client.chat.completions.create.call_count == 1

    def test_value_error_no_retry(self):
        """ValueError 不触发重试"""
        client = MagicMock()
        client.chat.completions.create.side_effect = ValueError("bad param")
        with pytest.raises(ValueError):
            chat_completion_with_retry(client, model="test")
        assert client.chat.completions.create.call_count == 1

    def test_keyboard_interrupt_no_retry(self):
        """KeyboardInterrupt 不触发重试"""
        client = MagicMock()
        client.chat.completions.create.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            chat_completion_with_retry(client, model="test")
        assert client.chat.completions.create.call_count == 1

    def test_runtime_error_no_retry(self):
        """RuntimeError 不触发重试"""
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("internal")
        with pytest.raises(RuntimeError):
            chat_completion_with_retry(client, model="test")
        assert client.chat.completions.create.call_count == 1


# ==================== 参数透传 ====================


class TestKwargsPassthrough:
    """验证所有参数正确透传给底层 API"""

    def test_kwargs_forwarded_to_create(self):
        """model/messages/max_tokens 等参数原样传递"""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response("ok")

        kwargs = {
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        chat_completion_with_retry(client, **kwargs)
        client.chat.completions.create.assert_called_once_with(**kwargs)

    def test_no_extra_args_added(self):
        """重试机制不注入额外参数"""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_response("ok")

        chat_completion_with_retry(client, model="m")
        args, kwargs = client.chat.completions.create.call_args
        assert args == ()
        assert kwargs == {"model": "m"}


# ==================== 配置验证 ====================


class TestRetryConfiguration:
    """验证重试配置常量"""

    def test_retryable_exceptions_tuple(self):
        """RETRYABLE_EXCEPTIONS 包含三种可恢复异常"""
        assert APITimeoutError in RETRYABLE_EXCEPTIONS
        assert APIConnectionError in RETRYABLE_EXCEPTIONS
        assert RateLimitError in RETRYABLE_EXCEPTIONS

    def test_retryable_exceptions_length(self):
        """仅包含 3 种异常，不多不少"""
        assert len(RETRYABLE_EXCEPTIONS) == 3

    def test_auth_error_not_retryable(self):
        """认证错误不在重试范围"""
        assert AuthenticationError not in RETRYABLE_EXCEPTIONS


# ==================== 脚本集成测试 ====================


class TestScriptIntegration:
    """验证实验脚本和 app.py 已正确集成重试函数"""

    def test_vlm_test_uses_retry(self):
        """contrast_VLM_test.py 导入了 chat_completion_with_retry"""
        with open("scripts/contrast_VLM_test.py", encoding="utf-8") as f:
            content = f.read()
        assert "from modules.vlm.retry import chat_completion_with_retry" in content
        assert "chat_completion_with_retry(" in content
        # 确保旧的直接调用已被替换
        assert "client.chat.completions.create(" not in content

    def test_vlm_cv_test_v2_uses_retry(self):
        """contrast_VLM_CV_test_v2.py 导入了 chat_completion_with_retry"""
        with open("scripts/contrast_VLM_CV_test_v2.py", encoding="utf-8") as f:
            content = f.read()
        assert "from modules.vlm.retry import chat_completion_with_retry" in content
        assert "chat_completion_with_retry(" in content
        assert "client.chat.completions.create(" not in content

    def test_app_uses_retry(self):
        """app.py 导入了 chat_completion_with_retry"""
        with open("app.py", encoding="utf-8") as f:
            content = f.read()
        assert "from modules.vlm.retry import chat_completion_with_retry" in content
        assert "chat_completion_with_retry(" in content
        assert "ocr_client.chat.completions.create(" not in content
