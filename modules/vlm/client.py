"""VLM API 客户端池管理

提供 OpenAI 兼容客户端的资源池创建和 Round-Robin 任务分发。
API 密钥统一从 modules/config/settings.py 加载，避免硬编码。
"""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

from modules.config import settings


def create_client_pool(
    base_url: Optional[str] = None,
    api_keys: Optional[list[str]] = None,
) -> list[OpenAI]:
    """创建 OpenAI 兼容客户端池

    Args:
        base_url: API 端点，默认从环境变量加载
        api_keys: 密钥列表，默认从环境变量加载
    """
    url = base_url or settings.API_BASE_URL
    keys = api_keys or settings.VLM_API_KEYS
    if not keys:
        raise ValueError("未配置 VLM API 密钥，请设置 VLM_API_KEYS 环境变量")
    return [OpenAI(base_url=url, api_key=k) for k in keys]


def distribute_tasks(
    items: list,
    clients: list[OpenAI],
    extra_args: tuple = (),
) -> list[tuple]:
    """Round-Robin 将任务项均匀分配到客户端池

    Args:
        items: 待处理元素列表（每项为 (image_name, folder_path) 或其他元组）
        clients: 客户端池
        extra_args: 附加到每个任务末尾的参数
    Returns:
        [(item..., client, *extra_args), ...]
    """
    tasks = []
    for i, item in enumerate(items):
        client = clients[i % len(clients)]
        if isinstance(item, tuple):
            tasks.append((*item, client, *extra_args))
        else:
            tasks.append((item, client, *extra_args))
    return tasks
