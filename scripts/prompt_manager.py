"""
提示词管理模块

功能：
1. 从 YAML 文件加载提示词
2. 列出所有可用提示词
3. 提供统一的提示词访问接口

用法：
    from prompt_manager import PromptManager

    pm = PromptManager()
    prompt = pm.get("cv_enhanced_p4")
    print(prompt.content)
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class Prompt:
    """提示词数据类"""

    name: str
    version: str
    description: str
    content: str
    author: str = "Unknown"
    created: str = ""

    def __str__(self):
        return f"Prompt({self.name} v{self.version})"


class PromptManager:
    """提示词管理器"""

    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = os.path.join(script_dir, "..", "prompts")

        self.prompts_dir = prompts_dir
        self._cache: Dict[str, Prompt] = {}

    def list_prompts(self) -> List[str]:
        """列出所有可用的提示词名称"""
        if not os.path.exists(self.prompts_dir):
            return []

        prompts = []
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                name = os.path.splitext(filename)[0]
                prompts.append(name)
        return sorted(prompts)

    def get(self, name: str) -> Prompt:
        if name in self._cache:
            return self._cache[name]

        yaml_path = os.path.join(self.prompts_dir, f"{name}.yaml")
        yml_path = os.path.join(self.prompts_dir, f"{name}.yml")

        if os.path.exists(yaml_path):
            filepath = yaml_path
        elif os.path.exists(yml_path):
            filepath = yml_path
        else:
            available = self.list_prompts()
            raise FileNotFoundError(f"提示词 '{name}' 不存在。可用: {available}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        required_fields = ["name", "content"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"提示词文件缺少必填字段: {field}")

        prompt = Prompt(
            name=data.get("name", name),
            version=str(data.get("version", "1.0")),
            description=data.get("description", ""),
            content=data.get("content", ""),
            author=data.get("author", "Unknown"),
            created=data.get("created", ""),
        )

        self._cache[name] = prompt
        return prompt

    def get_content(self, name: str) -> str:
        return self.get(name).content

    def reload(self, name: Optional[str] = None):
        if name is None:
            self._cache.clear()
        elif name in self._cache:
            del self._cache[name]

    def info(self, name: str) -> Dict:
        prompt = self.get(name)
        return {
            "name": prompt.name,
            "version": prompt.version,
            "description": prompt.description,
            "author": prompt.author,
            "created": prompt.created,
            "content_length": len(prompt.content),
        }


_default_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptManager()
    return _default_manager


def load_prompt(name: str) -> str:
    return get_prompt_manager().get_content(name)


def list_prompts() -> List[str]:
    return get_prompt_manager().list_prompts()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提示词管理工具")
    parser.add_argument("action", choices=["list", "show", "info"])
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()

    pm = PromptManager()

    if args.action == "list":
        for p in pm.list_prompts():
            info = pm.info(p)
            print(f"  - {p} (v{info['version']})")
    elif args.action == "show" and args.name:
        print(pm.get_content(args.name))
    elif args.action == "info" and args.name:
        info = pm.info(args.name)
        for k, v in info.items():
            print(f"{k}: {v}")
