"""
统一配置管理模块

功能：
1. 从环境变量加载敏感配置
2. 提供类型安全的配置访问
3. 支持默认值和验证

使用方式:
    from config.settings import settings
    
    api_keys = settings.VLM_API_KEYS  # 列表
    model = settings.VLM_MODEL
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    
    # 查找项目根目录的 .env 文件
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[Settings] 已加载环境变量: {env_path}")
    else:
        print(f"[Settings] 未找到 .env 文件，使用系统环境变量")
        
except ImportError:
    print("[Settings] python-dotenv 未安装，仅使用系统环境变量")
    project_root = Path(__file__).parent.parent


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """获取环境变量，支持必填验证"""
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"必填环境变量 {key} 未设置")
    return value


def get_env_list(key: str, default: List[str] = None, sep: str = ",") -> List[str]:
    """获取列表类型环境变量（逗号分隔）"""
    value = os.getenv(key, "")
    if not value:
        return default or []
    return [v.strip() for v in value.split(sep) if v.strip()]


def get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔类型环境变量"""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """获取整数类型环境变量"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """应用配置类"""
    
    # API Keys (支持多个)
    VLM_API_KEYS: List[str]
    OCR_API_KEY: str
    
    # API 端点
    API_BASE_URL: str
    
    # 模型配置
    VLM_MODEL: str
    OCR_MODEL: str
    YOLO_WEIGHTS: str
    INFERENCE_DEVICE: str
    
    # 服务配置
    FLASK_PORT: int
    DEBUG_MODE: bool
    MAX_WORKERS: int
    
    # 路径配置
    PROJECT_ROOT: Path
    
    @property
    def VLM_API_KEY(self) -> str:
        """兼容属性：返回第一个 VLM API Key"""
        return self.VLM_API_KEYS[0] if self.VLM_API_KEYS else ""
    
    @classmethod
    def load(cls) -> "Settings":
        """从环境变量加载配置"""
        project_root = Path(__file__).parent.parent
        
        return cls(
            # API Keys - 敏感信息从环境变量获取
            VLM_API_KEYS=get_env_list("VLM_API_KEYS"),
            OCR_API_KEY=get_env("OCR_API_KEY", ""),
            
            # API 端点
            API_BASE_URL=get_env("API_BASE_URL", "https://api.ppinfra.com/openai"),
            
            # 模型配置
            VLM_MODEL=get_env("VLM_MODEL", "qwen/qwen3-vl-30b-a3b-instruct"),
            OCR_MODEL=get_env("OCR_MODEL", "qwen/qwen3-vl-8b-instruct"),
            YOLO_WEIGHTS=get_env("YOLO_WEIGHTS", str(project_root / "weights" / "best.pt")),
            INFERENCE_DEVICE=get_env("INFERENCE_DEVICE", "cuda:0"),
            
            # 服务配置
            FLASK_PORT=get_env_int("FLASK_PORT", 5000),
            DEBUG_MODE=get_env_bool("DEBUG_MODE", False),
            MAX_WORKERS=get_env_int("MAX_WORKERS", 15),
            
            # 路径
            PROJECT_ROOT=project_root,
        )
    
    def validate(self) -> list:
        """验证配置完整性，返回警告列表"""
        warnings = []
        
        if not self.VLM_API_KEYS:
            warnings.append("VLM_API_KEYS 未设置，VLM 调用将失败")
        if not self.OCR_API_KEY:
            warnings.append("OCR_API_KEY 未设置，OCR 调用将失败")
        if not Path(self.YOLO_WEIGHTS).exists():
            warnings.append(f"YOLO 权重文件不存在: {self.YOLO_WEIGHTS}")
            
        return warnings
    
    def print_status(self):
        """打印配置状态（隐藏敏感信息）"""
        def mask(s: str, show: int = 4) -> str:
            if not s or len(s) <= show:
                return "***"
            return s[:show] + "*" * (len(s) - show)
        
        print("=" * 50)
        print("[配置状态]")
        print(f"  VLM_API_KEYS: {len(self.VLM_API_KEYS)} 个")
        for i, k in enumerate(self.VLM_API_KEYS):
            print(f"    [{i}] {mask(k)}")
        print(f"  OCR_API_KEY: {mask(self.OCR_API_KEY)}")
        print(f"  API_BASE_URL: {self.API_BASE_URL}")
        print(f"  VLM_MODEL: {self.VLM_MODEL}")
        print(f"  YOLO_WEIGHTS: {self.YOLO_WEIGHTS}")
        print(f"  INFERENCE_DEVICE: {self.INFERENCE_DEVICE}")
        print(f"  MAX_WORKERS: {self.MAX_WORKERS}")
        print("=" * 50)
        
        warnings = self.validate()
        if warnings:
            print("[警告]")
            for w in warnings:
                print(f"  - {w}")


# 全局配置实例（延迟加载）
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


# 便捷访问
settings = get_settings()


if __name__ == "__main__":
    # 测试配置加载
    settings.print_status()
