"""
实验配置管理模块

功能：
1. 从 YAML 文件加载实验配置
2. 验证配置完整性
3. 保存配置到实验输出目录

用法：
    from modules.experiment.config import load_config, save_config

    config = load_config("configs/default.yaml")
    save_config(config, "/path/to/output/")
"""

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """实验配置数据类"""

    # 基本信息
    exp_name: str
    description: str = ""

    # VLM 配置
    model: str = "qwen/qwen3-vl-30b-a3b-instruct"
    prompt_id: str = "cv_enhanced_p4"

    # 图像处理
    max_size: tuple = (768, 768)
    quality: int = 80

    # 模型配置
    segmentor_weights: str = "/root/XiaoanNew/assets/weights/best.pt"
    segmentor_device: str = "cuda:0"
    conf_threshold: float = 0.6

    # 数据目录
    data_folders: List[str] = field(
        default_factory=lambda: [
            "/root/XiaoanNew/data/Compliance_test_data/no_val",
            "/root/XiaoanNew/data/Compliance_test_data/yes_val",
        ]
    )

    # 输出配置
    output_root: str = "/root/XiaoanNew/outputs/test_outputs"
    save_visuals: bool = True

    # API 配置
    max_workers: int = 15

    # 运行时生成
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d["max_size"] = list(d["max_size"])  # tuple -> list for YAML
        return d

    @property
    def exp_dir(self) -> str:
        """获取实验输出目录路径"""
        return os.path.join(self.output_root, f"exp_{self.timestamp}_{self.exp_name}")

    @property
    def vis_dir(self) -> str:
        """获取可视化目录路径"""
        return os.path.join(self.exp_dir, "visuals")


def load_config(config_path: str) -> ExperimentConfig:
    """
    从 YAML 文件加载实验配置

    Args:
        config_path: 配置文件路径

    Returns:
        ExperimentConfig 对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 处理 max_size
    if "max_size" in data and isinstance(data["max_size"], list):
        data["max_size"] = tuple(data["max_size"])

    # 创建配置对象
    config = ExperimentConfig(**data)

    return config


def save_config(config: ExperimentConfig, output_dir: Optional[str] = None) -> str:
    """
    保存配置到实验输出目录

    Args:
        config: ExperimentConfig 对象
        output_dir: 输出目录，默认为 config.exp_dir

    Returns:
        保存的配置文件路径
    """
    if output_dir is None:
        output_dir = config.exp_dir

    # 支持传入文件路径或目录路径
    if output_dir.endswith(".yaml") or output_dir.endswith(".yml"):
        config_path = output_dir
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "experiment_config.yaml")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return config_path


def create_experiment_dirs(config: ExperimentConfig) -> tuple:
    """
    创建实验目录结构

    Returns:
        (exp_dir, vis_dir) 元组
    """
    os.makedirs(config.exp_dir, exist_ok=True)
    if config.save_visuals:
        os.makedirs(config.vis_dir, exist_ok=True)

    # 保存配置副本
    save_config(config)

    return config.exp_dir, config.vis_dir


def list_configs(configs_dir: Optional[str] = None) -> List[str]:
    """列出所有可用的配置文件"""
    if configs_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        configs_dir = os.path.join(script_dir, "..", "..", "assets", "configs")

    if not os.path.exists(configs_dir):
        return []

    configs = []
    for filename in os.listdir(configs_dir):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            configs.append(filename)
    return sorted(configs)


# 命令行工具
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验配置管理工具")
    parser.add_argument(
        "action", choices=["list", "show", "create"], help="操作: list(列出), show(显示), create(创建模板)"
    )
    parser.add_argument("name", nargs="?", help="配置文件名")

    args = parser.parse_args()

    if args.action == "list":
        configs = list_configs()
        print("可用的实验配置:")
        for c in configs:
            print(f"  - {c}")

    elif args.action == "show" and args.name:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "..", "assets", "configs", args.name)
        if not config_path.endswith(".yaml"):
            config_path += ".yaml"
        config = load_config(config_path)
        print(yaml.dump(config.to_dict(), allow_unicode=True, default_flow_style=False))

    elif args.action == "create":
        # 创建默认配置模板
        config = ExperimentConfig(exp_name="my_experiment")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "..", "..", "assets", "configs", "template.yaml")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config.to_dict(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"模板已创建: {output_path}")
