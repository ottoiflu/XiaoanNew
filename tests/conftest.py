"""共享 pytest fixtures"""

import csv

import numpy as np
import pytest
import yaml
from PIL import Image


@pytest.fixture()
def tmp_dir(tmp_path):
    """提供临时目录路径（字符串）"""
    return str(tmp_path)


@pytest.fixture()
def sample_image_rgb():
    """生成 100x100 RGB numpy 数组"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture()
def sample_image_pil(sample_image_rgb):
    """生成 PIL Image 对象"""
    return Image.fromarray(sample_image_rgb)


@pytest.fixture()
def sample_image_path(tmp_path, sample_image_pil):
    """将样本图片保存到临时路径并返回路径字符串"""
    p = tmp_path / "test_img.jpg"
    sample_image_pil.save(str(p))
    return str(p)


@pytest.fixture()
def sample_rgba_image():
    """生成带 alpha 通道的 RGBA PIL 图片"""
    arr = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture()
def binary_mask_pair():
    """生成两个有部分重叠的二值 mask"""
    m1 = np.zeros((100, 100), dtype=bool)
    m2 = np.zeros((100, 100), dtype=bool)
    m1[10:60, 10:60] = True
    m2[30:80, 30:80] = True
    return m1, m2


@pytest.fixture()
def scoring_config_yaml(tmp_path):
    """生成一个合法的 scoring YAML 配置文件并返回路径"""
    data = {
        "score_map": {
            "composition": {"[合规]": 1.0, "[不合规-构图]": 0.0},
            "angle": {"[合规]": 1.0, "[不合规-角度]": 0.0},
            "distance": {"[完全合规]": 1.0, "[不合规-超界]": 0.0},
            "context": {"[合规]": 1.0, "[不合规-环境]": 0.0},
        },
        "weights": {
            "composition": 0.1,
            "angle": 0.3,
            "distance": 0.3,
            "context": 0.3,
        },
        "threshold": 0.5,
        "composition_gate": True,
    }
    p = tmp_path / "scoring.yaml"
    with open(str(p), "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return str(p)


@pytest.fixture()
def sample_prompt_dir(tmp_path):
    """创建带若干提示词 YAML 的临时目录"""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    p1 = {
        "name": "test_prompt",
        "version": "1.0",
        "description": "A test prompt",
        "content": "你是一名测试员。",
        "author": "tester",
        "created": "2026-01-01",
    }
    with open(str(prompts_dir / "test_prompt.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(p1, f, allow_unicode=True)

    p2 = {"version": "1.0", "description": "no name or content"}
    with open(str(prompts_dir / "bad_prompt.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(p2, f, allow_unicode=True)

    return str(prompts_dir)


@pytest.fixture()
def sample_labels_dir(tmp_path):
    """创建带 labels.txt 和若干图片的临时目录"""
    d = tmp_path / "images"
    d.mkdir()

    with open(str(d / "labels.txt"), "w", encoding="utf-8") as f:
        f.write("img001.jpg, yes\n")
        f.write("img002.jpg, no\n")
        f.write("img003.jpg, 合规\n")

    for name in ("img001.jpg", "img002.jpg", "img003.jpg", "img004.jpg"):
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(str(d / name))

    return str(d)


@pytest.fixture()
def experiment_csv(tmp_path):
    """生成实验结果 CSV 用于 scoring 批量评估"""
    p = tmp_path / "results.csv"
    headers = ["image", "ground_truth", "composition", "angle", "distance", "context"]
    rows = [
        ["img1.jpg", "yes", "[合规]", "[合规]", "[完全合规]", "[合规]"],
        ["img2.jpg", "no", "[不合规-构图]", "[合规]", "[完全合规]", "[合规]"],
        ["img3.jpg", "yes", "[合规]", "[不合规-角度]", "[完全合规]", "[合规]"],
        ["img4.jpg", "no", "[合规]", "[合规]", "[不合规-超界]", "[不合规-环境]"],
    ]
    with open(str(p), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return str(p)


@pytest.fixture()
def flask_app():
    """创建 Flask 测试客户端 (mock 掉模型加载)"""
    import os
    from unittest.mock import MagicMock, patch

    mock_ai = MagicMock()
    mock_ai.predict_memory.return_value = __import__("io").BytesIO(b"\x89PNG\r\n\x1a\n")
    mock_ai.predict_static_json.return_value = {
        "detections": [
            {"id": 1, "label": "Electric bike", "confidence": 0.92, "bbox": [0, 0, 10, 10], "area_ratio": 0.1}
        ],
        "mask_base64": "AAAA",
    }

    with (
        patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "testing",
                "VLM_API_KEYS": "k1",
                "OCR_API_KEY": "ocr",
                "API_BASE_URL": "https://test.com",
                "VLM_MODEL": "m",
                "OCR_MODEL": "om",
            },
        ),
        patch("openai.OpenAI", return_value=MagicMock()),
    ):
        # 将 ai_engine 注入
        import app as app_module

        app_module.ai_engine = mock_ai
        app_module.ocr_client = MagicMock()
        app_module.app.config["TESTING"] = True
        yield app_module.app.test_client(), mock_ai, app_module
