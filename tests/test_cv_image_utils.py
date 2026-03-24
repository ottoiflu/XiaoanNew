"""modules.cv.image_utils 单元测试"""

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules.cv.image_utils import (
    calculate_iou_and_overlap,
    combine_masks,
    draw_wireframe_visual,
    encode_image_to_base64,
)


class TestEncodeImageToBase64:
    def test_from_file_str(self, sample_image_path):
        r = encode_image_to_base64(sample_image_path)
        assert len(base64.b64decode(r)) > 0

    def test_from_pathlib(self, sample_image_path):
        assert isinstance(encode_image_to_base64(Path(sample_image_path)), str)

    def test_from_numpy(self, sample_image_rgb):
        assert isinstance(encode_image_to_base64(sample_image_rgb), str)

    def test_from_pil(self, sample_image_pil):
        assert isinstance(encode_image_to_base64(sample_image_pil), str)

    def test_rgba(self, sample_rgba_image):
        assert isinstance(encode_image_to_base64(sample_rgba_image), str)

    def test_bad_type(self):
        with pytest.raises(TypeError, match="不支持"):
            encode_image_to_base64(12345)

    def test_max_size(self):
        big = np.zeros((2000, 3000, 3), dtype=np.uint8)
        r = encode_image_to_base64(big, max_size=(100, 100))
        img = Image.open(io.BytesIO(base64.b64decode(r)))
        assert img.width <= 100 and img.height <= 100

    def test_single_pixel(self):
        tiny = np.array([[[255, 0, 0]]], dtype=np.uint8)
        assert isinstance(encode_image_to_base64(tiny), str)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/image.jpg")

    def test_grayscale(self):
        gray = Image.fromarray(np.zeros((50, 50), dtype=np.uint8), mode="L")
        assert isinstance(encode_image_to_base64(gray), str)


class TestCalculateIouAndOverlap:
    def test_identical(self):
        m = np.ones((50, 50), dtype=bool)
        iou, overlap = calculate_iou_and_overlap(m, m)
        assert iou == 1.0 and overlap == 1.0

    def test_no_overlap(self):
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[:50, :] = True
        m2[50:, :] = True
        iou, ov = calculate_iou_and_overlap(m1, m2)
        assert iou == 0.0 and ov == 0.0

    def test_partial(self, binary_mask_pair):
        iou, ov = calculate_iou_and_overlap(*binary_mask_pair)
        assert 0 < iou < 1 and 0 < ov < 1

    def test_both_empty(self):
        m = np.zeros((10, 10), dtype=bool)
        assert calculate_iou_and_overlap(m, m) == (0, 0)

    def test_m1_subset(self):
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[20:40, 20:40] = True
        m2[10:50, 10:50] = True
        _, ov = calculate_iou_and_overlap(m1, m2)
        assert ov == 1.0


class TestCombineMasks:
    def test_single(self):
        m = np.ones((10, 10), dtype=bool)
        r = combine_masks([{"label": "bike", "mask": m}], "bike")
        assert np.array_equal(r, m)

    def test_union(self):
        m1 = np.zeros((10, 10), dtype=bool)
        m2 = np.zeros((10, 10), dtype=bool)
        m1[:5] = True
        m2[5:] = True
        r = combine_masks([{"label": "b", "mask": m1}, {"label": "b", "mask": m2}], "b")
        assert r.sum() == 100

    def test_filter(self):
        r = combine_masks(
            [
                {"label": "a", "mask": np.ones((5, 5), dtype=bool)},
                {"label": "b", "mask": np.ones((5, 5), dtype=bool)},
            ],
            "a",
        )
        assert r is not None

    def test_no_match(self):
        assert combine_masks([{"label": "a", "mask": np.ones((5, 5), dtype=bool)}], "b") is None

    def test_empty(self):
        assert combine_masks([], "x") is None

    def test_none_mask_skipped(self):
        r = combine_masks(
            [
                {"label": "a", "mask": None},
                {"label": "a", "mask": np.ones((3, 3), dtype=bool)},
            ],
            "a",
        )
        assert r is not None and r.shape == (3, 3)


class TestDrawWireframeVisual:
    def test_basic(self, sample_image_rgb):
        m = np.zeros((100, 100), dtype=bool)
        m[20:80, 20:80] = True
        r = draw_wireframe_visual(sample_image_rgb, [{"label": "Electric bike", "mask": m}])
        assert r.shape == sample_image_rgb.shape

    def test_empty_objects(self, sample_image_rgb):
        r = draw_wireframe_visual(sample_image_rgb, [])
        assert r.shape == sample_image_rgb.shape

    def test_none_mask(self, sample_image_rgb):
        r = draw_wireframe_visual(sample_image_rgb, [{"label": "x", "mask": None}])
        assert r.shape == sample_image_rgb.shape

    def test_no_modify_input(self, sample_image_rgb):
        orig = sample_image_rgb.copy()
        m = np.ones((100, 100), dtype=bool)
        draw_wireframe_visual(sample_image_rgb, [{"label": "Electric bike", "mask": m}])
        assert np.array_equal(sample_image_rgb, orig)
