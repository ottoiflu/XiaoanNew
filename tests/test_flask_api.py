"""Flask API 端点集成测试 (mock 模型)"""

import io
from unittest.mock import patch

import numpy as np
from PIL import Image


def _make_image_bytes(fmt="JPEG", size=(100, 100)):
    """生成内存中的图片字节"""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


class TestHealthCheck:
    def test_ok(self, flask_app):
        client, ai, _ = flask_app
        resp = client.get("/api/health")
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_model_type(self, flask_app):
        client, _, _ = flask_app
        data = client.get("/api/health").get_json()
        assert data["model_type"] == "MagicMock"


class TestCollectUpload:
    def test_basic(self, flask_app, tmp_path):
        client, _, app_mod = flask_app
        old_root = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        try:
            resp = client.post(
                "/api/collect/upload",
                data={"file": (_make_image_bytes(), "photo.jpg"), "label": "test", "ground_truth": "yes"},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200
            assert resp.get_json()["status"] == "success"
        finally:
            app_mod.UPLOAD_ROOT = old_root

    def test_no_file(self, flask_app):
        client, _, _ = flask_app
        resp = client.post("/api/collect/upload")
        assert resp.status_code == 400

    def test_custom_path(self, flask_app, tmp_path):
        client, _, app_mod = flask_app
        old_root = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        try:
            resp = client.post(
                "/api/collect/upload",
                data={"file": (_make_image_bytes(), "p.jpg"), "custom_path": "my/folder"},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200
        finally:
            app_mod.UPLOAD_ROOT = old_root

    def test_path_traversal_stripped(self, flask_app, tmp_path):
        """确保 ../ 路径遍历被过滤"""
        client, _, app_mod = flask_app
        old_root = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        try:
            resp = client.post(
                "/api/collect/upload",
                data={"file": (_make_image_bytes(), "x.jpg"), "custom_path": "../../etc/passwd"},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200
            path = resp.get_json()["path"]
            assert ".." not in path
        finally:
            app_mod.UPLOAD_ROOT = old_root


class TestDetectMaskRealtime:
    def test_returns_png(self, flask_app):
        client, ai, _ = flask_app
        buf = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        ai.predict_memory.return_value = buf
        resp = client.post(
            "/api/segmentation/detect",
            data={"file": (_make_image_bytes(), "img.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_no_file(self, flask_app):
        client, _, _ = flask_app
        resp = client.post("/api/segmentation/detect")
        assert resp.status_code == 400

    def test_model_not_loaded(self, flask_app):
        client, _, app_mod = flask_app
        old = app_mod.ai_engine
        app_mod.ai_engine = None
        try:
            resp = client.post(
                "/api/segmentation/detect",
                data={"file": (_make_image_bytes(), "i.jpg")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 500
        finally:
            app_mod.ai_engine = old

    def test_engine_exception(self, flask_app):
        client, ai, _ = flask_app
        ai.predict_memory.side_effect = RuntimeError("GPU OOM")
        resp = client.post(
            "/api/segmentation/detect",
            data={"file": (_make_image_bytes(), "i.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500
        ai.predict_memory.side_effect = None


class TestDetectStatic:
    def test_ok(self, flask_app):
        client, _, _ = flask_app
        resp = client.post(
            "/api/segmentation/detect_static",
            data={"file": (_make_image_bytes(), "i.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert len(data["data"]["detections"]) == 1

    def test_no_file(self, flask_app):
        client, _, _ = flask_app
        assert client.post("/api/segmentation/detect_static").status_code == 400

    def test_model_not_loaded(self, flask_app):
        client, _, app_mod = flask_app
        old = app_mod.ai_engine
        app_mod.ai_engine = None
        try:
            resp = client.post(
                "/api/segmentation/detect_static",
                data={"file": (_make_image_bytes(), "i.jpg")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 500
        finally:
            app_mod.ai_engine = old


class TestCheckParking:
    def test_no_file(self, flask_app):
        client, _, _ = flask_app
        assert client.post("/api/test/check_parking").status_code == 400

    def test_no_plate(self, flask_app, tmp_path):
        """OCR 未识别到车牌"""
        client, ai, app_mod = flask_app
        old_root = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        with patch.object(app_mod, "recognize_license_plate", return_value=None):
            resp = client.post(
                "/api/test/check_parking",
                data={"file": (_make_image_bytes(), "car.jpg")},
                content_type="multipart/form-data",
            )
            data = resp.get_json()
            assert data["is_valid"] is False
            assert "车牌" in data["message"]
        app_mod.UPLOAD_ROOT = old_root

    def test_with_plate_parking_lane(self, flask_app, tmp_path):
        """识别到车牌且检测到停车线"""
        client, ai, app_mod = flask_app
        old_root = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        ai.predict_static_json.return_value = {
            "detections": [
                {"label": "parking lane", "confidence": 0.9},
                {"label": "Electric bike", "confidence": 0.8},
            ]
        }
        with patch.object(app_mod, "recognize_license_plate", return_value="京A12345"):
            resp = client.post(
                "/api/test/check_parking",
                data={"file": (_make_image_bytes(), "car.jpg")},
                content_type="multipart/form-data",
            )
            data = resp.get_json()
            assert data["is_valid"] is True
            assert data["plate_number"] == "京A12345"
            assert "停车线" in data["message"]
        app_mod.UPLOAD_ROOT = old_root

    def test_with_plate_curb(self, flask_app, tmp_path):
        """识别到车牌且检测到马路牙子"""
        client, ai, app_mod = flask_app
        old = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        ai.predict_static_json.return_value = {"detections": [{"label": "Curb", "confidence": 0.8}]}
        with patch.object(app_mod, "recognize_license_plate", return_value="沪B222"):
            data = client.post(
                "/api/test/check_parking",
                data={"file": (_make_image_bytes(), "c.jpg")},
                content_type="multipart/form-data",
            ).get_json()
            assert "马路牙子" in data["message"]
        app_mod.UPLOAD_ROOT = old

    def test_with_plate_no_markers(self, flask_app, tmp_path):
        """识别到车牌但无停车标识"""
        client, ai, app_mod = flask_app
        old = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        ai.predict_static_json.return_value = {"detections": []}
        with patch.object(app_mod, "recognize_license_plate", return_value="粤C333"):
            data = client.post(
                "/api/test/check_parking",
                data={"file": (_make_image_bytes(), "c.jpg")},
                content_type="multipart/form-data",
            ).get_json()
            assert data["is_valid"] is True
            assert "车牌清晰" in data["message"]
        app_mod.UPLOAD_ROOT = old

    def test_ocr_network_error_returns_no_plate(self, flask_app, tmp_path):
        """OCR 网络故障导致返回 None"""
        client, _, app_mod = flask_app
        old = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        with patch.object(app_mod, "recognize_license_plate", side_effect=None, return_value=None):
            data = client.post(
                "/api/test/check_parking",
                data={"file": (_make_image_bytes(), "c.jpg")},
                content_type="multipart/form-data",
            ).get_json()
            assert data["is_valid"] is False
        app_mod.UPLOAD_ROOT = old

    def test_tactile_paving_detected(self, flask_app, tmp_path):
        """检测到盲道时的行为"""
        client, ai, app_mod = flask_app
        old = app_mod.UPLOAD_ROOT
        app_mod.UPLOAD_ROOT = str(tmp_path)
        ai.predict_static_json.return_value = {
            "detections": [
                {"label": "parking lane", "confidence": 0.9},
                {"label": "Tactile paving", "confidence": 0.8},
            ]
        }
        with patch.object(app_mod, "recognize_license_plate", return_value="京D444"):
            data = client.post(
                "/api/test/check_parking",
                data={"file": (_make_image_bytes(), "c.jpg")},
                content_type="multipart/form-data",
            ).get_json()
            assert data["detections"]["tactile_paving"] is True
            assert data["is_valid"] is True
        app_mod.UPLOAD_ROOT = old
