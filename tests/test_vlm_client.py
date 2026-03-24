"""modules.vlm.client 单元测试"""

from unittest.mock import MagicMock, patch

import pytest

from modules.vlm.client import create_client_pool, distribute_tasks


class TestCreateClientPool:
    def test_creates_correct_count(self):
        clients = create_client_pool(base_url="https://example.com", api_keys=["k1", "k2", "k3"])
        assert len(clients) == 3

    def test_single_key(self):
        clients = create_client_pool(base_url="https://example.com", api_keys=["k1"])
        assert len(clients) == 1

    def test_empty_keys_raises(self):
        mock_settings = MagicMock()
        mock_settings.API_BASE_URL = "https://test.com"
        mock_settings.VLM_API_KEYS = []
        with patch("modules.vlm.client.settings", mock_settings):
            with pytest.raises(ValueError, match="未配置"):
                create_client_pool(base_url="https://example.com", api_keys=[])

    def test_uses_settings_defaults(self):
        mock_settings = MagicMock()
        mock_settings.API_BASE_URL = "https://test.com"
        mock_settings.VLM_API_KEYS = ["default_key"]
        with patch("modules.vlm.client.settings", mock_settings):
            clients = create_client_pool()
            assert len(clients) == 1


class TestDistributeTasks:
    def _clients(self, n):
        return [MagicMock(name=f"c{i}") for i in range(n)]

    def test_round_robin(self):
        cs = self._clients(2)
        items = [("a",), ("b",), ("c",), ("d",)]
        tasks = distribute_tasks(items, cs)
        assert tasks[0][1] is cs[0]
        assert tasks[1][1] is cs[1]
        assert tasks[2][1] is cs[0]

    def test_single_client(self):
        cs = self._clients(1)
        tasks = distribute_tasks([("a",), ("b",)], cs)
        assert all(t[1] is cs[0] for t in tasks)

    def test_tuple_unpacked(self):
        cs = self._clients(1)
        tasks = distribute_tasks([("img.jpg", "/path")], cs)
        assert tasks[0] == ("img.jpg", "/path", cs[0])

    def test_non_tuple(self):
        cs = self._clients(1)
        tasks = distribute_tasks(["img.jpg"], cs)
        assert tasks[0] == ("img.jpg", cs[0])

    def test_extra_args(self):
        cs = self._clients(1)
        tasks = distribute_tasks([("img.jpg",)], cs, extra_args=("model",))
        assert tasks[0] == ("img.jpg", cs[0], "model")

    def test_empty_items(self):
        assert distribute_tasks([], self._clients(2)) == []

    def test_more_clients_than_items(self):
        cs = self._clients(5)
        tasks = distribute_tasks([("a",)], cs)
        assert len(tasks) == 1
