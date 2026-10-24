from types import SimpleNamespace

from backend.api.health_api import health


def test_health_is_degraded_for_stale_camera_or_disabled_detector():
    class Manager:
        default_camera_id = "one"
        cameras = []

        def default_runtime(self):
            return SimpleNamespace(camera_id="one", camera=SimpleNamespace(status=lambda: {}),
                detector_manager=SimpleNamespace(names=lambda: [], worker_status=lambda: []))

        def list_status(self):
            return self.cameras

    manager = Manager()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(camera_manager=manager)))
    manager.cameras = [{"camera": {"frame_fresh": True},
                         "detector_workers": [{"alive": True, "operational": False, "last_error": None}]}]
    assert health(request)["status"] == "degraded"
    manager.cameras[0]["detector_workers"][0]["operational"] = True
    assert health(request)["status"] == "ok"
    manager.cameras[0]["camera"]["frame_fresh"] = False
    assert health(request)["status"] == "degraded"
