import csv
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.detector.airway_model import AIRWAY_LABELS, check_model_contract
from scripts.airway_model import initialize, validate, load_model


def dataset(root):
    initialize(root)
    rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        for index, state in enumerate(("clear", "abnormal", "unknown")):
            name = f"images/{split}/{index}.png"
            cv2.imwrite(str(root / name), np.full((20, 20, 3), split_index*10+index, np.uint8))
            (root / "labels" / split / f"{index}.txt").write_text(
                "\n".join(f"{cls} 0.5 0.5 0.2 0.2" for cls in range(8)))
            rows.append([name, f"session-{split}", state])
    with (root / "samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "session", "assessment"))
        writer.writerows(rows)
    return root


def test_empty_template_is_not_training_ready(tmp_path):
    initialize(tmp_path)
    report, _ = validate(tmp_path)
    assert not report["valid"]
    assert any("lacks annotations" in e for e in report["errors"])


def test_valid_layout_and_idempotent_init(tmp_path):
    dataset(tmp_path)
    initialize(tmp_path)
    report, records = validate(tmp_path)
    assert report["valid"], report
    assert len(records) == 9


@pytest.mark.parametrize("box", ["0 nan 0.5 0.2 0.2", "8 0.5 0.5 0.2 0.2", "0 0.1 0.5 0.9 0.2", "0 0.5 0.5 -0.2 0.2"])
def test_reject_invalid_boxes(tmp_path, box):
    dataset(tmp_path)
    (tmp_path / "labels/train/0.txt").write_text(box)
    report, _ = validate(tmp_path)
    assert any("invalid YOLO box" in e for e in report["errors"])


def test_missing_label_is_not_silently_treated_as_negative(tmp_path):
    dataset(tmp_path)
    (tmp_path / "labels/train/0.txt").unlink()
    report, _ = validate(tmp_path)
    assert any("missing reviewed label" in e for e in report["errors"])


def test_session_and_pixel_leakage(tmp_path):
    dataset(tmp_path)
    manifest = tmp_path / "samples.csv"
    manifest.write_text(manifest.read_text().replace("session-val", "session-train"))
    (tmp_path / "images/test/0.png").write_bytes((tmp_path / "images/train/0.png").read_bytes())
    report, _ = validate(tmp_path)
    assert any("session leakage" in e for e in report["errors"])
    assert any("duplicate decoded image" in e for e in report["errors"])


def test_reject_path_escape(tmp_path):
    initialize(tmp_path)
    (tmp_path / "samples.csv").write_text("image,session,assessment\n../escape.png,s,clear\n")
    report, _ = validate(tmp_path)
    assert "row 2: image outside dataset" in report["errors"]


def test_model_contract_rejects_face_only_and_pose():
    with pytest.raises(ValueError, match="blanket_on_nose"):
        check_model_contract(SimpleNamespace(task="detect", names={0: "face"}))
    with pytest.raises(ValueError, match="detect"):
        check_model_contract(SimpleNamespace(task="pose", names=AIRWAY_LABELS))
    check_model_contract(SimpleNamespace(task="detect", names=dict(enumerate(AIRWAY_LABELS))))


def test_missing_weights_never_trigger_download(tmp_path):
    with pytest.raises(ValueError, match="no automatic download"):
        load_model(tmp_path / "missing.pt")


def test_archive_extracts_only_data_and_reports_classes(tmp_path):
    import zipfile
    from scripts.inspect_cribhd import inspect_archive
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("B/data.yaml", "names: [blanket]\n")
        handle.writestr("run.py", "raise RuntimeError()")
        handle.writestr("model.pt", "not a real model")
    output = tmp_path / "out"
    report = inspect_archive(archive, output)
    assert report["datasets"][0]["classes"] == ["blanket"]
    assert not (output / "run.py").exists()
    assert not (output / "model.pt").exists()
    assert report["weights"] == ["model.pt"]
    assert not report["ready_for_airway_training"]


def test_archive_rejects_path_traversal(tmp_path):
    import zipfile
    from scripts.inspect_cribhd import inspect_archive
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        inspect_archive(archive, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()


def test_evaluation_counts_unknown_as_missed_obstruction(tmp_path, monkeypatch):
    from scripts import airway_model
    dataset(tmp_path)
    _, records = validate(tmp_path)

    class Detector:
        predict_fn = True

        def __init__(self, **kwargs):
            pass

        def detect(self, frame):
            # On test images: clear->abnormal, abnormal->unknown, unknown->unknown.
            state = "abnormal" if int(frame[0, 0, 0]) == 20 else "unknown"
            return SimpleNamespace(extra={"assessment": state}, message="test")

    monkeypatch.setattr("backend.detector.face_cover.FaceCoverDetector", Detector)
    monkeypatch.setattr(airway_model, "inspect", lambda path: {"compatible": True})
    report = airway_model.evaluate(tmp_path / "model.pt", records, tmp_path / "report.json")
    assert len(report["predictions"]) == 3  # held-out split only
    assert report["missed_obstruction_rate"] == 1
    assert report["false_alert_rate"] == 0.5
    assert report["unknown_rate"] == pytest.approx(2/3)
    assert report["deployment_ready"] is False
