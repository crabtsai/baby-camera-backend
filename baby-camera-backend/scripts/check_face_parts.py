"""Check the configured Roboflow service using a synthetic frame, never a camera."""
import json

import numpy as np

from backend.config import get_settings
from backend.detector.baby_face_parts import BabyFacePartsDetector


def main():
    detector = BabyFacePartsDetector.from_settings(get_settings())
    result = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    print(json.dumps({"model_id": detector.model_id, "service_responded": detector.operational,
                      "message": result.message,
                      "note": "Synthetic connectivity check only, not an accuracy test."}, ensure_ascii=False, indent=2))
    return 0 if detector.operational else 1


if __name__ == "__main__":
    raise SystemExit(main())
