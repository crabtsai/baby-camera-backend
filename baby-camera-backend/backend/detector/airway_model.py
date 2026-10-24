"""Shared class contract for airway training and runtime inference."""

AIRWAY_LABELS = (
    "face", "nose", "mouth", "pacifier_in_mouth", "blanket",
    "blanket_on_nose", "blanket_on_mouth", "covered_face",
)


def check_model_contract(model):
    names = model.names
    labels = set(names.values() if isinstance(names, dict) else names)
    missing = sorted(set(AIRWAY_LABELS) - labels)
    if model.task != "detect":
        raise ValueError(f"口鼻模型必須為 detect，目前為 {model.task}")
    if missing:
        raise ValueError("口鼻模型缺少類別：" + ", ".join(missing))
