from pathlib import Path


_HALF_SUFFIX = "_half"


def _model_name_key(model_family: str, model_size: str) -> str:
    model_names = {
        "yolo11": {"nano": "yolo11n", "small": "yolo11s", "medium": "yolo11m", "large": "yolo11l", "xlarge": "yolo11x"},
        "yolo12": {"nano": "yolo12n", "small": "yolo12s", "medium": "yolo12m", "large": "yolo12l", "xlarge": "yolo12x"},
        "rtdetr": {"nano": "rtdetr-l", "large": "rtdetr-l", "xlarge": "rtdetr-x"},
    }
    return model_names.get(model_family, model_names["yolo11"]).get(model_size, "yolo11n")


def engine_path(model_family: str, model_size: str, half: bool = True) -> str:
    name = _model_name_key(model_family, model_size)
    suffix = _HALF_SUFFIX if half else ""
    return f"{name}{suffix}.engine"


def pt_path(model_family: str, model_size: str) -> str:
    name = _model_name_key(model_family, model_size)
    return f"{name}.pt"


def has_engine(model_family: str, model_size: str, half: bool = True) -> bool:
    return Path(engine_path(model_family, model_size, half)).exists()


def export_to_engine(
    model_family: str = "yolo11",
    model_size: str = "nano",
    half: bool = True,
    device: int = 0,
    force: bool = False,
) -> str:
    export_path = engine_path(model_family, model_size, half)
    if not force and Path(export_path).exists():
        return str(Path(export_path).absolute())
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for TensorRT export")

    pt = pt_path(model_family, model_size)
    model = YOLO(pt)
    model.export(format="engine", half=half, device=device, verbose=False)
    if not Path(export_path).exists():
        found = list(Path(".").glob(f"{_model_name_key(model_family, model_size)}*.engine"))
        if found:
            return str(found[0].absolute())
    return str(Path(export_path).absolute())
