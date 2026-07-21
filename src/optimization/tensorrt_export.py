from pathlib import Path


def _model_name_key(model_family: str, model_size: str) -> str:
    model_names = {
        "yolo11": {"nano": "yolo11n", "small": "yolo11s", "medium": "yolo11m", "large": "yolo11l", "xlarge": "yolo11x"},
        "yolo12": {"nano": "yolo12n", "small": "yolo12s", "medium": "yolo12m", "large": "yolo12l", "xlarge": "yolo12x"},
        "rtdetr": {"nano": "rtdetr-l", "large": "rtdetr-l", "xlarge": "rtdetr-x"},
    }
    return model_names.get(model_family, model_names["yolo11"]).get(model_size, "yolo11n")


def engine_path(model_family: str, model_size: str, half: bool = True, batch_size: int = 1) -> str:
    name = _model_name_key(model_family, model_size)
    suffix = f"_b{batch_size}" if batch_size > 1 else ""
    return f"{name}{suffix}.engine"


def pt_path(model_family: str, model_size: str) -> str:
    name = _model_name_key(model_family, model_size)
    return f"{name}.pt"


def has_engine(model_family: str, model_size: str, half: bool = True, batch_size: int = 1) -> bool:
    return Path(engine_path(model_family, model_size, half, batch_size)).exists()


def export_to_engine(
    model_family: str = "yolo11",
    model_size: str = "nano",
    half: bool = True,
    device: int = 0,
    force: bool = False,
    batch_size: int = 1,
) -> str:
    export_path = engine_path(model_family, model_size, half, batch_size)
    if not force and Path(export_path).exists():
        return str(Path(export_path).absolute())
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required for TensorRT export")

    pt = pt_path(model_family, model_size)
    model = YOLO(pt)
    export_kwargs = dict(
        format="engine",
        half=half,
        device=device,
        verbose=False,
        imgsz=640,
        workspace=4,
    )
    if batch_size > 1:
        export_kwargs["batch"] = batch_size
    model.export(**export_kwargs)

    # Export saves to default name (e.g. yolo11x.engine). Rename to batch-specific name.
    default_name = f"{_model_name_key(model_family, model_size)}.engine"
    default_path = Path(default_name)
    target_path = Path(export_path)

    if default_path.exists() and target_path != default_path:
        if target_path.exists():
            target_path.unlink()
        default_path.rename(target_path)
    elif not target_path.exists():
        # Fallback: glob for any engine file
        found = list(Path(".").glob(f"{_model_name_key(model_family, model_size)}*.engine"))
        if found:
            return str(found[0].absolute())

    return str(target_path.absolute())
