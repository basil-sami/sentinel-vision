from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CameraStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    RECORDING = "recording"


@dataclass
class Camera:
    id: str
    name: str
    source: str
    fps: int = 25
    resolution: tuple[int, int] = (640, 360)
    location: str = ""
    gps: tuple[float, float] | None = None
    timezone: str = "UTC"
    status: CameraStatus = CameraStatus.OFFLINE
    calibration_config: dict | None = None
    zones_config: dict | None = None
    topology: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "location": self.location,
            "gps": list(self.gps) if self.gps else None,
            "timezone": self.timezone,
            "status": self.status.value,
            "topology": self.topology,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        data.pop("status", None)
        res = data.pop("resolution", [640, 360])
        gps = data.pop("gps", None)
        data["resolution"] = tuple(res) if isinstance(res, list) else res
        data["gps"] = tuple(gps) if gps and isinstance(gps, list) else gps
        return cls(**data)


class CameraRegistry:
    def __init__(self):
        self._cameras: dict[str, Camera] = {}

    def register(self, camera: Camera):
        self._cameras[camera.id] = camera

    def get(self, camera_id: str) -> Camera | None:
        return self._cameras.get(camera_id)

    def all(self) -> list[Camera]:
        return list(self._cameras.values())

    def online(self) -> list[Camera]:
        return [c for c in self._cameras.values() if c.status == CameraStatus.ONLINE]

    def from_config(self, config_path: str | Path):
        import json
        data = json.loads(Path(config_path).read_text())
        for entry in data.get("cameras", []):
            self.register(Camera.from_dict(entry))

    def save(self, path: str | Path):
        import json
        data = {"cameras": [c.to_dict() for c in self._cameras.values()]}
        Path(path).write_text(json.dumps(data, indent=2))
