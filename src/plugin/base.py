from abc import ABC, abstractmethod
from typing import Any


class AnalyticsPlugin(ABC):
    name: str = "base_plugin"
    version: str = "0.1.0"

    @abstractmethod
    def initialize(self, config: dict | None = None):
        pass

    @abstractmethod
    def process_frame(self, frame: Any, tracks: list, frame_index: int) -> Any:
        pass

    @abstractmethod
    def process_track(self, track: Any, frame_index: int) -> list:
        pass

    @abstractmethod
    def process_event(self, event: dict) -> dict | None:
        pass

    @abstractmethod
    def shutdown(self):
        pass


class PluginRegistry:
    def __init__(self):
        self._plugins: dict[str, AnalyticsPlugin] = {}

    def register(self, plugin: AnalyticsPlugin):
        self._plugins[plugin.name] = plugin
        plugin.initialize()

    def get(self, name: str) -> AnalyticsPlugin | None:
        return self._plugins.get(name)

    def all(self) -> list[AnalyticsPlugin]:
        return list(self._plugins.values())

    def process_frame(self, frame, tracks: list, frame_index: int):
        results = {}
        for name, plugin in self._plugins.items():
            results[name] = plugin.process_frame(frame, tracks, frame_index)
        return results

    def process_track(self, track, frame_index: int) -> list:
        events = []
        for plugin in self._plugins.values():
            events.extend(plugin.process_track(track, frame_index))
        return events

    def process_event(self, event: dict) -> list[dict]:
        results = []
        for plugin in self._plugins.values():
            r = plugin.process_event(event)
            if r:
                results.append(r)
        return results

    def shutdown(self):
        for plugin in self._plugins.values():
            plugin.shutdown()
