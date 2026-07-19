import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class EventMessage:
    topic: str
    data: dict[str, Any]
    camera_id: str = ""
    source: str = ""
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    utc_time: str = ""

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "data": self.data,
            "camera_id": self.camera_id,
            "source": self.source,
            "priority": self.priority.name.lower(),
            "timestamp": self.timestamp,
            "utc_time": self.utc_time,
        }


SubscriberFn = Callable[[EventMessage], None]


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[SubscriberFn]] = defaultdict(list)
        self._wildcard_subscribers: list[SubscriberFn] = []
        self._history: list[EventMessage] = []
        self._max_history: int = 1000

    def subscribe(self, topic: str | None, callback: SubscriberFn):
        if topic is None:
            self._wildcard_subscribers.append(callback)
        else:
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: SubscriberFn):
        if callback in self._subscribers.get(topic, []):
            self._subscribers[topic].remove(callback)

    def publish(self, message: EventMessage):
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        for cb in self._wildcard_subscribers:
            cb(message)

        matched = self._subscribers.get(message.topic, [])
        for cb in matched:
            cb(message)

        parts = message.topic.split(".")
        for i in range(len(parts) - 1, 0, -1):
            wild = ".".join(parts[:i]) + ".*"
            for cb in self._subscribers.get(wild, []):
                cb(message)

    def history(self, topic: str | None = None, limit: int = 50) -> list[EventMessage]:
        if topic is None:
            return list(self._history[-limit:])
        return [m for m in self._history[-limit:] if m.topic == topic]

    def clear(self):
        self._history.clear()

    def count(self, topic: str | None = None) -> int:
        if topic is None:
            return len(self._history)
        return sum(1 for m in self._history if m.topic == topic)
