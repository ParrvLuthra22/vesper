"""
Sensors — local-only observation of machine state for the Proactive Engine.

All sensing is local only: nothing observed ever leaves the machine except
as short context lines handed to the LLM planner. Each sensor is
individually toggleable via its own `sensors.<name>.enabled` config flag
(see sensors/base_sensor.py).
"""

from sensors.base_sensor import BaseSensor
from sensors.calendar_sensor import CalendarSensor
from sensors.focus_sensor import FocusSensor
from sensors.inbox_sensor import InboxSensor

__all__ = [
    "BaseSensor",
    "FocusSensor",
    "CalendarSensor",
    "InboxSensor",
]
