"""Switches for the optional pipeline safeguards.

Each safeguard can be turned on or off on its own, so runs can be compared
with any combination of them.
"""

from dataclasses import asdict, dataclass

OUTLINE_MODES = ("off", "fresh", "rolling")


@dataclass(frozen=True)
class SafeguardConfig:
    # Turned on together by --safeguard.
    grounded_feedback: bool = False
    output_recovery: bool = False
    delivery_preservation: bool = False
    # "off": no outline. "fresh": the planner and every repair write an
    # outline before the JSON. "rolling": as "fresh", and each repair also
    # receives the outline of the schedule it is revising.
    outline: str = "off"
    # Send the simulator's vessel and shipment timelines to the repair model.
    trace: bool = False

    def __post_init__(self):
        if self.outline not in OUTLINE_MODES:
            raise ValueError(f"outline must be one of {OUTLINE_MODES}, got {self.outline!r}")

    @property
    def uses_outline(self) -> bool:
        return self.outline != "off"

    @classmethod
    def from_flags(cls, *, safeguard=False, outline="off", trace=False):
        return cls(
            grounded_feedback=safeguard,
            output_recovery=safeguard,
            delivery_preservation=safeguard,
            outline=outline,
            trace=trace,
        )

    def to_dict(self) -> dict:
        return asdict(self)
