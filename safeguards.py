"""Grouped and individual switches for the optional pipeline safeguards."""

from dataclasses import asdict, dataclass

OUTLINE_MODES = ("off", "fresh", "rolling")


@dataclass(frozen=True)
class SafeguardConfig:
    # Individually testable; grouped by purpose in from_flags().
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
    def from_flags(
        cls, *, safeguard=False, grounded_feedback=False,
        output_recovery=False, delivery_preservation=False,
        outline="off", trace=False, planning_support=False,
        guided_repair=False, repair_guards=False, all_safeguards=False,
    ):
        return cls(
            grounded_feedback=safeguard or grounded_feedback or guided_repair or all_safeguards,
            output_recovery=safeguard or output_recovery or repair_guards or all_safeguards,
            delivery_preservation=safeguard or delivery_preservation or repair_guards or all_safeguards,
            outline=outline if outline != "off" else "rolling" if planning_support or all_safeguards else "off",
            trace=trace or guided_repair or all_safeguards,
        )

    def to_dict(self) -> dict:
        return asdict(self)
