"""What happened in the landing burn, in the numbers that explain a touchdown speed.

The stop height is where the hard part of the burn first brought the descent below the target
speed; the tail then has to sink the rocket from there. Stopping too low means residual speed,
stopping too high means a climb on leftover thrust or a long slow sink the tail may not finish.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BurnSummary:
    command_time: float | None
    thrust_start_time: float | None
    height_estimate_error: float | None
    speed_estimate_error: float | None
    thrust_start_height: float | None
    thrust_start_speed: float | None
    stop_height: float | None
    climb_after_stop: float
    burn_time_left: float | None

    def describe(self) -> str:
        if self.command_time is None:
            return "landing burn: never commanded"
        parts = [f"landing burn: commanded at {self.command_time:.2f} s"]
        if self.height_estimate_error is not None and self.speed_estimate_error is not None:
            parts.append(
                f"estimate error height {self.height_estimate_error:+.2f} m, speed {self.speed_estimate_error:+.2f} m/s"
            )
        if self.thrust_start_height is not None and self.thrust_start_speed is not None:
            parts.append(f"thrust from {self.thrust_start_height:.1f} m at {self.thrust_start_speed:.1f} m/s")
        if self.stop_height is not None:
            parts.append(f"stopped at {self.stop_height:.2f} m, climbed {self.climb_after_stop:.2f} m after")
        else:
            parts.append("never reached the target speed")
        if self.burn_time_left is not None:
            parts.append(f"burn left at touchdown {self.burn_time_left:.1f} s")
        return "; ".join(parts)


class BurnTracker:
    """Collects the burn summary from the true state one physics step at a time."""

    def __init__(self, target_speed: float, burn_time: float) -> None:
        self.target_speed = target_speed
        self.burn_time = burn_time
        self.reset()

    def reset(self) -> None:
        self.command_time: float | None = None
        self.thrust_start_time: float | None = None
        self.height_estimate_error: float | None = None
        self.speed_estimate_error: float | None = None
        self.thrust_start_height: float | None = None
        self.thrust_start_speed: float | None = None
        self.stop_height: float | None = None
        self.peak_height_after_stop: float | None = None
        self.burn_time_left: float | None = None

    def commanded(self, t: float, height_error: float, speed_error: float) -> None:
        """The igniter command went out: remember the estimate errors at that instant."""
        if self.command_time is None:
            self.command_time = t
            self.height_estimate_error = height_error
            self.speed_estimate_error = speed_error

    def update(self, t: float, feet_height: float, speed_down: float, ignition_time: float | None) -> None:
        """One physics step of truth: feet height, downward speed and the motor's ignition time if known."""
        if ignition_time is None or t < ignition_time:
            return
        if self.thrust_start_time is None:
            self.thrust_start_time = ignition_time
            self.thrust_start_height, self.thrust_start_speed = feet_height, speed_down
        if self.stop_height is None:
            if speed_down <= self.target_speed:
                self.stop_height = self.peak_height_after_stop = feet_height
        elif self.peak_height_after_stop is not None:
            self.peak_height_after_stop = max(self.peak_height_after_stop, feet_height)

    def touchdown(self, t: float) -> None:
        if self.thrust_start_time is not None:
            self.burn_time_left = max(self.burn_time - (t - self.thrust_start_time), 0.0)

    @property
    def summary(self) -> BurnSummary:
        climb = 0.0
        if self.stop_height is not None and self.peak_height_after_stop is not None:
            climb = self.peak_height_after_stop - self.stop_height
        return BurnSummary(
            command_time=self.command_time,
            thrust_start_time=self.thrust_start_time,
            height_estimate_error=self.height_estimate_error,
            speed_estimate_error=self.speed_estimate_error,
            thrust_start_height=self.thrust_start_height,
            thrust_start_speed=self.thrust_start_speed,
            stop_height=self.stop_height,
            climb_after_stop=climb,
            burn_time_left=self.burn_time_left,
        )
