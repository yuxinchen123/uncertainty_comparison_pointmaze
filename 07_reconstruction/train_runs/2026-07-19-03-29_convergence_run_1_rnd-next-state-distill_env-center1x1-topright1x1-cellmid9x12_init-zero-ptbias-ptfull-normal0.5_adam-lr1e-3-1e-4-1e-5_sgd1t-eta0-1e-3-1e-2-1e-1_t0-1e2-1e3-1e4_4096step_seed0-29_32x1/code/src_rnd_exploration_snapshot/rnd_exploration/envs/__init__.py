"""PointMaze environment wrappers."""
from .point_maze_wrappers import (
    FixedGoalWrapper,
    FixedStartWrapper,
    RemoveGoalWrapper,
    TerminateOnTimeLimitWrapper,
    PositionVisitCountWrapper,
    PositionVelocityVisitCountWrapper,
    ComputeIntrinsicRewardWrapper,
)

__all__ = [
    "FixedGoalWrapper",
    "FixedStartWrapper",
    "RemoveGoalWrapper",
    "TerminateOnTimeLimitWrapper",
    "PositionVisitCountWrapper",
    "PositionVelocityVisitCountWrapper",
    "ComputeIntrinsicRewardWrapper",
]
