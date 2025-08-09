#%% CELL 00 – MODULE OVERVIEW
"""
BehaviorScoringConfig.py

Purpose
=======
Define thresholds, smoothing windows, and tolerance settings for the behavior scoring pipeline.
Imported by BehaviorScoringFunctions and related modules to parameterize classification logic.

"""

#%% CELL 01 – BEHAVIOR SCORING VARIABLES

# THRESHOLDS
HIGH_SPEED = 75  # mm/s threshold for jump detection
LOW_SPEED  = 4   # mm/s threshold for walk vs stationary

# SMOOTHING WINDOWS (SECONDS)
LAYER2_AVG_WINDOW_SEC = 0.1  # small smoothing window for layer 2

# TOLERANCES
NOISE_TOLERANCE         = 2       # frames to smooth binary signals
NAN_TOLERANCE           = 0.0001  # max fraction of NaNs in centroid data
POSE_TRACKING_TOLERANCE = 0.05    # max fraction of missing pose frames
LAYER1_TOLERANCE        = 0.05    # max fraction of unassigned layer‑1 frames
BASELINE_EXPLORATION    = 0.2     # min fraction of walking during baseline

# OUTPUT COLUMNS
SCORED_COLUMNS = [  # columns saved in scored CSVs
    "FrameIndex", "VisualStim", "Stim0", "Stim1",
    "Position_X", "Position_Y", "Speed", "Motion",
    "Layer1", "Layer2",
    "Behavior",
]

SCORED_POSE_COLUMNS = [  # additional columns when POSE_SCORING is True
    "FrameIndex", "Orientation", "View", "View_X", "View_Y",
    "Head_X", "Head_Y", "Thorax_X", "Thorax_Y", "Abdomen_X", "Abdomen_Y",
    "LeftWing_X", "LeftWing_Y", "RightWing_X", "RightWing_Y",
]


#%% CELL 02 – EXPORTS

__all__ = [
    "HIGH_SPEED",
    "LOW_SPEED",
    "LAYER2_AVG_WINDOW_SEC",
    "NOISE_TOLERANCE",
    "NAN_TOLERANCE",
    "POSE_TRACKING_TOLERANCE",
    "LAYER1_TOLERANCE",
    "BASELINE_EXPLORATION",
    "SCORED_COLUMNS",
    "SCORED_POSE_COLUMNS",
]