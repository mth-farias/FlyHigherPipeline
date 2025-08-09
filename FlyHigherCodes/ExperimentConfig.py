#%% CELL 00 – MODULE OVERVIEW
"""
ExperimentConfig.py

Purpose
=======
Define general experimental settings and metadata for the Drosophila defensive-behaviour analysis toolkit.
All pipeline stages import this file to access common constants.
"""

#%% CELL 01 – GENERAL EXPERIMENT VARIABLES

# Enable pose-derived metrics processing
POSE_SCORING = True  # include pose‑derived metrics from SLEAP

# Stimulus alignment configuration
ALIGNMENT_COL         = "VisualStim"  # column holding stimulus pulses (0→1)
STIMULUS_NUMBER       = 20            # expected onsets per run
STIMULUS_DURATION_SEC = 0.5           # stimulus length (sec)
EXPECTED_STIMULUS     = STIMULUS_NUMBER + 3  # extra events (e.g. lights‑off)


# Timing & arena dimensions
FRAME_RATE      = 60   # frames per second
ARENA_WIDTH_MM  = 30   # arena width (millimetres)
ARENA_HEIGHT_MM = 30   # arena height (millimetres)

# Experimental periods durations (sec)
EXPERIMENTAL_PERIODS = {
    "Baseline":    {"duration_sec": 300},
    "Stimulation": {"duration_sec": 300},
    "Recovery":    {"duration_sec": 300},
}

# Filename and grouping metadata
FILENAME_STRUCTURE = [  # order of fields in scored filenames
    "Experimenter", "Genotype", "Protocol", "Sex", "Age",
    "Setup", "Camera", "Date", "FlyID", "Extension",
]

GROUP_IDENTIFIER = "Protocol"  # metadata field used for grouping runs

EXPERIMENTAL_GROUPS = {
    "Control": {
        "label": "Control",               # group name
        "idValue": "20Control_3BlackOut", # identifier in filename metadata
        "color": "#645769",               # color for plotting
    },
    "Loom": {
        "label": "Loom",
        "idValue": "20Loom_3BlackOut",
        "color": "#E35B29",
    },
}


#%% CELL 02 - TIMING VARIABLES

"""
CELL 02: Derived Timing Variables

This cell calculates and defines various timing-related variables derived from the frame rate and 
experiment periods. These derived variables include frame spans, total experiment length, 
and period durations in frames. These variables are crucial for segmenting 
and aligning data during the analysis.

EXPERIMENTAL_PERIODS = {
    'PeriodName': {
        'label': 'Human-readable label',  # Descriptive label for the period
        'duration_sec': <float>,  # Duration of the period in seconds
        'duration_frames': <int>,  # Duration of the period in frames
        'range_sec': (<float>, <float>),  # Tuple indicating the start and end time in seconds
        'range_frames': (<int>, <int>)  # Tuple indicating the start and end frame numbers
    },
    # Additional periods follow the same structure...
}
"""

# Derived Timing Variables
frame_span_sec = 1 / FRAME_RATE  # Duration of a single frame (in seconds)
stimulus_duration_frames = STIMULUS_DURATION_SEC * FRAME_RATE  # Duration of stimulus (in frames)

# Initialize variables for cumulative time calculation
current_time_sec = 0
current_time_frames = 0

# Update EXPERIMENTAL_PERIODS dictionary to include duration in frames, range in frames, and range in seconds
for label, info in EXPERIMENTAL_PERIODS.items():
    # Calculate duration in frames
    info['duration_frames'] = info['duration_sec'] * FRAME_RATE
    
    # Define the range in seconds
    info['range_sec'] = (current_time_sec, current_time_sec + info['duration_sec'])
    
    # Define the range in frames
    info['range_frames'] = (current_time_frames, current_time_frames + info['duration_frames'])
    
    # Update the cumulative time
    current_time_sec += info['duration_sec']
    current_time_frames += info['duration_frames']

# Calculate the total experiment duration in seconds and frames
total_duration_sec = current_time_sec
total_duration_frames = current_time_frames

# Add an 'Experiment' entry to the EXPERIMENTAL_PERIODS dictionary
EXPERIMENTAL_PERIODS['Experiment'] = {
    'label': 'Experiment',
    'duration_sec': total_duration_sec,
    'duration_frames': total_duration_frames,
    'range_sec': (0, total_duration_sec),
    'range_frames': (0, total_duration_frames)
}


#%% CELL 03 – EXPORTS

__all__ = [
    "POSE_SCORING",
    "ALIGNMENT_COL", "STIMULUS_NUMBER", "STIMULUS_DURATION_SEC", "EXPECTED_STIMULUS",
    "FRAME_RATE", "ARENA_WIDTH_MM", "ARENA_HEIGHT_MM",
    "EXPERIMENTAL_PERIODS",
    "FILENAME_STRUCTURE", "GROUP_IDENTIFIER", "EXPERIMENTAL_GROUPS",
]
