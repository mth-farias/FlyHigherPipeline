#%%% CELL 00 – MODULE OVERVIEW
"""
BehaviorScoringFunctions.py

Purpose
This module provides utilities for the Drosophila behavior pipeline. It loads and
checks files, transforms signals, computes kinematics, classifies behaviors, and
writes safe outputs. All helpers are kept small and composable for clarity.

Steps
- Declare error registry and checkpoint helpers.
- Provide skip-logic for already-processed files.
- Offer binary clean-up utilities and bout duration tools.
- Compute speed and orientation; pick best view per frame.
- Smooth signals, enforce hierarchy, and classify behaviors.
- Mark resistant bouts with full startle-window overlap.
- Write CSVs atomically and format reporting blocks.

Output
- Error registry (CHECKPOINT_ERRORS) and formatting helpers.
- Data transforms, classifiers, and atomic write function.
- Public helpers used by BehaviorScoringMain.py.
"""

#%%% CELL 01 – IMPORTS
"""
Purpose
Import required libraries. Keep the surface minimal and standard.

Steps
- Import os, numpy, pandas.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

#%%% CELL 02 – ERROR DEFINITIONS & CHECKPOINT
"""
Purpose
Define standard error keys/messages and provide a checkpoint handler that
writes an error CSV using Path-safe atomic I/O and increments counters.

Steps
- Declare CHECKPOINT_ERRORS with file suffixes and messages.
- Build error output path with Path and write atomically.
- Print a standardized error line and return the updated counter.
"""

CHECKPOINT_ERRORS = {
    "ERROR_READING_FILE": {
        "message": "Error reading tracked file.",
        "file_end": "error_reading.csv",
    },
    "WRONG_STIMULUS_COUNT": {
        "message": "Wrong stimulus count detected.",
        "file_end": "wrong_stim_count.csv",
    },
    "WRONG_STIMULUS_DURATION": {
        "message": "Wrong stimulus duration detected.",
        "file_end": "wrong_stim_duration.csv",
    },
    "LOST_CENTROID_POSITION": {
        "message": "Too many centroid NaNs detected.",
        "file_end": "many_nans.csv",
    },
    "POSE_MISMATCH": {
        "message": "Mismatch between tracked and pose data lengths.",
        "file_end": "no_match_pose.csv",
    },
    "MISSING_POSE_FILE": {
        "message": "Pose file is missing.",
        "file_end": "missing_pose.csv",
    },
    "VIEW_NAN_EXCEEDED": {
        "message": "Too many NaNs in view data.",
        "file_end": "view_nan_exceeded.csv",
    },
    "UNASSIGNED_BEHAVIOR": {
        "message": "Too many unassigned behaviors detected.",
        "file_end": "unassigned_behaviors.csv",
    },
    "NO_EXPLORATION": {
        "message": "Insufficient exploration during baseline period.",
        "file_end": "too_little_exploration.csv",
    },
    "OUTPUT_LEN_SHORT": {
        "message": "Tracked file length is shorter than expected.",
        "file_end": "tracked_len_short.csv",
    },
}


def checkpoint_fail(df,
                    filename_tracked: str,
                    error_key: str,
                    error_counter: int,
                    error_dir) -> int:
    """
    Handle a failed checkpoint: write df with the error suffix, print a line,
    and increment the counter. Returns the updated counter.

    notes
    - downstream readers should ignore *.tmp files created during atomic write
    """
    info = CHECKPOINT_ERRORS[error_key]
    error_file = filename_tracked.replace("tracked.csv", info["file_end"])
    error_path = Path(error_dir) / error_file  # build with Path
    write_csv_atomic(df, error_path, header=True, index=False)
    print(report_error_line(info["message"]))  # standardized error line
    return error_counter + 1

#%%% CELL 03 – FILE STATUS CHECK
"""
Purpose
Check whether a tracked file is already processed or errored using flat-root
folders. Paths are handled with pathlib.Path for clarity and safety.

Steps
- Resolve output roots from PATHconfig as Path objects.
- Map tracked name to expected scored/error files.
- Update counters when a match is found and return True/False.
"""


def is_file_already_processed(filename_tracked,
                              pose_scoring,
                              processed_counters,
                              PATHconfig) -> bool:
    """
    Determine if a tracked file was already scored or labeled as error.

    parameters
    - filename_tracked: tracked filename as string
    - pose_scoring: bool flag for pose scoring destination
    - processed_counters: dict with 'scored' and 'error' counters
    - PATHconfig: config with pScored, pScoredPose, pScoredError

    returns
    - True if a scored file or an error file already exists; else False
    """
    scored_root = Path(PATHconfig.pScoredPose) if pose_scoring else Path(PATHconfig.pScored)
    error_root = Path(PATHconfig.pScoredError)

    # optional safety: if destinations are missing, treat as not processed
    if not scored_root.exists() or not error_root.exists():
        return False

    # scored name mapping
    scored_name = filename_tracked.replace(
        "tracked.csv", "scored_pose.csv" if pose_scoring else "scored.csv"
    )
    scored_path = scored_root / scored_name
    if scored_path.exists():
        processed_counters["scored"] += 1
        return True

    # any matching error file suffices (prefix match on base)
    base = filename_tracked.replace("tracked.csv", "")
    try:
        for err in error_root.iterdir():
            if err.is_file() and err.name.startswith(base):
                processed_counters["error"] += 1
                return True
    except Exception:
        pass  # transient FS errors; assume not processed

    return False

#%%% CELL 04 – BINARY CLEANERS & BOUT UTIL
"""
Purpose
Provide simple binary cleaners and a utility to compute bout durations.

Steps
- Implement fill_zeros and clean_ones (to be replaced by morphology later).
- Implement bout_duration to return lengths of 1-runs in frames.
"""

def fill_zeros(df, column, max_length):
    """
    Fill gaps in a binary column by setting isolated zeros to one when they sit
    inside short gaps. Future version will use windowed morphology (dilation).
    """
    x = df[column].to_numpy()
    n = len(x)

    # scan forward and fill single zeros inside short gaps
    for i in range(n - max_length - 1):
        if x[i] == 1 and x[i + 1] == 0:
            if x[i + 1:i + max_length + 1].sum() > 0:  # any 1 shortly after
                x[i + 1] = 1
    df[column] = x


def clean_ones(df, column, min_length):
    """
    Remove short 1-runs below a length threshold. Future version will use
    windowed morphology (opening) for robustness.
    """
    x = df[column].to_numpy()
    n = len(x)

    # scan forward and zero out spikes shorter than min_length
    for i in range(n - min_length - 1):
        if x[i] == 0 and x[i + 1] == 1:
            if x[i + 1:i + 1 + min_length + 1].sum() < 3:  # short run → drop
                x[i + 1] = 0
    df[column] = x


def bout_duration(df, column):
    """
    Return the frame lengths of each continuous 1-bout in a binary column.
    """
    x = df[column].to_numpy()
    durations, count = [], 0

    for val in x:
        if val == 1:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0

    if count > 0:  # capture an open bout at the end
        durations.append(count)
    return durations

#%%% CELL 05 – KINEMATICS & VIEW/ORIENTATION
"""
Purpose
Compute speed and orientation and select a per-frame best view for pose use.

Steps
- Calculate speed in mm/s as floats (rounding done at the call site).
- Determine view from confidences or use vertical fallback logic.
- Compute orientation in degrees with 0 at North.
"""

def calculate_speed(column_x, column_y, frame_span_sec):
    """
    Return speed in mm/s as float; rounding is applied at the call site.

    Parameters
    - column_x, column_y: coordinate Series in millimetres.
    - frame_span_sec: duration of a single frame in seconds.
    """
    dx = column_x.diff()
    dy = column_y.diff()
    distance = np.sqrt(dx ** 2 + dy ** 2)  # per-frame displacement in mm
    speed = distance / frame_span_sec      # mm/s as float
    return speed.astype(float)


def determine_view(row):
    """
    Choose a view per frame using confidences. When all three body parts are
    present, pick the view (Left/Right/Top) with highest confidence. For
    vertical cases, prefer Head if present, else Abdomen; otherwise return NaN.
    """
    # full key positions available → pick by confidence
    if (pd.notna(row.get("Head.Position.X")) and
        pd.notna(row.get("Thorax.Position.X")) and
        pd.notna(row.get("Abdomen.Position.X"))):
        confidences = {
            "Left": row.get("Left.Confidence", 0),
            "Right": row.get("Right.Confidence", 0),
            "Top": row.get("Top.Confidence", 0)
        }
        selected = max(confidences, key=confidences.get)
        vx = row.get(f"{selected}.Position.X", np.nan)
        vy = row.get(f"{selected}.Position.Y", np.nan)
        return selected, vx, vy

    # vertical fallback using head or abdomen
    if pd.notna(row.get("Head.Position.X")) or pd.notna(row.get("Abdomen.Position.X")):
        if pd.notna(row.get("Head.Position.X")):       # prefer head when present
            return "Vertical", row.get("Head.Position.X", np.nan), row.get("Head.Position.Y", np.nan)
        return "Vertical", row.get("Abdomen.Position.X", np.nan), row.get("Abdomen.Position.Y", np.nan)

    # no valid coordinates
    return np.nan, np.nan, np.nan


def calculate_orientation(pointA_x, pointA_y, pointB_x, pointB_y):
    """
    Compute orientation from A→B in degrees, normalized to [0, 360) with 0 = North.
    """
    dx = pointB_x - pointA_x
    dy = pointA_y - pointB_y  # invert y to set 0 = North
    angle = np.arctan2(dy, dx)
    deg = np.degrees(angle)
    deg = (deg + 360) % 360
    deg = (deg + 90) % 360  # shift so 0 corresponds to North
    return np.round(deg, 2)

#%%% CELL 06 – SMOOTHING & HIERARCHY
"""
Purpose
Smooth binary traces with a centered running average and enforce exclusivity.

Steps
- Calculate centered running means for given columns.
- Keep at most one positive behavior per frame via hierarchy.
"""

def calculate_center_running_average(df, cols, output_cols, window_size):
    """
    Add centered running means for each column in cols into output_cols.
    """
    for col, out in zip(cols, output_cols):
        df[out] = df[col].rolling(window=(window_size + 1), center=True).mean()
    return df


def hierarchical_classifier(df, columns):
    """
    Keep only the first positive flag per frame across the given columns.
    """
    arr = df[columns].to_numpy(copy=True)
    cumsum = np.cumsum(arr, axis=1)  # row-wise cumulative positives
    arr[cumsum > 1] = 0              # zero out after the first positive
    df[columns] = arr
    return df

#%%% CELL 07 – CLASSIFIERS
"""
Purpose
Provide vectorized selection for layer labels and mark resistant bouts.

Steps
- Choose dominant label by row-wise argmax with a >0 guard.
- Mark resistant bouts when they fully cover the startle window.
"""

def classify_layer_behaviors(df, average_columns):
    """
    Vectorized pick of the column with the maximum averaged value per row.
    Returns a list of column names or NaN where the max value is ≤ 0.
    """
    vals = df[average_columns].to_numpy()
    idx = np.argmax(vals, axis=1)
    max_vals = vals[np.arange(vals.shape[0]), idx]
    out = np.array(average_columns, dtype=object)[idx]
    out[max_vals <= 0] = np.nan  # no positive evidence → NaN
    return out.tolist()


def classify_resistant_behaviors(df, RESISTANT_COLUMNS, STARTLE_WINDOW_LEN_FRAMES):
    """
    Mark resistant bouts that fully overlap a single startle window.

    Notes
    - Requires the bout to fully cover one startle window (full-overlap rule).
    """
    for col in RESISTANT_COLUMNS:
        base = col.replace("resistant_", "")             # walk / stationary / freeze
        layer2 = f"layer2_{base}"
        df[col] = 0

        # find onsets and offsets in the layer-2 trace
        on = df[df[layer2].diff() == 1].index
        off = df[df[layer2].diff() == -1].index
        if len(off) < len(on):                           # open bout at file end
            off = np.hstack((off, len(df)))

        # flag any bout with full startle-window overlap
        for a, b in zip(on, off):
            overlap = df.loc[a:b, "Startle_window"].sum() >= STARTLE_WINDOW_LEN_FRAMES
            if overlap:                                  # full overlap → resistant
                df.loc[a:b - 1, col] = 1
    return df

#%%% CELL 08 – ATOMIC WRITES
"""
Purpose
Write CSVs atomically using a temporary file and an atomic replace. Accept
Path-like destinations and convert once inside the function.

Steps
- Normalize final_path to Path.
- Write to <final>.tmp, fsync, then os.replace to the final path.
- Use str(...) only at the os.replace boundary.
"""


def write_csv_atomic(df, final_path, **to_csv_kwargs) -> None:
    """
    Write a CSV atomically via <final_path>.tmp and os.replace.

    notes
    - downstream code should ignore any *.tmp files during syncing
    """
    final_path = Path(final_path)  # normalize to Path
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    df.to_csv(tmp_path, **to_csv_kwargs)  # write tmp
    # ensure bytes hit disk before replace
    with open(tmp_path, "rb") as _f:
        os.fsync(_f.fileno())

    os.replace(str(tmp_path), str(final_path))  # atomic rename to final

#%%% CELL 09 – REPORTING
"""
Purpose
Format consistent header, progress, error, and final summary report strings.

Steps
- Provide small helpers for aligned counters and standardized lines.
"""

def _format_count_line(label: str, value: str, total_width: int) -> str:
    """
    Return "{label}   -----   {value}" aligned so value ends at total_width.
    """
    hyphens = total_width - len(label) - 6 - len(value)
    return f"{label}{' ' * 3}{'-' * hyphens}{' ' * 3}{value}"


def report_header(experiment_path: str,
                  pose_scoring: bool,
                  total_files: int,
                  to_score: int,
                  skipped_count: int,
                  scored_files: int,
                  error_count: int,
                  total_width: int = 61) -> str:
    """
    Build and return the opening report block for a scoring run.
    """
    lines = [
        f"PROCESSING: {experiment_path}",
        f"POSE:    {str(pose_scoring).upper()}",
        "",
        _format_count_line("FILES FOUND", str(total_files), total_width),
        _format_count_line("TO SCORE",    str(to_score),    total_width),
        _format_count_line("SKIPPING",    str(skipped_count), total_width),
        f"---   scored: {scored_files}   ---   errors: {error_count}   ---",
        "",
        ""
    ]
    return "\n".join(lines)


def report_scoring_line(idx: int,
                        total_files: int,
                        delta_s: float,
                        eta: str,
                        basename: str) -> str:
    """
    Format one file’s progress line with per-file time and ETA.
    """
    return (f"SCORING: file {idx}/{total_files} "
            f"({delta_s:.2f} s/file – {eta} eta)  |  {basename}")


def report_error_line(msg: str) -> str:
    """
    Return a standardized error text line for the current file.
    """
    return f"---   ERROR: {msg}   ---\n"


def report_final_summary(time_scoring: str,
                         total_files: int,
                         scored_files: int,
                         error_count: int,
                         error_reading_file: int,
                         missing_pose_file: int,
                         wrong_stimulus_count: int,
                         wrong_stimulus_duration: int,
                         lost_centroid_position: int,
                         pose_mismatch: int,
                         view_nan_exceeded: int,
                         unassigned_behavior: int,
                         no_exploration: int,
                         output_len_short: int,
                         total_width: int = 61) -> str:
    """
    Build and return the concluding summary block for the run.
    """
    pct = f"{int(round(error_count / total_files * 100))}%"
    lines = [
        _format_count_line("TIME SCORING",       time_scoring,             total_width),
        "",
        _format_count_line("FILES FOUND",        str(total_files),         total_width),
        _format_count_line("FILES SCORED",       str(scored_files),        total_width),
        "",
        _format_count_line("ERRORS",             f"{error_count} ({pct})", total_width),
        _format_count_line("---   error reading file",     str(error_reading_file),     total_width),
        _format_count_line("---   missing pose file",      str(missing_pose_file),      total_width),
        _format_count_line("---   wrong stim count",       str(wrong_stimulus_count),   total_width),
        _format_count_line("---   wrong stim duration",    str(wrong_stimulus_duration),total_width),
        _format_count_line("---   lost centroid position", str(lost_centroid_position), total_width),
        _format_count_line("---   pose length mismatch",   str(pose_mismatch),          total_width),
        _format_count_line("---   many view NaNs",         str(view_nan_exceeded),      total_width),
        _format_count_line("---   unassigned behavior",    str(unassigned_behavior),    total_width),
        _format_count_line("---   no exploration",         str(no_exploration),         total_width),
        _format_count_line("---   tracked file short",     str(output_len_short),       total_width),
    ]
    return "\n".join(lines)


#Keep the done duck. It celebrates the end of a run. Whitespace art is sacred.
def done_duck(i=15):return f"""\n\n\n{' '*(i+9)}__(·)<    ,\n{' '*(i+6)}O  \\_) )   c|_|\n{' '*i}{'~'*27}"""
