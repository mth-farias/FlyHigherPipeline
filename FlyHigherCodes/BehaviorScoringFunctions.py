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
Define error keys/messages and provide a checkpoint handler that writes an
error CSV atomically. The handler returns the updated counter and a formatted
error line; Main controls how the line is printed (tree layout).

Steps
- Declare CHECKPOINT_ERRORS with file suffixes and messages.
- Build error output path with Path and write atomically.
- Return (counter+1, 'ERROR: ...' or 'ERROR: ... (details)').
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
                    error_dir,
                    details: str | None = None) -> tuple[int, str]:
    """
    Handle a failed checkpoint: write df with the error suffix and return
    the updated counter and a formatted error line.

    returns
    - (new_counter, "ERROR: <message>" or "ERROR: <message> (<details>)")
    """
    info = CHECKPOINT_ERRORS[error_key]
    error_file = filename_tracked.replace("tracked.csv", info["file_end"])
    error_path = Path(error_dir) / error_file  # build with Path
    write_csv_atomic(df, error_path, header=True, index=False)

    base = f"ERROR: {info['message']}"
    line = f"{base} ({details})" if details else base
    return error_counter + 1, line

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
Compute speed and select a per-frame best view for pose use.

Steps
- Calculate speed in mm/s as floats (rounding done at the call site).
- Determine view from confidences or use vertical fallback logic.
"""

import numpy as np
import pandas as pd

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
    present, pick the view (Left/Right/Top/Bottom) with highest confidence.
    For vertical cases, prefer Head if present, else Abdomen; otherwise NaN.
    """
    # full key positions available → pick by confidence
    if (pd.notna(row.get("Head.Position.X")) and
        pd.notna(row.get("Thorax.Position.X")) and
        pd.notna(row.get("Abdomen.Position.X"))):
        confidences = {
            "Left": row.get("Left.Confidence", 0),
            "Right": row.get("Right.Confidence", 0),
            "Top": row.get("Top.Confidence", 0),
            "Bottom": row.get("Bottom.Confidence", 0),
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

#%%% CELL 09 – REPORTING & GLOBAL STATS
"""
Purpose
Provide helpers to format:
- Section 2 header (exact 75/72 alignment).
- Section 3 progress lines with a static 28-char bar and modular ETA.
- Section 3 error lines using short labels and 72-col dash-fill.
- Section 4 dual SESSION|GLOBAL summary with fixed-width numeric cells.
Also contains robust orientation + small utilities used in Main.

Global formatting rules (agreed):
- Banner width: 75 (centered, ALL CAPS)
- Content width: 72 (each content line starts at 2 spaces and ends at col 72)
- KV rule: dash-fill starts 2 spaces after the longest label in the block and
  ends 2 spaces before the value; internal separators have 2 spaces around.
- Timestamps stay HH:MM:SS; computed durations use lettered style.
- Center truncation for overlong values to fit width 72.
"""

from pathlib import Path
from collections import Counter
import re
import numpy as np

# -------------------------
# Shared formatting constants
# -------------------------
BANNER_WIDTH   = 75
CONTENT_WIDTH  = 72
INDENT         = "  "   # two spaces
VALUE_SEP      = "  "   # two spaces around grouped values and '---'

BAR_TOTAL = 25
BAR_LEFT  = "["
BAR_RIGHT = "]"
BAR_FILL  = "#"
BAR_EMPTY = "."

# -------------------------
# Low-level helpers
# -------------------------
def _banner_75(title: str) -> str:
    t = title.strip().upper()
    pad = max(BANNER_WIDTH - len(t) - 2, 0)
    left = pad // 2
    right = pad - left
    return "=" * left + " " + t + " " + "=" * right

def _truncate_center(s: str, max_len: int) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    keep_left = (max_len - 3) // 2
    keep_right = max_len - 3 - keep_left
    return s[:keep_left] + "..." + s[-keep_right:]

def _kv_line_72(label: str, value: str, longest_label: int) -> str:
    """
    Return a single content row (exactly 72 visible chars):
      '␣␣' + label + gap + dashes + '␣␣' + value
    where gap = (longest_label - len(label)) + 2
    and dash-fill ends 2 spaces before value.
    """
    label = str(label)
    value = str(value)

    left = INDENT + label
    gap = (max(longest_label - len(label), 0) + 2)
    left += " " * gap

    max_value_len = CONTENT_WIDTH - len(left) - 2
    v = _truncate_center(value, max_value_len)

    dash_len = CONTENT_WIDTH - len(left) - 2 - len(v)
    if dash_len < 0:
        v = _truncate_center(v, max_value_len + dash_len)
        dash_len = max(CONTENT_WIDTH - len(left) - 2 - len(v), 0)

    return left + ("-" * dash_len) + "  " + v

def _fmt_eta_modular(seconds: float) -> str:
    """
    ETA format:
      - < 1h → 'MMmSSs'
      - >=1h → 'HHhMMm' (minutes rounded from seconds; ≥30s → +1m)
    """
    s = int(round(max(seconds, 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        if sec >= 30:
            m = (m + 1) % 60
            if m == 0:
                h += 1
        return f"{h:02d}h{m:02d}m"
    return f"{m:02d}m{sec:02d}s"

def _fmt_eta_from_hhmm_str(hhmm: str) -> str:
    """
    Back-compat: convert 'HHhMM' to modular style.
      - If HH == '00' → 'MMm00s'
      - Else → 'HHhMMm'
    """
    m = re.match(r'^(\d{2})h(\d{2})$', str(hhmm))
    if not m:
        return str(hhmm)
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh == 0:
        return f"{mm:02d}m00s"
    return f"{hh:02d}h{mm:02d}m"

def _progress_bar(idx: int, total: int) -> str:
    """
    Build a fixed-width bar of BAR_TOTAL (includes the brackets).
    Interior width = BAR_TOTAL - 2.
    """
    total = max(1, int(total))
    idx = max(0, min(int(idx), total))
    inner = BAR_TOTAL - 2
    filled = int(round(inner * (idx / total)))
    filled = max(0, min(filled, inner))
    return f"{BAR_LEFT}{BAR_FILL*filled}{BAR_EMPTY*(inner - filled)}{BAR_RIGHT}"

# -------------------------
# Back-compat helpers (for other sections)
# -------------------------
_SPACER   = "   "
_BASE_BAR = 50

def _end_col(pad_to: int, max_val_width: int, base_bar: int = _BASE_BAR) -> int:
    return CONTENT_WIDTH

def _kv_line_aligned(label: str, value: str, pad_to: int, end_column: int) -> str:
    return _kv_line_72(label, value, longest_label=pad_to)

# -------------------------
# Orientation & smoothing (robust)
# -------------------------
def calculate_orientation(pointA_x, pointA_y, pointB_x, pointB_y):
    """
    Compute orientation from A→B in degrees, normalized to [0, 360) with 0 = North.

    Robustness:
    - Local NumPy import (_np) so notebook-level 'np' shadowing can’t break it.
    - Convert inputs to NumPy arrays to bypass pandas __array_ufunc__ dispatch.
    - Preserve index if inputs are pandas Series.
    """
    import numpy as _np

    idx = getattr(pointA_x, "index", None)

    ax = _np.asarray(pointA_x, dtype=float)
    ay = _np.asarray(pointA_y, dtype=float)
    bx = _np.asarray(pointB_x, dtype=float)
    by = _np.asarray(pointB_y, dtype=float)

    dx = bx - ax
    dy = ay - by  # invert y to set 0 = North

    angle = _np.arctan2(dy, dx)
    deg = _np.degrees(angle)
    deg = (deg + 360) % 360
    deg = (deg + 90) % 360  # shift so 0 corresponds to North
    deg = _np.round(deg, 2)

    if idx is not None and deg.shape == (len(idx),):
        import pandas as _pd
        return _pd.Series(deg, index=idx, name="Orientation")
    return deg

def calculate_center_running_average(df, cols, output_cols, window_size):
    """
    Centered running average over 'cols' into 'output_cols' with window_size.
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

# -------------------------
# Header (Section 2)
# -------------------------
def report_header(experimental_root,
                  pose_scoring: bool,
                  total_found: int,
                  to_score: int,
                  skipped: int,
                  already_scored: int,
                  already_errors: int) -> str:
    root = str(experimental_root)
    pose_flag = "TRUE" if pose_scoring else "FALSE"

    out = []
    out.append(_banner_75("SCORING STARTED"))
    out.append("")

    labels_top = ["PROCESSING", "POSE SCORING"]
    L1 = max(len(s) for s in labels_top)
    out.append(_kv_line_72("PROCESSING", _truncate_center(root, 9999), L1))
    out.append(_kv_line_72("POSE SCORING", pose_flag, L1))
    out.append("")

    labels_bot = ["FILES FOUND", "TO SCORE", "SKIPPING"]
    L2 = max(len(s) for s in labels_bot)
    out.append(_kv_line_72("FILES FOUND", f"{total_found}", L2))
    out.append(_kv_line_72("TO SCORE",    f"{to_score}",    L2))
    out.append(_kv_line_72("SKIPPING",    f"{skipped}",     L2))

    summary_val = f"scored: {already_scored}{VALUE_SEP}---{VALUE_SEP}errors: {already_errors}"
    out.append(_kv_line_72("", summary_val, L2))

    return "\n".join(out) + "\n\n"

# -------------------------
# Section 3 – Progress + Error lines
# -------------------------
def report_scoring_line(idx: int,
                        total_session: int,
                        sec_per_file,
                        eta) -> str:
    """
    Build the 72-col SCORING line:
      '␣␣' + 'SCORING: {i}/{n} ' + [bar28] + ' ---- ' + payload
    - If sec_per_file is None/0 or eta is None → payload is 'estimating…'
    - Otherwise:
        * sec_per_file printed as '{sec:.2f}s/file'
        * eta can be a float seconds or a legacy 'HHhMM' string.
          We render modular: '<1h → MMmSSs, ≥1h → HHhMMm'
    """
    w = len(str(max(1, total_session)))
    left = f"{INDENT}SCORING: {idx:>{w}}/{total_session:{w}} "
    bar = _progress_bar(idx, total_session)
    mid = " - "

    # Determine payload
    payload = "estimating…"
    if sec_per_file is not None and sec_per_file > 0 and eta is not None:
        if isinstance(eta, (int, float)):
            eta_str = _fmt_eta_modular(float(eta))
        else:
            eta_str = _fmt_eta_from_hhmm_str(str(eta))
        payload = f"{float(sec_per_file):.2f}s/file | {eta_str} eta"

    line = left + bar + mid + payload

    # Enforce visible width = 72
    if len(line) < CONTENT_WIDTH:
        line = line + " " * (CONTENT_WIDTH - len(line))
    else:
        line = line[:CONTENT_WIDTH]

    return line

# ---- error line helpers (short labels + detail extraction) ----

# Final short labels (exact strings you approved)
_SUMMARY_LABELS = {
    "ERROR_READING_FILE":     "error reading file",
    "WRONG_STIMULUS_COUNT":   "wrong stim count",
    "WRONG_STIMULUS_DURATION":"wrong stim duration",
    "LOST_CENTROID_POSITION": "many centroid NaNs",
    "MISSING_POSE_FILE":      "missing pose file",
    "POSE_MISMATCH":          "tracked/pose mismatch",
    "VIEW_NAN_EXCEEDED":      "many sleap view NaNs",
    "UNASSIGNED_BEHAVIOR":    "many unassigned behavior",
    "NO_EXPLORATION":         "low baseline exploration",
    "OUTPUT_LEN_SHORT":       "short output length",
}

# Render order (match your target order)
_SUMMARY_ORDER = [
    "ERROR_READING_FILE",
    "WRONG_STIMULUS_COUNT",
    "WRONG_STIMULUS_DURATION",
    "LOST_CENTROID_POSITION",
    "MISSING_POSE_FILE",
    "POSE_MISMATCH",
    "VIEW_NAN_EXCEEDED",
    "UNASSIGNED_BEHAVIOR",
    "NO_EXPLORATION",
    "OUTPUT_LEN_SHORT",
]

def _label_for_summary(error_key: str) -> str:
    # Falls back to CHECKPOINT_ERRORS long message if needed (must exist earlier in module)
    return _SUMMARY_LABELS.get(error_key,
                               CHECKPOINT_ERRORS[error_key]["message"].lower())

def _extract_detail_for_error(error_key: str, err_text: str) -> str:
    """
    err_text may be like: 'ERROR: Long message. (Detail text)'
    We pull what's inside '(...)'. For NO_EXPLORATION we compact 'Walk 9% | > 20% allowed'
    to '9% (> 20% allowed)'. For others, we keep the inside as-is.
    """
    start = err_text.find("(")
    end = err_text.rfind(")")
    inside = ""
    if start != -1 and end != -1 and end > start:
        inside = err_text[start + 1:end].strip()

    if error_key == "NO_EXPLORATION" and inside:
        inside = inside.replace("Walk ", "")
        inside = inside.replace("| >", "(>")
        if not inside.endswith(")"):
            inside += ")"
    return inside

def report_error_line(error_key: str, err_text: str) -> str:
    """
    Print the short-label error line as a 72-col KV row with tree prefix:
      '␣␣└ ERROR: {short_label} {dash}  {value}'
    """
    short_label = _label_for_summary(error_key)
    value = _extract_detail_for_error(error_key, err_text)
    label = f"└ ERROR: {short_label}"
    return _kv_line_72(label, value, longest_label=len(label))

def report_error_filename(basename: str) -> str:
    """Second tree line; ONLY line allowed to exceed 72 cols."""
    return f"{INDENT}  └ {basename}\n"

# -------------------------
# Section 4 – Summary table
# -------------------------
def _center_width(s: str, w: int = 9) -> str:
    """Return s centered to width w (truncate if somehow longer)."""
    s = str(s)
    if len(s) > w:
        return s[:w]
    left = (w - len(s)) // 2
    right = w - len(s) - left
    return (" " * left) + s + (" " * right)

def _errors_table_header(longest_left: int) -> str:
    """'  ERRORS   ---|  SESSION  |---|  GLOBAL  |' aligned to width 72."""
    left = INDENT + "ERRORS"
    gap = (max(longest_left - len("ERRORS"), 0) + 2)
    left += " " * gap
    tail = "|" + _center_width("SESSION") + "|---|" + _center_width("GLOBAL") + "|"
    dash_len = CONTENT_WIDTH - len(left) - 1 - len(tail)
    return left + ("-" * max(dash_len, 0)) + " " + tail

def _errors_total_row(longest_left: int, sess_total_str: str, glob_total_str: str) -> str:
    """'  TOTAL    ---|  2 (15%)  |---| 30 (8%)  |' aligned to width 72."""
    left = INDENT + "TOTAL"
    gap = (max(longest_left - len("TOTAL"), 0) + 2)
    left += " " * gap
    tail = "|" + _center_width(sess_total_str) + "|---|" + _center_width(glob_total_str) + "|"
    dash_len = CONTENT_WIDTH - len(left) - 1 - len(tail)
    return left + ("-" * max(dash_len, 0)) + " " + tail

def _errors_detail_row(stub_label: str, longest_left: int, sess_cnt: int, glob_cnt: int) -> str:
    """
    '  -----    <label>---|   NNN   |---|   MMM   |' aligned to width 72.
    The left text for alignment is the whole '-----    <label>' string.
    """
    left_text = "-----    " + stub_label
    left = INDENT + left_text
    gap = (max(longest_left - len(left_text), 0) + 2)
    left += " " * gap
    tail = "|" + _center_width(str(sess_cnt)) + "|---|" + _center_width(str(glob_cnt)) + "|"
    dash_len = CONTENT_WIDTH - len(left) - 1 - len(tail)
    return left + ("-" * max(dash_len, 0)) + " " + tail

def _pct_int(numer: int, denom: int) -> int:
    if denom <= 0:
        return 0
    return int(round(100 * numer / denom))

def scan_global_stats(PATHconfig) -> dict:
    """
    Return global counts under the experiment root:
      { "total": scored+pose+error (files already materialized),
        "errors": error count,
        "per_type": Counter({...}) }
    """
    p_scored = Path(PATHconfig.pScored)
    p_pose   = Path(PATHconfig.pScoredPose)
    p_error  = Path(PATHconfig.pScoredError)

    total_scored = sum(1 for _ in p_scored.rglob("*.csv")) if p_scored.exists() else 0
    total_pose   = sum(1 for _ in p_pose.rglob("*.csv"))   if p_pose.exists()   else 0
    total_error  = sum(1 for _ in p_error.rglob("*.csv"))  if p_error.exists()  else 0

    per_type = Counter()
    if p_error.exists():
        suffix2key = {v["file_end"]: k for k, v in CHECKPOINT_ERRORS.items()}
        for f in p_error.rglob("*.csv"):
            for suffix, key in suffix2key.items():
                if f.name.endswith(suffix):
                    per_type[key] += 1
                    break

    return {"total": total_scored + total_pose + total_error,
            "errors": total_error,
            "per_type": per_type}

def report_final_summary_dual(*,
                              files_found: int,
                              files_processed_session: int,
                              files_scored_session: int,
                              session_per_type: Counter,
                              global_stats: dict) -> str:
    """
    Build the Section 4 summary exactly as specified.

    - FILES FOUND, FILES PROCESSED, FILES SCORED: KV rows (72 cols).
    - Table header: ERRORS | SESSION |---| GLOBAL | with fixed numeric cells (w=9).
    - TOTAL row shows 'N (P%)' for session (vs files_processed_session) and global
      (vs files_found).
    - Detail rows listed in _SUMMARY_ORDER; values taken from counters.

    Returns the whole block including the 75-col banner and trailing blank lines.
    """
    out = []    
    out.append("")
    out.append("")
    out.append(_banner_75("SESSION SUMMARY"))
    out.append("")

    # Top KV block
    labels = ["FILES FOUND", "FILES PROCESSED", "FILES SCORED"]
    L = max(len(s) for s in labels)
    out.append(_kv_line_72("FILES FOUND",      f"{files_found}",              L))
    out.append(_kv_line_72("FILES PROCESSED",  f"{files_processed_session}",  L))
    out.append(_kv_line_72("FILES SCORED",     f"{files_scored_session}",     L))
    out.append("")

    # Table header
    detail_labels = ["-----    " + _label_for_summary(k) for k in _SUMMARY_ORDER]
    longest_left = max(len("ERRORS"), len("TOTAL"), *(len(s) for s in detail_labels))
    out.append(_errors_table_header(longest_left))

    # Totals
    session_errors = sum(session_per_type.values())
    s_pct = _pct_int(session_errors, files_processed_session)
    g_errors = global_stats.get("errors", 0)
    g_pct = _pct_int(g_errors, files_found)  # global % vs total tracked in folder

    out.append(_errors_total_row(longest_left,
                                 f"{session_errors} ({s_pct}%)",
                                 f"{g_errors} ({g_pct}%)"))

    # Detail rows in fixed order
    global_per = global_stats.get("per_type", Counter())
    for key in _SUMMARY_ORDER:
        label = _label_for_summary(key)
        s_cnt = session_per_type.get(key, 0)
        g_cnt = global_per.get(key, 0)
        out.append(_errors_detail_row(label, longest_left, s_cnt, g_cnt))

    # Trailing spacing; duck printed by caller
    return "\n".join(out)


# Keep the done duck. It celebrates the end of a run.
def done_duck(i=24): return f"""\n\n\n\n{' '*(i+9)}__(·)<    ,\n{' '*(i+6)}O  \\_) )   c|_|\n{' '*i}{'~'*27}"""
