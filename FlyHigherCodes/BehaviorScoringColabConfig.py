#%%% CELL 00 – MODULE OVERVIEW
"""
BehaviorScoringColabConfig.py

Purpose
This module stages the behavior-scoring pipeline for Google Colab. It mirrors
the Drive folder tree into /content for fast I/O, selectively copies inputs
that still need work, creates placeholders so skip logic remains valid, and
syncs fresh outputs back to Drive. It can also warm up throughput and run a
quiet background sync during long runs.

Steps
- Load canonical paths from PathConfig and validate Drive inputs.
- Create local mirrors under /content with an identical structure.
- Detect already-processed files and copy only remaining inputs.
- Create placeholders for skipped inputs and existing outputs.
- Rebase a PathConfig-like namespace to the local mirrors.
- Sync new outputs back to Drive and optionally run background sync.

Output
- Data classes for Drive and local paths.
- StageSummary without scoredpose (tracked, pose, scored, error_items).
- Public functions to stage, rebase paths, and sync results.
"""

from __future__ import annotations

#%%% CELL 01 – IMPORTS
"""
Purpose
Import standard libraries and typing helpers required for staging and syncing.

Steps
- Import subprocess, shutil, time, dataclasses, pathlib, types, typing, threading.
"""

import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple, List, Set

# Background sync
import threading

#%%% CELL 02 – DATA CLASSES
"""
Purpose
Provide containers for Drive and local paths plus a concise staging summary.

Steps
- Define DrivePaths and LocalPaths with canonical members.
- Define StageSummary with counts (no scoredpose field).
"""

@dataclass(frozen=True)
class DrivePaths:
    """Canonical locations on Google Drive (from PathConfig)."""
    root: Path
    tracked: Path
    pose: Path
    scored: Path
    scoredpose: Path
    error: Path
    codes: Path


@dataclass(frozen=True)
class LocalPaths:
    """Local mirrors under /content for fast I/O during the run."""
    root: Path
    tracked: Path
    pose: Path
    scored: Path
    scoredpose: Path
    error: Path


@dataclass(frozen=True)
class StageSummary:
    """Counts of files staged into local mirrors (inputs + existing outputs)."""
    tracked: int
    pose: int
    scored: int
    error_items: int

#%%% CELL 03 – PUBLIC API
"""
Purpose
Expose functions to load configs, validate inputs, create local mirrors, and
rebase PathConfig-like objects for use inside Colab.

Steps
- Implement load_configs, validate_inputs, and local_mirrors.
"""

def load_configs(PathConfig) -> Tuple[Path, DrivePaths]:
    """Return (drive_root, drive_paths) using the canonical PathConfig."""
    drive_root = Path(PathConfig.pExperimentalRoot)
    drive_paths = DrivePaths(
        root=drive_root,
        tracked=Path(PathConfig.pTracked),
        pose=Path(PathConfig.pPose),
        scored=Path(PathConfig.pScored),
        scoredpose=Path(PathConfig.pScoredPose),
        error=Path(PathConfig.pScoredError),
        codes=Path(PathConfig.pCodes),
    )
    return drive_root, drive_paths


def validate_inputs(
    drive_paths: DrivePaths, pose_scoring: Optional[bool] = None, *, verbose: bool = False
) -> bool:
    """
    Validate required inputs on Drive and auto-detect pose_scoring when None.
    Returns final pose_scoring (auto-detected if None).
    """
    _require_csv_folder(drive_paths.tracked, "Tracked inputs not found or empty")

    # Auto-detect: pose scoring only if pose folder exists and has CSVs
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    if pose_scoring:
        _require_csv_folder(
            drive_paths.pose,
            "POSE_SCORING=True but Pose inputs not found or empty. "
            "Set POSE_SCORING=False or add Pose CSVs.",
        )

    if verbose:
        print(f"Validation OK. pose_scoring={pose_scoring}")

    return pose_scoring


def local_mirrors(
    drive_root: Path, drive_paths: DrivePaths, local_root: Optional[Path] = None, *, verbose: bool = False
) -> LocalPaths:
    """
    Create local mirrors under /content with an identical substructure.
    """
    # Namespace per experiment under /content/exp_runs/<experiment_name>
    base = local_root or (Path("/content/exp_runs") / drive_root.name)

    def mirror(p: Path) -> Path:
        return base / p.relative_to(drive_root)

    local = LocalPaths(
        root=base,
        tracked=mirror(drive_paths.tracked),
        pose=mirror(drive_paths.pose),
        scored=mirror(drive_paths.scored),
        scoredpose=mirror(drive_paths.scoredpose),
        error=mirror(drive_paths.error),
    )

    # Create all required dirs; idempotent behavior
    for d in (local.tracked, local.pose, local.scored, local.scoredpose, local.error):
        d.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Virtual paths created")
    return local

#%%% CELL 04 – PROCESSED DETECTION & SELECTIVE COPY
"""
Purpose
Provide helpers to list CSVs, detect processed items on Drive, and copy only
the files that still need work.

Steps
- List CSVs in a tree.
- Map tracked→scored names and detect any matching error file.
- Copy a selected list using rsync or a Python fallback.
"""

def _list_csvs(src: Path) -> List[Path]:
    if not src.exists():
        return []
    return [p for p in src.rglob("*.csv") if p.is_file()]


def _scored_name_for(tracked_name: str, pose_scoring: bool) -> str:
    if tracked_name.endswith("tracked.csv"):
        return tracked_name.replace(
            "tracked.csv", "scored_pose.csv" if pose_scoring else "scored.csv"
        )
    return tracked_name.replace(
        ".csv", "_scored_pose.csv" if pose_scoring else "_scored.csv"
    )


def _has_matching_error(tracked_name: str, error_dir: Path) -> bool:
    base = tracked_name.replace("tracked.csv", "")
    if not error_dir.exists():
        return False
    try:
        for f in error_dir.iterdir():
            if f.is_file() and f.name.startswith(base):
                return True  # any *_error.csv matches
    except Exception:
        pass  # ignore transient read errors and continue
    return False


def _is_processed_on_drive(tracked_rel: str, drive_paths: DrivePaths, pose_scoring: bool) -> bool:
    scored_name = _scored_name_for(tracked_rel, pose_scoring)
    scored_hit = (drive_paths.scoredpose if pose_scoring else drive_paths.scored) / scored_name
    if scored_hit.exists():
        return True
    if _has_matching_error(tracked_rel, drive_paths.error):
        return True
    return False


def _copy_selected(src: Path, dst: Path, rel_files: List[str]) -> int:
    """Copy only the files listed in rel_files; return count successfully copied."""
    if not rel_files:
        return 0
    dst.mkdir(parents=True, exist_ok=True)

    # Prefer rsync when available; it handles trees efficiently
    if shutil.which("rsync"):
        files_list = dst / ".rsync_files.txt"
        try:
            files_list.write_text("\n".join(rel_files) + "\n")
            cmd = [
                "rsync", "-a", "--no-compress", "--prune-empty-dirs",
                "--files-from", str(files_list), str(src) + "/", str(dst) + "/"
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or "rsync failed")  # bubble up
        except Exception:
            # Fallback: copy one-by-one; robust but slower
            for rel in rel_files:
                s = src / rel
                d = dst / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                if s.exists():
                    shutil.copy2(s, d)
        finally:
            try:
                files_list.unlink(missing_ok=True)
            except Exception:
                pass
        return len(rel_files)

    # Pure-Python fallback if rsync is not present
    copied = 0
    for rel in rel_files:
        s = src / rel
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.exists():
            shutil.copy2(s, d)
            copied += 1
    return copied


#%%% CELL 04A – FORMAT CONSTANTS & HELPERS
"""
Purpose
Provide shared formatting helpers for consistent console output:
- Banners at 75 chars
- Content rows at 72 chars with 2-space indent
- Dash-fill that starts 2 spaces after the longest label in the block and
  ends 2 spaces before the value
- Lettered duration formatting (MMmSSs or HHhMMm)
- Center truncation for long values
"""

# Shared constants & rules (for implementation)
BANNER_WIDTH = 75
CONTENT_WIDTH = 72
INDENT = "  "         # two spaces
VALUE_SEP = "  "      # two spaces before/after value groups & '---'

# Progress bar constants (not used in this cell, here for global consistency)
BAR_TOTAL = 28        # includes brackets
BAR_LEFT = "["
BAR_RIGHT = "]"
BAR_FILL = "#"
BAR_EMPTY = "."

def _banner(title: str) -> str:
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

def _fmt_duration_lettered(seconds: float) -> str:
    """
    Return lettered duration:
      - < 1h  -> 'MMmSSs'
      - >=1h  -> 'HHhMMm'
    Seconds are rounded to nearest second; minutes are rounded from the seconds.
    """
    s = int(round(max(seconds, 0)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        # Round minutes from seconds (>=30s -> +1m)
        if sec >= 30:
            m = (m + 1) % 60
            if m == 0:
                h += 1
        return f"{h:02d}h{m:02d}m"
    return f"{m:02d}m{sec:02d}s"

def _kv_line(label: str, value: str, longest_label: int) -> str:
    """
    Build a single content row (72 chars total):
      '␣␣' + label + gap + dashes + '␣␣' + value
    where
      gap = (longest_label - len(label)) + 2  # 2 spaces after longest label
    and the dash-fill ends 2 spaces before the value.
    If value overflows, it's center-truncated to fit exactly 72 cols.
    """
    label = str(label)
    value = str(value)

    # Left part: indent + label + gap to align dash start
    left = INDENT + label
    gap = (max(longest_label - len(label), 0) + 2)
    left += " " * gap

    # Compute max space for value (2 spaces reserved before value)
    max_value_len = CONTENT_WIDTH - len(left) - 2
    v = _truncate_center(value, max_value_len)

    # Dash-fill length so the final line ends exactly at CONTENT_WIDTH
    dash_len = CONTENT_WIDTH - len(left) - 2 - len(v)
    if dash_len < 0:
        # (Shouldn't happen because we truncated v, but guard anyway)
        v = _truncate_center(v, max_value_len + dash_len)
        dash_len = max(CONTENT_WIDTH - len(left) - 2 - len(v), 0)

    return left + ("-" * dash_len) + "  " + v



#%%% CELL 05 – PLACEHOLDERS & WARMUP
"""
Purpose
Create local placeholders so skip logic works without copying all outputs and
measure Drive→/content throughput to estimate staging time.

Steps
- Touch zero-byte placeholders for existing outputs on Drive.
- Read a short sample of real inputs to estimate MB/s.
"""

def _mirror_placeholders(src: Path, dst: Path, patterns: Optional[Iterable[str]]) -> int:
    """Create zero-byte placeholders in dst for files that exist in src."""
    if not src.exists():
        return 0

    if patterns is None:
        files = [p for p in src.rglob("*") if p.is_file()]
    else:
        files = []
        for pat in patterns:
            files.extend(src.rglob(pat))
        files = [p for p in files if p.is_file()]

    n = 0
    for f in files:
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.touch()  # zero-byte placeholder
            n += 1
    return n


def _warmup_measure_speed_mbps(
    drive_paths: DrivePaths,
    to_copy_tracked: List[str],
    to_copy_pose: List[str],
    *,
    min_seconds: float = 5.0,
    min_megabytes: float = 100.0,
    max_seconds: float = 10.0,
    chunk_size: int = 8 * 1024 * 1024,  # 8 MB
) -> float | None:
    """
    Measure Drive→/content throughput by reading the actual to-copy files for a
    short warm-up. Data is discarded; no writes occur. Returns MB/s or None.
    """
    import itertools

    # Build absolute paths on Drive for the to-copy inputs
    candidates: List[Path] = []
    for rel in to_copy_tracked:
        p = drive_paths.tracked / rel
        if p.exists() and p.is_file():
            candidates.append(p)
    for rel in to_copy_pose:
        p = drive_paths.pose / rel
        if p.exists() and p.is_file():
            candidates.append(p)

    if not candidates:
        return None

    target_bytes = int(min_megabytes * 1024 * 1024)
    start = time.perf_counter()
    read_bytes = 0

    # Cycle files until thresholds or the max time is reached
    for file_path in itertools.cycle(candidates):
        if time.perf_counter() - start >= max_seconds:
            break
        try:
            with open(file_path, "rb") as f:
                while True:
                    if (time.perf_counter() - start) >= max_seconds:
                        break
                    if ((time.perf_counter() - start) >= min_seconds and
                            read_bytes >= target_bytes):
                        break
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break  # end of file
                    read_bytes += len(chunk)
        except Exception:
            continue  # skip unreadable files and continue

        # Early stop once we hit both thresholds
        if ((time.perf_counter() - start) >= min_seconds and
                read_bytes >= target_bytes):
            break

    elapsed = max(time.perf_counter() - start, 1e-6)
    if read_bytes <= 0:
        return None
    return (read_bytes / (1024 * 1024)) / elapsed

#%%% CELL 06 – STAGING (ETA, PLACEHOLDERS, INPUT COPY)
"""
Purpose
Stage inputs into local mirrors with a live-updating **two-line** STAGING block
(72 cols), then print the aligned KV rows and the READY banner once done.

Shape (only two lines update during staging):
======================== LOADING FILES FROM DRIVE =========================

  SOURCE DRIVE PATH  --  /content/drive/My Drive/FHAnalysisPipeline/10DB
  RUN STARTED AT     -----------------------------------------  09:00:59


  STAGING [##########################.................................]
  -------------------------  60.8/60.8 MB  ---  2.2 MB/s  ---  00m00s eta


  Loaded            ------------------------  tracked: 13  ---  pose: 13
  Skipped           -------------------------  scored: 0  ---  errors: 0
  Loading time      --------------------------------------------  00m14s

============================== READY TO RUN ===============================

Notes
- Banners are width 75; content rows are width 72 (exact).
- SOURCE DRIVE PATH is left-truncated (…/tail) to fit 72.
- Two STAGING lines updated in place via IPython display handle.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import os, sys, time

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# Expect earlier cells to define:
#   BANNER_WIDTH, CONTENT_WIDTH, INDENT, VALUE_SEP
#   _banner(title), _kv_line(label, value, longest_label), _fmt_duration_lettered
#   _list_csvs(root), _is_processed_on_drive(rel, drive_paths, pose_scoring)
#   _mirror_placeholders(src_root, dst_root, patterns)
#   DrivePaths, LocalPaths, StageSummary dataclasses

# Cache last measured speed across calls in this process
try:
    _LAST_MEASURED_SPEED_MBPS
except NameError:
    _LAST_MEASURED_SPEED_MBPS = None

def _truncate_left(s: str, max_len: int) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[-max_len:]
    return "..." + s[-(max_len - 3):]

def _fmt_bytes_scaled(num_bytes: int) -> Tuple[str, str]:
    n = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while i < len(units) - 1 and n >= 1024.0:
        n /= 1024.0
        i += 1
    if n >= 100:
        return f"{int(round(n))}", units[i]
    return f"{n:.1f}", units[i]

def _fmt_bytes_pair(done: int, total: int) -> str:
    _, unit = _fmt_bytes_scaled(total)
    scale = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3, "TB":1024**4}[unit]
    dn = done / scale
    tn = total / scale
    if tn >= 100:
        d_str = f"{int(round(dn))}"
        t_str = f"{int(round(tn))}"
    else:
        d_str = f"{dn:.1f}"
        t_str = f"{tn:.1f}"
    return f"{d_str}/{t_str} {unit}"

def _staging_two_lines(done_bytes: int, total_bytes: int,
                       rate_mbps: float, eta_seconds: float) -> Tuple[str, str]:
    """
    Two 72-col lines aligned at the bar column:
      1) "  STAGING [bar]"
      2) "            dashes  <done/total>  ---  <rate> MB/s  ---  <eta> eta"
    """
    # line 1: bar (no colon)
    label = f"{INDENT}STAGING "
    bar_width = CONTENT_WIDTH - len(label)  # includes brackets
    inner = max(0, bar_width - 2)
    frac = 1.0 if total_bytes <= 0 else min(max(done_bytes / float(max(1, total_bytes)), 0.0), 1.0)
    filled = int(round(inner * frac))
    line1 = label + "[" + ("#" * filled) + ("." * (inner - filled)) + "]"
    if len(line1) < CONTENT_WIDTH:
        line1 += " " * (CONTENT_WIDTH - len(line1))
    elif len(line1) > CONTENT_WIDTH:
        line1 = line1[:CONTENT_WIDTH]

    # line 2: metrics aligned to the bar start
    right_start_width = len("STAGING  ")
    prefix = INDENT + (" " * right_start_width)

    bytes_pair = _fmt_bytes_pair(done_bytes, total_bytes)
    rate_str = f"{max(0.0, float(rate_mbps)):.1f} MB/s"
    eta_str  = _fmt_duration_lettered(max(0.0, float(eta_seconds)))
    payload  = f"{bytes_pair}  ---  {rate_str}  ---  {eta_str} eta"

    dash_len = max(0, CONTENT_WIDTH - len(prefix) - 2 - len(payload))
    line2 = prefix + ("-" * dash_len) + "  " + payload
    if len(line2) < CONTENT_WIDTH:
        line2 += " " * (CONTENT_WIDTH - len(line2))
    elif len(line2) > CONTENT_WIDTH:
        # Rare: drop decimals on rate, then re-fit
        rate_str = f"{int(round(max(0.0, float(rate_mbps))))} MB/s"
        payload  = f"{bytes_pair}  ---  {rate_str}  ---  {eta_str} eta"
        dash_len = max(0, CONTENT_WIDTH - len(prefix) - 2 - len(payload))
        line2 = prefix + ("-" * dash_len) + "  " + payload
        if len(line2) > CONTENT_WIDTH:
            line2 = line2[:CONTENT_WIDTH]

    return line1, line2

def _sum_sizes(paths: List[Path]) -> int:
    tot = 0
    for p in paths:
        try:
            tot += p.stat().st_size
        except OSError:
            pass
    return tot

def _copy_file_with_progress(src: Path, dst: Path, tick) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    CHUNK = 1024 * 1024  # 1 MiB
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            buf = fi.read(CHUNK)
            if not buf:
                break
            fo.write(buf)
            tick(len(buf))
        fo.flush()
        os.fsync(fo.fileno())
    tmp.replace(dst)

def _copy_selected_progress(src_root: Path, dst_root: Path,
                            rel_list: List[str], update_bytes) -> int:
    copied = 0
    for rel in rel_list:
        src = src_root / rel
        if not src.exists():
            continue
        dst = dst_root / rel
        try:
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
        except OSError:
            pass
        try:
            _copy_file_with_progress(src, dst, update_bytes)
            copied += 1
        except Exception:
            continue
    return copied

def stage_to_local(
    drive_paths: DrivePaths,
    local_paths: LocalPaths,
    pose_scoring: Optional[bool] = None,
    *,
    verbose: bool = True,
    prior_speed_mbps: Optional[float] = None,
) -> StageSummary:
    """Stage inputs selectively and mirror existing outputs as placeholders."""
    global _LAST_MEASURED_SPEED_MBPS

    # ---- BANNER ----
    print(_banner("LOADING FILES FROM DRIVE"))
    print()

    # Detect pose scoring
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    # Gather files & build work lists
    tracked_files = _list_csvs(drive_paths.tracked)
    pose_files    = _list_csvs(drive_paths.pose) if pose_scoring else []
    rel_tracked = [str(p.relative_to(drive_paths.tracked)) for p in tracked_files]

    to_copy_tracked, skipped_already_processed = [], []
    for rel in rel_tracked:
        if _is_processed_on_drive(rel, drive_paths, pose_scoring):
            skipped_already_processed.append(rel)
        else:
            to_copy_tracked.append(rel)

    to_copy_pose = []
    if pose_scoring and drive_paths.pose.exists():
        for rel in to_copy_tracked:
            pose_rel = rel.replace("tracked.csv", "pose.csv")
            if (drive_paths.pose / pose_rel).exists():
                to_copy_pose.append(pose_rel)

    # Top block (immediate)
    if ZoneInfo is not None:
        run_started_at = datetime.now(ZoneInfo("Europe/Lisbon")).strftime("%H:%M:%S")
    else:
        run_started_at = datetime.now().strftime("%H:%M:%S")

    head = f"{INDENT}SOURCE DRIVE PATH  --  "
    tail_max = CONTENT_WIDTH - len(head)
    tail_str = _truncate_left(str(drive_paths.root), tail_max)
    src_line = head + tail_str
    if len(src_line) < CONTENT_WIDTH:
        src_line += " " * (CONTENT_WIDTH - len(src_line))
    print(src_line)

    block1_labels = ["SOURCE DRIVE PATH", "RUN STARTED AT"]
    L1 = max(len(x) for x in block1_labels)
    print(_kv_line("RUN STARTED AT", run_started_at, L1))
    print(); print()

    # Compute total bytes & show live two-line STAGING
    to_copy_tracked_paths = [drive_paths.tracked / rel for rel in to_copy_tracked]
    to_copy_pose_paths    = [drive_paths.pose / rel    for rel in to_copy_pose]
    total_bytes = _sum_sizes(to_copy_tracked_paths) + _sum_sizes(to_copy_pose_paths)

    DEFAULT_SPEED_MBPS = 2.6
    quick_speed = (prior_speed_mbps if (prior_speed_mbps and prior_speed_mbps > 0)
                   else (_LAST_MEASURED_SPEED_MBPS if _LAST_MEASURED_SPEED_MBPS else DEFAULT_SPEED_MBPS))
    eta_guess_s = ((total_bytes / (1024 * 1024)) / quick_speed) if (quick_speed and total_bytes > 0) else 0.0

    # Display handle for exactly two lines
    display_handle = None
    can_display = False
    try:
        from IPython.display import display, Markdown
        can_display = True
        l1, l2 = _staging_two_lines(0, int(total_bytes), float(quick_speed or 0.0), float(eta_guess_s))
        display_handle = display(Markdown(f"```text\n{l1}\n{l2}\n```"), display_id=True)
    except Exception:
        l1, l2 = _staging_two_lines(0, int(total_bytes), float(quick_speed or 0.0), float(eta_guess_s))
        print(l1); print(l2); print(); print()

    # Copy with updates
    t0 = time.perf_counter()
    bytes_done = 0
    last_refresh = 0.0
    REFRESH_EVERY = 0.25

    def _tick(n):
        nonlocal bytes_done, last_refresh
        bytes_done += n
        now = time.perf_counter()
        if (now - last_refresh) >= REFRESH_EVERY or bytes_done >= total_bytes:
            last_refresh = now
            elapsed = max(1e-6, now - t0)
            rate_mbps = (bytes_done / elapsed) / (1024 * 1024)
            remaining = max(0, int(total_bytes) - bytes_done)
            eta_s = (remaining / (rate_mbps * 1024 * 1024)) if rate_mbps > 0 else 0.0
            l1, l2 = _staging_two_lines(bytes_done, int(total_bytes), rate_mbps, eta_s)
            if can_display and display_handle is not None:
                try:
                    display_handle.update(Markdown(f"```text\n{l1}\n{l2}\n```"))
                except Exception:
                    pass

    n_tracked = _copy_selected_progress(drive_paths.tracked, local_paths.tracked, to_copy_tracked, _tick)
    n_pose = 0
    if pose_scoring and to_copy_pose:
        n_pose = _copy_selected_progress(drive_paths.pose, local_paths.pose, to_copy_pose, _tick)

    for rel in skipped_already_processed:
        p = local_paths.tracked / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()
    if pose_scoring and drive_paths.pose.exists():
        for rel in skipped_already_processed:
            pose_rel = rel.replace("tracked.csv", "pose.csv")
            pose_src = drive_paths.pose / pose_rel
            if pose_src.exists():
                q = local_paths.pose / pose_rel
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists():
                    q.touch()

    n_scored = _mirror_placeholders(drive_paths.scored, local_paths.scored, patterns=("*.csv",))
    if pose_scoring:
        _ = _mirror_placeholders(drive_paths.scoredpose, local_paths.scoredpose, patterns=("*.csv",))
    n_error = _mirror_placeholders(drive_paths.error, local_paths.error, patterns=None)

    # Final state & cache speed
    t1 = time.perf_counter()
    elapsed = max(1e-6, t1 - t0)
    globals()["_LAST_LOADING_SECONDS"] = elapsed
    measured_mbps = (bytes_done / elapsed) / (1024 * 1024)
    _LAST_MEASURED_SPEED_MBPS = measured_mbps

    # Finalize the two-line block
    try:
        if can_display and display_handle is not None:
            l1, l2 = _staging_two_lines(int(total_bytes), int(total_bytes), measured_mbps, 0.0)
            display_handle.update(Markdown(f"```text\n{l1}\n{l2}\n```"))
    except Exception:
        pass

    # Tail printing
    if verbose:
        block2_labels = ["LOADED", "SKIPPED", "LOAD TIME"]
        L2 = max(len(x) for x in block2_labels)
        loaded_value  = f"tracked: {n_tracked}{VALUE_SEP}---{VALUE_SEP}pose: {n_pose}"
        skipped_value = f"scored: {n_scored}{VALUE_SEP}---{VALUE_SEP}errors: {n_error}"
        loading_time_value = _fmt_duration_lettered(elapsed)

        print(); print()
        print(_kv_line("LOADED",       loaded_value,       L2))
        print(_kv_line("SKIPPED",      skipped_value,      L2))
        print(_kv_line("LOAD TIME",    loading_time_value, L2))
        print()
        print(_banner("READY TO RUN"))
        print()

    return StageSummary(tracked=n_tracked, pose=n_pose, scored=n_scored, error_items=n_error)





#%%% CELL 07 – LOCAL PATHCONFIG REBASE
"""
Purpose
Return a PathConfig-like object rebased under the local root using Path
objects (not strings) and inject convenience globs computed from those Paths.
This lets the pipeline run unchanged while targeting local mirrors. Globs are
generators; iterate them after rebasing.

Steps
- Convert Drive-based paths to local Path objects.
- Preserve non-path attributes and callables.
- Inject gTracked/gPose/gScored/gScoredPose/gError using .rglob().
"""

def make_local_pathconfig(PathConfig, local_paths: LocalPaths):
    """
    Rebase every Drive path under the local root and return a namespace with
    Path objects plus fresh convenience globs. Globs are generators; iterate
    them after calling this function.
    """
    drive_root = Path(PathConfig.pExperimentalRoot)
    local_root = Path(local_paths.root)

    rebased = {}

    for name, value in vars(PathConfig).items():
        if name.startswith("__"):
            continue  # skip dunder attributes
        if callable(value):
            rebased[name] = value
            continue
        try:
            p = Path(value)
            rel = p.relative_to(drive_root)
            rebased[name] = local_root / rel  # keep as Path, not str
        except Exception:
            rebased[name] = value  # keep non-path values verbatim

    # Keep Codes path pointing to Drive for clarity in notebooks
    if hasattr(PathConfig, "pCodes"):
        rebased["pCodes"] = PathConfig.pCodes

    # Inject convenience globs rebased to local mirrors
    try:
        p_tracked = (rebased["pTracked"] if isinstance(rebased["pTracked"], Path)
                     else Path(rebased["pTracked"]))
        p_pose = (rebased["pPose"] if isinstance(rebased["pPose"], Path)
                  else Path(rebased["pPose"]))
        p_scored = (rebased["pScored"] if isinstance(rebased["pScored"], Path)
                    else Path(rebased["pScored"]))
        p_scoredpose = (rebased["pScoredPose"] if isinstance(rebased["pScoredPose"], Path)
                        else Path(rebased["pScoredPose"]))
        p_error = (rebased["pScoredError"] if isinstance(rebased["pScoredError"], Path)
                   else Path(rebased["pScoredError"]))
    except KeyError:
        # if any required path is missing, skip globs silently
        pass
    else:
        # note: these are generators; they are evaluated when iterated
        rebased["gTracked"] = p_tracked.rglob("*tracked.csv")
        rebased["gPose"] = p_pose.rglob("*pose.csv")
        rebased["gScored"] = p_scored.rglob("*.csv")
        rebased["gScoredPose"] = p_scoredpose.rglob("*.csv")
        rebased["gError"] = p_error.rglob("*.csv")

    return SimpleNamespace(**rebased)


#%%% CELL 08 – SYNC OUTPUTS BACK TO DRIVE
"""
Purpose
Copy outputs from local → Drive in bulk, skipping placeholders and avoiding
overwrites when running in upload mode. Then print the final 75/72-formatted
banner with the destination path and timing breakdown — **only if scoring
completed** (we consider it completed when a positive 'scoring_seconds' is
passed in).

Formatting (matches your global rules)
- Banner: 75 chars
- Content rows: 72 chars (2-space indent; dash-fill starts 2 spaces after the
  longest label in the block and ends 2 spaces before the value)
- Only durations use lettered format (MMmSSs or HHhMMm).

Inputs
- run_started_at: optional 'HH:MM:SS' (for provenance; not used in math)
- loading_seconds: optional numeric duration from Section 1 staging
- scoring_seconds: optional numeric duration computed in Main just before sync
- dest_path_override: optional alternative path string for the 'SAVED IN DRIVE' row
- verbose: if True, we may print — but only when scoring_seconds > 0
"""

# Keep the done duck. It celebrates the end of a run.
def done_duck(i=24): return f"""\n\n\n{' '*(i+9)}__(·)<    ,\n{' '*(i+6)}O  \\_) )   c|_|\n{' '*i}{'~'*27}"""

def sync_outputs_back(
    local_paths: LocalPaths,
    drive_paths: DrivePaths,
    pose_scoring: Optional[bool] = None,
    *,
    verbose: bool = True,
    run_started_at: Optional[str] = None,
    loading_seconds: Optional[float] = None,
    scoring_seconds: Optional[float] = None,
    dest_path_override: Optional[str] = None,
) -> None:
    """
    Copy outputs from local → Drive in bulk (resilient, skip placeholders),
    then (optionally) print the final 'SCORING AND SAVING COMPLETE' block with:
      - SAVED IN DRIVE
      - SESSION TIME (Loading + Scoring)              <-- per your latest spec
      - Subrows: LOADING, SCORING                     <-- (no SAVING subrow)
      - Closing 75-char '=' banner

    Printing occurs only if 'verbose' is True **and** scoring_seconds > 0.
    """
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    # --- Copy + measure saving time (kept but not shown as a subrow) ---
    t0 = time.perf_counter()

    _copy_tree(local_paths.scored, drive_paths.scored, patterns=("*.csv",), upload_mode=True)
    _copy_tree(local_paths.error, drive_paths.error, patterns=None, upload_mode=True)
    if pose_scoring:
        _copy_tree(local_paths.scoredpose, drive_paths.scoredpose, patterns=("*.csv",), upload_mode=True)

    t1 = time.perf_counter()
    saving_seconds = max(0.0, t1 - t0)

    # --- Only show the pretty final block when scoring_seconds indicates completion ---
    if verbose and scoring_seconds and scoring_seconds > 0.0:
        # Duck + banner
        print(done_duck())
        print(_banner("SCORING AND SAVING COMPLETE"))
        print()

        # Row: SAVED IN DRIVE
        dest_str = str(dest_path_override) if dest_path_override else str(drive_paths.root)
        L_saved = len("SAVED IN DRIVE")
        print(_kv_line("SAVED IN DRIVE", dest_str, L_saved))
        print()

        # If caller didn't pass loading_seconds, fall back to last measured staging time
        if loading_seconds is None:
            loading_seconds = globals().get("_LAST_LOADING_SECONDS", 0.0)

        # Durations (we follow your current definition of SESSION = Loading + Scoring + Saving)
        load_s  = max(0.0, float(loading_seconds or 0.0))
        score_s = max(0.0, float(scoring_seconds or 0.0))
        save_s  = max(0.0, float(saving_seconds))
        session_seconds = load_s + score_s

        session_str = _fmt_duration_lettered(session_seconds)
        load_str    = _fmt_duration_lettered(load_s)
        score_str   = _fmt_duration_lettered(score_s)

        # Align SESSION TIME dashes with SAVED IN DRIVE (use L_saved)
        print(_kv_line("SESSION TIME", session_str, L_saved))

        # Subrows: LOADING and SCORING only (uppercase labels, as requested)
        sub_labels = [
            "------------    LOADING",
            "------------    SCORING",
        ]
        L_sub = max(len(s) for s in sub_labels)
        print(_kv_line("------------    LOADING", load_str,  L_sub))
        print(_kv_line("------------    SCORING", score_str, L_sub))

        print()
        print("=" * 75)



#%%% CELL 09 – BACKGROUND SYNC (SILENT, FINAL FILES ONLY)
"""
Purpose
Provide a silent background thread that syncs after exactly N new final files
appear. This reduces manual sync invocations during long runs.

Steps
- Track seen outputs and compare with the current set every few seconds.
- Sync after an exact batch of new files is detected.
- Allow explicit start/stop around a run.
"""

_bg_state = {
    "thread": None,
    "stop": threading.Event(),
    "seen": set(),        # set[str] of files already accounted for
    "batch_size": 30,
}


def _final_csvs_in_dir(p: Path) -> Set[str]:
    if not p.exists():
        return set()
    return {str(f.resolve()) for f in p.rglob("*.csv") if not f.name.endswith(".tmp")}


def _final_outputs_set(local_paths: LocalPaths, pose_scoring: bool) -> Set[str]:
    s = _final_csvs_in_dir(local_paths.scored) | _final_csvs_in_dir(local_paths.error)
    if pose_scoring:
        s |= _final_csvs_in_dir(local_paths.scoredpose)
    return s


def start_background_sync(
    local_paths: LocalPaths,
    drive_paths: DrivePaths,
    pose_scoring: Optional[bool] = None,
    batch_size: int = 30
) -> None:
    """Start a silent background sync that triggers after EXACTLY batch_size new files."""
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    if _bg_state["thread"] and _bg_state["thread"].is_alive():
        return  # already running

    _bg_state["stop"].clear()
    _bg_state["batch_size"] = int(batch_size)
    _bg_state["seen"] = _final_outputs_set(local_paths, pose_scoring)

    def _worker():
        while not _bg_state["stop"].is_set():
            time.sleep(5)
            try:
                current = _final_outputs_set(local_paths, pose_scoring)
                new_files = current - _bg_state["seen"]
                if len(new_files) >= _bg_state["batch_size"]:
                    try:
                        sync_outputs_back(local_paths, drive_paths, pose_scoring, verbose=False)
                    finally:
                        _bg_state["seen"] = current
            except Exception:
                pass  # stay quiet and keep trying

    t = threading.Thread(target=_worker, daemon=True)
    _bg_state["thread"] = t
    t.start()


def stop_background_sync() -> None:
    """Stop the background sync thread."""
    if _bg_state["thread"]:
        _bg_state["stop"].set()
        _bg_state["thread"].join(timeout=15)
        _bg_state["thread"] = None

#%%% CELL 10 – INTERNAL HELPERS (RESILIENT COPY)
"""
Purpose
Implement resilient copying using rsync when available and a Python fallback
when needed. Handle Drive mount drops with a remount attempt.

Steps
- Validate CSV folders, copy trees with patterns, and remount on failure.
- Provide rsync wrapper and simple formatting helpers.
"""

def _require_csv_folder(folder: Path, message: str) -> None:
    if not folder.exists():
        raise RuntimeError(f"{message}: {folder} (folder not found)")
    if not any(folder.rglob("*.csv")):
        raise RuntimeError(f"{message}: {folder} (no CSV files found)")


def _copy_tree(
    src: Path, dst: Path, patterns: Optional[Iterable[str]], *, upload_mode: bool = False
) -> int:
    """
    Copy files from src to dst; return count of files attempted to copy.

    - Prefer rsync. If the mount drops, remount once and retry.
    - Upload mode: do not overwrite Drive files; skip zero-byte placeholders.
    """
    if not src.exists():
        return 0

    dst.mkdir(parents=True, exist_ok=True)

    # Build candidate file list
    if patterns is None:
        candidates = [p for p in src.rglob("*") if p.is_file()]
    else:
        candidates = []
        for pat in patterns:
            candidates.extend(src.rglob(pat))
        candidates = [p for p in candidates if p.is_file()]

    file_count = len(candidates)
    if file_count == 0:
        return 0

    # Prefer rsync
    if shutil.which("rsync"):
        try:
            _rsync_copy(src, dst, patterns, upload_mode=upload_mode)
            return file_count
        except _MountDropError:
            if _remount_drive():
                _rsync_copy(src, dst, patterns, upload_mode=upload_mode)
                return file_count
            raise RuntimeError(f"Drive remount failed while copying: {src} → {dst}")
        except Exception:
            pass  # fall through to Python copy

    # Python fallback with the same rules as rsync mode
    did_remount = False
    for path in candidates:
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if upload_mode:
                try:
                    if path.stat().st_size == 0:
                        continue  # skip placeholders
                except OSError:
                    continue
                if out.exists():
                    continue  # do not overwrite Drive files
            shutil.copy2(path, out)
        except OSError as oe:
            # 107 is a typical "transport endpoint" error on /content/drive
            if not did_remount and getattr(oe, "errno", None) == 107:
                did_remount = _remount_drive()
                if did_remount:
                    if upload_mode:
                        try:
                            if path.stat().st_size == 0:
                                continue
                        except OSError:
                            continue
                        if out.exists():
                            continue
                    shutil.copy2(path, out)
                    continue
            raise
    return file_count


def _rsync_copy(
    src: Path, dst: Path, patterns: Optional[Iterable[str]], *, upload_mode: bool = False
) -> None:
    """Copy using rsync and raise _MountDropError on mount-related failures."""
    cmd = ["rsync", "-a", "--no-compress", "--prune-empty-dirs"]

    # Upload mode: do not overwrite existing files; skip zero-byte placeholders
    if upload_mode:
        cmd += ["--ignore-existing", "--min-size=1"]

    if patterns is None:
        pass  # copy everything
    else:
        pats = set(patterns)
        if pats == {"*.csv"} or pats == {".csv", "*.csv"}:
            cmd += ["--include", "*/", "--include", "*.csv", "--exclude", "*"]
        else:
            # Fallback handled by caller with Python copy
            raise RuntimeError("Unsupported include pattern; falling back to Python copy.")

    dst.mkdir(parents=True, exist_ok=True)
    src_arg = str(src) + "/"
    dst_arg = str(dst) + "/"

    proc = subprocess.run(cmd + [src_arg, dst_arg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "") + " " + (proc.stdout or "")
        if ("Transport endpoint is not connected" in err) or ("Input/output error" in err):
            raise _MountDropError(err)  # signal a remount condition
        raise RuntimeError(f"rsync failed ({proc.returncode}): {err.strip()}")


class _MountDropError(RuntimeError):
    """Raised when the Drive mount drops during rsync operations."""
    pass


def _remount_drive() -> bool:
    """Attempt to remount /content/drive; return True when successful."""
    try:
        subprocess.run(["fusermount", "-u", "/content/drive"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        subprocess.run(["rm", "-rf", "/content/drive"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        from google.colab import drive as colab_drive
        colab_drive.mount("/content/drive", force_remount=True)
        return True
    except Exception:
        return False


def _fmt_seconds(s: float) -> str:
    """Return HH:MM:SS or MM:SS for short durations."""
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

#%%% CELL 11 – EXPORTS
"""
Purpose
Expose public names that notebooks and runners are expected to import.

Steps
- Provide data classes and the main staging/sync functions.
"""
__all__ = [
    "DrivePaths",
    "LocalPaths",
    "StageSummary",
    "load_configs",
    "validate_inputs",
    "local_mirrors",
    "stage_to_local",
    "make_local_pathconfig",
    "sync_outputs_back",
    "start_background_sync",
    "stop_background_sync",
]

#%%% CELL 12 – EXECUTION GUARD
"""
Purpose
Prevent accidental module execution in Colab; this file is imported by the
notebook or by a runner script that calls the public functions.

Steps
- Raise a clear error if executed directly.
"""
if __name__ == "__main__":
    raise RuntimeError("Direct execution not supported – use the Run notebook.")
