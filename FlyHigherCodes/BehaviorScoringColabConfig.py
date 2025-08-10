#%% CELL 00 – MODULE OVERVIEW
"""
BehaviorScoringColabConfig.py

Purpose
=======
Helpers for running the BehaviorScoring pipeline on Google Colab while keeping
the canonical folder structure on Google Drive. This module:

1) Reads canonical Drive paths from PathConfig
2) Validates required inputs (Tracked always; Pose iff present)
3) Creates local mirrors under /content for fast, quota‑friendly I/O
4) Stages inputs and existing outputs Drive → local (preserves skip logic)
   - Inputs: copy only files that still need processing
   - Existing outputs: create zero‑byte placeholders (fast) so Main can skip
5) Builds a temporary PathConfig-like object that points to the local mirrors
6) Syncs new outputs local → Drive after the run (skips placeholders)

Design Notes
------------
- Resilient copy: prefer rsync; if the Drive mount drops, remount once and retry.
- Silent by default; set verbose=True in calls if you want prints.
- No Analysis folder: we do not mirror or sync any analysis outputs.
"""

#%% CELL 01 – IMPORTS
from __future__ import annotations

import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple, List, Set

# Background sync
import threading


#%% CELL 02 – DATA CLASSES
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
    scoredpose: int
    error_items: int


#%% CELL 03 – PUBLIC API
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


def validate_inputs(drive_paths: DrivePaths, pose_scoring: Optional[bool] = None, *, verbose: bool = False) -> bool:
    """Validate required inputs on Drive.

    Returns final pose_scoring (auto‑detected if None).
    """
    _require_csv_folder(drive_paths.tracked, "Tracked inputs not found or empty")

    # Auto-detect if not given: pose is ON only if folder exists and has CSVs
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
    """Create local mirrors under /content with identical substructure."""
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

    for d in (local.tracked, local.pose, local.scored, local.scoredpose, local.error):
        d.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Virtual paths created")

    return local


# --------- Processed detection + selective copy helpers ---------
def _list_csvs(src: Path) -> List[Path]:
    if not src.exists():
        return []
    return [p for p in src.rglob("*.csv") if p.is_file()]


def _scored_name_for(tracked_name: str, pose_scoring: bool) -> str:
    if tracked_name.endswith("tracked.csv"):
        return tracked_name.replace("tracked.csv", "scored_pose.csv" if pose_scoring else "scored.csv")
    return tracked_name.replace(".csv", "_scored_pose.csv" if pose_scoring else "_scored.csv")


def _has_matching_error(tracked_name: str, error_dir: Path) -> bool:
    base = tracked_name.replace("tracked.csv", "")
    if not error_dir.exists():
        return False
    try:
        for f in error_dir.iterdir():
            if f.is_file() and f.name.startswith(base):
                return True
    except Exception:
        pass
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
    """Copy only the files listed in rel_files (paths relative to src). Return count copied."""
    if not rel_files:
        return 0
    dst.mkdir(parents=True, exist_ok=True)

    if shutil.which("rsync"):
        files_list = dst / ".rsync_files.txt"
        try:
            files_list.write_text("\n".join(rel_files) + "\n")
            cmd = [
                "rsync", "-a", "--no-compress", "--prune-empty-dirs",
                "--files-from", str(files_list),
                str(src) + "/", str(dst) + "/"
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or "rsync failed")
        except Exception:
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

    # Python fallback
    copied = 0
    for rel in rel_files:
        s = src / rel
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.exists():
            shutil.copy2(s, d)
            copied += 1
    return copied


# --------- Placeholder creation ---------
def _mirror_placeholders(src: Path, dst: Path, patterns: Optional[Iterable[str]]) -> int:
    """Create zero-byte placeholders in dst matching files in src. Return count created."""
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
            out.touch()
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
    Measure Drive→/content read throughput (MB/s) by streaming data from the *actual*
    to-copy set for a short warm-up window and discarding it (no writes to disk).

    Stops when (elapsed >= min_seconds AND bytes_read >= min_megabytes) OR elapsed >= max_seconds.
    Returns MB/s (float) or None if no readable files.
    """
    import itertools

    # Build absolute paths on Drive for the to-copy inputs
    candidates: List[Path] = []
    for rel in to_copy_tracked:
        p = (drive_paths.tracked / rel)
        if p.exists() and p.is_file():
            candidates.append(p)
    for rel in to_copy_pose:
        p = (drive_paths.pose / rel)
        if p.exists() and p.is_file():
            candidates.append(p)

    if not candidates:
        return None

    target_bytes = int(min_megabytes * 1024 * 1024)
    start = time.perf_counter()
    read_bytes = 0

    # Cycle over files until thresholds or max_seconds reached
    for file_path in itertools.cycle(candidates):
        # Stop if time cap reached
        if time.perf_counter() - start >= max_seconds:
            break
        try:
            with open(file_path, "rb") as f:
                while True:
                    # Stop if thresholds reached or time cap reached
                    if (time.perf_counter() - start) >= max_seconds:
                        break
                    if (time.perf_counter() - start) >= min_seconds and read_bytes >= target_bytes:
                        break
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    read_bytes += len(chunk)
        except Exception:
            # Skip unreadable files and continue
            continue

        # Early exit if thresholds reached
        if (time.perf_counter() - start) >= min_seconds and read_bytes >= target_bytes:
            break

    elapsed = max(time.perf_counter() - start, 1e-6)
    if read_bytes <= 0:
        return None
    return (read_bytes / (1024 * 1024)) / elapsed



# --------- Staging (with your requested print format) ---------
def stage_to_local(
    drive_paths: DrivePaths,
    local_paths: LocalPaths,
    pose_scoring: Optional[bool] = None,
    *,
    verbose: bool = True
) -> StageSummary:
    """Stage inputs selectively (skip already-processed), create input placeholders for skipped,
    and mirror existing outputs as placeholders. Prints ETA before copying.
    """
    # Auto-detect pose_scoring if not given
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    print("================= LOADING FILES FROM DRIVE =================")

    # Gather all input candidates
    tracked_files = _list_csvs(drive_paths.tracked)
    total_tracked = len(tracked_files)

    pose_files = _list_csvs(drive_paths.pose) if pose_scoring else []
    total_pose = len(pose_files) if pose_scoring else 0

    # Build worklist: only unprocessed inputs
    rel_tracked = [str(p.relative_to(drive_paths.tracked)) for p in tracked_files]
    to_copy_tracked: List[str] = []
    skipped_already_processed: List[str] = []

    for rel in rel_tracked:
        if _is_processed_on_drive(rel, drive_paths, pose_scoring):
            skipped_already_processed.append(rel)
        else:
            to_copy_tracked.append(rel)

    # Pose files to copy only for the unprocessed tracked set
    to_copy_pose: List[str] = []
    if pose_scoring and drive_paths.pose.exists():
        for rel in to_copy_tracked:
            pose_rel = rel.replace("tracked.csv", "pose.csv")
            if (drive_paths.pose / pose_rel).exists():
                to_copy_pose.append(pose_rel)

    # ---- ETA for inputs we will actually copy (not placeholders) ----
    est_files_to_copy = len(to_copy_tracked) + len(to_copy_pose)
    total_input_files = total_tracked + total_pose

    total_bytes = 0
    for rel in to_copy_tracked:
        f = drive_paths.tracked / rel
        try: total_bytes += f.stat().st_size
        except OSError: pass
    for rel in to_copy_pose:
        f = drive_paths.pose / rel
        try: total_bytes += f.stat().st_size
        except OSError: pass

    # Adaptive warm-up over *actual to-copy* list: 5–10s and/or ≥100 MB
    speed_mbps = _warmup_measure_speed_mbps(
        drive_paths,
        to_copy_tracked=to_copy_tracked,
        to_copy_pose=to_copy_pose,
        min_seconds=5.0,
        min_megabytes=100.0,
        max_seconds=10.0,
    ) or 12.0  # conservative fallback

    if est_files_to_copy > 0 and total_bytes > 0:
        est_seconds = (total_bytes / (1024 * 1024)) / max(speed_mbps, 0.1)
        print(f"\n   Estimated: ~{_fmt_seconds(est_seconds)} at {speed_mbps:.1f} MB/s "
              f"for {est_files_to_copy}/{total_input_files} files")
    else:
        print("\n   Estimated: No new input files to copy")

    # ---- Perform staging ----
    t0 = time.perf_counter()

    # Inputs (copy only unprocessed)
    n_tracked = _copy_selected(drive_paths.tracked, local_paths.tracked, to_copy_tracked)
    n_pose = 0
    if pose_scoring and to_copy_pose:
        n_pose = _copy_selected(drive_paths.pose, local_paths.pose, to_copy_pose)

    # Inputs (create ZERO-BYTE PLACEHOLDERS for the ones we skipped, so Main sees them)
    # Tracked placeholders
    for rel in skipped_already_processed:
        p = local_paths.tracked / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()

    # Pose placeholders corresponding to skipped tracked (only if pose file exists on Drive)
    if pose_scoring and drive_paths.pose.exists():
        for rel in skipped_already_processed:
            pose_rel = rel.replace("tracked.csv", "pose.csv")
            pose_src = drive_paths.pose / pose_rel
            if pose_src.exists():
                q = local_paths.pose / pose_rel
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists():
                    q.touch()

    # Existing outputs (placeholders only)
    n_scored = _mirror_placeholders(drive_paths.scored, local_paths.scored, patterns=("*.csv",))
    # ScoredPose placeholders are created for correctness but not reported in summary
    _ = _mirror_placeholders(drive_paths.scoredpose, local_paths.scoredpose, patterns=("*.csv",)) if pose_scoring else 0
    n_error = _mirror_placeholders(drive_paths.error, local_paths.error, patterns=None)

    if verbose:
        t1 = time.perf_counter()
        print(f"\n   Inputs        : Tracked={n_tracked}" + (f", Pose={n_pose}" if pose_scoring else ""))
        print(f"   Existing outs : Scored={n_scored}, ScoredError={n_error}")
        print(f"   Staging time  : { _fmt_seconds(t1 - t0) }")
        print("\n======================= READY TO RUN =======================")

    return StageSummary(
        tracked=n_tracked,
        pose=n_pose,
        scored=n_scored,
        scoredpose=0,    # ScoredPose placeholders were created but not reported here
        error_items=n_error,
    )



def make_local_pathconfig(PathConfig, local_paths: LocalPaths):
    """Return a PathConfig-like object by rebasing ALL PathConfig attributes under the local root.

    - Any attribute that is a filesystem path under pExperimentalRoot is rebased to local.
    - Non-path values and callables are copied as-is.
    - pCodes is explicitly kept pointing to Drive.
    """
    drive_root = Path(PathConfig.pExperimentalRoot)
    local_root = Path(local_paths.root)

    rebased = {}

    for name, value in vars(PathConfig).items():
        if name.startswith("__"):
            continue
        if callable(value):
            rebased[name] = value
            continue
        try:
            p = Path(value)
            rel = p.relative_to(drive_root)
            rebased[name] = str(local_root / rel)
        except Exception:
            rebased[name] = value

    if hasattr(PathConfig, "pCodes"):
        rebased["pCodes"] = PathConfig.pCodes

    return SimpleNamespace(**rebased)


def sync_outputs_back(
    local_paths: LocalPaths,
    drive_paths: DrivePaths,
    pose_scoring: Optional[bool] = None,
    *,
    verbose: bool = False
) -> None:
    """Copy outputs from local → Drive in bulk (resilient, skip placeholders)."""
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    t0 = time.perf_counter()

    _copy_tree(local_paths.scored,     drive_paths.scored,     patterns=("*.csv",), upload_mode=True)
    _copy_tree(local_paths.error,      drive_paths.error,      patterns=None,        upload_mode=True)
    if pose_scoring:
        _copy_tree(local_paths.scoredpose, drive_paths.scoredpose, patterns=("*.csv",), upload_mode=True)

    if verbose:
        t1 = time.perf_counter()
        print("\n\n================ SCORING AND SAVING COMPLETE ================")
        print("\n              Synced outputs back to Drive.")
        print(f"                 Sync time     : { _fmt_seconds(t1 - t0) }")


#%% CELL 04 – BACKGROUND SYNC (SILENT, FINAL-FILES ONLY, EXACT BATCHING)
_bg_state = {
    "thread": None,
    "stop": threading.Event(),
    "seen": set(),        # set[str] of file paths we have already accounted for
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
    """Start a silent background thread that syncs after EXACTLY `batch_size` new final files appear."""
    if pose_scoring is None:
        pose_scoring = drive_paths.pose.exists() and any(drive_paths.pose.rglob("*.csv"))

    if _bg_state["thread"] and _bg_state["thread"].is_alive():
        return
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
                pass

    t = threading.Thread(target=_worker, daemon=True)
    _bg_state["thread"] = t
    t.start()


def stop_background_sync() -> None:
    """Stop the background sync thread."""
    if _bg_state["thread"]:
        _bg_state["stop"].set()
        _bg_state["thread"].join(timeout=15)
        _bg_state["thread"] = None


#%% CELL 05 – INTERNAL HELPERS (RESILIENT COPY)
def _require_csv_folder(folder: Path, message: str) -> None:
    if not folder.exists():
        raise RuntimeError(f"{message}: {folder} (folder not found)")
    if not any(folder.rglob("*.csv")):
        raise RuntimeError(f"{message}: {folder} (no CSV files found)")


def _copy_tree(src: Path, dst: Path, patterns: Optional[Iterable[str]], *, upload_mode: bool = False) -> int:
    """Copy files from *src* to *dst*; return count of files attempted to copy.

    - Prefer rsync; if mount drops, remount once and retry.
    - For uploads (upload_mode=True): don't overwrite existing Drive files and skip zero-byte placeholders.
    - Fallback to Python file-by-file copy with the same rules.
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

    # Python fallback
    did_remount = False
    for path in candidates:
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if upload_mode:
                # Skip placeholders and don't overwrite existing files on Drive
                try:
                    if path.stat().st_size == 0:
                        continue
                except OSError:
                    continue
                if out.exists():
                    continue
            shutil.copy2(path, out)
        except OSError as oe:
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


def _rsync_copy(src: Path, dst: Path, patterns: Optional[Iterable[str]], *, upload_mode: bool = False) -> None:
    """Copy using rsync. Raises _MountDropError on mount-related failures."""
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
            # fallback handled by caller
            raise RuntimeError("Unsupported include pattern for rsync path; fallback to Python copy.")

    dst.mkdir(parents=True, exist_ok=True)
    src_arg = str(src) + "/"
    dst_arg = str(dst) + "/"

    proc = subprocess.run(cmd + [src_arg, dst_arg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "") + " " + (proc.stdout or "")
        if "Transport endpoint is not connected" in err or "Input/output error" in err:
            raise _MountDropError(err)
        raise RuntimeError(f"rsync failed ({proc.returncode}): {err.strip()}")


class _MountDropError(RuntimeError):
    pass


def _remount_drive() -> bool:
    try:
        subprocess.run(["fusermount", "-u", "/content/drive"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        subprocess.run(["rm", "-rf", "/content/drive"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        from google.colab import drive as colab_drive
        colab_drive.mount("/content/drive", force_remount=True)
        return True
    except Exception:
        return False


def _fmt_seconds(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


#%% CELL 06 – EXPORTS
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


#%% CELL 07 – EXECUTION GUARD
if __name__ == "__main__":
    raise RuntimeError("Direct execution not supported – use the Run notebook.")
