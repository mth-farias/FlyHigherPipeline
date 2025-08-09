#%% CELL 00 – MODULE OVERVIEW
"""
PathConfig.py

Purpose
=======
Define and export experiment folder paths and individual file locations for all downstream scripts.
A single placeholder "__EXP_ROOT__" marks where the experimental root will be injected.
All other modules import these constants to locate data, code, and notebooks.


Usage
-----
After cloning/downloading, run an injection step to replace "__EXP_ROOT__" with the real path,
then any script can import folder and file constants directly:

    from PathConfig import pRawData, pScored, BSR, EXPconfig
    from PathConfig import *
    # pRawData and pScored are directory Paths;                             
    # BSR and EXPconfig are Path objects to files.

    # Example: list scored CSVs
    csvs = list(pScored.glob("*_scored.csv"))
    # Example: get path to ExperimentConfig.py
    config_path = EXPconfig


Structure
---------
ExperimentalFolder/
├── Protocols/
│   ├── Codes/
│   │   ├── BehaviorScoringRun.ipynb
│   │   ├── CreateDataFramesRun.ipynb
│   │   ├── BehaviorScoring/
│   │   │   ├── BehaviorScoringFunctions.py
│   │   │   └── BehaviorScoringMain.py
│   │   ├── CreateDataFrames/
│   │   │   ├── CreateDataFramesFunctions.py
│   │   │   └── CreateDataFramesMain.py
│   │   └── Config/
│   │       ├── ExperimentConfig.py
│   │       ├── BehaviorScoringConfig.py
│   │       ├── CreateDataFramesConfig.py
│   │       ├── ParamsConfig.py
│   │       ├── ColorsConfig.py
│   │       ├── TimeConfig.py
│   │       └── PathConfig.py
│   └── Bonfly/
│       ├── Protocol/
│       └── Tracker/
├── RawData/
├── PostProcessing/
│   ├── Arenas/
│   ├── CropVideo/
│   ├── Tracked/
│   ├── Pose/
│   ├── Scored/
│   ├── ScoredPose/
│   └── Error/
└── Analysis/
    ├── DataFrames/
    ├── ZoomDataFrames/
    └── Plots/
"""

#%% CELL 01 – IMPORTS & ROOT PLACEHOLDER
from pathlib import Path

# AUTO-INJECT this line at download-time:
pExperimentalRoot = Path("__EXP_ROOT__")

# helper to build subpaths
def _p(sub: str) -> Path:
    """Return a path under the experimental root."""
    return pExperimentalRoot / sub

#%% CELL 02 – STANDARD FOLDER CONSTANTS
# Protocols and Codes
pProtocols        = _p("Protocols")
pCodes            = pProtocols / "Codes"
pBehaviorScoring  = pCodes / "BehaviorScoring"
pCreateDataFrames = pCodes / "CreateDataFrames"
pConfig           = pCodes / "Config"

# RawData
pRawData          = _p("RawData")

# PostProcessing (subfolders)
pPostProcessing   = _p("PostProcessing")
pArenas           = pPostProcessing / "Arenas"
pCropVideo        = pPostProcessing / "CropVideo"
pTracked          = pPostProcessing / "Tracked"
pPose             = pPostProcessing / "Pose"
pScored           = pPostProcessing / "Scored"
pScoredPose       = pPostProcessing / "ScoredPose"
pError           = pPostProcessing / "Error"

# Analysis (subfolders)
pAnalysis         = _p("Analysis")
pDataFrames       = pAnalysis / "DataFrames"
pZoomDataFrames   = pAnalysis / "ZoomDataFrames"
pPlots            = pAnalysis / "Plots"

#%% CELL 03 – INDIVIDUAL FILE LOCATIONS
# Notebooks
BSR  = pCodes / "BehaviorScoringRun.ipynb"
CDFR = pCodes / "CreateDataFramesRun.ipynb"

# BehaviorScoring code
BSF  = pBehaviorScoring / "BehaviorScoringFunctions.py"
BSM  = pBehaviorScoring / "BehaviorScoringMain.py"

# CreateDataFrames code
CDFF = pCreateDataFrames / "CreateDataFramesFunctions.py"
CDFM = pCreateDataFrames / "CreateDataFramesMain.py"

# Shared configs
EXPconfig = pConfig / "ExperimentConfig.py"        # import as EXPconfig
BSconfig  = pConfig / "BehaviorScoringConfig.py"   # import as BSconfig
CDFconfig = pConfig / "CreateDataFramesConfig.py"  # import as CDFconfig
PARAMconfig = pConfig / "ParamsConfig.py"          # import as PARAMconfig
COLORconfig = pConfig / "ColorsConfig.py"          # import as COLORconfig
TIMEconfig = pConfig / "TimeConfig.py"            # import as TIMEconfig
PATHconfig = pConfig / "PathConfig.py"            # import as PATHconfig

#%% CELL 04 – EXPORTS
__all__ = [
    # root
    "pExperimentalRoot",
    # folders
    "pProtocols", "pCodes", "pBehaviorScoring", "pCreateDataFrames", "pConfig",
    "pRawData",
    "pPostProcessing", "pArenas", "pCropVideo", "pTracked", "pPose", "pScored", "pScoredPose", "pError",
    "pAnalysis", "pDataFrames", "pZoomDataFrames", "pPlots",
    # individual files (shorthands)
    "BSR", "CDFR",
    "BSF", "BSM",
    "CDFF", "CDFM",
    "EXPconfig", "BSconfig", "CDFconfig",
    "PARAMconfig", "COLORconfig", "TIMEconfig", "PATHconfig",
]
