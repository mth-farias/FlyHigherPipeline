#%% CELL 00 - DESCRIPTION
"""
BehaviorScoringMain.py

This script processes and classifies Drosophila defensive behaviors in response
to visual stimuli.  The experimental folder must contain:
    - 'Tracked': tracked‑data CSVs (…tracked.csv)
    - 'Pose'    (optional): pose CSVs if POSE_SCORING is enabled.

Pipeline overview
    1. Data Loading:
         - Read tracked data (and pose data when enabled).
    2. Pre‑processing & Validation:
         - Noise filtering, alignment corrections, QC checkpoints.
    3. Data Transformation:
         - Convert coordinates to millimetres; compute speed & motion; derive
           orientation when pose is available.
    4. Layered Behaviour Classification:
         - Multi‑layer thresholds & smoothing → transient/sustained/resistant.
    5. Output Generation:
         - Save scored data into 'Scored' / 'ScoredPose' and print a summary.
"""


#%% CELL 01 - IMPORT LIBRARIES AND SETUP
"""
Imports the required libraries for the pipeline.
"""

def behavior_scoring_main(PATHconfig, EXPconfig, BSconfig, BSF):

    import numpy as np
    import os
    import pandas as pd
    import time
    
    
    #%% CELL 02 - COMPUTE DERIVED VALUES
    """
    Calculate derived values for frame-based parameters used throughout the pipeline.
    Assumes experimental parameters are defined in the run file:
      - FRAME_RATE, EXPERIMENTAL_PERIODS,
        STARTLE_WINDOW_SEC, MIN_PERSISTENT_DURATION_SEC,
        LAYER2_AVG_WINDOW_SEC, LAYER3_AVG_WINDOW_SEC, LAYER4_AVG_WINDOW_SEC.
    
    Derived values include:
      - FRAME_SPAN_SEC: Duration of a single frame.
      - EXPERIMENTAL_PERIODS: Frame counts for each phase.
      - STARTLE_WINDOW_FRAMES: Number of frames defining the startle period.
      - MIN_PERSISTENT_DURATION_FRAMES: Minimum frames for sustained/resistant behavior.
      - NUMBER_FRAMES: Total number of frames in the trial.
      - LAYER2_AVG_WINDOW: Smoothing windows (in frames)
        for behavior classification in the respective layers.
    """
    
    # Calculate basic frame durations and counts
    FRAME_SPAN_SEC = 1 / EXPconfig.FRAME_RATE
    NUMBER_FRAMES = EXPconfig.EXPERIMENTAL_PERIODS['Experiment']['duration_frames']
    
    # Compute smoothing windows (in frames) for each layer classification
    LAYER2_AVG_WINDOW = int(BSconfig.LAYER2_AVG_WINDOW_SEC * EXPconfig.FRAME_RATE)


    #%% CELL 03 - DEFINE CHECKPOINT ERRORS & INITIALIZE COUNTERS
    """
    Define checkpoint error and initialize counters.
    These counters track error occurrences and successfully scored files.
    """
    
    # Initialize error and success counters
    error_reading_file = 0
    wrong_stim_count = 0
    wrong_stim_duration = 0
    lost_centroid_position = 0
    pose_mismatch = 0
    missing_pose_file = 0
    unassigned_behavior = 0
    no_exploration = 0
    view_nan_exceeded = 0
    output_len_short = 0
    scored_files = 0

    
    #%% CELL 04 - LOAD EXPERIMENTAL FOLDER & SETUP DIRECTORIES
    """
    Prompt the user to select the experimental folder and initialize key directory paths.
    Expected folder structure:
      - 'Tracked': Contains tracked data files (ending with 'tracked.csv').
      - 'Pose': Contains corresponding pose data files (if POSE_SCORING is enabled).
      
    Depending on the POSE_SCORING flag, create the following output directories under the selected folder:
      - If POSE_SCORING is True: 'ScoredPose', 'Scored', 'Error'
      - Otherwise: 'Scored', 'Error'
      
    These directories store the processed/scored data and error logs.
    """
    
    # Use helper function to prompt folder selection
    PATH = PATHconfig.pPostProcessing
    
    # Define input folders for tracked and, optionally, pose data
    tracked_folder = PATHconfig.pTracked
    pose_folder = PATHconfig.pPose
    
    # Determine and create required output folders based on POSE_SCORING flag
    if EXPconfig.POSE_SCORING:
        output_folders = ['ScoredPose', 'Scored', 'Error']
    else:
        output_folders = ['Scored', 'Error']
    
    for folder in output_folders:
        os.makedirs(os.path.join(PATH, folder), exist_ok=True)
    
    
    #%% CELL 05 - PROCESS EACH FILE (CHECKPOINTS, TRANSFORMATIONS, CLASSIFICATION)
    """
    Process each tracked file in the 'Tracked' folder and execute the full pipeline for data validation, transformation, 
    and behavior classification. This cell is divided into subcells, each focusing on a specific stage of processing.
    """
    
    #%%% CELL 05a - FILE ITERATION & INITIAL CHECKS
    """
    Iterate over tracked files, perform initial checks, print the header report,
    and then enter the main loop to report per-file progress.
    """
    
    # Gather and count all tracked.csv files
    tracked_files = sorted(f for f in os.listdir(tracked_folder) if f.endswith('tracked.csv'))
    total_files = len(tracked_files)

    # Pre-filter processed vs. pending files
    processed_counters = {'scored': 0, 'error': 0}
    skipped_count = 0
    to_process = []
    for filename_tracked in tracked_files:
        if BSF.is_file_already_processed(filename_tracked, "",
            EXPconfig.POSE_SCORING, processed_counters, PATHconfig):
            skipped_count += 1
        else:
            to_process.append(filename_tracked)

    header_scored = processed_counters.get("scored", 0)
    header_errors = processed_counters.get("error", 0)

    # Initial summary report
    print(BSF.report_header(PATHconfig.pExperimentalRoot, EXPconfig.POSE_SCORING,
        total_files, len(to_process), skipped_count, header_scored, header_errors))

    start_time = time.time()

    # Prepare timing for ETA calculations
    prev_time = start_time

    # Main processing loop with per-file progress reporting
    for idx, filename_tracked in enumerate(to_process, start=1):
        now = time.time()
        if idx == 1:
            delta_s = 0.0
        else:
            delta_s = now - prev_time
        prev_time = now

        remaining = total_files - idx
        eta_seconds = delta_s * remaining
        eta = f"{int(eta_seconds // 3600):02d}h{int((eta_seconds % 3600) // 60):02d}"

        # Strip the '_tracked.csv' suffix for display
        basename = filename_tracked.replace('_tracked.csv', '')

        # Print the per-file scoring line
        print(BSF.report_scoring_line(idx, len(to_process), delta_s, eta, basename))

        
        #%%% CELL 05b - LOAD TRACKED DATA & INITIAL ALIGNMENT
        """
        Load the tracked data from the current file, apply alignment corrections,
        check stimulus durations, and compute stimulus indices for quality control.
    
        Steps:
          - Read the tracked CSV file.
          - Apply alignment corrections on the alignment column.
          - Measure each stimulus bout duration and verify it matches expected duration.
          - Compute stimulus indices by detecting changes in the alignment column.
          - Validate that at least one stimulus exists and count matches expected.
          - Define first_stim as the first stimulus index.
          - Check for excessive missing centroid data using NaN thresholds.
    
        If any check fails, print an error line and skip to the next file.
        """
    
        # Load tracked data from the current file
        tracked_file_path = os.path.join(tracked_folder, filename_tracked)
        try:
            tracked_df = pd.read_csv(tracked_file_path)
        except Exception:
            error_reading_file = BSF.checkpoint_fail(pd.DataFrame(), filename_tracked, 'ERROR_READING_FILE',
                                                     error_reading_file, PATHconfig.pError)
            continue


        # Apply alignment corrections on the alignment column
        BSF.fill_zeros(tracked_df, EXPconfig.ALIGNMENT_COL, BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(tracked_df, EXPconfig.ALIGNMENT_COL, BSconfig.NOISE_TOLERANCE)


        # Compute and validate stimulus indices/count
        stim_indices = tracked_df.index[tracked_df[EXPconfig.ALIGNMENT_COL].diff() > 0].tolist()
        if not stim_indices or len(stim_indices) != EXPconfig.EXPECTED_STIMULUS:
            wrong_stim_count = BSF.checkpoint_fail(tracked_df, filename_tracked, 'WRONG_LOOM_COUNT',
                                                   wrong_stim_count, PATHconfig.pError)
            continue
        
        
        # Check stimulus bout durations
        expected_duration_frames = int(EXPconfig.STIMULUS_DURATION_SEC * EXPconfig.FRAME_RATE)
        durations = BSF.bout_duration(tracked_df, EXPconfig.ALIGNMENT_COL)
        if any(abs(d - expected_duration_frames) > BSconfig.NOISE_TOLERANCE for d in durations):
            wrong_stim_duration = BSF.checkpoint_fail(tracked_df, filename_tracked, 'WRONG_STIMULUS_DURATION',
                                                      wrong_stim_duration, PATHconfig.pError)
            continue
        

        # Define first_stim as the first stimulus index
        first_stim = stim_indices[0]

        # Validate centroid data by checking the number of NaNs in 'NormalizedCentroidX'
        pos_nan_count = tracked_df['NormalizedCentroidX'].isna().sum()
        if pos_nan_count > (NUMBER_FRAMES * BSconfig.NAN_TOLERANCE):
            lost_centroid_position = BSF.checkpoint_fail(tracked_df, filename_tracked, 'LOST_CENTROID_POSITION',
                                                         lost_centroid_position, PATHconfig.pError)
            continue


        #%%% CELL 05c - TRACKED DATA TRANSFORMATIONS
        """
        Transform tracked data into a format for further analysis.
        
        Steps:
          - Align and clean stimulus columns (VisualStim, Stim0, Stim1).
          - Convert normalized centroid positions to millimeter space.
          - Calculate motion (based on pixel changes) and speed (from positional changes).
        """
        
        # Initialize a new DataFrame for transformed data
        transform_df = pd.DataFrame()
        
        # Copy the FrameIndex column
        transform_df['FrameIndex'] = tracked_df['FrameIndex']
        
        # Copy and clean the VisualStim column
        transform_df['VisualStim'] = tracked_df['VisualStim']
        BSF.fill_zeros(transform_df, 'VisualStim', BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, 'VisualStim', BSconfig.NOISE_TOLERANCE)
        
        # Copy and clean the Stim0 column
        transform_df['Stim0'] = tracked_df['Stim0']
        BSF.fill_zeros(transform_df, 'Stim0', BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, 'Stim0', BSconfig.NOISE_TOLERANCE)
        
        # Conpy and clean the Stim1 column
        transform_df['Stim1'] = tracked_df['Stim1']
        BSF.fill_zeros(transform_df, 'Stim1', BSconfig.NOISE_TOLERANCE)
        BSF.clean_ones(transform_df, 'Stim1', BSconfig.NOISE_TOLERANCE)
        
        # Convert normalized pixel positions to millimeter space
        transform_df['Position_X'] = tracked_df['NormalizedCentroidX'] * EXPconfig.ARENA_WIDTH_MM
        transform_df['Position_Y'] = (tracked_df['NormalizedCentroidY'] * EXPconfig.ARENA_HEIGHT_MM) * -1 + EXPconfig.ARENA_HEIGHT_MM
        
        # Binarize motion based on pixel change
        transform_df['Motion'] = np.where(tracked_df['PixelChange'] > 0, 1, 0)
        
        # Calculate speed based on positional changes
        transform_df['Speed'] = BSF.calculate_speed(transform_df['Position_X'], transform_df['Position_Y'], FRAME_SPAN_SEC).round(2)


        #%%% CELL 05d - PROCESS POSE DATA (IF ENABLED)
        """
        If POSE_SCORING is enabled, load and preprocess the corresponding pose data.
        
        Steps:
          - Construct the pose file name by replacing 'tracked.csv' with 'pose.csv'.
          - Validate that the pose file exists. If not, save the tracked data to the Error folder,
            increment the missing_pose_file counter, print an error, and skip the file.
          - Load the pose data using pandas (with comma as the separator).
          - Temporarily drop bottom points from the pose data.
          - Validate that the tracked data length matches (pose data length - 1). If not,
            save the tracked data to the Error folder, increment the pose_mismatch counter,
            print an error, and skip the file.
        """
        
        if EXPconfig.POSE_SCORING:
            filename_pose = filename_tracked.replace("tracked.csv", "pose.csv")
            pose_file_path = os.path.join(pose_folder, filename_pose)
            
            # Validate that the pose file exists
            if not os.path.exists(pose_file_path):
                missing_pose_file = BSF.checkpoint_fail(tracked_df, filename_tracked, 'MISSING_POSE_FILE',
                                                        missing_pose_file, PATHconfig.pError)
                continue
    
            # Load pose data
            pose_df = pd.read_csv(pose_file_path, sep=',')
            
            # Validate length match between tracked and pose
            if len(tracked_df) != len(pose_df) - 1:
                pose_mismatch = BSF.checkpoint_fail(tracked_df, filename_tracked, 'POSE_MISMATCH',
                                                    pose_mismatch, PATHconfig.pError)
                continue


        #%%% CELL 05e - POSE DATA TRANSFORMATIONS
        """
        If pose scoring is enabled, transform the pose data for further analysis.
        
        Steps:
          - Determine view orientation (including Bottom as Top) by highest confidence.
          - Map view to coordinates.
          - Apply vertical fallback.
          - Validate NaN proportion against tolerance.
        """
        
        if EXPconfig.POSE_SCORING:
            # Identify rows with full key points
            valid_mask = pose_df[['Head.Position.X','Thorax.Position.X','Abdomen.Position.X']].notna().all(axis=1)
            
            # Pick the view by highest confidence, including 'Bottom'
            pose_df['Selected_View'] = 'Vertical'
            pose_df.loc[valid_mask, 'Selected_View'] = (pose_df.loc[valid_mask, [
                    'Left.Confidence','Right.Confidence',
                    'Top.Confidence','Bottom.Confidence']]
                .idxmax(axis=1).str.replace('.Confidence','', regex=False))
            
            # Map each selected view to its coordinates
            for v in ['Left','Right','Top','Bottom']:
                m = pose_df['Selected_View'] == v
                pose_df.loc[m, 'View_X'] = pose_df.loc[m, f'{v}.Position.X']
                pose_df.loc[m, 'View_Y'] = pose_df.loc[m, f'{v}.Position.Y']
            
            # Treat any 'Bottom' view as 'Top' - Low confidence on the bottom point identification
            pose_df['View'] = pose_df['Selected_View'].replace({'Bottom':'Top'})
            
            # Vertical fallback (head then abdomen)
            m = pose_df['View'] == 'Vertical'
            pose_df.loc[m, 'View_X'] = pose_df.loc[m, 'Head.Position.X'].fillna(pose_df.loc[m, 'Abdomen.Position.X'])
            pose_df.loc[m, 'View_Y'] = pose_df.loc[m, 'Head.Position.Y'].fillna(pose_df.loc[m, 'Abdomen.Position.Y'])
            
            # --- NEW: keep the categorical view label in transform_df for saving ---
            transform_df['View'] = pose_df['View']
        
            # QC: too many NaNs in view?
            if pose_df['View_X'].isna().sum() > NUMBER_FRAMES * BSconfig.POSE_TRACKING_TOLERANCE:
                view_nan_exceeded = BSF.checkpoint_fail(tracked_df, filename_tracked, 'VIEW_NAN_EXCEEDED', 
                                                        view_nan_exceeded, PATHconfig.pError)
                continue
        
            # Convert to millimetres
            transform_df['View_X'] = pose_df['View_X'] * EXPconfig.ARENA_WIDTH_MM
            transform_df['View_Y'] = (pose_df['View_Y'] * EXPconfig.ARENA_HEIGHT_MM) * -1 + EXPconfig.ARENA_HEIGHT_MM
            
            for part in ['Head','Thorax','Abdomen','LeftWing','RightWing']:
                transform_df[f'{part}_X'] = pose_df[f'{part}.Position.X'] * EXPconfig.ARENA_WIDTH_MM
                transform_df[f'{part}_Y'] = (pose_df[f'{part}.Position.Y'] * EXPconfig.ARENA_HEIGHT_MM) * -1 + EXPconfig.ARENA_HEIGHT_MM
            
            # Compute orientation
            transform_df['Orientation'] = BSF.calculate_orientation(transform_df['Thorax_X'], transform_df['Thorax_Y'],
                                                                    transform_df['View_X'],   transform_df['View_Y'])
        
                
        
        #%%% CELL 05f - LAYER 1 - MOVEMENT THRESHOLD
        """
        Classify and denoise behaviors at Layer 1 based on speed and motion thresholds.
        Steps:
          - Define Layer 1 behavior columns.
          - Classify behaviors using defined speed and motion thresholds.
          - Categorize each frame into a single behavior using np.select.
          - Check for excessive unassigned behaviors and log an error if the number exceeds tolerance.
        """
        
        # Define Layer 1 behavior columns.
        LAYER1_COLUMNS = ['layer1_jump', 'layer1_walk', 'layer1_stationary', 'layer1_freeze', 'layer1_none']
        
        # Classify behaviors based on speed/motion thresholds.
        transform_df['layer1_jump'] = np.where(transform_df['Speed'] >= BSconfig.HIGH_SPEED, 1, 0)
        transform_df['layer1_walk'] = np.where((transform_df['Speed'] >= BSconfig.LOW_SPEED) & 
                                               (transform_df['Speed'] < BSconfig.HIGH_SPEED), 1, 0)
        transform_df['layer1_stationary'] = np.where((transform_df['Speed'] < BSconfig.LOW_SPEED) & 
                                                     (transform_df['Motion'] > 0), 1, 0)
        transform_df['layer1_freeze'] = np.where(transform_df['Motion'] == 0, 1, 0)
        
        # Categorize each frame into a single behavior.
        conditions = [
            transform_df['layer1_jump'] == 1,
            transform_df['layer1_walk'] == 1,
            transform_df['layer1_stationary'] == 1,
            transform_df['layer1_freeze'] == 1,
        ]
        choices = ['Layer1_Jump', 'Layer1_Walk', 'Layer1_Stationary', 'Layer1_Freeze']
        transform_df['Layer1'] = np.select(conditions, choices, default=None)
        
        # Check if too many behaviors remain unassigned.
        total_unassigned_behaviors = transform_df['Layer1'].isna().sum()
        if total_unassigned_behaviors > (NUMBER_FRAMES * BSconfig.LAYER1_TOLERANCE):
            unassigned_behavior = BSF.checkpoint_fail(tracked_df, filename_tracked, 'UNASSIGNED_BEHAVIOR',
                                                      unassigned_behavior, PATHconfig.pError)
            continue



    
        #%%% CELL 05g - LAYER 2 - APPLYING SMALL SMOOTHING
        """
        This section handles the classification and denoising of behaviors at Layer 2.
        It applies a small smoothing window to the behaviors classified in Layer 1, further refines the classification, and categorizes the behaviors. 
        Finally, it checks if there was sufficient exploration (walking behavior) during the baseline period.
        
        Steps:
          - Initialize columns for Layer 2 behaviors.
          - Apply a centered running average to smooth the Layer 1 classifications.
          - Compute the row-wise maximum of the smoothed values to determine the dominant behavior.
          - Set binary flags for each behavior, with jump taking precedence.
          - Retain a single behavior per frame using hierarchical classification.
          - Categorize each frame's behavior into a single label.
          - Verify sufficient exploration during the baseline period.
        """
        
        LAYER2_COLUMNS = ['layer2_jump', 'layer2_walk', 'layer2_stationary', 'layer2_freeze', 'layer2_none']
        
        # Initialize Layer 2 columns to zero.
        for behavior in LAYER2_COLUMNS:
            transform_df[behavior] = 0
        
        # Apply smoothing to Layer 1 classifications.
        layer2_avg_columns = ['layer2_jump_avg', 'layer2_walk_avg', 'layer2_stationary_avg', 'layer2_freeze_avg']
        transform_df = BSF.calculate_center_running_average(transform_df, LAYER1_COLUMNS, layer2_avg_columns, LAYER2_AVG_WINDOW)
        
        # Compute row-wise maximum and the corresponding averaged behavior.
        temp = transform_df[layer2_avg_columns].fillna(-np.inf)
        max_values2 = temp.max(axis=1)
        max_behavior2 = temp.idxmax(axis=1)
        # Set rows where all values were -np.inf or maximum value is <= 0 to None.
        max_behavior2[max_values2 == -np.inf] = None
        max_behavior2[max_values2 <= 0] = None
    
        # Set binary flags with jump taking precedence.
        transform_df['layer2_jump'] = (transform_df['layer2_jump_avg'] > 0).astype(int)
        mask = transform_df['layer2_jump'] == 0
        transform_df.loc[mask, 'layer2_walk'] = (max_behavior2[mask] == 'layer2_walk_avg').astype(int)
        transform_df.loc[mask, 'layer2_stationary'] = (max_behavior2[mask] == 'layer2_stationary_avg').astype(int)
        transform_df.loc[mask, 'layer2_freeze'] = (max_behavior2[mask] == 'layer2_freeze_avg').astype(int)
        
        # Assign 'none' flag.
        transform_df['layer2_none'] = np.where(transform_df[LAYER2_COLUMNS[:-1]].sum(axis=1) == 0, 1, 0)
        
        # Retain single behavior per frame.
        transform_df = BSF.hierarchical_classifier(transform_df, LAYER2_COLUMNS)
        
        # Layer 2 categorization using vectorized mapping.
        conditions2 = [
            transform_df['layer2_jump'] == 1,
            transform_df['layer2_walk'] == 1,
            transform_df['layer2_stationary'] == 1,
            transform_df['layer2_freeze'] == 1,
        ]
        choices2 = ['Layer2_Jump', 'Layer2_Walk', 'Layer2_Stationary', 'Layer2_Freeze']
        transform_df['Layer2'] = np.select(conditions2, choices2, default=None)
        
        # CHECKPOINT - TOO LITTLE EXPLORATION DURING BASELINE
        baseline_start = max(0, first_stim - EXPconfig.EXPERIMENTAL_PERIODS['Baseline']['duration_frames'])
        baseline_end   = first_stim
        walk_count_baseline = transform_df.loc[baseline_start:baseline_end, 'Layer2'].eq('Layer2_Walk').sum()

        if walk_count_baseline < BSconfig.BASELINE_EXPLORATION * EXPconfig.EXPERIMENTAL_PERIODS['Baseline']['duration_frames']:
            no_exploration = BSF.checkpoint_fail(transform_df, filename_tracked, 'NO_EXPLORATION',
                                                 no_exploration, PATHconfig.pError)
            continue

    
        #%%% CELL 05j - RESISTANT BEHAVIORS
        """
        This section handles the classification of resistant behaviors.
        Resistant behaviors are those that persist during a startle window following stimuli.
        
        Steps:
          - Define the startle window based on the stimulus indices (-1s to 2s from onset).
          - Classify resistant behaviors (Walk, Stationary, Freeze) based on their persistence during the startle window.
          - Apply the classification for resistant behaviors.
          - Determine frames with no resistant behavior.
          - Categorize each frame's resistant behavior into a single label.
        """
        
        # Define startle window by setting frames around each stimulus onset (-1s to 2s from onset).
        transform_df['Startle_window'] = 0
        for onset in stim_indices:
            start = max(0, onset - EXPconfig.FRAME_RATE)
            end = min(len(transform_df) - 1, onset + (EXPconfig.FRAME_RATE * 2))
            transform_df.loc[start:end, 'Startle_window'] = 1
        
        # Define resistant behavior columns.
        RESISTANT_COLUMNS = ['resistant_walk', 'resistant_stationary', 'resistant_freeze', 'resistant_none']
        
        # Classify resistant behaviors using the helper function from BSF.
        BSF.classify_resistant_behaviors(transform_df, RESISTANT_COLUMNS, EXPconfig.FRAME_RATE*3)
        
        # Determine frames with no resistant behavior.
        transform_df['resistant_none'] = np.where(transform_df[RESISTANT_COLUMNS[:-1]].sum(axis=1) == 0, 1, 0)
        
        # Categorize resistant behaviors.
        transform_df['Resistant'] = pd.Series(dtype='object')
        for i in range(len(transform_df)):
            if transform_df.loc[i, 'resistant_walk'] == 1:
                transform_df.loc[i, 'Resistant'] = 'Resistant_Walk'
            elif transform_df.loc[i, 'resistant_stationary'] == 1:
                transform_df.loc[i, 'Resistant'] = 'Resistant_Stationary'
            elif transform_df.loc[i, 'resistant_freeze'] == 1:
                transform_df.loc[i, 'Resistant'] = 'Resistant_Freeze'


        #%%% CELL 05n - SIMPLIFIED FINAL BEHAVIOR
        """
        This section creates a simplified final `Behavior` column using Layer2 classification,
        with special‐case overrides for any resistant behavior (walk, stationary, or freeze).
        
        Steps:
          - Initialize the Behavior column as object dtype.
          - For each frame:
              - If Layer2_Jump    → Behavior = 'Jump'
              - If Layer2_Walk    → Behavior = 'Walk'
              - If Layer2_Stationary → Behavior = 'Stationary'
              - If Layer2_Freeze  → Behavior = 'Resistant_Freeze' if resistant_freeze==1 else 'Freeze'
        """
        
        # Initialize the Behavior column
        transform_df['Behavior'] = pd.Series(dtype='object')
        
        # Frame-wise classification
        for i in range(len(transform_df)):
            layer2 = transform_df.loc[i, 'Layer2']
            res   = transform_df.loc[i, 'Resistant']
        
            if layer2 == 'Layer2_Jump':
                transform_df.loc[i, 'Behavior'] = 'Jump'
        
            elif layer2 == 'Layer2_Walk':
                transform_df.loc[i, 'Behavior'] = 'Walk'
        
            elif layer2 == 'Layer2_Stationary':
                transform_df.loc[i, 'Behavior'] = 'Stationary'
        
            elif layer2 == 'Layer2_Freeze':
                if res == 'Resistant_Freeze':
                    transform_df.loc[i, 'Behavior'] = 'Resistant_Freeze'
                else:
                    transform_df.loc[i, 'Behavior'] = 'Freeze'



        #%%% CELL 05m - SAVE SCORED FILE
        """
        This section handles saving the scored data to a CSV file.
        Depending on whether pose scoring is enabled, different sets of columns are saved.
        The output is aligned based on the stimulus timing, and the file is saved in the appropriate directory.
        """
        
        # Always output the regular scored file from the transformed DataFrame
        output_df = transform_df[BSconfig.SCORED_COLUMNS]
        
        # Align the output DataFrame to include only the frames around the stimulus
        aligned_output_df = output_df.iloc[
            int(first_stim - EXPconfig.EXPERIMENTAL_PERIODS['Baseline']['duration_frames']):
            int(first_stim + (EXPconfig.EXPERIMENTAL_PERIODS['Experiment']['duration_frames']
                              - EXPconfig.EXPERIMENTAL_PERIODS['Baseline']['duration_frames'])), : ].reset_index(drop=True)
    
        # CHECKPOINT: ensure aligned length matches the full experiment duration
        expected_len = EXPconfig.EXPERIMENTAL_PERIODS['Experiment']['duration_frames']
        if len(aligned_output_df) != expected_len:
            output_len_short = BSF.checkpoint_fail(aligned_output_df, filename_tracked, 'OUTPUT_LEN_SHORT', 
                                                   output_len_short, PATHconfig.pError)
            continue
    
        # Now save the scored file
        scored_file   = filename_tracked.replace('tracked.csv', 'scored.csv')
        scored_folder = os.path.join(PATHconfig.pScored)
        os.makedirs(scored_folder, exist_ok=True)
        aligned_output_df.to_csv(os.path.join(scored_folder, scored_file), header=True, index=False)
        
        # Additionally output the pose-scored file if POSE_SCORING is enabled
        if EXPconfig.POSE_SCORING:
            output_pose_df = transform_df[BSconfig.SCORED_POSE_COLUMNS]
            aligned_output_pose_df = output_pose_df.iloc[
                int(first_stim - EXPconfig.EXPERIMENTAL_PERIODS['Baseline']['duration_frames']):
                int(first_stim + (EXPconfig.EXPERIMENTAL_PERIODS['Experiment']['duration_frames']
                                  - EXPconfig.EXPERIMENTAL_PERIODS['Baseline']['duration_frames'])), : ].reset_index(drop=True)
    
            scored_pose_file   = filename_tracked.replace('tracked.csv', 'scored_pose.csv')
            scored_pose_folder = os.path.join(PATHconfig.pScoredPose)
            os.makedirs(scored_pose_folder, exist_ok=True)
            aligned_output_pose_df.to_csv(os.path.join(scored_pose_folder, scored_pose_file), header=True, index=False)
    
        scored_files += 1

    
    #%% CELL 06 - PRINT SUMMARY
    """
    Print the final summary report.
    """
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = f"{int(total_time//3600):02d}h{int((total_time%3600)//60):02d}"
    
    # Compute total error count
    error_count = (error_reading_file + wrong_stim_count +  wrong_stim_duration + lost_centroid_position +
                   pose_mismatch + missing_pose_file + view_nan_exceeded + unassigned_behavior +
                   no_exploration + output_len_short)
    
    print(BSF.report_final_summary(total_time_str, total_files, scored_files,
                                   error_count, error_reading_file, missing_pose_file,
                                   wrong_stim_count, wrong_stim_duration, lost_centroid_position,
                                   pose_mismatch, view_nan_exceeded, unassigned_behavior, no_exploration,
                                   output_len_short) + BSF.done_duck())
    

if __name__ == "__main__":
    behavior_scoring_main()