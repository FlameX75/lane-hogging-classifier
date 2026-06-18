================================================================
        LANE HOGGING CLASSIFIER — highD Dataset
        README & Usage Instructions
================================================================

WHAT THIS DOES
--------------
This script reads highD highway drone dataset files and classifies
every vehicle as a lane hogger (True) or not (False) based on a
4-stage decision flowchart.

Output is a simple CSV with one row per vehicle and a True/False label.


----------------------------------------------------------------
FOLDER STRUCTURE (set this up before running)
----------------------------------------------------------------

Put everything like this:

    data/
    ├── 01_tracks.csv
    ├── 01_tracksMeta.csv
    ├── 01_recordingMeta.csv
    ├── 02_tracks.csv
    ├── 02_tracksMeta.csv
    ├── 02_recordingMeta.csv
    ├── ... (all your highD CSV files go here)
    │
    └── script/
        ├── lane_hogging_classifier.py
        ├── merge_results.py
        └── README.txt  ← you are here

The script needs all 3 files for each recording:
  XX_tracks.csv          — frame-by-frame vehicle data
  XX_tracksMeta.csv      — vehicle class (Car/Truck)
  XX_recordingMeta.csv   — recording info (frame rate etc.)

If any of the 3 is missing for a recording, that recording is skipped.


----------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------

- Python 3.8 or higher
- pandas and numpy libraries

To install the libraries, open a terminal and run:

    pip install pandas numpy

You only need to do this once.


----------------------------------------------------------------
STEP 1 — Run the classifier
----------------------------------------------------------------

Open a terminal (Command Prompt or PowerShell on Windows).
Navigate to the script folder:

    cd path\to\data\script

Then run:

    python lane_hogging_classifier.py --data_dir .. --output_dir ../results

  --data_dir ..          tells the script to look for CSV files in the
                         parent folder (i.e. the data/ folder)

  --output_dir ../results  saves all results into data/results/
                           (this folder is created automatically)

The script will process every recording it finds and print progress.
It may take a few minutes depending on how many recordings you have.


----------------------------------------------------------------
STEP 2 — Merge all results into one file
----------------------------------------------------------------

After Step 1 finishes, run:

    python merge_results.py

This combines all individual recording results into one single file:

    data/results/ALL_classifications.csv

This is the main deliverable — every vehicle from every recording
with a True/False label.


----------------------------------------------------------------
OUTPUT FILES EXPLAINED
----------------------------------------------------------------

Inside data/results/ you will find:

1. ALL_classifications.csv          ← START HERE
   The combined output. One row per vehicle.
   Columns:
     recording       — which recording the vehicle is from (e.g. 12)
     id              — vehicle ID within that recording
     class           — Car or Truck
     is_lane_hogger  — TRUE or FALSE  ← the answer

2. XX_results_per_vehicle.csv       (one file per recording)
   Detailed per-vehicle stats including:
     hogging_sec     — how many seconds the vehicle was hogging
     stage3_sec      — how many seconds it had a chance to return left
     hogging_pct     — hogging as % of eligible time

3. XX_results_per_frame.csv         (one file per recording)
   Full frame-by-frame breakdown. Shows exactly which stage of the
   flowchart each frame exited at. Useful for debugging or deeper analysis.

4. 00_summary.csv
   One row per recording. Quick overview of total vehicles and
   how many were classified as lane hoggers.


----------------------------------------------------------------
PARAMETERS (if you want to adjust the thresholds)
----------------------------------------------------------------

Open lane_hogging_classifier.py in any text editor.
Near the top of the file (around line 55) you will see:

    T_CF           = 4.0    # seconds — car-following threshold
    V_T            = 1.0    # m/s     — overtaking speed difference
    D              = 50.0   # metres  — left-lane look-ahead distance
    OPP_DUR        = 5.0    # seconds — min opportunity duration (from flowchart)
    T_PERS_PCT     = 0.50   # 0 to 1  — min % of eligible time spent hogging
    MIN_STAGE3_SEC = 5.0    # seconds — min time needed to classify a vehicle

Change any value, save the file, and re-run Step 1 and Step 2.

NOTE ON T_PERS_PCT:
  The original flowchart uses a fixed persistence time (e.g. 60 seconds).
  The highD camera only covers ~420 metres of road, so vehicles are visible
  for just 10-36 seconds on average. A 60 second threshold would classify
  zero vehicles as hoggers. Instead, this script uses a proportional
  threshold: a vehicle is a hogger if it spends >= 50% of its eligible
  observed time hogging. This is the standard adaptation for short aerial
  observation datasets.


----------------------------------------------------------------
QUICK REFERENCE — commands to copy-paste
----------------------------------------------------------------

  # Install libraries (first time only)
  pip install pandas numpy

  # Navigate to script folder
  cd path\to\data\script

  # Run the classifier
  python lane_hogging_classifier.py --data_dir .. --output_dir ../results

  # Merge all results into one file
  python merge_results.py

  # To re-run with different parameters:
  # 1. Edit the values near line 55 of lane_hogging_classifier.py
  # 2. Delete the results/ folder
  # 3. Run the two commands above again


----------------------------------------------------------------
TROUBLESHOOTING
----------------------------------------------------------------

"No module named pandas"
  → Run: pip install pandas numpy

"No *_tracks.csv files found"
  → Make sure you are running from inside the script/ folder
  → Make sure --data_dir is pointing to where your CSVs are

"Missing companion file: XX_tracksMeta.csv"
  → That recording is missing its meta file. It will be skipped.
  → Check that all 3 files exist for each recording number.

"No result files found" (when running merge_results.py)
  → Run Step 1 (lane_hogging_classifier.py) first before merge_results.py

Results seem too low (very few True values)
  → This is expected. Most vehicles are car-following or actively
    overtaking, so they are correctly classified as False.
  → If you want more sensitivity, lower T_PERS_PCT to 0.40 or 0.35

================================================================