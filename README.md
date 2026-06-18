# lane-hogging-classifier

HOW TO USE — Lane Hogging Classifier
======================================

SETUP
-----
1. Make sure Python is installed on your computer.

2. Install the required libraries (only once):
   Open terminal / command prompt and run:

       pip install pandas numpy

3. Place your highD CSV files in a folder (e.g. "data").
   Each recording needs all 3 files:
       XX_tracks.csv
       XX_tracksMeta.csv
       XX_recordingMeta.csv

4. Put the script files inside a subfolder called "script" inside "data":
       data/
       ├── 01_tracks.csv
       ├── 01_tracksMeta.csv
       ├── 01_recordingMeta.csv
       ├── ... all other CSVs ...
       └── script/
           ├── lane_hogging_classifier.py
           └── merge_results.py


RUNNING
-------
Open terminal, go to the script folder:

    cd path\to\data\script

Step 1 — Run the classifier:

    python lane_hogging_classifier.py --data_dir .. --output_dir ../results

   This processes all recordings and saves results to data/results/.
   Takes a few minutes depending on how many files you have.

Step 2 — Combine all results into one file:

    python merge_results.py


OUTPUT
------
Open data/results/ALL_classifications.csv

This has every vehicle from every recording with a True/False label
in the "is_lane_hogger" column. That's the final answer.
