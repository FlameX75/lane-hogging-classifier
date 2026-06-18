import pandas as pd
import glob
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
results_dir = os.path.abspath(results_dir)

print(f"Looking for results in: {results_dir}")

files = glob.glob(os.path.join(results_dir, '*_per_vehicle.csv'))

if not files:
    print("ERROR: No result files found. Make sure you ran lane_hogging_classifier.py first.")
    print(f"Expected files like: {results_dir}\\12_results_per_vehicle.csv")
else:
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['recording'] = os.path.basename(f).split('_')[0]
        dfs.append(df[['recording', 'id', 'class', 'is_lane_hogger']])

    combined = pd.concat(dfs)
    out_path = os.path.join(results_dir, 'ALL_classifications.csv')
    combined.to_csv(out_path, index=False)

    print(f"Done! Saved to: {out_path}")
    print(f"Total vehicles : {len(combined)}")
    print(f"Hoggers (True) : {combined['is_lane_hogger'].sum()}")
    print(f"Normal  (False): {(~combined['is_lane_hogger']).sum()}")