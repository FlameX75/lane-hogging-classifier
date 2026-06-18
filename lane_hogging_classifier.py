"""
Lane Hogging Classifier — highD Dataset
========================================
Classifies each vehicle in highD recordings as a lane hogger (True/False)
using the 4-stage flowchart decision logic.

OUTPUTS (saved to --output_dir):
  XX_results_per_vehicle.csv  — one row per vehicle (primary deliverable)
  XX_results_per_frame.csv    — frame-by-frame labels (for detailed analysis)
  00_summary.csv              — aggregate stats per recording

USAGE:
  python lane_hogging_classifier.py --data_dir /path/to/data --output_dir /path/to/output

  Each recording XX must have: XX_tracks.csv, XX_tracksMeta.csv, XX_recordingMeta.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETERS (edit below if supervisor specifies different values)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  T_CF             = 4.0 s      THW threshold: car-following vs free-flow
  V_T              = 1.0 m/s    Speed-diff threshold: overtaking intent
  D                = 50.0 m     Look-ahead for left-lane overtaking check
  OPP_DUR          = 5.0 s      Min continuous safe return opportunity (Stage 3)
  T_PERS_PCT       = 0.50       Stage 4: min fraction of stage-3 time spent hogging
  MIN_STAGE3_SEC   = 5.0 s      Min time reaching Stage 3 to classify a vehicle

NOTE ON T_PERS_PCT:
  The original flowchart uses an absolute T (e.g. 60s) for persistence.
  The highD camera covers only ~420m of road, so vehicles are visible for
  just 10–36 seconds — making any fixed T >= 30s impossible for nearly all
  vehicles. This script uses a proportional threshold instead:
    vehicle is a hogger if hogging_frames / stage3_frames >= T_PERS_PCT
  This faithfully captures "the vehicle stayed in the middle/fast lane and
  ignored safe return opportunities for most of its observed time."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOWCHART → CODE MAPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 1 — Lane Eligibility
  CSV column : laneId
  Logic      : rightmost lane (upper→laneId=2, lower→laneId=8) → FALSE
               middle/fast lanes → eligible, continue

Stage 2 — Driving State Check
  CSV columns: precedingId, thw, xVelocity, precedingXVelocity
  Logic      :
    precedingId == 0 → no lead → Overtaking Check (side box)
    thw < T_CF       → car-following, constrained → FALSE
    thw >= T_CF:
      Δv = |xVelocity| - |precedingXVelocity|
      Δv > V_T         → overtaking state, constrained → FALSE
      Δv <= V_T        → free, not overtaking → Stage 3

Overtaking Check (side box — when no lead vehicle in current lane)
  CSV columns: leftPrecedingId, x, width, xVelocity
  Logic      : if leftPrecedingId exists within D metres → FALSE
               else → FALSE
               (both branches exit False; SV is free-moving but context check fails)

Stage 3 — Return Left Opportunity
  CSV columns: leftPrecedingId, leftAlongsideId, leftFollowingId,
               x, width, xVelocity (for gap/TH/TTC computation)
  tracksMeta : class (Car / Truck)
  Logic      :
    No left-lane vehicles → gap open → increment opportunity streak
    left-lane vehicles present:
      leftAlongsideId != 0          → unsafe (blocked) → reset streak
      leftFollowingId class==Truck  → HDV → unsafe → reset streak
      TH_left <= 2s OR TTC_left <= 5s → unsafe → reset streak
      leftPrecedingId class==Truck  → HDV → unsafe → reset streak
      else                          → gap safe → increment streak
    opportunity_streak > OPP_DUR (frames) → frame is a hogging candidate

Stage 4 — Persistence (proportional, adapted for short observation window)
  vehicle is a hogger if:
    stage3_frames >= MIN_STAGE3_SEC × frameRate  AND
    hogging_frames / stage3_frames >= T_PERS_PCT
  where stage3_frames = frames that passed Stage 1 + Stage 2
        hogging_frames = frames with opportunity streak > OPP_DUR
"""

import os
import sys
import argparse
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
T_CF           = 4.0    # s     car-following THW threshold
V_T            = 1.0    # m/s   overtaking speed-difference threshold
D              = 50.0   # m     left-lane look-ahead for overtaking check
OPP_DUR        = 5.0    # s     min continuous safe return opportunity (Stage 3)
T_PERS_PCT     = 0.50   # 0–1   fraction of stage-3 time that must be hogging
MIN_STAGE3_SEC = 5.0    # s     min stage-3 time required to classify a vehicle

MAX_VALID_THW  = 60.0   # s     cap for corrupt THW values (artefact in some recordings)


# ─────────────────────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────────────────────

def get_recording_id(tracks_path: str) -> str:
    return os.path.basename(tracks_path).replace("_tracks.csv", "")


def load_recording(tracks_path: str, data_dir: str):
    rec_id = get_recording_id(tracks_path)
    meta_path     = os.path.join(data_dir, f"{rec_id}_tracksMeta.csv")
    rec_meta_path = os.path.join(data_dir, f"{rec_id}_recordingMeta.csv")
    for p in [meta_path, rec_meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
    tracks      = pd.read_csv(tracks_path)
    tracks_meta = pd.read_csv(meta_path)
    rec_meta    = pd.read_csv(rec_meta_path)
    frame_rate  = int(rec_meta["frameRate"].iloc[0])
    return tracks, tracks_meta, frame_rate


# ─────────────────────────────────────────────────────────────
# LANE TOPOLOGY
# ─────────────────────────────────────────────────────────────

def get_lane_topology(tracks: pd.DataFrame) -> dict:
    """
    Upper direction (xVelocity < 0): rightmost = lowest laneId (e.g. 2).
    Lower direction (xVelocity > 0): rightmost = highest laneId (e.g. 8).
    Returns lane1 (rightmost) and eligible set for each direction.
    """
    up  = sorted(tracks.loc[tracks["xVelocity"] < 0, "laneId"].unique())
    lo  = sorted(tracks.loc[tracks["xVelocity"] > 0, "laneId"].unique())
    return {
        "upper": {"lane1": up[0],  "eligible": set(up[1:])},
        "lower": {"lane1": lo[-1], "eligible": set(lo[:-1])},
    }


# ─────────────────────────────────────────────────────────────
# POSITION LOOKUP
# ─────────────────────────────────────────────────────────────

def build_pos_lookup(tracks: pd.DataFrame) -> dict:
    """(frame, id) → {'x', 'xVelocity', 'width'} for gap computation."""
    lookup = {}
    for r in tracks[["frame", "id", "x", "xVelocity", "width"]].itertuples(index=False):
        lookup[(r.frame, r.id)] = {"x": r.x, "xVelocity": r.xVelocity, "width": r.width}
    return lookup


# ─────────────────────────────────────────────────────────────
# CORE CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_recording(tracks, tracks_meta, frame_rate, verbose=True):
    """
    Apply the 4-stage flowchart to every vehicle frame-by-frame.
    Returns (per_frame_df, per_vehicle_df).
    """
    class_lookup = dict(zip(tracks_meta["id"], tracks_meta["class"]))
    pos_lookup   = build_pos_lookup(tracks)
    topology     = get_lane_topology(tracks)

    upper_lane1, upper_eligible = topology["upper"]["lane1"], topology["upper"]["eligible"]
    lower_lane1, lower_eligible = topology["lower"]["lane1"], topology["lower"]["eligible"]

    opp_frames = int(OPP_DUR * frame_rate)   # e.g. 5s × 25fps = 125 frames

    results = []
    vgroups = list(tracks.groupby("id"))
    total   = len(vgroups)

    for v_idx, (vid, vdf) in enumerate(vgroups):
        if verbose and v_idx % 500 == 0:
            print(f"    {v_idx+1}/{total} vehicles processed...")

        vdf = vdf.sort_values("frame").reset_index(drop=True)
        vcls = class_lookup.get(vid, "Car")
        opp_streak = 0

        for row in vdf.itertuples(index=False):
            frame         = row.frame
            lane_id       = row.laneId
            x             = row.x
            width         = row.width
            xvel          = row.xVelocity
            prec_id       = row.precedingId
            thw           = row.thw
            prec_xvel     = row.precedingXVelocity
            lp_id         = row.leftPrecedingId
            la_id         = row.leftAlongsideId
            lf_id         = row.leftFollowingId

            # ── Determine direction ─────────────────────────────
            if   xvel < 0: direction, lane1, eligible = "upper", upper_lane1, upper_eligible
            elif xvel > 0: direction, lane1, eligible = "lower", lower_lane1, lower_eligible
            else:
                opp_streak = 0
                results.append({"frame": frame, "id": vid, "laneId": lane_id,
                                 "class": vcls, "eligible": False,
                                 "reached_stage3": False, "stage_exit": "stationary",
                                 "opp_streak": 0, "is_lane_hogging": False})
                continue

            # ── STAGE 1: Lane Eligibility ───────────────────────
            if lane_id == lane1 or lane_id not in eligible:
                opp_streak = 0
                results.append({"frame": frame, "id": vid, "laneId": lane_id,
                                 "class": vcls, "eligible": False,
                                 "reached_stage3": False, "stage_exit": "S1_lane1",
                                 "opp_streak": 0, "is_lane_hogging": False})
                continue

            # ── STAGE 2: Driving State Check ────────────────────
            sv_speed = abs(xvel)

            if prec_id == 0:
                # No lead vehicle → Overtaking Check (side box)
                exit_reason = "S2_no_lead"
                if lp_id != 0:
                    lp = pos_lookup.get((frame, lp_id))
                    if lp is not None:
                        dist = (x - (lp["x"] + lp["width"]) if direction == "upper"
                                else lp["x"] - (x + width))
                        if 0 <= dist <= D:
                            exit_reason = "S2_overtake_check"
                opp_streak = 0
                results.append({"frame": frame, "id": vid, "laneId": lane_id,
                                 "class": vcls, "eligible": True,
                                 "reached_stage3": False, "stage_exit": exit_reason,
                                 "opp_streak": 0, "is_lane_hogging": False})
                continue

            # Has lead vehicle
            clean_thw = thw if abs(thw) <= MAX_VALID_THW else 0.0

            if clean_thw < T_CF:
                opp_streak = 0
                results.append({"frame": frame, "id": vid, "laneId": lane_id,
                                 "class": vcls, "eligible": True,
                                 "reached_stage3": False, "stage_exit": "S2_car_following",
                                 "opp_streak": 0, "is_lane_hogging": False})
                continue

            delta_v = sv_speed - abs(prec_xvel)
            if delta_v > V_T:
                opp_streak = 0
                results.append({"frame": frame, "id": vid, "laneId": lane_id,
                                 "class": vcls, "eligible": True,
                                 "reached_stage3": False, "stage_exit": "S2_overtaking",
                                 "opp_streak": 0, "is_lane_hogging": False})
                continue

            # ── STAGE 3: Return Left Opportunity ────────────────
            # Vehicle is free-flowing but not overtaking → check if it CAN return left
            has_left = (lp_id != 0 or la_id != 0 or lf_id != 0)
            gap_safe = True
            s3_exit  = "S3_open_gap"

            if has_left:
                s3_exit = "S3_safe_gap"

                # Alongside → blocked
                if la_id != 0:
                    gap_safe = False; s3_exit = "S3_blocked_alongside"

                # Left follower checks
                if gap_safe and lf_id != 0:
                    if class_lookup.get(lf_id, "Car") == "Truck":
                        gap_safe = False; s3_exit = "S3_lf_hdv"
                    else:
                        lf = pos_lookup.get((frame, lf_id))
                        if lf is None:
                            gap_safe = False; s3_exit = "S3_lf_missing"
                        else:
                            gap = (lf["x"] - (x + width) if direction == "upper"
                                   else x - (lf["x"] + lf["width"]))
                            if gap <= 0:
                                gap_safe = False; s3_exit = "S3_overlap"
                            else:
                                th_l  = gap / sv_speed if sv_speed > 0 else np.inf
                                ar    = abs(lf["xVelocity"]) - sv_speed
                                ttc_l = gap / ar if ar > 0 else np.inf
                                if th_l <= 2.0:
                                    gap_safe = False; s3_exit = "S3_th_small"
                                elif ttc_l <= 5.0:
                                    gap_safe = False; s3_exit = "S3_ttc_small"

                # Left preceding HDV check
                if gap_safe and lp_id != 0:
                    if class_lookup.get(lp_id, "Car") == "Truck":
                        gap_safe = False; s3_exit = "S3_lp_hdv"

            if not gap_safe:
                opp_streak = 0
                results.append({"frame": frame, "id": vid, "laneId": lane_id,
                                 "class": vcls, "eligible": True,
                                 "reached_stage3": True, "stage_exit": s3_exit,
                                 "opp_streak": 0, "is_lane_hogging": False})
                continue

            # Gap is safe → opportunity exists
            opp_streak += 1

            if opp_streak <= opp_frames:
                # Opportunity not sustained long enough yet
                results.append({"frame": frame, "id": vid, "laneId": lane_id,
                                 "class": vcls, "eligible": True,
                                 "reached_stage3": True, "stage_exit": "S3_opp_building",
                                 "opp_streak": opp_streak, "is_lane_hogging": False})
                continue

            # ── STAGE 4: Frame-level hogging flag ───────────────
            # Opportunity has been available > OPP_DUR continuously.
            # Vehicle-level verdict is determined after all frames are processed.
            results.append({"frame": frame, "id": vid, "laneId": lane_id,
                             "class": vcls, "eligible": True,
                             "reached_stage3": True, "stage_exit": "S4_hogging",
                             "opp_streak": opp_streak, "is_lane_hogging": True})

    # ── Build per-frame DataFrame ────────────────────────────────────────
    per_frame_df = pd.DataFrame(results)

    if per_frame_df.empty:
        return per_frame_df, pd.DataFrame()

    # ── Per-vehicle aggregation ──────────────────────────────────────────
    agg = (
        per_frame_df.groupby(["id", "class"])
        .agg(
            total_frames    = ("eligible",        "count"),
            eligible_frames = ("eligible",        "sum"),
            stage3_frames   = ("reached_stage3",  "sum"),
            hogging_frames  = ("is_lane_hogging", "sum"),
        )
        .reset_index()
    )

    agg["total_sec"]    = agg["total_frames"]    / frame_rate
    agg["eligible_sec"] = agg["eligible_frames"] / frame_rate
    agg["stage3_sec"]   = agg["stage3_frames"]   / frame_rate
    agg["hogging_sec"]  = agg["hogging_frames"]  / frame_rate

    # Stage 4 persistence: proportional threshold on stage-3-reached frames
    # denominator = stage3_frames (frames that passed Stage 1 + 2)
    min_s3_frames = int(MIN_STAGE3_SEC * frame_rate)

    agg["hogging_pct"] = np.where(
        agg["stage3_frames"] > 0,
        (agg["hogging_frames"] / agg["stage3_frames"] * 100).round(2),
        0.0
    )

    agg["is_lane_hogger"] = (
        (agg["stage3_frames"] >= min_s3_frames) &
        (agg["hogging_frames"] / agg["stage3_frames"].clip(lower=1) >= T_PERS_PCT)
    )

    per_vehicle_df = agg[[
        "id", "class", "is_lane_hogger",
        "hogging_frames", "hogging_sec",
        "stage3_frames", "stage3_sec",
        "eligible_frames", "eligible_sec",
        "total_frames", "total_sec",
        "hogging_pct",
    ]].copy()

    # Push vehicle-level verdict back to per-frame table
    hogger_ids = set(per_vehicle_df.loc[per_vehicle_df["is_lane_hogger"], "id"])
    per_frame_df["vehicle_is_hogger"] = per_frame_df["id"].isin(hogger_ids)

    return per_frame_df, per_vehicle_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(data_dir: str, output_dir: str, verbose: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    tracks_files = sorted(glob.glob(os.path.join(data_dir, "*_tracks.csv")))
    if not tracks_files:
        print(f"[ERROR] No *_tracks.csv files found in: {data_dir}")
        sys.exit(1)

    print(f"\nFound {len(tracks_files)} recording(s).")
    print(
        f"Parameters: T_CF={T_CF}s | V_T={V_T}m/s | D={D}m | OPP_DUR={OPP_DUR}s\n"
        f"            T_PERS_PCT={T_PERS_PCT*100:.0f}% | MIN_STAGE3={MIN_STAGE3_SEC}s\n"
    )

    summary_rows = []

    for tracks_path in tracks_files:
        rec_id = get_recording_id(tracks_path)
        print(f"{'='*55}\nRecording: {rec_id}\n{'='*55}")

        try:
            tracks, tracks_meta, frame_rate = load_recording(tracks_path, data_dir)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}\n")
            continue

        print(f"  Rows: {len(tracks):,} | Vehicles: {tracks['id'].nunique()} | FPS: {frame_rate}")

        per_frame_df, per_vehicle_df = classify_recording(
            tracks, tracks_meta, frame_rate, verbose=verbose
        )

        frame_out   = os.path.join(output_dir, f"{rec_id}_results_per_frame.csv")
        vehicle_out = os.path.join(output_dir, f"{rec_id}_results_per_vehicle.csv")
        per_frame_df.to_csv(frame_out,    index=False)
        per_vehicle_df.to_csv(vehicle_out, index=False)

        n_veh     = len(per_vehicle_df)
        n_hog     = int(per_vehicle_df["is_lane_hogger"].sum()) if n_veh > 0 else 0
        pct       = n_hog / n_veh * 100 if n_veh > 0 else 0.0

        print(f"  Vehicles: {n_veh} | Hoggers: {n_hog} ({pct:.1f}%)")
        print(f"  → {vehicle_out}\n  → {frame_out}\n")

        summary_rows.append({
            "recording_id":   rec_id,
            "total_vehicles": n_veh,
            "lane_hoggers":   n_hog,
            "hogging_pct":    round(pct, 2),
        })

    if summary_rows:
        summary_df  = pd.DataFrame(summary_rows)
        summary_out = os.path.join(output_dir, "00_summary.csv")
        summary_df.to_csv(summary_out, index=False)
        print(f"{'='*55}\nSUMMARY → {summary_out}")
        print(summary_df.to_string(index=False))
        print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane hogging classifier — highD dataset")
    parser.add_argument("--data_dir",   required=True, help="Folder with highD CSV files")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, verbose=not args.quiet)
