#!/usr/bin/env bash
# Asegura activar Conda (ajusta la ruta si usas miniforge)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ge-falls

python src/yolo_pose_infer.py \
  --glob "data/samples/*.mp4" \
  --list opp.mov cam1.avi cam3.avi cam5.avi cam7.avi \
  --conf 0.25 \
  --drop_px_per_s 600 \
  --angle_th 55 \
  --aspect_th 1.4 \
  --horiz_min_frames 6 \
  --save_video \
  --events_csv runs_falls.csv
