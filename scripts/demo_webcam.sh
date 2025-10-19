#!/usr/bin/env bash
conda activate ge-falls
python src/yolo_pose_infer.py --source 0 --conf 0.25
