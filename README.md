# 🛰️ Guardian Eye – Pose-Based Fall Detection Baseline

## 🧠 Overview
**Guardian Eye** is an AI-powered monitoring system designed to detect human falls in real time.  
This module represents the **pose-based baseline**, developed to analyze the **geometry and motion** of the human body rather than only detecting objects.  
Using **Ultralytics YOLO Pose**, the system infers 17 human keypoints (COCO format) and calculates spatio-temporal metrics such as **body angle**, **aspect ratio**, and **vertical velocity**, enabling early and interpretable fall detection.

---

## ⚙️ Technical Stack
- **Framework:** [Ultralytics YOLOv11-Pose](https://docs.ultralytics.com)
- **Language:** Python 3.11  
- **Libraries:** PyTorch 2.8, OpenCV 4.9, NumPy, Pandas, Scikit-learn  
- **Hardware Tested:** MacBook M4 Pro (24 GB RAM, CPU mode)  
- **Dataset:** Roboflow “Fall-Oscar v3” (571 labeled images)  

---

## 🧩 Model Description
The model estimates human pose and derives three key metrics:

| Variable | Description | Purpose |
|-----------|--------------|----------|
| `angle_deg (ang)` | Angle between shoulders–hips axis and the vertical | Detect posture tilt |
| `aspect (asp)` | Ratio between skeleton width and height | Identify standing vs lying |
| `vy_hps` | Vertical velocity normalized by person’s height | Detect sudden vertical drops |
| `ema_alpha` | Temporal smoothing factor | Reduce noise from pose jitter |

**Fall Detection Heuristic**
```
(vy_hps > 0.9) AND (angle_deg > 55 OR aspect > 1.4)
AND posture remains horizontal for ≥ 6 frames
```

---

## 🧪 Experimental Results

### ✅ Observations
- Successfully detects falls in **frontal** camera angles.  
- Decreased accuracy in **side or distant** viewpoints.  
- Achieves **~15 FPS** on CPU (resolution 448×448).  
- Visualization overlays allow effective threshold tuning and debugging.

### ⚡ Strengths
- High interpretability: every decision can be visualized (`ang`, `asp`, `vy_hps`).  
- Runs in real time without GPU acceleration.  
- Fully explainable rule-based logic.

### ⚠️ Limitations
- Sensitive to camera angle and distance.  
- Misses partial occlusions or falls leaving the frame.  
- Purely heuristic — lacks learned context or adaptability.

---

## 🧠 Insights and Key Decisions

| Area | Decision | Rationale |
|-------|-----------|------------|
| Execution backend | Forced CPU | Avoid Apple MPS pose bug |
| Motion representation | Use normalized vertical velocity (h/s) | Scale-independent measurement |
| Posture indicator | Combine angle + aspect ratio | Capture both tilt and flattening |
| Detection logic | Two-stage: sudden drop + horizontal posture | Balance precision vs recall |
| Future step | Add supervised classifier (Roboflow dataset) | Improve robustness and context-awareness |

---

## 📚 Lessons Learned
1. Pose-based methods are **interpretable and explainable**, providing measurable physical context.  
2. Pure heuristics require **manual tuning** and are angle-dependent.  
3. Normalizing velocity by body height increases **robustness to distance and scale**.  
4. **EMA smoothing** greatly reduces false positives from jittery keypoints.  
5. Integration with a **supervised posture classifier** is the logical next evolution.

---

## 🔮 Next Steps (Phase 2: Supervised Posture Classification)
The next phase builds on the current baseline using the **Roboflow “Fall-Oscar v3” dataset**.

**Planned Actions**
1. Extract pose features (`angle`, `aspect`, `var_y`, `ratio_hw`) from each image.  
2. Assemble a labeled CSV linking features and ground-truth classes (`fall`, `normal`).  
3. Train a small supervised model (Logistic Regression / XGBoost).  
4. Integrate the classifier as a **posture verification module** within the runtime.  
5. Fine-tune using real-world Guardian Eye camera data.

---

## 🚀 How to Run

### 1️⃣ Environment Setup
```bash
conda env create -f environment.yml
conda activate ge-falls
```

### 2️⃣ Run Fall Detection
```bash
python src/yolo_pose_infer.py   --source data/samples/opp.MOV   --imgsz 320 --vid_stride 2 --max_det 2   --conf 0.30 --save_video
```

### 3️⃣ Batch Mode
```bash
python src/yolo_pose_infer.py   --glob "data/samples/*.mp4"   --save_video --events_csv runs_falls.csv
```

**Output:**  
Annotated videos → `data/samples/<filename>_pose.mp4`  
Event logs → `runs_falls.csv`


---

## 📈 Related Work
- **YOLOv12 Two-Class Model:** Direct detection of `fall` and `normal`.  
- **Pose-Based Baseline (this work):** Rule-based + interpretable geometry.  
- **Next Step:** Hybrid system (pose geometry + supervised posture learning).


---

## 📚 References
- Guardian Eye Project Team (2025). *AI-Powered Monitoring for Safer Independent Aging.*  
- Ultralytics. (2025). *YOLO Pose Documentation.* https://docs.ultralytics.com  
- Roboflow. (2024). *Annotation Platform.* https://roboflow.com  
- Sarker, S. (2022). *Multiple Cameras Fall Dataset.* Kaggle.  
- Charfi, I. et al. (2013). *Optimized Spatio-Temporal Descriptors for Real-Time Fall Detection.* Journal of Electronic Imaging, 22(4), 041106.  

---

> 🧩 *This pose-based baseline serves as a bridge between rule-based motion analysis and data-driven classification, paving the way for a deployable Guardian Eye prototype.*

