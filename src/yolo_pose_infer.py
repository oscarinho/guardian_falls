# src/yolo_pose_infer.py
import argparse, math, time, os, csv
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ------------------------------
# Config general
# ------------------------------
cv2.setNumThreads(2)  # evita overhead de hilos en CPU


def get_device():
    # Forzamos CPU por bug conocido MPS + Pose en macOS
    print("⚙️  Ejecutando en CPU (bug MPS conocido en modelos Pose).")
    return torch.device("cpu")


def ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)


def body_metrics_from_keypoints(kps):
    """
    kps: (17,3) xyv del modelo COCO keypoints (Ultralytics Pose)
    Retorna:
      angle_deg: ángulo del eje hombros->caderas respecto a vertical (0° = vertical, 90° = horizontal)
      aspect:    razón ancho/alto del bbox del esqueleto
      center_y:  centro vertical del esqueleto (px)
      w, h:      ancho y alto del bbox del esqueleto (px)
    """
    # índices COCO típicos (0 nariz; 5 hombro izq; 6 hombro der; 11 cadera izq; 12 cadera der)
    L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
    pts = kps[:, :2]
    valid = kps[:, 2] > 0

    xs = pts[valid, 0]
    ys = pts[valid, 1]
    if xs.size < 4:
        return None, None, None, None, None

    w = float((xs.max() - xs.min()) + 1e-6)
    h = float((ys.max() - ys.min()) + 1e-6)
    aspect = w / h

    shoulder_mid = pts[[L_SHO, R_SHO]].mean(axis=0)
    hip_mid = pts[[L_HIP, R_HIP]].mean(axis=0)

    vec = hip_mid - shoulder_mid  # hacia abajo idealmente
    # Ángulo respecto a vertical (eje Y): si vertical -> 0°, si horizontal -> 90°
    angle_rad = math.atan2(abs(vec[0]), abs(vec[1]))
    angle_deg = float(np.degrees(angle_rad))

    center_y = float(ys.mean())
    return angle_deg, aspect, center_y, w, h

def make_out_path(path, suffix="_pose"):
    root, _ = os.path.splitext(path)
    return f"{root}{suffix}.mp4"   # fuerza mp4 para compatibilidad


def run_on_source(model, source, args, events_writer=None):
    # Abrimos fuente
    cap_source = 0 if source == "0" else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir: {source}")
        if not (source == "0") and not os.path.exists(source):
            print("  • La ruta no existe. Pasa ruta correcta/absoluta o mueve el archivo a data/samples/")
        else:
            print("  • Probable backend/codec. Considera transcodificar a MP4 H.264 yuv420p (avc1).")
        return

    # Info de la fuente
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Estado por persona (pid ~ índice detección; idealmente usar tracker)
    last_cy = {}
    vel_norm_ema = {}           # vy normalizada (h/s) con EMA
    just_dropped = defaultdict(int)
    horiz_frames = defaultdict(int)
    cooldown = defaultdict(int)
    COOLDOWN_FRAMES = args.cooldown

    # Cola de últimos eventos
    recent_falls = deque(maxlen=5)

    # Writer de salida opcional
    writer = None
    if args.save_video and isinstance(cap_source, str):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = make_out_path(source, suffix="_pose")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Guardando video anotado en: {out_path}")

    # Tipografías
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = max(0.7, min(1.6, width / 1280))
    thick = max(2, int(2 * base_scale))

    device = "cpu"  # fijo por bug MPS+Pose

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inferencia pose (Ultralytics)
        results = model.predict(
            frame,
            device=device,
            conf=args.conf,
            imgsz=args.imgsz,
            vid_stride=args.vid_stride,
            max_det=args.max_det,
            verbose=False
        )
        r = results[0]
        kps = getattr(r, "keypoints", None)
        boxes = getattr(r, "boxes", None)

        if kps is not None and boxes is not None:
            n = len(kps)
            now = time.time()

            for i in range(n):
                xyv = kps[i].data.cpu().numpy().squeeze()  # (17,3)
                angle_deg, aspect, cy, bw, bh = body_metrics_from_keypoints(xyv)
                if angle_deg is None:  # keypoints insuficientes
                    continue

                pid = i
                # estimación de velocidad vertical normalizada por altura de la persona (h/s)
                prev_cy = last_cy.get(pid, cy)
                dt = 1.0 / max(fps, 1.0)  # estable; evita depender de time.time()
                vy_px_s = (cy - prev_cy) / dt
                last_cy[pid] = cy

                vy_norm = vy_px_s / max(bh, 1e-6)  # alturas por segundo (h/s)
                vel_norm_ema[pid] = ema(vel_norm_ema.get(pid), vy_norm, args.ema_alpha)

                # Reglas (dos etapas)
                is_fast_drop = (vel_norm_ema[pid] is not None) and (vel_norm_ema[pid] > args.drop_hps)
                is_horizontal = (angle_deg > args.angle_th) or (aspect > args.aspect_th)

                # Memorias
                if is_fast_drop:
                    just_dropped[pid] = args.drop_memory
                else:
                    just_dropped[pid] = max(0, just_dropped[pid] - 1)

                if is_horizontal:
                    horiz_frames[pid] += 1
                else:
                    horiz_frames[pid] = 0

                fall = False
                if cooldown[pid] == 0 and just_dropped[pid] > 0 and horiz_frames[pid] >= args.horiz_min_frames:
                    fall = True
                    cooldown[pid] = COOLDOWN_FRAMES
                    recent_falls.append({
                        "t": now, "pid": pid,
                        "ang": angle_deg, "asp": aspect,
                        "vy_hps": vel_norm_ema[pid]
                    })
                    if events_writer:
                        events_writer.writerow({
                            "source": source,
                            "time_s": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                            "pid": pid,
                            "angle_deg": angle_deg,
                            "aspect": aspect,
                            "vy_hps": vel_norm_ema[pid]
                        })
                else:
                    cooldown[pid] = max(0, cooldown[pid] - 1)

                # Dibujo
                bb = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
                color = (0, 0, 255) if fall else ((0, 165, 255) if is_fast_drop else (0, 255, 0))
                cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), color, thick)

                # Texto grande y legible
                txt1 = f"id:{pid} ang:{angle_deg:.0f} asp:{aspect:.2f}"
                txt2 = f"vy:{vel_norm_ema[pid]:.2f} h/s"
                cv2.putText(frame, txt1, (bb[0], max(22, bb[1]-30)), font, 0.7*base_scale, color, thick)
                cv2.putText(frame, txt2, (bb[0], max(42, bb[1]-8)),  font, 0.7*base_scale, color, thick)
                if fall:
                    cv2.putText(frame, "FALL!", (bb[0], bb[1]-50), font, 0.9*base_scale, (0,0,255), max(2, thick+1))

        # HUD
        hud = f"FPS:{fps:.1f} | DROP>{args.drop_hps:.2f} h/s | ANG>{args.angle_th} | ASP>{args.aspect_th}"
        cv2.putText(frame, hud, (10, int(28*base_scale)), font, 0.7*base_scale, (255,255,255), thick)

        y = int(54*base_scale)
        for ev in list(recent_falls):
            cv2.putText(frame, f"fall pid:{ev['pid']} vy:{ev['vy_hps']:.2f} h/s",
                        (10, y), font, 0.7*base_scale, (0,0,255), thick)
            y += int(22*base_scale)

        # Mostrar/guardar
        if not args.headless:
            cv2.imshow(f"Pose: {os.path.basename(source)}", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    # Fuente única (webcam o archivo)
    ap.add_argument("--source", default="0", help="0 webcam o ruta de video (ej. data/samples/opp.MOV)")

    # Modelo
    ap.add_argument("--model", default="yolo11n-pose.pt", help="modelo pose Ultralytics (*-pose.pt)")
    ap.add_argument("--conf", type=float, default=0.25)

    # Rendimiento
    ap.add_argument("--imgsz", type=int, default=448, help="tamaño de entrada (320–640). 320/384 para CPU rápido")
    ap.add_argument("--vid_stride", type=int, default=1, help="salto de frames de entrada (1–3)")
    ap.add_argument("--max_det", type=int, default=3, help="máximo de personas por frame")
    ap.add_argument("--ema_alpha", type=float, default=0.35,
                help="alpha del suavizado EMA para velocidad normalizada (0.2–0.5)")

    # Heurística caída
    ap.add_argument("--drop_hps", type=float, default=0.90, help="umbral de descenso en alturas por segundo (h/s)")
    ap.add_argument("--angle_th", type=float, default=55.0, help="ángulo tronco vs vertical para horizontalidad")
    ap.add_argument("--aspect_th", type=float, default=1.40, help="ancho/alto que sugiere horizontalidad")
    ap.add_argument("--horiz_min_frames", type=int, default=6, help="frames consecutivos horizontal tras el drop")
    ap.add_argument("--drop_memory", type=int, default=15, help="ventana (frames) donde se recuerda el drop")
    ap.add_argument("--cooldown", type=int, default=30, help="frames de enfriamiento por ID")

    # Visualización / salida
    ap.add_argument("--headless", action="store_true", help="no mostrar ventana (útil para medir FPS)")
    ap.add_argument("--save_video", action="store_true", help="guardar video anotado (solo si source es archivo)")
    ap.add_argument("--events_csv", default=None, help="CSV para loguear eventos de fall")

    args = ap.parse_args()

    device = get_device()  # imprime aviso y fija CPU
    model = YOLO(args.model)
    model.fuse()

    # CSV de eventos
    events_writer = None
    f = None
    if args.events_csv:
        new_file = not os.path.exists(args.events_csv)
        f = open(args.events_csv, "a", newline="")
        events_writer = csv.DictWriter(f, fieldnames=["source","time_s","pid","angle_deg","aspect","vy_hps"])
        if new_file:
            events_writer.writeheader()

    run_on_source(model, args.source, args, events_writer)

    if f:
        f.close()


if __name__ == "__main__":
    main()
