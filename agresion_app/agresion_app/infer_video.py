
"""
infer_video.py
══════════════════════════════════════════════════════════════════════════════
Script de inferencia — Clasificación de comportamiento agresivo en video

Uso:
    python infer_video.py --video ruta/al/video.mp4 --checkpoint modelo.pt

Opciones:
    --video       Ruta al video a clasificar
    --checkpoint  Ruta al archivo .pt del modelo (default: ./msg3d_f3_lr3e-04_bs64_T30_best.pt)
    --model_arch  Arquitectura: msg3d (default: msg3d)
    --T           Ventana temporal usada en entrenamiento (default: 30)
    --n_frames    Frames a extraer del video          (default: 64)
    --device      cuda o cpu                          (default: auto)
    --save_video  Guardar video con esqueletos + etiqueta superpuesta
    --verbose     Mostrar detalle de keypoints por frame

Requisitos:
    pip install torch ultralytics opencv-python-headless numpy scipy tqdm

Flujo:
    1. Extraer N_FRAMES frames uniformes del video (OpenCV)
    2. Detectar 2 personas con YOLOv8l-pose → keypoints [T, 2, 17, 3]
    3. Normalizar esqueletos (centro de cadera, escala)
    4. Añadir canales de velocidad y aceleración → [T, 2, 17, 5]
    5. Ajustar a TARGET_T frames (crop o padding)
    6. Cargar MSG3D y hacer forward → logits → softmax → clase + confianza
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Parámetros por defecto (deben coincidir con el entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "bilstm_f3_lr1e-04_bs32_T40_best.pt"
DEFAULT_T          = 40        # TARGET_T con el que se entrenó el modelo
DEFAULT_N_FRAMES   = 64        # Frames a extraer del video
N_PERSONS          = 2         # El modelo siempre espera exactamente 2 personas
CONF_THRESHOLD     = 0.30      # Umbral de confianza para detección YOLOv8
VIS_THRESH         = 0.30      # Umbral de visibilidad de keypoints
TOP_K              = 8         # Máximo de detecciones a considerar por frame
LABELS             = ["No agresivo", "Agresivo"]
LABEL_COLORS       = [(0, 200, 0), (0, 0, 220)]   # Verde / Rojo (BGR)

COCO_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]

# Índices de keypoints COCO para visualización del esqueleto
SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),           # Cabeza
    (5,6),(5,7),(7,9),(6,8),(8,10),    # Brazos
    (5,11),(6,12),(11,12),             # Torso
    (11,13),(13,15),(12,14),(14,16),   # Piernas
]
PERSON_COLORS = [(255, 100, 0), (0, 180, 255)]  # Azul / Cian (BGR)


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 1 — Extracción de frames con OpenCV
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, n_frames: int = DEFAULT_N_FRAMES):
    """
    Extrae exactamente n_frames frames distribuidos uniformemente a lo largo
    del video usando OpenCV (sin dependencia de FFmpeg).

    Retorna:
        frames      : list de arrays BGR [H, W, 3]
        total_orig  : número total de frames en el video original
        fps_orig    : FPS del video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el video: {video_path}")

    total_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig   = cap.get(cv2.CAP_PROP_FPS)
    duration   = total_orig / fps_orig if fps_orig > 0 else 0

    print(f"\n  Video cargado:")
    print(f"    Ruta      : {video_path}")
    print(f"    Frames    : {total_orig}  |  FPS: {fps_orig:.2f}  |  Duración: {duration:.2f}s")

    # Índices uniformes de frames a leer
    if total_orig <= n_frames:
        indices = list(range(total_orig))
    else:
        indices = [int(i * total_orig / n_frames) for i in range(n_frames)]

    frames = []
    for idx in tqdm(indices, desc="  Extrayendo frames", ncols=70):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    print(f"    Frames extraídos: {len(frames)}")
    return frames, total_orig, fps_orig


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 2 — Normalización de pose (igual que en entrenamiento)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_pose(kps: np.ndarray) -> np.ndarray:
    """
    Normaliza un esqueleto centrado en la cadera y escalado por dispersión media.
    kps: [17, 3]  (x, y, visibility)
    """
    kps = kps.copy()
    left_hip, right_hip = 11, 12

    if kps[left_hip, 2] > VIS_THRESH and kps[right_hip, 2] > VIS_THRESH:
        center = (kps[left_hip, :2] + kps[right_hip, :2]) / 2
    else:
        center = kps[:, :2].mean(axis=0)

    kps[:, :2] -= center
    scale = np.linalg.norm(kps[:, :2], axis=1).mean()
    if scale > 1e-3:
        kps[:, :2] /= (scale + 1e-6)
        kps[:, :2] *= 0.5

    return kps


def pose_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Distancia media entre keypoints visibles de dos poses."""
    mask = (p1[:, 2] > VIS_THRESH) & (p2[:, 2] > VIS_THRESH)
    if mask.sum() == 0:
        return 1e6
    return float(np.linalg.norm(p1[mask, :2] - p2[mask, :2]))


def match_poses(prev: np.ndarray, current: list) -> np.ndarray:
    """
    Asigna detecciones del frame actual a las personas del frame anterior
    mediante Hungarian matching para mantener identidad de persona.
    prev    : [2, 17, 3]
    current : list de arrays [17, 3]
    """
    out = prev.copy()
    if len(current) == 0:
        return prev

    cost = np.zeros((N_PERSONS, len(current)))
    for i in range(N_PERSONS):
        for j in range(len(current)):
            cost[i, j] = pose_distance(prev[i], current[j])

    try:
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            out[r] = current[c]
    except Exception:
        return prev

    return out


def temporal_smooth(seq: np.ndarray) -> np.ndarray:
    """Suavizado temporal con ventana de 5 frames (igual que en extracción)."""
    out = seq.copy()
    T = seq.shape[0]
    for t in range(2, T - 2):
        out[t, :, :, :2] = (
            seq[t-2, :, :, :2] + seq[t-1, :, :, :2] +
            seq[t,   :, :, :2] +
            seq[t+1, :, :, :2] + seq[t+2, :, :, :2]
        ) / 5
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 3 — Extracción de keypoints con YOLOv8l-pose
# ══════════════════════════════════════════════════════════════════════════════

def extract_keypoints(frames: list, model, batch_size: int = 16,
                      verbose: bool = False) -> np.ndarray:
    """
    Detecta 2 personas en cada frame y retorna [T, 2, 17, 3].
    Lógica idéntica a fase2_2_2_2_extract_keypoints.py:
      - Si hay 0 detecciones → repite la pose del frame anterior
      - Si hay 1 detección   → duplica como segunda persona (rara vez real)
      - Si hay 2+            → asigna por Hungarian matching

    Retorna:
        seq : np.ndarray [T, 2, 17, 3]  (x, y, visibility)
    """
    sequence = []
    prev = np.zeros((N_PERSONS, 17, 3), dtype=np.float32)
    detection_counts = []

    for i in tqdm(range(0, len(frames), batch_size),
                  desc="  Extrayendo keypoints", ncols=70):
        batch   = frames[i:i + batch_size]
        results = model(batch, conf=CONF_THRESHOLD, verbose=False)

        for res in results:
            detections = []

            if res.keypoints is not None:
                kps   = res.keypoints.data.cpu().numpy()
                boxes = res.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idxs  = np.argsort(areas)[::-1][:TOP_K]
                for idx in idxs:
                    detections.append(kps[idx])

            detection_counts.append(len(detections))

            # Manejo de detecciones
            if len(detections) == 1:
                detections.append(detections[0].copy())   # duplicar
            elif len(detections) == 0:
                detections = [prev[0].copy(), prev[1].copy()]

            current = match_poses(prev, detections)
            for p in range(N_PERSONS):
                current[p] = normalize_pose(current[p])

            sequence.append(current)
            prev = current

    seq = np.stack(sequence, axis=0)        # [T, 2, 17, 3]
    seq = temporal_smooth(seq)

    if verbose:
        unique, counts = np.unique(detection_counts, return_counts=True)
        print(f"\n  Detecciones por frame:")
        for u, c in zip(unique, counts):
            print(f"    {u} personas: {c} frames")
        frames_with_1 = sum(1 for d in detection_counts if d == 1)
        if frames_with_1 > len(frames) * 0.3:
            print(f"\n  ⚠️  ATENCIÓN: {frames_with_1}/{len(frames)} frames ({100*frames_with_1/len(frames):.0f}%)"
                  f" con solo 1 persona detectada.")
            print("     El modelo espera 2 personas. La segunda fue duplicada.")
            print("     Considera verificar que el video contiene 2 personas visibles.")

    return seq


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 4 — Canal de velocidad y aceleración (igual que dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

def add_velocity_channels(seq: np.ndarray) -> np.ndarray:
    """
    seq : [T, 2, 17, 3]  →  [T, 2, 17, 5]
    Canal 3: magnitud de velocidad (diferencia primera de x, y)
    Canal 4: magnitud de aceleración (diferencia segunda)
    """
    T, P, V, C = seq.shape
    out = np.zeros((T, P, V, 5), dtype=np.float32)
    out[:, :, :, :3] = seq

    vel = np.zeros_like(seq[:, :, :, :2])
    vel[1:] = seq[1:, :, :, :2] - seq[:-1, :, :, :2]
    out[:, :, :, 3] = np.linalg.norm(vel, axis=-1)

    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    out[:, :, :, 4] = np.linalg.norm(acc, axis=-1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 5 — Ajustar a TARGET_T frames
# ══════════════════════════════════════════════════════════════════════════════

def resize_temporal(seq: np.ndarray, target_T: int) -> np.ndarray:
    """
    Ajusta la secuencia a exactamente target_T frames:
      - Si T > target_T : recorte centrado
      - Si T < target_T : padding repitiendo el último frame
      - Si T == target_T: sin cambios
    """
    T = seq.shape[0]
    if T == target_T:
        return seq
    elif T > target_T:
        start = (T - target_T) // 2
        return seq[start:start + target_T]
    else:
        pad = np.repeat(seq[-1:], target_T - T, axis=0)
        return np.concatenate([seq, pad], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  PASO 6 — Cargar modelo y hacer inferencia
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device,
               arch: str = "bilstm", T: int = DEFAULT_T) -> nn.Module:

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    print(f"\n  Cargando modelo desde: {checkpoint_path}")

    # ─────────────────────────────────────────────
    # IMPORT CORRECTO
    # ─────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    try:
        from lstm_tsm import BiLSTMClassifier, TSMClassifier
    except ImportError:
        try:
            from models.lstm_tsm import BiLSTMClassifier, TSMClassifier
        except ImportError:
            raise ImportError(
                "No se pudo importar lstm_tsm.py. Verifica que exista en el proyecto."
            )

    # ─────────────────────────────────────────────
    # CREAR MODELO SEGÚN ARQUITECTURA
    # ─────────────────────────────────────────────
    if arch == "bilstm":
        model = BiLSTMClassifier()
    elif arch == "tsm":
        model = TSMClassifier()
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")

    # ─────────────────────────────────────────────
    # CARGAR CHECKPOINT
    # ─────────────────────────────────────────────
    raw = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw

        model.load_state_dict(state_dict, strict=True)
        print("  Formato: state_dict ✓")

    elif isinstance(raw, nn.Module):
        model = raw
        print("  Formato: modelo completo ✓")

    else:
        raise ValueError(f"Formato de checkpoint no reconocido: {type(raw)}")

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros totales: {n_params:,}")

    return model


@torch.no_grad()
def predict(model: nn.Module, seq: np.ndarray,
            device: torch.device, target_T: int) -> dict:
    """
    Ejecuta el forward pass y retorna la predicción con confianza.

    seq     : [T, 2, 17, 5]
    Retorna : dict con clase, confianza, logits y probabilidades
    """
    seq_T = resize_temporal(seq, target_T)               # [T, 2, 17, 5]
    x = torch.from_numpy(seq_T).unsqueeze(0).to(device)  # [1, T, 2, 17, 5]

    logits = model(x)                                    # [1, 2]
    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    return {
        "class_id":    pred_class,
        "class_label": LABELS[pred_class],
        "confidence":  confidence,
        "probs":       probs,
        "logits":      logits.squeeze(0).cpu().numpy(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  OPCIONAL — Guardar video con esqueleto y etiqueta superpuesta
# ══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(frame: np.ndarray, kps: np.ndarray,
                  color: tuple, alpha: float = 0.85) -> np.ndarray:
    """
    Dibuja esqueleto COCO sobre el frame (in-place).
    kps : [17, 3] en coordenadas NORMALIZADAS → se re-escalan al frame
    """
    H, W = frame.shape[:2]
    # Desnormalizar (las coordenadas originales vienen en píxeles de YOLOv8)
    # OJO: aquí kps ya vienen normalizados; los usamos solo para visualización
    # relativa. En el video de salida se usan los kps en coordenadas de imagen.
    for i, j in SKELETON_EDGES:
        if kps[i, 2] > VIS_THRESH and kps[j, 2] > VIS_THRESH:
            x0, y0 = int(kps[i, 0]), int(kps[i, 1])
            x1, y1 = int(kps[j, 0]), int(kps[j, 1])
            cv2.line(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    for k in range(17):
        if kps[k, 2] > VIS_THRESH:
            cv2.circle(frame, (int(kps[k, 0]), int(kps[k, 1])),
                       4, color, -1, cv2.LINE_AA)
    return frame


def save_annotated_video(video_path: str, frames: list,
                         keypoints_raw: np.ndarray,
                         result: dict, out_path: str,
                         fps: float = 15.0):
    """
    Guarda video con esqueletos y etiqueta de clasificación superpuesta.
    keypoints_raw : [T, 2, 17, 3] en coordenadas ORIGINALES (antes de normalizar)
    """
    if not frames:
        return

    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    label    = result["class_label"].upper()
    conf     = result["confidence"]
    color    = LABEL_COLORS[result["class_id"]]
    text     = f"{label}  ({conf*100:.1f}%)"

    for t, frame in enumerate(frames):
        vis = frame.copy()

        # Dibujar esqueletos (coordenadas originales de imagen)
        if t < len(keypoints_raw):
            for p in range(N_PERSONS):
                draw_skeleton(vis, keypoints_raw[t, p], PERSON_COLORS[p])

        # Etiqueta de clasificación
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(vis, (10, 10), (10 + tw + 12, 10 + th + 12), (0,0,0), -1)
        cv2.putText(vis, text, (16, 10 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Barra de probabilidad
        bar_w = int((W - 20) * conf)
        cv2.rectangle(vis, (10, H-25), (W-10, H-10), (50,50,50), -1)
        cv2.rectangle(vis, (10, H-25), (10 + bar_w, H-10), color, -1)

        writer.write(vis)

    writer.release()
    print(f"\n  Video anotado guardado en: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia LSTM — Clasificación de comportamiento agresivo en video"
    )
    parser.add_argument("--video",      required=True,
                        help="Ruta al video a clasificar")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"Ruta al .pt del modelo (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--model_arch", default="bilstm",
                        choices=["bilstm", "tsm"],
                        help="Arquitectura del modelo (bilstm | tsm)"),
    parser.add_argument("--T",          type=int, default=DEFAULT_T,
                        help=f"Ventana temporal T usada en entrenamiento (default: {DEFAULT_T})")
    parser.add_argument("--n_frames",   type=int, default=DEFAULT_N_FRAMES,
                        help=f"Frames a extraer del video (default: {DEFAULT_N_FRAMES})")
    parser.add_argument("--device",     default="auto",
                        help="cuda | cpu | auto (default: auto)")
    parser.add_argument("--save_video", action="store_true",
                        help="Guardar video anotado con esqueletos")
    parser.add_argument("--verbose",    action="store_true",
                        help="Mostrar detalle de detecciones por frame")
    args = parser.parse_args()

    t0 = time.time()

    print("\n" + "═"*60)
    print("  LSTM — INFERENCIA DE COMPORTAMIENTO AGRESIVO")
    print("═"*60)

    # ── Dispositivo ────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Dispositivo   : {device}")
    print(f"  Modelo T      : {args.T} frames")
    print(f"  Frames video  : {args.n_frames}")

    # ── Paso 1: Extraer frames ──────────────────────────────────────
    print("\n[1/5] Extracción de frames")
    frames, total_orig, fps_orig = extract_frames(args.video, args.n_frames)

    if len(frames) == 0:
        print("❌ No se pudieron extraer frames del video.")
        sys.exit(1)

    # ── Paso 2: Detectar keypoints con YOLOv8l-pose ─────────────────
    print("\n[2/5] Detección de keypoints (YOLOv8l-pose)")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics no está instalado. Ejecuta: pip install ultralytics")
        sys.exit(1)

    yolo = YOLO("yolov8l-pose.pt")
    yolo.to(device)

    seq_raw = extract_keypoints(frames, yolo, verbose=args.verbose)
    # seq_raw : [T, 2, 17, 3] — coordenadas normalizadas

    # ── Paso 3: Canales de velocidad y aceleración ──────────────────
    print("\n[3/5] Añadiendo canales de velocidad y aceleración")
    seq_5ch = add_velocity_channels(seq_raw)       # [T, 2, 17, 5]
    print(f"  Shape final: {seq_5ch.shape}  (T={seq_5ch.shape[0]}, P=2, V=17, C=5)")

    # ── Paso 4: Cargar modelo MSG3D ─────────────────────────────────
    print("\n[4/5] Cargando modelo LSTM")
    model = load_model(args.checkpoint, device, args.model_arch, args.T)

    # ── Paso 5: Predicción ──────────────────────────────────────────
    print("\n[5/5] Clasificando secuencia")
    result = predict(model, seq_5ch, device, args.T)

    elapsed = time.time() - t0

    # ── Resultado ───────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  RESULTADO DE CLASIFICACIÓN")
    print("═"*60)

    label_display = result["class_label"].upper()
    conf_pct      = result["confidence"] * 100
    color_symbol  = "🔴" if result["class_id"] == 1 else "🟢"

    print(f"\n  {color_symbol}  Predicción : {label_display}")
    print(f"     Confianza  : {conf_pct:.2f}%")
    print(f"\n  Probabilidades por clase:")
    for i, (lbl, p) in enumerate(zip(LABELS, result["probs"])):
        bar = "█" * int(p * 30)
        marker = " ← pred" if i == result["class_id"] else ""
        print(f"    {lbl:15s}  [{bar:<30s}] {p*100:5.1f}%{marker}")

    print(f"\n  Logits brutos : {result['logits']}")
    print(f"  Tiempo total  : {elapsed:.2f}s")
    print("═"*60)

    # ── Diagnóstico de confianza ────────────────────────────────────
    print("\n  Diagnóstico:")
    if conf_pct >= 90:
        print("  ✅ Predicción muy segura (≥90%) — el modelo discrimina con claridad.")
    elif conf_pct >= 70:
        print("  ✅ Predicción confiable (≥70%).")
    elif conf_pct >= 55:
        print("  ⚠️  Predicción incierta (55-70%). El video puede estar en zona ambigua.")
        print("     Considera usar --save_video para revisar los esqueletos detectados.")
    else:
        print("  ❌ Predicción muy incierta (<55%). Posibles causas:")
        print("     • El video no contiene claramente 2 personas visibles.")
        print("     • La resolución o duración del video es muy baja.")
        print("     • El contenido es atípico para el dataset de entrenamiento.")
        print("     Usa --verbose para revisar las detecciones por frame.")

    # ── Guardar video anotado (opcional) ───────────────────────────
    if args.save_video:
        out_path = os.path.splitext(args.video)[0] + "_clasificado.mp4"
        print("\n  Generando video anotado...")
        # Para el video anotado usamos las coordenadas originales
        # (seq_raw tiene kps normalizados; el video anotado será esquemático)
        save_annotated_video(args.video, frames, seq_raw, result,
                             out_path, fps=fps_orig if fps_orig > 0 else 15.0)

    return result


if __name__ == "__main__":
    main()

