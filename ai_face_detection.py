#!/usr/bin/env python3
"""
PowerFace V6 - Optimized for low-end devices + high accuracy
Changes from V5:
  - buffalo_s model (smaller, faster) with fallback
  - Frame skipping to reduce CPU/GPU load
  - Lower detection resolution
  - Removed expensive bilateral filter
  - Optimized CLAHE (only in extreme dark)
  - Reduced camera resolution
  - Fewer registration samples needed
  - Smarter track cleanup
  - Kalman-like velocity smoothing for walking detection
"""

import cv2
import numpy as np
import os
import csv
import json
import pickle
import logging
import time
import threading
import queue
import traceback
from datetime import datetime
from collections import deque
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import insightface
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Limit CPU thread explosion to prevent CPU 100% core saturation
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
cv2.setNumThreads(4)

FACE_DB_FILE = "face_database_v6.pkl"
ATTENDANCE_FILE = "attendance_v6.csv"
REGISTERED_FILE = "registered_v6.json"

# ---- Performance & Hardware Load-Sharing Configuration ----
PERFORMANCE_MODE = False          # False = Max accuracy + GPU offloading
FRAME_SKIP = 1                   # 1 = detect every frame (GPU handles it smoothly)
DET_SIZE = (640, 640)            # 640x640 high resolution face detection
LOW_RES_CAM = (1280, 720)        # 720p HD webcam capture
INITIAL_PROCESS_SCALE = 0.65     # 65% scale for crisp primary detection pass
USE_BILATERAL_FILTER = False     # Bilateral filter toggle
CLAHE_LOW_LIGHT_ONLY = True      # Dynamic adaptive lighting enhancement

# ---- Recognition & Confidence Thresholds (Tuned for Motion & Walking) ----
THRESH_HIGH_CONF = 0.38
THRESH_MEDIUM_CONF = 0.32
THRESH_LOW_CONF = 0.24

# ---- Attendance thresholds (Motion-adaptive) ----
ATTENDANCE_THRESH_STILL = 0.50     # Standard still face
ATTENDANCE_THRESH_MOVING = 0.42    # Moving slightly
ATTENDANCE_THRESH_WALKING = 0.36   # Walking past camera
WALKING_VELOCITY_THRESHOLD = 18    # Pixels/frame to trigger walking mode
MOVING_VELOCITY_THRESHOLD = 8

MIN_QUALITY_FOR_RELAXED_ATTENDANCE = 0.22

# ---- Matching ----
MATCH_MIN_SCORE = 0.28
MATCH_MARGIN = 0.03
MATCH_TOPK = 5                   # Top-5 sample average for maximum identity accuracy
MIN_FACE_SIZE = 18
TRACK_TIMEOUT = 45               # Keep track memory active for 45 frames during movement & head turns

# ---- EMA tracking ----
EMA_ALPHA_BASE = 0.25
EMA_ALPHA_MAX = 0.65             # Faster adaptation for moving faces

NIGHT_BRIGHT_THRESHOLD = 60      # Auto low-light trigger
CLAHE_CLIP_LIMIT = 3.5

# ---- Registration ----
NUM_REGISTRATION_CAPTURES = 20   # Multi-angle guided capture
NUM_REGISTRATION_KEEP = 10       # Keep 10 highest-quality clean samples
MIN_REG_QUALITY = 0.30

CONFIRM_FRAMES = 2               # Faster confirmation for moving subjects
ATTENDANCE_COOLDOWN = 30
FLASH_DURATION = 20              # 20 frames glowing green blink animation

os.makedirs("registered_photos_v6", exist_ok=True)


def open_camera(preferred_width=640, preferred_height=480, preferred_index=None):
    """
    Robust camera initialization with DirectShow prioritization for Windows
    and automatic index fallback (0, 1, 2).
    """
    indices = [preferred_index] if preferred_index is not None else [0, 1, 2]
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]

    for idx in indices:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        logging.info(f"Camera successfully connected on index {idx} (backend: {backend})")
                        return cap
                    cap.release()
            except Exception as e:
                logging.debug(f"Failed opening camera index {idx} with backend {backend}: {e}")
    return None


def draw_translucent_rect(img, pt1, pt2, color, alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text_shadow(img, text, pos, font_scale=0.6, color=(255,255,255), thickness=2):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_apple_face_box(display, bbox, name, score, quality, velocity_mag, is_marked, is_lost, flash_remaining=0, needed_thresh=0.50):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = max(10, x2 - x1)
    h = max(10, y2 - y1)
    cl = min(24, max(12, w // 4), max(12, h // 4))

    is_flashing = flash_remaining > 0

    # Apple Cupertino Palette (BGR for OpenCV)
    if is_marked or is_flashing:
        base_color = (60, 220, 50)   # Apple Vibrant Neon Green
        status_text = f"✓ {name} ({score:.0%})"
    elif name != "Unknown":
        if score >= needed_thresh:
            base_color = (60, 220, 50)
            status_text = f"✓ {name} ({score:.0%})"
        else:
            base_color = (10, 180, 255)  # Apple Amber/Yellow: candidate
            status_text = f"{name} ({score:.0%} < {needed_thresh:.0%})"
    else:
        base_color = (50, 60, 255)   # Apple Red
        status_text = "Unknown"

    if is_lost:
        base_color = (160, 160, 165)
        status_text = f"{name} (tracking...)"

    # 1. Pulsating Stroboscopic / Glowing Green Blink when attendance is recorded
    if is_flashing:
        progress = flash_remaining / FLASH_DURATION
        blink_osc = (np.sin(flash_remaining * 0.9) + 1.0) * 0.5  # 0.0 to 1.0 oscillation
        pulse_alpha = float(np.clip(0.18 + 0.32 * blink_osc * progress, 0.0, 0.65))

        # Translucent full box highlight
        overlay = display.copy()
        cv2.rectangle(overlay, (max(0, x1 - 8), max(0, y1 - 8)), 
                               (min(display.shape[1] - 1, x2 + 8), min(display.shape[0] - 1, y2 + 8)), 
                               (60, 220, 50), -1)
        cv2.addWeighted(overlay, pulse_alpha, display, 1 - pulse_alpha, 0, display)

        # Pulsating glowing outer neon border
        outer_thick = 2 + int(3 * blink_osc)
        cv2.rectangle(display, (max(0, x1 - 4), max(0, y1 - 4)), 
                               (min(display.shape[1] - 1, x2 + 4), min(display.shape[0] - 1, y2 + 4)), 
                               (140, 255, 120), outer_thick)

    # 2. Main Face Box
    cv2.rectangle(display, (x1, y1), (x2, y2), base_color, 2)

    # 3. Apple FaceID Scanner Corner Brackets
    thick = 3 if not is_flashing else 4
    bracket_color = (140, 255, 120) if is_flashing else base_color
    cv2.line(display, (x1, y1), (x1 + cl, y1), bracket_color, thick)
    cv2.line(display, (x1, y1), (x1, y1 + cl), bracket_color, thick)
    cv2.line(display, (x2, y1), (x2 - cl, y1), bracket_color, thick)
    cv2.line(display, (x2, y1), (x2, y1 + cl), bracket_color, thick)
    cv2.line(display, (x1, y2), (x1 + cl, y2), bracket_color, thick)
    cv2.line(display, (x1, y2), (x1, y2 - cl), bracket_color, thick)
    cv2.line(display, (x2, y2), (x2 - cl, y2), bracket_color, thick)
    cv2.line(display, (x2, y2), (x2, y2 - cl), bracket_color, thick)

    # 4. Apple Status Tag Pill
    motion_tag = " • WALK" if velocity_mag > WALKING_VELOCITY_THRESHOLD else (" • MOVE" if velocity_mag > MOVING_VELOCITY_THRESHOLD else "")
    full_tag = f" {status_text}{motion_tag} "
    (tw, th), _ = cv2.getTextSize(full_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    tag_y = max(th + 10, y1)

    cv2.rectangle(display, (x1, tag_y - th - 8), (x1 + tw + 4, tag_y), base_color, -1)
    cv2.putText(display, full_tag, (x1 + 2, tag_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


# ========================= QUALITY SCORING =========================
def blur_score(gray_crop):
    if gray_crop.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
    return float(np.clip(lap / 400.0, 0.0, 1.0))

def pose_score(kps):
    if kps is None or len(kps) < 5:
        return 0.5
    le, re, nose, lm, rm = kps[:5]
    eye_dist = np.linalg.norm(re - le) + 1e-6
    eye_mid_x = (le[0] + re[0]) / 2.0
    yaw_ratio = abs(nose[0] - eye_mid_x) / eye_dist
    mouth_mid_y = (lm[1] + rm[1]) / 2.0
    eye_mid_y = (le[1] + re[1]) / 2.0
    vert_ratio = abs((mouth_mid_y - eye_mid_y) - eye_dist * 0.9) / eye_dist
    yaw_score = float(np.clip(1.0 - yaw_ratio / 0.55, 0.0, 1.0))
    pitch_score = float(np.clip(1.0 - vert_ratio / 0.9, 0.0, 1.0))
    return 0.5 * yaw_score + 0.5 * pitch_score

def size_score(bbox, min_good=120):
    w = bbox[2] - bbox[0]
    return float(np.clip(w / min_good, 0.0, 1.0))

def brightness_score(gray_crop):
    if gray_crop.size == 0:
        return 0.0
    mean = np.mean(gray_crop)
    return float(np.clip(1.0 - abs(mean - 130) / 130.0, 0.0, 1.0))

def face_quality_score(frame_gray, bbox, kps):
    x1, y1, x2, y2 = [max(0, v) for v in bbox]
    crop = frame_gray[y1:y2, x1:x2]
    b = blur_score(crop)
    p = pose_score(kps)
    s = size_score(bbox)
    br = brightness_score(crop)
    return 0.4 * b + 0.3 * p + 0.15 * s + 0.15 * br


# ========================= TRACK OBJECT (with Kalman-like smoothing) =========================
class Track:
    def __init__(self, tid, bbox, embedding, name, score, quality):
        self.tid = tid
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.velocity = (0.0, 0.0)
        self.velocity_mag = 0.0
        self.smooth_center = list(self.center)

        self.ema_embedding = embedding
        self.name = name
        self.stable_name = name
        self.score = score
        self.quality = quality
        self.quality_hist = deque([quality], maxlen=10)

        self.lost_frames = 0
        self.frames_seen = 1
        self.confirmed = False
        self.attendance_marked = False
        self.mark_time = None

    def predict(self):
        px = self.smooth_center[0] + self.velocity[0]
        py = self.smooth_center[1] + self.velocity[1]
        self.smooth_center = [px, py]
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = [int(px - w/2), int(py - h/2), int(px + w/2), int(py + h/2)]
        self.center = (px, py)

    def update(self, bbox, embedding, name, score, quality):
        old_center = self.center
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        vx = self.center[0] - old_center[0]
        vy = self.center[1] - old_center[1]
        self.velocity = (0.4 * self.velocity[0] + 0.6 * vx, 0.4 * self.velocity[1] + 0.6 * vy)
        self.velocity_mag = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

        # Kalman-like position smoothing
        self.smooth_center[0] = 0.7 * self.smooth_center[0] + 0.3 * self.center[0]
        self.smooth_center[1] = 0.7 * self.smooth_center[1] + 0.3 * self.center[1]

        self.quality = quality
        self.quality_hist.append(quality)

        if embedding is not None:
            alpha = EMA_ALPHA_BASE + (EMA_ALPHA_MAX - EMA_ALPHA_BASE) * quality
            self.ema_embedding = (1 - alpha) * self.ema_embedding + alpha * embedding
            norm = np.linalg.norm(self.ema_embedding)
            if norm > 0:
                self.ema_embedding = self.ema_embedding / norm

        self.name = name
        self.score = score
        self.lost_frames = 0
        self.frames_seen += 1
        if self.frames_seen >= CONFIRM_FRAMES:
            self.confirmed = True

    @property
    def avg_quality(self):
        return float(np.mean(self.quality_hist)) if self.quality_hist else 0.0


# ========================= MAIN SYSTEM =========================
class PowerFaceV5:
    def __init__(self, performance_mode=PERFORMANCE_MODE):
        self.performance_mode = performance_mode
        logging.info(f"Initializing PowerFace V6 (Performance Mode: {performance_mode})...")
        self.use_gpu = False

        det_size = (640, 640) if not performance_mode else DET_SIZE
        det_thresh = 0.38

        # Try GPU DirectML first, fallback to CPU
        models_to_try = ('buffalo_s', 'buffalo_l')
        gpu_ok = False

        for model_name in models_to_try:
            try:
                logging.info(f"Attempting GPU DirectML acceleration with model: {model_name}...")
                self.app = FaceAnalysis(
                    name=model_name,
                    allowed_modules=['detection', 'recognition'],
                    providers=['DmlExecutionProvider', 'CPUExecutionProvider']
                )
                self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                _ = self.app.get(dummy)
                self.use_gpu = True
                gpu_ok = True
                logging.info(f"⚡ GPU DirectML Active with model: {model_name}")
                break
            except Exception as e:
                err_str = str(e)
                logging.warning(f"GPU DirectML {model_name} failed ({err_str[:100]}).")

        if not gpu_ok:
            for model_name in models_to_try:
                try:
                    logging.info(f"Attempting CPU inference with model: {model_name}...")
                    self.app = FaceAnalysis(
                        name=model_name,
                        allowed_modules=['detection', 'recognition'],
                        providers=['CPUExecutionProvider']
                    )
                    self.app.prepare(ctx_id=-1, det_size=det_size, det_thresh=det_thresh)
                    self.use_gpu = False
                    logging.info(f"💻 CPU Execution Provider Active with model: {model_name}")
                    break
                except Exception as e:
                    logging.warning(f"CPU {model_name} failed ({e}).")

        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

        self.known_faces = {}
        self.db_matrix = None
        self.db_names = []
        self.load_database()

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.running = False
        self.detection_thread = None

        self.tracks = {}
        self.next_track_id = 0
        self.flash_effects = {}
        self.frame_count = 0
        self.current_scale = INITIAL_PROCESS_SCALE

    def load_database(self):
        # Compatibility bridge for pickle files created across numpy versions
        try:
            import sys
            import numpy.core
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.numeric'] = numpy.core.numeric
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        except Exception:
            pass

        loaded = False
        for db_file in [FACE_DB_FILE, "face_database_v5.pkl", "face_database_v4.pkl"]:
            if os.path.exists(db_file):
                try:
                    with open(db_file, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, dict) and data:
                            self.known_faces = data
                            self._rebuild_match_cache()
                            logging.info(f"Database Loaded from '{db_file}': {len(self.known_faces)} identities ({sum(len(v) for v in self.known_faces.values())} sample embeddings).")
                            loaded = True
                            break
                except Exception as e:
                    logging.warning(f"Could not load {db_file}: {e}")
        if not loaded:
            self.known_faces = {}
            self.db_matrix = None
            self.db_names = []

    def save_database(self):
        with open(FACE_DB_FILE, 'wb') as f:
            pickle.dump(self.known_faces, f)
        self._rebuild_match_cache()

    def _rebuild_match_cache(self):
        if not self.known_faces:
            self.db_matrix = None
            self.db_names = []
            return
        matrices = []
        names = []
        for name, samples in self.known_faces.items():
            matrices.append(np.array(samples, dtype=np.float32))
            names.extend([name] * len(samples))
        if matrices:
            self.db_matrix = np.vstack(matrices)
            norms = np.linalg.norm(self.db_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.db_matrix = self.db_matrix / norms
            self.db_names = names

    # ==================== BATCH MATCHING ====================
    def batch_match(self, embeddings_list):
        if not embeddings_list or self.db_matrix is None:
            return [("Unknown", 0.0)] * len(embeddings_list)

        emb_matrix = np.array(embeddings_list, dtype=np.float32)
        emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1.0
        emb_matrix = emb_matrix / emb_norms

        sims = emb_matrix @ self.db_matrix.T
        names_arr = np.array(self.db_names)
        unique_names = sorted(set(self.db_names))

        results = []
        for i in range(len(embeddings_list)):
            row_sims = sims[i]
            per_identity = []
            for name in unique_names:
                idxs = np.where(names_arr == name)[0]
                if len(idxs) == 0:
                    continue
                s = row_sims[idxs]
                k = min(MATCH_TOPK, len(s))
                top_vals = np.partition(s, -k)[-k:]
                per_identity.append((name, float(np.mean(top_vals))))

            per_identity.sort(key=lambda x: x[1], reverse=True)
            best_name, best_score = per_identity[0]
            second_score = per_identity[1][1] if len(per_identity) > 1 else -1.0

            if best_score < MATCH_MIN_SCORE or (best_score - second_score) < MATCH_MARGIN:
                results.append(("Unknown", best_score))
            else:
                results.append((best_name, best_score))
        return results

    # ==================== HUNGARIAN TRACKING ====================
    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-8)

    def update_tracks(self, detections):
        for tid, trk in self.tracks.items():
            if trk.lost_frames > 0:
                trk.predict()

        active_tids = [tid for tid, trk in self.tracks.items() if trk.lost_frames < TRACK_TIMEOUT]
        active_tracks = [self.tracks[tid] for tid in active_tids]
        new_tracks = {}

        if not active_tracks:
            for det in detections:
                self.next_track_id += 1
                new_tracks[self.next_track_id] = Track(
                    self.next_track_id, det['bbox'], det['embedding'], det['name'],
                    det['score'], det['quality'])
        elif not detections:
            for trk in active_tracks:
                trk.lost_frames += 1
                if trk.lost_frames < TRACK_TIMEOUT:
                    new_tracks[trk.tid] = trk
        else:
            num_tracks = len(active_tracks)
            num_dets = len(detections)
            # Only build cost matrix if both sides have entries
            if num_tracks > 0 and num_dets > 0:
                cost_matrix = np.zeros((num_tracks, num_dets))
                for i, trk in enumerate(active_tracks):
                    tc = trk.smooth_center if hasattr(trk, 'smooth_center') else trk.center
                    for j, det in enumerate(detections):
                        dc = det.get('smooth_center', ((det['bbox'][0] + det['bbox'][2]) / 2.0,
                                                        (det['bbox'][1] + det['bbox'][3]) / 2.0))
                        emb_dist = 1.0 - np.dot(trk.ema_embedding, det['embedding'] /
                                                 (np.linalg.norm(det['embedding']) + 1e-8))
                        iou = self._iou(trk.bbox, det['bbox'])
                        spatial_cost = 1.0 - iou
                        name_penalty = 0 if (det['name'] == "Unknown" or
                                            trk.stable_name == "Unknown" or
                                            det['name'] == trk.stable_name) else 1.0
                        cost_matrix[i, j] = (0.5 * emb_dist) + (0.3 * spatial_cost) + 0.2 * name_penalty

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                assigned_tracks = set()
                assigned_dets = set()

                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1.2:
                        trk = active_tracks[r]
                        det = detections[c]
                        if det['score'] > 0.35 and det['name'] != "Unknown":
                            trk.stable_name = det['name']
                        elif trk.stable_name == "Unknown" and det['name'] != "Unknown":
                            trk.stable_name = det['name']
                        trk.update(det['bbox'], det['embedding'], det['name'], det['score'], det['quality'])
                        new_tracks[trk.tid] = trk
                        assigned_tracks.add(r)
                        assigned_dets.add(c)

                for i, trk in enumerate(active_tracks):
                    if i not in assigned_tracks:
                        trk.lost_frames += 1
                        if trk.lost_frames < TRACK_TIMEOUT:
                            new_tracks[trk.tid] = trk

                for j, det in enumerate(detections):
                    if j not in assigned_dets:
                        self.next_track_id += 1
                        new_tracks[self.next_track_id] = Track(
                            self.next_track_id, det['bbox'], det['embedding'], det['name'],
                            det['score'], det['quality'])

        self.tracks = new_tracks

    # ==================== PREPROCESSING (optimized) ====================
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < NIGHT_BRIGHT_THRESHOLD:
            if CLAHE_LOW_LIGHT_ONLY or not self.performance_mode:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = self.clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        if USE_BILATERAL_FILTER and self.current_scale > 0.5:
            frame = cv2.bilateralFilter(frame, 5, 50, 50)
        return frame

    # ==================== REGISTRATION ====================
    def register_person(self):
        name = input("Enter full name: ").strip()
        if not name:
            print("Name cannot be empty!")
            return
        if name in self.known_faces:
            print(f"'{name}' already exists with {len(self.known_faces[name])} samples.")
            if input("Add more samples? (y/n): ").lower() != 'y':
                return
        self._register_capture_loop(name)

    def register_person_gui(self, name):
        self._register_capture_loop(name)

    def _register_capture_loop(self, name):
        cap = open_camera(640, 480)
        if cap is None:
            print("\n[ERROR] Could not access webcam! Please verify camera connection and Windows privacy permissions.\n")
            return

        prompts = ["FRONT", "Left", "Right", "Up", "Down",
                   "Smile", "Serious", "Tilt", "Bad Light", "Neutral"]
        candidates = []
        photo_dir = f"registered_photos_v6/{name}"
        os.makedirs(photo_dir, exist_ok=True)

        print(f"\nRegistering: {name} | SPACE=capture | ESC=Finish\n")
        print(f"Target: {NUM_REGISTRATION_CAPTURES} captures, keep best {NUM_REGISTRATION_KEEP}\n")

        while len(candidates) < NUM_REGISTRATION_CAPTURES:
            ret, frame = cap.read()
            if not ret:
                continue
            proc_frame = self.preprocess_frame(frame)
            faces = self.app.get(proc_frame)

            display = frame.copy()
            gray_full = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            live_quality = None
            for f in faces:
                b = f.bbox.astype(int)
                q = face_quality_score(gray_full, b, getattr(f, 'kps', None))
                live_quality = q
                qcolor = (0, 255, 0) if q >= MIN_REG_QUALITY else (0, 140, 255)
                cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), qcolor, 2)
                put_text_shadow(display, f"Q:{q:.2f}", (b[0], b[1]-8), 0.5, qcolor)

            idx = min(len(candidates), len(prompts)-1)
            draw_translucent_rect(display, (0, 0), (420, 90), (30, 30, 30))
            put_text_shadow(display, f"{len(candidates)}/{NUM_REGISTRATION_CAPTURES}: {prompts[idx]}",
                            (15, 30), 0.6, (0, 255, 255))
            if live_quality is not None:
                put_text_shadow(display, f"Quality: {live_quality:.2f}", (15, 60), 0.5,
                                (0, 255, 0) if live_quality >= MIN_REG_QUALITY else (0, 140, 255))

            cv2.imshow('Register - SPACE=Capture  ESC=Finish', display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(' ') and len(faces) == 1:
                f = faces[0]
                b = f.bbox.astype(int)
                w = b[2] - b[0]
                q = face_quality_score(gray_full, b, getattr(f, 'kps', None))
                if w >= MIN_FACE_SIZE:
                    if q < MIN_REG_QUALITY:
                        print(f"  x Rejected (quality {q:.2f})")
                    else:
                        candidates.append((q, f.embedding, frame.copy(), b))
                        print(f"  + Captured (quality {q:.2f}, {w}px)")
                else:
                    print(f"  x Rejected (face too small: {w}px)")

        cap.release()
        cv2.destroyAllWindows()

        if not candidates:
            print("\nNo samples captured.")
            return

        candidates.sort(key=lambda c: c[0], reverse=True)
        keep = candidates[:NUM_REGISTRATION_KEEP]
        embeddings = [c[1] for c in keep]
        for i, (q, emb, frm, b) in enumerate(keep):
            cv2.imwrite(f"{photo_dir}/{i+1:02d}_q{q:.2f}.jpg", frm)

        if name not in self.known_faces:
            self.known_faces[name] = []
        self.known_faces[name].extend(embeddings)
        self.save_database()
        avg_q = np.mean([c[0] for c in keep])
        print(f"\nSUCCESS: {name} registered with {len(keep)} samples "
              f"(avg quality {avg_q:.2f}, discarded {len(candidates)-len(keep)}).\n")

    # ==================== LIST & DELETE ====================
    def list_registered(self):
        if not self.known_faces:
            print("\nNo people registered yet!")
            return
        print(f"\n{'='*40}")
        print("  REGISTERED PEOPLE")
        print(f"{'='*40}")
        for name, samples in sorted(self.known_faces.items()):
            print(f"  - {name} ({len(samples)} samples)")
        print(f"{'='*40}\n")

    def delete_person(self):
        if not self.known_faces:
            print("\nNo people registered to delete!")
            return
        print("\n--- Registered People ---")
        for name in sorted(self.known_faces.keys()):
            print(f"  - {name}")
        print("-------------------------")
        name = input("Enter the EXACT name to delete: ").strip()
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_database()
            print(f"Successfully deleted: {name}")
        else:
            print("Name not found! Check spelling.")

    # ==================== DETECTION LOOP (with adaptive frame skip) ====================
    def detection_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                t0 = time.time()
                proc_frame = self.preprocess_frame(frame)

                scale = self.current_scale
                small = cv2.resize(proc_frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else proc_frame
                faces = self.app.get(small)
                inv_scale = 1.0 / scale if scale != 1.0 else 1.0

                # Adaptive re-detection at full resolution if first pass found nothing
                if len(faces) == 0 and scale < 1.0:
                    faces = self.app.get(proc_frame)
                    inv_scale = 1.0

                gray_full = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

                embeddings_to_match = []
                dets = []
                for f in faces:
                    bbox = f.bbox.astype(int)
                    if inv_scale != 1.0:
                        bbox = [int(v * inv_scale) for v in bbox]
                    w = bbox[2] - bbox[0]
                    if w < MIN_FACE_SIZE:
                        continue
                    q = face_quality_score(gray_full, bbox, getattr(f, 'kps', None))
                    dets.append({'bbox': bbox, 'embedding': f.embedding,
                                 'face_width': w, 'quality': q})
                    embeddings_to_match.append(f.embedding)

                batch_results = self.batch_match(embeddings_to_match)
                for i, det in enumerate(dets):
                    det['name'] = batch_results[i][0]
                    det['score'] = batch_results[i][1]

                dets.sort(key=lambda x: x['score'], reverse=True)
                keep = []
                for d in dets:
                    if all(self._iou(d['bbox'], k['bbox']) <= 0.5 for k in keep):
                        keep.append(d)

                self.update_tracks(keep)
                detect_ms = (time.time() - t0) * 1000

                # Adaptive scale based on crowd
                if len(keep) > 30:
                    self.current_scale = 0.30
                elif len(keep) > 15:
                    self.current_scale = 0.40
                else:
                    self.current_scale = INITIAL_PROCESS_SCALE

                try:
                    while not self.result_queue.empty():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait({
                        'tracks': self.tracks,
                        'detect_ms': detect_ms,
                        'num_faces': len(keep),
                        'night_mode': np.mean(gray_full) < NIGHT_BRIGHT_THRESHOLD
                    })
                except queue.Full:
                    pass

            except Exception as e:
                logging.error(f"Error in detection loop: {e}")
                logging.error(traceback.format_exc())

    # ==================== TAKE ATTENDANCE ====================
    def take_attendance(self, subject="General"):
        if not self.known_faces:
            print("=" * 50)
            print("WARNING: No registered people found!")
            print("The camera will open, but all faces will show as 'Unknown'.")
            print("Please close the camera and select Option 1 to register.")
            print("=" * 50)
            time.sleep(2)

        cap = open_camera(640, 480)
        if cap is None:
            print("\n[ERROR] Could not open webcam! Please verify camera connection and Windows privacy permissions.\n")
            return
        cv2.namedWindow('PowerFace V6', cv2.WINDOW_AUTOSIZE)

        self.running = True
        self.tracks = {}
        self.flash_effects = {}
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

        marked = {}
        attendance_log = []
        today = datetime.now().strftime("%Y-%m-%d")
        frame_times = deque(maxlen=30)
        display_fps = 0.0
        detect_fps = 0.0
        last_info = {'num_faces': 0, 'night_mode': False, 'detect_ms': 0, 'tracks': {}}
        frame_counter = 0

        print(f"\n=== Attendance: {subject} ===\nQ=Quit\n")

        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_count += 1
            frame_counter += 1

            # FRAME SKIP: only send every Nth frame to the detection thread
            if frame_counter % FRAME_SKIP == 0 or self.frame_count < 5:
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            try:
                last_info = self.result_queue.get_nowait()
                detect_fps = 1000.0 / max(last_info['detect_ms'], 1)
            except queue.Empty:
                pass

            tracks = last_info.get('tracks', {})
            is_night = last_info.get('night_mode', False)
            display = frame.copy()
            current_attendance_thresh = ATTENDANCE_THRESH_STILL

            for tid, trk in tracks.items():
                if trk.lost_frames > 0 and trk.frames_seen < 2:
                    continue

                x1, y1, x2, y2 = trk.bbox
                score = trk.score
                stable_name = trk.stable_name
                is_lost = trk.lost_frames > 0
                velocity_mag = trk.velocity_mag
                quality = trk.avg_quality

                if velocity_mag > WALKING_VELOCITY_THRESHOLD:
                    threshold = THRESH_LOW_CONF
                elif velocity_mag > MOVING_VELOCITY_THRESHOLD:
                    threshold = THRESH_MEDIUM_CONF
                else:
                    threshold = THRESH_HIGH_CONF

                quality_ok_for_relax = quality >= MIN_QUALITY_FOR_RELAXED_ATTENDANCE
                if velocity_mag > WALKING_VELOCITY_THRESHOLD and quality_ok_for_relax:
                    current_attendance_thresh = ATTENDANCE_THRESH_WALKING
                elif velocity_mag > MOVING_VELOCITY_THRESHOLD and quality_ok_for_relax:
                    current_attendance_thresh = ATTENDANCE_THRESH_MOVING
                else:
                    current_attendance_thresh = ATTENDANCE_THRESH_STILL

                is_marked = trk.attendance_marked
                if recognized and score >= current_attendance_thresh:
                    now = time.time()
                    if trk.confirmed and not trk.attendance_marked and not is_lost:
                        if stable_name not in marked or (now - marked[stable_name]) > ATTENDANCE_COOLDOWN:
                            marked[stable_name] = now
                            trk.attendance_marked = True
                            is_marked = True
                            trk.mark_time = now
                            self.flash_effects[tid] = FLASH_DURATION
                            ts = datetime.now().strftime("%H:%M:%S")
                            attendance_log.append({
                                'date': today, 'time': ts, 'name': stable_name,
                                'subject': subject, 'confidence': round(score, 4),
                                'quality': round(quality, 3)
                            })
                            logging.info(f"MARKED: {stable_name} at {score:.0%} "
                                         f"(Vel: {velocity_mag:.0f}, Qual: {quality:.2f})")

                flash_rem = self.flash_effects.get(tid, 0)
                draw_apple_face_box(
                    display=display,
                    bbox=trk.bbox,
                    name=stable_name,
                    score=score,
                    quality=quality,
                    velocity_mag=velocity_mag,
                    is_marked=is_marked,
                    is_lost=is_lost,
                    flash_remaining=flash_rem,
                    needed_thresh=current_attendance_thresh
                )

                if flash_rem > 0:
                    self.flash_effects[tid] -= 1

            self.flash_effects = {k: v for k, v in self.flash_effects.items() if v > 0}
            frame_times.append(time.time() - t_start)
            if sum(frame_times) > 0:
                display_fps = len(frame_times) / sum(frame_times)

            # Minimal HUD
            draw_translucent_rect(display, (0, 0), (430, 80), (30, 30, 30), 0.8)
            hw_str = "GPU (DirectML)" if self.use_gpu else "CPU Mode"
            put_text_shadow(display, f"FPS:{display_fps:.0f} Det:{detect_fps:.0f} | {hw_str}",
                            (10, 25), 0.5, (0, 255, 200) if self.use_gpu else (200, 200, 200))
            put_text_shadow(display, f"Thresh: {current_attendance_thresh:.0%} | {'NIGHT' if is_night else 'DAY'} | Crowd:{last_info['num_faces']}",
                            (10, 50), 0.45, (0, 150, 255) if is_night else (50, 205, 50))

            if marked:
                rx = display.shape[1] - 220
                draw_translucent_rect(display, (rx, 0), (display.shape[1], len(marked) * 24 + 35),
                                      (30, 30, 30), 0.85)
                put_text_shadow(display, f"PRESENT:{len(marked)}", (rx + 10, 22), 0.55, (50, 205, 50))
                for i, mname in enumerate(sorted(marked.keys())):
                    put_text_shadow(display, f"  {mname}", (rx + 15, 50 + i * 24), 0.45, (255, 255, 255))

            cv2.imshow('PowerFace V6', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        self.save_attendance(attendance_log)

    # ==================== SAVE & VIEW ====================
    def save_attendance(self, records):
        if not records:
            return
        exists = os.path.exists(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['date', 'time', 'name', 'subject', 'confidence', 'quality'])
            if not exists:
                w.writeheader()
            for r in records:
                w.writerow(r)

    def view_attendance(self):
        if not os.path.exists(ATTENDANCE_FILE):
            print("No attendance records yet!")
            return
        print(f"\n{'DATE':<12}{'TIME':<10}{'NAME':<20}{'CONF':<8}{'QUAL':<6}")
        print("-" * 56)
        with open(ATTENDANCE_FILE, 'r') as f:
            for row in csv.DictReader(f):
                print(f"{row['date']:<12}{row['time']:<10}{row['name']:<20}"
                      f"{row['confidence']:<8}{row.get('quality', ''):<6}")


# ========================= MAIN MENU =========================
def main():
    system = PowerFaceV5()
    while True:
        print("\n" + "=" * 50)
        print("  POWER FACE V6 - LOW-END OPTIMIZED")
        print("=" * 50)
        print("1. Register Person")
        print("2. Take Attendance")
        print("3. View Registered People")
        print("4. Delete Person")
        print("5. View Attendance Records")
        print("6. Exit")
        print("-" * 50)

        c = input("Select (1-6): ").strip()

        if c == '1':
            system.register_person()
        elif c == '2':
            sub = input("Subject Name (Enter=General): ").strip() or "General"
            system.take_attendance(sub)
        elif c == '3':
            system.list_registered()
        elif c == '4':
            system.delete_person()
        elif c == '5':
            system.view_attendance()
        elif c == '6':
            print("Exiting System...")
            break
        else:
            print("Invalid choice! Select 1-6.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 50)
        print("CRITICAL STARTUP ERROR!")
        print("!" * 50)
        traceback.print_exc()
        input("Press Enter to exit...")
