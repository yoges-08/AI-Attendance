#!/usr/bin/env python3
"""
PowerFace V4 - WALKING OPTIMIZED ENGINE
Updates: Motion-Adaptive Thresholds (60% still / 40% walking) | Fast EMA | Hyper-Sensitive Detection
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
import insightface
from insightface.app import FaceAnalysis

# ========================= LOGGING & CONFIG =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FACE_DB_FILE = "face_database_v4.pkl"
ATTENDANCE_FILE = "attendance_v4.csv"
REGISTERED_FILE = "registered_v4.json"

THRESH_HIGH_CONF = 0.40
THRESH_MEDIUM_CONF = 0.35
THRESH_LOW_CONF = 0.28

# ATTENDANCE THRESHOLDS (Adaptive based on movement)
ATTENDANCE_THRESH_STILL = 0.60   # 60% required when standing still
ATTENDANCE_THRESH_WALKING = 0.40 # 40% required when walking (blur compensation)
WALKING_VELOCITY_THRESHOLD = 30  # Pixels per frame to be considered "walking"

MIN_FACE_SIZE = 20
TRACK_TIMEOUT = 40
EMA_ALPHA_STILL = 0.3
EMA_ALPHA_WALKING = 0.6 # Update facial memory faster when moving

INITIAL_PROCESS_SCALE = 0.6
DISPLAY_SCALE = 1.0

NIGHT_BRIGHT_THRESHOLD = 70
CLAHE_CLIP_LIMIT = 4.0

NUM_REGISTRATION_SAMPLES = 15
CONFIRM_FRAMES = 3
ATTENDANCE_COOLDOWN = 30
FLASH_DURATION = 15

os.makedirs("registered_photos_v4", exist_ok=True)

# ========================= UI HELPERS =========================
def draw_translucent_rect(img, pt1, pt2, color, alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text_shadow(img, text, pos, font_scale=0.6, color=(255,255,255), thickness=2):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# ========================= TRACK OBJECT =========================
class Track:
    def __init__(self, tid, bbox, embedding, name, score):
        self.tid = tid
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.velocity = (0.0, 0.0)
        self.velocity_mag = 0.0
        
        self.ema_embedding = embedding
        self.name = name
        self.stable_name = name
        self.score = score
        
        self.lost_frames = 0
        self.frames_seen = 1
        self.confirmed = False
        self.attendance_marked = False
        self.mark_time = None

    def predict(self):
        px = self.center[0] + self.velocity[0]
        py = self.center[1] + self.velocity[1]
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = [int(px - w/2), int(py - h/2), int(px + w/2), int(py + h/2)]
        self.center = (px, py)

    def update(self, bbox, embedding, name, score):
        old_center = self.center
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        
        vx = self.center[0] - old_center[0]
        vy = self.center[1] - old_center[1]
        self.velocity = (0.5 * self.velocity[0] + 0.5 * vx, 0.5 * self.velocity[1] + 0.5 * vy)
        self.velocity_mag = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        
        if embedding is not None:
            # WALKING FIX: Update memory faster when moving to adapt to blur/angles
            alpha = EMA_ALPHA_WALKING if self.velocity_mag > WALKING_VELOCITY_THRESHOLD else EMA_ALPHA_STILL
            self.ema_embedding = (1 - alpha) * self.ema_embedding + alpha * embedding
            norm = np.linalg.norm(self.ema_embedding)
            if norm > 0: self.ema_embedding = self.ema_embedding / norm
            
        self.name = name
        self.score = score
        self.lost_frames = 0
        self.frames_seen += 1
        if self.frames_seen >= CONFIRM_FRAMES:
            self.confirmed = True

# ========================= MAIN SYSTEM =========================
class PowerFaceV4:
    def __init__(self):
        logging.info("Initializing PowerFace V4 Walking Optimized Engine...")
        self.use_gpu = False
        
        try:
            logging.info("Attempting Intel Iris Xe GPU Activation...")
            self.app = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
            # WALKING FIX: Lower det_thresh to 0.35 to catch blurry moving faces
            self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.app.get(dummy)
            self.use_gpu = True
            logging.info("GPU Activated Successfully.")
        except Exception as e:
            logging.warning(f"GPU Failed ({e}). Falling back to CPU.")
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.35)

        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        
        self.known_faces = {}
        self.db_matrix = None
        self.db_names = []
        self.load_database()

        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        self.detection_thread = None

        self.tracks = {}
        self.next_track_id = 0
        self.flash_effects = {}
        self.frame_count = 0
        self.current_scale = INITIAL_PROCESS_SCALE

    def load_database(self):
        if os.path.exists(FACE_DB_FILE):
            with open(FACE_DB_FILE, 'rb') as f: self.known_faces = pickle.load(f)
            self._rebuild_match_cache()
            logging.info(f"Database Loaded: {len(self.known_faces)} identities.")

    def save_database(self):
        with open(FACE_DB_FILE, 'wb') as f: pickle.dump(self.known_faces, f)
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

    # ==================== VECTORIZED BATCH MATCHING ====================
    def batch_match(self, embeddings_list):
        if not embeddings_list or self.db_matrix is None:
            return [("Unknown", 0.0)] * len(embeddings_list)

        emb_matrix = np.array(embeddings_list, dtype=np.float32)
        emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1.0
        emb_matrix = emb_matrix / emb_norms

        sims = emb_matrix @ self.db_matrix.T
        
        results = []
        for i in range(len(embeddings_list)):
            row_sims = sims[i]
            best_idx = np.argmax(row_sims)
            best_score = row_sims[best_idx]
            best_name = self.db_names[best_idx]
            
            if best_score < 0.25:
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
            if trk.lost_frames > 0: trk.predict()

        active_tids = [tid for tid, trk in self.tracks.items() if trk.lost_frames < TRACK_TIMEOUT]
        active_tracks = [self.tracks[tid] for tid in active_tids]
        new_tracks = {}

        if not active_tracks:
            for det in detections:
                self.next_track_id += 1
                new_tracks[self.next_track_id] = Track(self.next_track_id, det['bbox'], det['embedding'], det['name'], det['score'])
        elif not detections:
            for trk in active_tracks:
                trk.lost_frames += 1
                if trk.lost_frames < TRACK_TIMEOUT: new_tracks[trk.tid] = trk
        else:
            num_tracks = len(active_tracks)
            num_dets = len(detections)
            cost_matrix = np.zeros((num_tracks, num_dets))

            for i, trk in enumerate(active_tracks):
                for j, det in enumerate(detections):
                    emb_dist = 1.0 - np.dot(trk.ema_embedding, det['embedding'] / (np.linalg.norm(det['embedding']) + 1e-8))
                    iou = self._iou(trk.bbox, det['bbox'])
                    spatial_cost = 1.0 - iou
                    name_penalty = 0 if (det['name'] == "Unknown" or trk.stable_name == "Unknown" or det['name'] == trk.stable_name) else 1.0
                    cost_matrix[i, j] = (0.6 * emb_dist) + (0.3 * spatial_cost) + name_penalty

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1.2: 
                    trk = active_tracks[r]
                    det = detections[c]
                    if det['score'] > 0.35 and det['name'] != "Unknown": trk.stable_name = det['name']
                    elif trk.stable_name == "Unknown" and det['name'] != "Unknown": trk.stable_name = det['name']
                    trk.update(det['bbox'], det['embedding'], det['name'], det['score'])
                    new_tracks[trk.tid] = trk
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            for i, trk in enumerate(active_tracks):
                if i not in assigned_tracks:
                    trk.lost_frames += 1
                    if trk.lost_frames < TRACK_TIMEOUT: new_tracks[trk.tid] = trk

            for j, det in enumerate(detections):
                if j not in assigned_dets:
                    self.next_track_id += 1
                    new_tracks[self.next_track_id] = Track(self.next_track_id, det['bbox'], det['embedding'], det['name'], det['score'])

        self.tracks = new_tracks

    # ==================== PREPROCESSING ====================
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < NIGHT_BRIGHT_THRESHOLD:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        if self.current_scale > 0.5:
            frame = cv2.bilateralFilter(frame, 5, 50, 50)
        return frame

    # ==================== REGISTRATION ====================
    def register_person(self):
        name = input("Enter full name: ").strip()
        if not name: print("Name cannot be empty!"); return
        if name in self.known_faces:
            print(f"'{name}' already exists with {len(self.known_faces[name])} samples.")
            if input("Add more samples? (y/n): ").lower() != 'y': return

        cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        prompts = ["FRONT Close", "FRONT Med", "FRONT Far", "Left", "Right", "Up", "Down", "Glasses On", "Glasses Off", "Smile", "Serious", "Squint", "Mouth Open", "Tilt", "Bad Light"]
        embeddings = []
        photo_dir = f"registered_photos_v4/{name}"; os.makedirs(photo_dir, exist_ok=True)
        print(f"\nRegistering: {name} | SPACE=capture | Q=Quit\n")

        while len(embeddings) < NUM_REGISTRATION_SAMPLES:
            ret, frame = cap.read(); 
            if not ret: continue
            proc_frame = self.preprocess_frame(frame)
            faces = self.app.get(proc_frame)
            
            display = frame.copy()
            for f in faces: 
                b = f.bbox.astype(int)
                cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), (255,255,255), 2)

            idx = min(len(embeddings), len(prompts)-1)
            draw_translucent_rect(display, (0, 0), (500, 100), (30,30,30))
            put_text_shadow(display, f"{len(embeddings)}/{NUM_REGISTRATION_SAMPLES}: {prompts[idx]}", (15, 35), 0.7, (0,255,255))
            
            cv2.imshow('Register - Q to Quit', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and len(faces) == 1:
                f = faces[0]; b = f.bbox.astype(int); w = b[2]-b[0]
                if w >= MIN_FACE_SIZE:
                    embeddings.append(f.embedding)
                    cv2.imwrite(f"{photo_dir}/{len(embeddings):02d}.jpg", frame)
                    print(f"  + Captured {w}px")
            elif key == ord('q'): break

        cap.release(); cv2.destroyAllWindows()
        if embeddings:
            if name not in self.known_faces: self.known_faces[name] = []
            self.known_faces[name].extend(embeddings)
            self.save_database()
            print(f"\nSUCCESS: {name} registered!\n")

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

    # ==================== DETECTION LOOP ====================
    def detection_loop(self):
        while self.running:
            try: frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty: continue

            try:
                t0 = time.time()
                proc_frame = self.preprocess_frame(frame)
                
                scale = self.current_scale
                small = cv2.resize(proc_frame, (0,0), fx=scale, fy=scale) if scale != 1.0 else proc_frame
                faces = self.app.get(small)
                inv_scale = 1.0 / scale if scale != 1.0 else 1.0

                embeddings_to_match = []
                dets = []
                for f in faces:
                    bbox = f.bbox.astype(int)
                    if inv_scale != 1.0: bbox = [int(v*inv_scale) for v in bbox]
                    w = bbox[2]-bbox[0]
                    if w < MIN_FACE_SIZE: continue
                    dets.append({'bbox': bbox, 'embedding': f.embedding, 'face_width': w})
                    embeddings_to_match.append(f.embedding)

                batch_results = self.batch_match(embeddings_to_match)
                for i, det in enumerate(dets):
                    det['name'] = batch_results[i][0]
                    det['score'] = batch_results[i][1]

                dets.sort(key=lambda x: x['score'], reverse=True)
                keep = []
                for d in dets:
                    if all(self._iou(d['bbox'], k['bbox']) <= 0.5 for k in keep): keep.append(d)
                
                self.update_tracks(keep)
                detect_ms = (time.time()-t0)*1000

                if len(keep) > 35: self.current_scale = 0.35
                elif len(keep) > 20: self.current_scale = 0.5
                else: self.current_scale = INITIAL_PROCESS_SCALE

                try:
                    while not self.result_queue.empty(): self.result_queue.get_nowait()
                    self.result_queue.put_nowait({'tracks': self.tracks, 'detect_ms': detect_ms, 'num_faces': len(keep), 'night_mode': np.mean(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)) < NIGHT_BRIGHT_THRESHOLD})
                except queue.Full: pass

            except Exception as e:
                logging.error(f"Error in detection loop: {e}")
                logging.error(traceback.format_exc())

    # ==================== TAKE ATTENDANCE ====================
    def take_attendance(self, subject="General"):
        if not self.known_faces:
            print("="*50)
            print("WARNING: No registered people found!")
            print("The camera will open, but all faces will show as 'Unknown'.")
            print("Please close the camera and select Option 1 to register.")
            print("="*50)
            time.sleep(2)

        cap = cv2.VideoCapture(0) 
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cv2.namedWindow('PowerFace V4 Enterprise', cv2.WINDOW_NORMAL)

        self.running = True; self.tracks = {}; self.flash_effects = {}
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

        marked = {}; attendance_log = []; today = datetime.now().strftime("%Y-%m-%d")
        frame_times = deque(maxlen=60); display_fps = 0.0; detect_fps = 0.0
        last_info = {'num_faces': 0, 'night_mode': False, 'detect_ms': 0, 'tracks': {}}

        print(f"\n=== Attendance: {subject} ===\nQ=Quit\n")

        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret: continue

            self.frame_count += 1
            try:
                if self.frame_queue.full(): self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Full: pass

            try: 
                last_info = self.result_queue.get_nowait()
                detect_fps = 1000.0/max(last_info['detect_ms'],1)
            except queue.Empty: pass

            tracks = last_info.get('tracks', {})
            is_night = last_info.get('night_mode', False)
            display = frame.copy()

            for tid, trk in tracks.items():
                if trk.lost_frames > 0 and trk.frames_seen < 2: continue
                
                x1, y1, x2, y2 = trk.bbox
                score = trk.score
                stable_name = trk.stable_name
                is_lost = trk.lost_frames > 0
                velocity_mag = trk.velocity_mag

                # WALKING FIX: Adaptive Thresholds based on Movement
                if velocity_mag > WALKING_VELOCITY_THRESHOLD:
                    threshold = THRESH_LOW_CONF
                    current_attendance_thresh = ATTENDANCE_THRESH_WALKING # 40% allowed while walking
                elif velocity_mag > 15:
                    threshold = THRESH_MEDIUM_CONF
                    current_attendance_thresh = 0.50 # 50% if moving slightly
                else:
                    threshold = THRESH_HIGH_CONF
                    current_attendance_thresh = ATTENDANCE_THRESH_STILL # 60% required if standing still

                recognized = stable_name != "Unknown" and score > threshold

                if recognized:
                    # --- DYNAMIC ATTENDANCE LOGIC ---
                    if score >= current_attendance_thresh:
                        color = (50,205,50) # Bright Green
                        label_name = stable_name
                        
                        now = time.time()
                        if trk.confirmed and not trk.attendance_marked and not is_lost:
                            if stable_name not in marked or (now - marked[stable_name]) > ATTENDANCE_COOLDOWN:
                                marked[stable_name] = now; trk.attendance_marked = True; trk.mark_time = now
                                self.flash_effects[tid] = FLASH_DURATION
                                ts = datetime.now().strftime("%H:%M:%S")
                                attendance_log.append({'date': today, 'time': ts, 'name': stable_name, 'subject': subject, 'confidence': round(score,4)})
                                logging.info(f"MARKED PRESENT: {stable_name} at {score:.0%} (Vel: {velocity_mag:.0f})")
                    else:
                        # Seen, but not high enough for attendance yet
                        color = (0,255,255) # Yellow
                        label_name = f"{stable_name} (need {current_attendance_thresh:.0%})"
                        
                else:
                    color, label_name = (0,0,255), "Unknown"

                if is_lost:
                    color, label_name = (200, 200, 200), f"{stable_name} (pred)" if trk.confirmed else "Tracking..."

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # WALKING UI TAG
                tag = f" {label_name} {score:.0%} "
                if velocity_mag > WALKING_VELOCITY_THRESHOLD: 
                    tag += "WALK "
                elif velocity_mag > 15: 
                    tag += "MOVE "
                
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x1, y1-th-8), (x1+tw, y1), color, -1)
                cv2.putText(display, tag, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                if tid in self.flash_effects and self.flash_effects[tid] > 0:
                    alpha = self.flash_effects[tid] / FLASH_DURATION
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x1-10, y1-10), (x2+10, y2+10), (0,255,0), -1)
                    cv2.addWeighted(overlay, alpha*0.4, display, 1-alpha*0.4, 0, display)
                    self.flash_effects[tid] -= 1

            self.flash_effects = {k:v for k,v in self.flash_effects.items() if v > 0}
            frame_times.append(time.time()-t_start)
            if sum(frame_times) > 0: display_fps = len(frame_times)/sum(frame_times)

            draw_translucent_rect(display, (0, 0), (450, 140), (30,30,30), 0.8)
            put_text_shadow(display, f"FPS: {display_fps:.0f} | Detect: {detect_fps:.0f} | Crowd: {last_info['num_faces']}", (10, 25), 0.55, (200,200,200))
            put_text_shadow(display, f"Mark At: {ATTENDANCE_THRESH_STILL:.0f} (Still) / {ATTENDANCE_THRESH_WALKING:.0f} (Walk)", (10, 50), 0.45, (150,150,150))
            put_text_shadow(display, f"Tracking: Hungarian | ReID: FastEMA", (10, 75), 0.45, (0,255,255))
            put_text_shadow(display, f"Lighting: {'NIGHT MODE' if is_night else 'ADAPTIVE'}", (10, 100), 0.45, (0,150,255) if is_night else (50,205,50))

            if marked:
                rx = display.shape[1] - 260
                draw_translucent_rect(display, (rx, 0), (display.shape[1], len(marked)*28 + 40), (30,30,30), 0.85)
                put_text_shadow(display, f"PRESENT: {len(marked)}", (rx+10, 25), 0.6, (50,205,50))
                for i, mname in enumerate(sorted(marked.keys())):
                    put_text_shadow(display, f"-> {mname}", (rx+15, 55 + i*28), 0.5, (255,255,255))

            cv2.imshow('PowerFace V4 Enterprise', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.running = False; cap.release(); cv2.destroyAllWindows()
        if self.detection_thread: self.detection_thread.join(timeout=2)
        self.save_attendance(attendance_log)

    # ==================== SAVE & VIEW ====================
    def save_attendance(self, records):
        if not records: return
        exists = os.path.exists(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['date','time','name','subject','confidence'])
            if not exists: w.writeheader()
            for r in records: w.writerow(r)

    def view_attendance(self):
        if not os.path.exists(ATTENDANCE_FILE): 
            print("No attendance records yet!")
            return
        print(f"\n{'DATE':<12}{'TIME':<10}{'NAME':<20}{'CONF':<8}")
        print("-"*50)
        with open(ATTENDANCE_FILE, 'r') as f:
            for row in csv.DictReader(f):
                print(f"{row['date']:<12}{row['time']:<10}{row['name']:<20}{row['confidence']:<8}")

# ========================= MAIN MENU =========================
def main():
    system = PowerFaceV4()
    while True:
        print("\n" + "="*50)
        print("  POWER FACE V4 - WALKING OPTIMIZED")
        print("="*50)
        print("1. Register Person")
        print("2. Take Attendance")
        print("3. View Registered People")
        print("4. Delete Person")
        print("5. View Attendance Records")
        print("6. Exit")
        print("-"*50)
        
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
        print("\n" + "!"*50)
        print("CRITICAL STARTUP ERROR!")
        print("!"*50)
        traceback.print_exc()
        input("Press Enter to exit...")
