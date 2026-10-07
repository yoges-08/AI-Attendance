#!/usr/bin/env python3
"""
PowerFace V6 - Modern Dashboard Edition
A sleek, modern, dark-themed GUI for real-time AI face recognition attendance.
Features:
  - Embedded real-time video player with fixed 4:3 natural aspect ratio (no horizontal distortion/over-wide stretching)
  - DirectML GPU acceleration status badge
  - In-app multi-angle guided registration with live quality scoring
  - Live session attendance roster & historical audit log
  - Searchable registered user database and CSV export
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import threading
import os
import csv
import sys
import time
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_face_detection import (
    PowerFaceV5,
    ATTENDANCE_FILE,
    FACE_DB_FILE,
    draw_apple_face_box,
    face_quality_score,
    MIN_REG_QUALITY,
    MIN_FACE_SIZE,
    NUM_REGISTRATION_CAPTURES,
    NUM_REGISTRATION_KEEP,
    THRESH_HIGH_CONF,
    THRESH_MEDIUM_CONF,
    THRESH_LOW_CONF,
    ATTENDANCE_THRESH_STILL,
    ATTENDANCE_THRESH_MOVING,
    ATTENDANCE_THRESH_WALKING,
    WALKING_VELOCITY_THRESHOLD,
    MOVING_VELOCITY_THRESHOLD,
    MIN_QUALITY_FOR_RELAXED_ATTENDANCE,
    ATTENDANCE_COOLDOWN,
    FLASH_DURATION
)


class PowerFaceApp:
    # ===== Apple Cupertino Dark Mode Color Palette =====
    BG_DARK = "#0D0E12"          # Ultra-dark Titanium background
    CARD_BG = "#161922"          # Cupertino surface card
    CARD_BG_LIGHT = "#1F2432"    # Inset surface / input fields
    CARD_BG_HOVER = "#272E40"    # Hover surface
    BORDER = "#2E3648"           # 1px subtle divider border
    BORDER_FOCUS = "#0A84FF"     # Apple Focus Blue
    
    APPLE_BLUE = "#0A84FF"       # Apple System Blue
    APPLE_BLUE_HOVER = "#0071E3"
    APPLE_GREEN = "#30D158"      # Apple Vibrant Neon Green
    APPLE_GREEN_HOVER = "#28B84D"
    APPLE_RED = "#FF453A"        # Apple Coral Red
    APPLE_RED_HOVER = "#D73A32"
    APPLE_ORANGE = "#FF9F0A"     # Apple Amber
    APPLE_PURPLE = "#BF5AF2"     # Apple Indigo/Purple
    
    TEXT_MAIN = "#F5F5F7"        # Apple Primary Text
    TEXT_MUTED = "#86868B"       # Apple Secondary Text
    TEXT_SUB = "#A1A1A6"         # Apple Tertiary Text

    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480

    def __init__(self, root):
        self.root = root
        self.root.title("PowerFace — Apple Intelligence Attendance")
        self.root.geometry("1200x780")
        self.root.minsize(1080, 700)
        self.root.configure(bg=self.BG_DARK)

        # Initialize core AI recognition engine
        self.system = PowerFaceV5()

        # State management
        self.cap = None
        self.is_running = False
        self.current_subject = tk.StringVar(value="General")
        self.marked_today = {}
        self.session_records = []
        self.fps_tracker = []
        self.last_detect_ms = 0.0
        self.active_photo = None
        self.status_var = tk.StringVar(value="● DirectML GPU Accelerated • System Ready" if self.system.use_gpu else "● CPU Mode • System Ready")

        self._setup_styles()
        self._build_header()
        self._build_stats_bar()
        self._build_main_workspace()
        self._build_statusbar()

        # Load data
        self._refresh_registered_list()
        self._refresh_history_table()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".",
                        font=("Segoe UI", 10),
                        background=self.BG_DARK,
                        foreground=self.TEXT_MAIN)

        # Tabs styling (Apple Segmented Control look)
        style.configure("TNotebook", background=self.BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab",
                        font=("Segoe UI", 10, "bold"),
                        padding=(18, 9),
                        background=self.CARD_BG,
                        foreground=self.TEXT_MUTED,
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", self.APPLE_BLUE), ("active", self.CARD_BG_LIGHT)],
                  foreground=[("selected", "#FFFFFF"), ("active", self.TEXT_MAIN)])

        # Treeview styling (Cupertino table)
        style.configure("Treeview",
                        font=("Segoe UI", 9),
                        rowheight=30,
                        background=self.CARD_BG,
                        fieldbackground=self.CARD_BG,
                        foreground=self.TEXT_MAIN,
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        font=("Segoe UI", 9, "bold"),
                        background=self.CARD_BG_LIGHT,
                        foreground=self.APPLE_BLUE,
                        borderwidth=0,
                        padding=(8, 6))
        style.map("Treeview.Heading", background=[("active", self.BORDER)])
        style.map("Treeview", background=[("selected", "#1A365D")], foreground=[("selected", "#FFFFFF")])

        # Combobox styling
        style.configure("TCombobox",
                        fieldbackground=self.CARD_BG_LIGHT,
                        background=self.CARD_BG_LIGHT,
                        foreground=self.TEXT_MAIN,
                        arrowcolor=self.APPLE_BLUE,
                        padding=6)

    def _make_button(self, parent, text, command, bg_color, hover_color, fg="#FFFFFF", width=None, icon=None):
        label_text = f"{icon}  {text}" if icon else text
        btn = tk.Button(parent, text=label_text, command=command,
                        font=("Segoe UI", 10, "bold"),
                        fg=fg, bg=bg_color,
                        activebackground=hover_color, activeforeground=fg,
                        relief="flat", bd=0, cursor="hand2",
                        padx=18, pady=8, width=width)
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg_color))
        return btn

    def _build_header(self):
        header = tk.Frame(self.root, bg=self.CARD_BG, height=66, padx=24, highlightbackground=self.BORDER, highlightthickness=1)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Left branding
        title_box = tk.Frame(header, bg=self.CARD_BG)
        title_box.pack(side=tk.LEFT, fill=tk.Y, pady=10)

        logo_lbl = tk.Label(title_box, text="⚡ PowerFace V6",
                            font=("Segoe UI", 16, "bold"),
                            fg=self.TEXT_MAIN, bg=self.CARD_BG)
        logo_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(title_box, text="Apple Intelligence Attendance Engine",
                           font=("Segoe UI", 10),
                           fg=self.TEXT_MUTED, bg=self.CARD_BG, padx=14)
        sub_lbl.pack(side=tk.LEFT, pady=(3, 0))

        # Right status badge & clock
        right_box = tk.Frame(header, bg=self.CARD_BG)
        right_box.pack(side=tk.RIGHT, fill=tk.Y, pady=14)

        gpu_text = "⚡ DirectML GPU" if self.system.use_gpu else "💻 CPU Mode"
        gpu_color = self.APPLE_GREEN if self.system.use_gpu else self.APPLE_ORANGE
        
        gpu_badge = tk.Label(right_box, text=gpu_text, font=("Segoe UI", 9, "bold"),
                             fg="#FFFFFF", bg=gpu_color, padx=12, pady=4)
        gpu_badge.pack(side=tk.RIGHT, padx=(12, 0))

        self.clock_label = tk.Label(right_box, text="", font=("Segoe UI", 10),
                                    fg=self.TEXT_MUTED, bg=self.CARD_BG)
        self.clock_label.pack(side=tk.RIGHT)
        self._update_clock()

    def _update_clock(self):
        now_str = time.strftime("%A, %b %d %Y  •  %H:%M:%S")
        self.clock_label.configure(text=now_str)
        self.root.after(1000, self._update_clock)

    def _build_stats_bar(self):
        stats_frame = tk.Frame(self.root, bg=self.BG_DARK)
        stats_frame.pack(fill=tk.X, padx=24, pady=(16, 12))

        self.stat_registered = self._create_stat_card(stats_frame, "👥 Enrolled Profiles", "0", self.APPLE_BLUE)
        self.stat_registered.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.stat_present = self._create_stat_card(stats_frame, "🟢 Verified Present", "0", self.APPLE_GREEN)
        self.stat_present.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.stat_subject = self._create_stat_card(stats_frame, "📖 Active Subject", "General", self.APPLE_PURPLE)
        self.stat_subject.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.stat_latency = self._create_stat_card(stats_frame, "⚡ Vision Engine", "-- ms  (0 FPS)", self.APPLE_ORANGE)
        self.stat_latency.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_stat_card(self, parent, title, value, accent_color):
        card = tk.Frame(parent, bg=self.CARD_BG, highlightbackground=self.BORDER,
                        highlightthickness=1, padx=18, pady=12)
        
        t_lbl = tk.Label(card, text=title, font=("Segoe UI", 9, "bold"),
                         fg=self.TEXT_MUTED, bg=self.CARD_BG, anchor="w")
        t_lbl.pack(fill=tk.X)

        v_lbl = tk.Label(card, text=value, font=("Segoe UI", 16, "bold"),
                         fg=accent_color, bg=self.CARD_BG, anchor="w")
        v_lbl.pack(fill=tk.X, pady=(2, 0))
        card._val = v_lbl
        return card

    def _build_main_workspace(self):
        main = tk.Frame(self.root, bg=self.BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 12))

        # ===== LEFT COLUMN: Video Stream Screen & Controls =====
        left_col = tk.Frame(main, bg=self.CARD_BG, highlightbackground=self.BORDER,
                            highlightthickness=1, padx=14, pady=14)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 12))

        # Video Canvas with strictly locked natural 4:3 aspect ratio (640x480)
        video_header = tk.Frame(left_col, bg=self.CARD_BG)
        video_header.pack(fill=tk.X, pady=(0, 8))
        tk.Label(video_header, text="📹 Live Camera Recognition", font=("Segoe UI", 11, "bold"),
                 fg=self.TEXT_MAIN, bg=self.CARD_BG).pack(side=tk.LEFT)

        self.live_indicator = tk.Label(video_header, text="● OFFLINE", font=("Segoe UI", 9, "bold"),
                                       fg=self.TEXT_MUTED, bg=self.CARD_BG)
        self.live_indicator.pack(side=tk.RIGHT)

        # Video container
        self.video_frame = tk.Frame(left_col, bg="#05070B", width=self.VIDEO_WIDTH, height=self.VIDEO_HEIGHT,
                                    highlightbackground=self.BORDER, highlightthickness=1)
        self.video_frame.pack_propagate(False)
        self.video_frame.pack(pady=(0, 12))

        self.video_label = tk.Label(self.video_frame, bg="#05070B")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self._show_offline_placeholder()

        # Subject & Action Controls
        control_bar = tk.Frame(left_col, bg=self.CARD_BG)
        control_bar.pack(fill=tk.X, pady=(0, 6))

        tk.Label(control_bar, text="Subject:", font=("Segoe UI", 10, "bold"),
                 fg=self.TEXT_MUTED, bg=self.CARD_BG).pack(side=tk.LEFT, padx=(0, 6))

        subjects = ["General", "Mathematics", "Physics", "Computer Science", "Chemistry", "English", "Biology"]
        self.subj_combo = ttk.Combobox(control_bar, textvariable=self.current_subject, values=subjects, width=13)
        self.subj_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.subj_combo.bind("<<ComboboxSelected>>", lambda e: self.stat_subject._val.configure(text=self.current_subject.get()))
        self.subj_combo.bind("<KeyRelease>", lambda e: self.stat_subject._val.configure(text=self.current_subject.get()))

        self.btn_start = self._make_button(control_bar, "Start Attendance", self._toggle_attendance,
                                           self.APPLE_GREEN, self.APPLE_GREEN_HOVER, icon="▶")
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_register = self._make_button(control_bar, "Enroll Person", self._open_register_dialog,
                                              self.APPLE_BLUE, self.APPLE_BLUE_HOVER, icon="➕")
        self.btn_register.pack(side=tk.LEFT)

        # ===== RIGHT COLUMN: Tabs (Live Session, Registered Users, History Log) =====
        right_col = tk.Frame(main, bg=self.CARD_BG, highlightbackground=self.BORDER,
                             highlightthickness=1, padx=12, pady=12)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(right_col)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- TAB 1: Live Today's Roster ---
        tab_live = tk.Frame(self.notebook, bg=self.CARD_BG, padx=8, pady=8)
        self.notebook.add(tab_live, text="  🟢 Today's Roster  ")
        self._build_live_roster_tab(tab_live)

        # --- TAB 2: Face Directory ---
        tab_registered = tk.Frame(self.notebook, bg=self.CARD_BG, padx=8, pady=8)
        self.notebook.add(tab_registered, text="  👥 Face Directory  ")
        self._build_registered_tab(tab_registered)

        # --- TAB 3: Audit History & CSV Export ---
        tab_history = tk.Frame(self.notebook, bg=self.CARD_BG, padx=8, pady=8)
        self.notebook.add(tab_history, text="  📜 Audit Log & Export  ")
        self._build_history_tab(tab_history)

    def _show_offline_placeholder(self):
        img = np.zeros((self.VIDEO_HEIGHT, self.VIDEO_WIDTH, 3), dtype=np.uint8)
        img[:] = (18, 14, 11)
        # Cupertino offline graphics
        cv2.circle(img, (self.VIDEO_WIDTH // 2, self.VIDEO_HEIGHT // 2 - 40), 45, (45, 52, 68), -1)
        cv2.circle(img, (self.VIDEO_WIDTH // 2, self.VIDEO_HEIGHT // 2 - 40), 45, (80, 95, 120), 2)
        cv2.putText(img, "LIVE", (self.VIDEO_WIDTH // 2 - 20, self.VIDEO_HEIGHT // 2 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 210), 1, cv2.LINE_AA)

        cv2.putText(img, "Camera is Offline", (self.VIDEO_WIDTH // 2 - 110, self.VIDEO_HEIGHT // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 225, 235), 2, cv2.LINE_AA)
        cv2.putText(img, "Click 'Start Attendance' to launch live recognition",
                    (self.VIDEO_WIDTH // 2 - 190, self.VIDEO_HEIGHT // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (134, 134, 139), 1, cv2.LINE_AA)
        self._render_numpy_to_label(img, self.video_label)

    def _render_numpy_to_label(self, cv_img, target_label):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        target_label.configure(image=tk_img)
        target_label.image = tk_img

    def _build_live_roster_tab(self, parent):
        top_bar = tk.Frame(parent, bg=self.CARD_BG)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        tk.Label(top_bar, text="Real-Time Attendance Log:", font=("Segoe UI", 10, "bold"),
                 fg=self.TEXT_MAIN, bg=self.CARD_BG).pack(side=tk.LEFT)

        tree_frame = tk.Frame(parent, bg=self.CARD_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("time", "name", "subject", "confidence", "quality")
        self.live_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        
        self.live_tree.heading("time", text="Time")
        self.live_tree.heading("name", text="Student Name")
        self.live_tree.heading("subject", text="Subject")
        self.live_tree.heading("confidence", text="Match Conf")
        self.live_tree.heading("quality", text="Face Quality")

        self.live_tree.column("time", width=75, anchor=tk.CENTER)
        self.live_tree.column("name", width=140, anchor=tk.W)
        self.live_tree.column("subject", width=95, anchor=tk.W)
        self.live_tree.column("confidence", width=85, anchor=tk.CENTER)
        self.live_tree.column("quality", width=80, anchor=tk.CENTER)

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.live_tree.yview)
        self.live_tree.configure(yscrollcommand=scroll.set)
        self.live_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_registered_tab(self, parent):
        top_bar = tk.Frame(parent, bg=self.CARD_BG)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        tk.Label(top_bar, text="Search Directory:", font=("Segoe UI", 10),
                 fg=self.TEXT_MUTED, bg=self.CARD_BG).pack(side=tk.LEFT, padx=(0, 6))

        self.reg_search_var = tk.StringVar()
        self.reg_search_var.trace("w", lambda *args: self._filter_registered_list())
        search_entry = tk.Entry(top_bar, textvariable=self.reg_search_var,
                                font=("Segoe UI", 10), bg=self.CARD_BG_LIGHT, fg=self.TEXT_MAIN,
                                insertbackground="#FFFFFF", relief="flat", highlightthickness=1,
                                highlightbackground=self.BORDER, highlightcolor=self.APPLE_BLUE, width=18)
        search_entry.pack(side=tk.LEFT, padx=(0, 12), ipady=3)

        btn_del = self._make_button(top_bar, "Delete Person", self._delete_person,
                                    self.APPLE_RED, self.APPLE_RED_HOVER, icon="🗑️")
        btn_del.pack(side=tk.RIGHT)

        btn_ref = self._make_button(top_bar, "Refresh", self._refresh_registered_list,
                                    self.CARD_BG_LIGHT, self.CARD_BG_HOVER, icon="🔄")
        btn_ref.pack(side=tk.RIGHT, padx=(0, 8))

        # Registered directory list
        list_frame = tk.Frame(parent, bg=self.CARD_BG)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.reg_listbox = tk.Listbox(list_frame, font=("Segoe UI", 10),
                                      bg=self.CARD_BG_LIGHT, fg=self.TEXT_MAIN,
                                      selectbackground=self.APPLE_BLUE, selectforeground="#FFFFFF",
                                      relief="flat", highlightthickness=1,
                                      highlightbackground=self.BORDER, bd=0, activestyle="none")
        scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.reg_listbox.yview,
                              bg=self.BORDER)
        self.reg_listbox.configure(yscrollcommand=scroll.set)
        self.reg_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_history_tab(self, parent):
        top_bar = tk.Frame(parent, bg=self.CARD_BG)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        btn_export = self._make_button(top_bar, "Export CSV Report", self._export_csv,
                                       self.APPLE_GREEN, self.APPLE_GREEN_HOVER, icon="📥")
        btn_export.pack(side=tk.RIGHT)

        btn_refresh = self._make_button(top_bar, "Reload History", self._refresh_history_table,
                                        self.CARD_BG_LIGHT, self.CARD_BG_HOVER, icon="🔄")
        btn_refresh.pack(side=tk.RIGHT, padx=(0, 8))

        tree_frame = tk.Frame(parent, bg=self.CARD_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("date", "time", "name", "subject", "confidence", "quality")
        self.hist_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.hist_tree.heading(c, text=c.title())
            w = 80 if c in ("date", "time", "confidence", "quality") else 125
            self.hist_tree.column(c, width=w, anchor=tk.CENTER if c in ("date", "time", "confidence", "quality") else tk.W)

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=scroll.set)
        self.hist_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_statusbar(self):
        status_bar = tk.Frame(self.root, bg=self.CARD_BG, height=32, padx=18, highlightbackground=self.BORDER, highlightthickness=1)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        lbl = tk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 9),
                       fg=self.TEXT_MUTED, bg=self.CARD_BG, anchor="w")
        lbl.pack(side=tk.LEFT, fill=tk.X)

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    # ==================== DATA SYNC ====================
    def _refresh_registered_list(self):
        self.reg_listbox.delete(0, tk.END)
        count = len(self.system.known_faces)
        for name in sorted(self.system.known_faces.keys()):
            num_samples = len(self.system.known_faces[name])
            self.reg_listbox.insert(tk.END, f"  {name}  ({num_samples} face embeddings)")
        self.stat_registered._val.configure(text=str(count))
        self._set_status(f"Database loaded with {count} registered individuals.")

    def _filter_registered_list(self):
        query = self.reg_search_var.get().lower().strip()
        self.reg_listbox.delete(0, tk.END)
        for name in sorted(self.system.known_faces.keys()):
            if query in name.lower():
                num_samples = len(self.system.known_faces[name])
                self.reg_listbox.insert(tk.END, f"  {name}  ({num_samples} embeddings)")

    def _refresh_history_table(self):
        for row in self.hist_tree.get_children():
            self.hist_tree.delete(row)

        if not os.path.exists(ATTENDANCE_FILE):
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        today_marked_set = set()

        try:
            with open(ATTENDANCE_FILE, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for row in reversed(rows):
                    self.hist_tree.insert("", tk.END, values=(
                        row.get('date', ''),
                        row.get('time', ''),
                        row.get('name', ''),
                        row.get('subject', ''),
                        row.get('confidence', ''),
                        row.get('quality', '')
                    ))
                    if row.get('date') == today_str:
                        today_marked_set.add(row.get('name'))
            self.stat_present._val.configure(text=str(len(today_marked_set)))
        except Exception as e:
            self._set_status(f"Error loading records: {e}")

    def _export_csv(self):
        if not os.path.exists(ATTENDANCE_FILE):
            messagebox.showinfo("Export", "No attendance records found to export.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                                            initialfile=f"Attendance_Export_{datetime.now().strftime('%Y%m%d')}.csv")
        if dest:
            try:
                import shutil
                shutil.copyfile(ATTENDANCE_FILE, dest)
                messagebox.showinfo("Success", f"Attendance records successfully exported to:\n{dest}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV: {e}")

    # ==================== ATTENDANCE STREAMING ====================
    def _toggle_attendance(self):
        if self.is_running:
            self._stop_attendance()
        else:
            self._start_attendance()

    def _start_attendance(self):
        if not self.system.known_faces:
            if not messagebox.askyesno("No Registered Faces",
                                       "No registered people found in database.\nAll faces will show as 'Unknown'.\nStart camera anyway?"):
                return

        self.is_running = True
        self.btn_start.configure(text="⏹ Stop Attendance", bg=self.DANGER, activebackground=self.DANGER_HOVER)
        self.live_indicator.configure(text="● LIVE RECOGNITION", fg=self.SUCCESS)
        self.btn_register.configure(state=tk.DISABLED)
        self.subj_combo.configure(state=tk.DISABLED)

        # Open camera in standard natural aspect ratio
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.VIDEO_HEIGHT)

        # Launch async AI worker
        self.system.running = True
        self.system.tracks = {}
        self.system.flash_effects = {}
        self.system.detection_thread = threading.Thread(target=self.system.detection_loop, daemon=True)
        self.system.detection_thread.start()

        self._set_status(f"Camera active. Subject: {self.current_subject.get()}. Matching live video...")
        self._video_loop()

    def _stop_attendance(self):
        self.is_running = False
        self.btn_start.configure(text="▶ Start Attendance", bg=self.SUCCESS, activebackground=self.SUCCESS_HOVER)
        self.live_indicator.configure(text="● OFFLINE", fg=self.TEXT_MUTED)
        self.btn_register.configure(state=tk.NORMAL)
        self.subj_combo.configure(state=tk.NORMAL)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.system.running = False
        self._show_offline_placeholder()
        self._refresh_history_table()
        self._set_status("Attendance session stopped.")

    def _video_loop(self):
        if not self.is_running or self.cap is None:
            return

        t0 = time.time()
        ret, frame = self.cap.read()
        if ret:
            # Strictly maintain natural 4:3 aspect ratio without stretching
            if frame.shape[1] != self.VIDEO_WIDTH or frame.shape[0] != self.VIDEO_HEIGHT:
                frame = cv2.resize(frame, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)

            # Send frame to background AI detector
            try:
                if self.system.frame_queue.full():
                    self.system.frame_queue.get_nowait()
                self.system.frame_queue.put_nowait(frame)
            except Exception:
                pass

            # Fetch latest detections from AI worker
            try:
                last_info = self.system.result_queue.get_nowait()
                self.last_detect_ms = last_info.get('detect_ms', 0.0)
            except Exception:
                last_info = {'tracks': self.system.tracks, 'detect_ms': self.last_detect_ms, 'num_faces': 0}

            tracks = last_info.get('tracks', {})
            subject = self.current_subject.get() or "General"
            today_str = datetime.now().strftime("%Y-%m-%d")
            display = frame.copy()

            for tid, trk in tracks.items():
                if trk.lost_frames > 0 and trk.frames_seen < 2:
                    continue

                score = trk.score
                stable_name = trk.stable_name
                is_lost = trk.lost_frames > 0
                velocity_mag = trk.velocity_mag
                quality = trk.avg_quality

                # Adaptive motion thresholds (detecting walking / moving subjects)
                if velocity_mag > WALKING_VELOCITY_THRESHOLD:
                    threshold = THRESH_LOW_CONF
                elif velocity_mag > MOVING_VELOCITY_THRESHOLD:
                    threshold = THRESH_MEDIUM_CONF
                else:
                    threshold = THRESH_HIGH_CONF

                quality_ok = quality >= MIN_QUALITY_FOR_RELAXED_ATTENDANCE
                if velocity_mag > WALKING_VELOCITY_THRESHOLD and quality_ok:
                    current_attendance_thresh = ATTENDANCE_THRESH_WALKING
                elif velocity_mag > MOVING_VELOCITY_THRESHOLD and quality_ok:
                    current_attendance_thresh = ATTENDANCE_THRESH_MOVING
                else:
                    current_attendance_thresh = ATTENDANCE_THRESH_STILL

                recognized = stable_name != "Unknown" and score > threshold
                is_marked = trk.attendance_marked

                if recognized and score >= current_attendance_thresh:
                    now = time.time()
                    if trk.confirmed and not trk.attendance_marked and not is_lost:
                        if stable_name not in self.marked_today or (now - self.marked_today[stable_name]) > ATTENDANCE_COOLDOWN:
                            self.marked_today[stable_name] = now
                            trk.attendance_marked = True
                            is_marked = True
                            self.system.flash_effects[tid] = FLASH_DURATION
                            ts = datetime.now().strftime("%H:%M:%S")

                            # Save to CSV
                            rec = {
                                'date': today_str, 'time': ts, 'name': stable_name,
                                'subject': subject, 'confidence': round(score, 4),
                                'quality': round(quality, 3)
                            }
                            self.system.save_attendance([rec])

                            # Update Live Roster Treeview
                            self.live_tree.insert("", 0, values=(
                                ts, stable_name, subject, f"{score:.1%}", f"{quality:.2f}"
                            ))
                            self.stat_present._val.configure(text=str(len(self.marked_today)))
                            self._set_status(f"✓ ATTENDANCE RECORDED: {stable_name} ({score:.0%})")

                flash_rem = self.system.flash_effects.get(tid, 0)
                # Render Apple-style FaceID reticle with pulsating green flash
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
                    self.system.flash_effects[tid] -= 1

            self.system.flash_effects = {k: v for k, v in self.system.flash_effects.items() if v > 0}

            # Render frame on embedded GUI label
            self._render_numpy_to_label(display, self.video_label)

            # Measure live FPS
            elapsed = time.time() - t0
            self.fps_tracker.append(elapsed)
            if len(self.fps_tracker) > 20:
                self.fps_tracker.pop(0)
            avg_fps = len(self.fps_tracker) / max(sum(self.fps_tracker), 0.001)
            self.stat_latency._val.configure(text=f"{self.last_detect_ms:.0f} ms  ({avg_fps:.0f} FPS)")

        self.root.after(10, self._video_loop)

    # ==================== APPLE FACEID REGISTRATION MODAL ====================
    def _open_register_dialog(self):
        if self.is_running:
            self._stop_attendance()

        name = simpledialog.askstring("Enroll Person", "Enter the full name of the student/person:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()

        # Build Apple-styled Modal
        reg_win = tk.Toplevel(self.root)
        reg_win.title(f"FaceID Enrollment: {name}")
        reg_win.geometry("700x650")
        reg_win.resizable(False, False)
        reg_win.configure(bg=self.BG_DARK)
        reg_win.grab_set()

        header = tk.Frame(reg_win, bg=self.CARD_BG, height=56, padx=18, highlightbackground=self.BORDER, highlightthickness=1)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text=f"👤 FaceID Guided Enrollment: {name}", font=("Segoe UI", 12, "bold"),
                 fg=self.TEXT_MAIN, bg=self.CARD_BG).pack(side=tk.LEFT, pady=12)

        # Video container
        v_container = tk.Frame(reg_win, bg="#05070B", width=640, height=480, highlightbackground=self.BORDER, highlightthickness=1)
        v_container.pack_propagate(False)
        v_container.pack(pady=12)

        v_label = tk.Label(v_container, bg="#05070B")
        v_label.pack(fill=tk.BOTH, expand=True)

        # Prompt & Progress Info
        info_frame = tk.Frame(reg_win, bg=self.BG_DARK)
        info_frame.pack(fill=tk.X, padx=24)

        prompt_lbl = tk.Label(info_frame, text="Pose 1/10: Look Straight at Camera", font=("Segoe UI", 11, "bold"),
                              fg=self.APPLE_BLUE, bg=self.BG_DARK)
        prompt_lbl.pack(side=tk.LEFT)

        progress_lbl = tk.Label(info_frame, text="0 / 10 Samples Captured", font=("Segoe UI", 10),
                                fg=self.TEXT_MUTED, bg=self.BG_DARK)
        progress_lbl.pack(side=tk.RIGHT)

        # Controls
        btn_box = tk.Frame(reg_win, bg=self.BG_DARK, pady=10)
        btn_box.pack(fill=tk.X, padx=24)

        cap_state = {
            'running': True,
            'candidates': [],
            'cap': cv2.VideoCapture(0),
            'flash_frames': 0
        }
        cap_state['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_state['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prompts = [
            "1/10: Look Straight at Camera",
            "2/10: Slight Smile",
            "3/10: Turn Face Slightly Left",
            "4/10: Turn Face Slightly Right",
            "5/10: Tilt Head Slightly Up",
            "6/10: Tilt Head Slightly Down",
            "7/10: Move Slightly Closer",
            "8/10: Move Slightly Back",
            "9/10: Natural Neutral Expression",
            "10/10: Final Frontal Verification"
        ]

        def on_capture():
            if not cap_state['running']:
                return
            ret, frame = cap_state['cap'].read()
            if not ret:
                return
            faces = self.system.app.get(frame)
            if len(faces) != 1:
                messagebox.showwarning("FaceID Guide", "Please ensure exactly ONE face is visible inside the scan area!", parent=reg_win)
                return
            f = faces[0]
            bbox = f.bbox.astype(int)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            q = face_quality_score(gray, bbox, getattr(f, 'kps', None))

            if q < MIN_REG_QUALITY:
                messagebox.showwarning("Quality Notice", f"Image quality too low ({q:.2f}). Please ensure good lighting and face camera directly.", parent=reg_win)
                return

            cap_state['candidates'].append((q, f.embedding, frame.copy(), bbox))
            cap_state['flash_frames'] = 8  # Trigger flash capture visual
            num = len(cap_state['candidates'])
            progress_lbl.configure(text=f"{num} / 10 Samples Captured")
            if num < len(prompts):
                prompt_lbl.configure(text=prompts[num], fg=self.APPLE_BLUE)
            else:
                prompt_lbl.configure(text="✅ All 10 samples collected! Saving...", fg=self.APPLE_GREEN)
                finish_reg()

        def finish_reg():
            if not cap_state['candidates']:
                close_win()
                return
            cap_state['running'] = False
            cap_state['candidates'].sort(key=lambda c: c[0], reverse=True)
            keep = cap_state['candidates'][:NUM_REGISTRATION_KEEP]
            embeddings = [c[1] for c in keep]

            photo_dir = f"registered_photos_v6/{name}"
            os.makedirs(photo_dir, exist_ok=True)
            for i, (q, emb, frm, b) in enumerate(keep):
                cv2.imwrite(f"{photo_dir}/{i+1:02d}_q{q:.2f}.jpg", frm)

            if name not in self.system.known_faces:
                self.system.known_faces[name] = []
            self.system.known_faces[name].extend(embeddings)
            self.system.save_database()

            messagebox.showinfo("Registration Complete",
                                f"Successfully enrolled '{name}' with {len(keep)} high-accuracy biometric samples!",
                                parent=reg_win)
            self._refresh_registered_list()
            close_win()

        def close_win():
            cap_state['running'] = False
            if cap_state['cap']:
                cap_state['cap'].release()
            reg_win.destroy()

        reg_win.protocol("WM_DELETE_WINDOW", close_win)

        btn_cap = self._make_button(btn_box, "Capture Sample (Space)", on_capture, self.APPLE_BLUE, self.APPLE_BLUE_HOVER, icon="📸")
        btn_cap.pack(side=tk.LEFT, padx=(0, 8))

        btn_fin = self._make_button(btn_box, "Save & Finish", finish_reg, self.APPLE_GREEN, self.APPLE_GREEN_HOVER, icon="✅")
        btn_fin.pack(side=tk.LEFT)

        btn_cancel = self._make_button(btn_box, "Cancel", close_win, self.CARD_BG_LIGHT, self.CARD_BG_HOVER)
        btn_cancel.pack(side=tk.RIGHT)

        reg_win.bind("<space>", lambda e: on_capture())

        # Live capture loop with Apple FaceID Guide
        def update_reg_video():
            if not cap_state['running']:
                return
            ret, frame = cap_state['cap'].read()
            if ret:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                display = frame.copy()
                faces = self.system.app.get(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Draw FaceID scan reticle in center
                cx, cy = 320, 240
                r = 130
                cv2.ellipse(display, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, (70, 80, 100), 2)

                for f in faces:
                    b = f.bbox.astype(int)
                    q = face_quality_score(gray, b, getattr(f, 'kps', None))
                    good_quality = q >= MIN_REG_QUALITY
                    color = (60, 220, 50) if good_quality else (50, 60, 255)
                    
                    # Apple corner brackets
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    w, h = max(10, x2 - x1), max(10, y2 - y1)
                    cl = min(20, w // 4, h // 4)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.line(display, (x1, y1), (x1 + cl, y1), color, 3)
                    cv2.line(display, (x1, y1), (x1, y1 + cl), color, 3)
                    cv2.line(display, (x2, y1), (x2 - cl, y1), color, 3)
                    cv2.line(display, (x2, y1), (x2, y1 + cl), color, 3)
                    cv2.line(display, (x1, y2), (x1 + cl, y2), color, 3)
                    cv2.line(display, (x1, y2), (x1, y2 - cl), color, 3)
                    cv2.line(display, (x2, y2), (x2 - cl, y2), color, 3)
                    cv2.line(display, (x2, y2), (x2, y2 - cl), color, 3)

                    q_tag = f" Quality: {q:.0%} ({'GOOD' if good_quality else 'LOW'}) "
                    (tw, th), _ = cv2.getTextSize(q_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(display, q_tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Capture flash effect
                if cap_state['flash_frames'] > 0:
                    overlay = display.copy()
                    overlay[:] = (255, 255, 255)
                    cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
                    cap_state['flash_frames'] -= 1

                self._render_numpy_to_label(display, v_label)
            reg_win.after(15, update_reg_video)

        update_reg_video()

    # ==================== DELETE PERSON ====================
    def _delete_person(self):
        sel = self.reg_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select Profile", "Please select a person from the directory list first.")
            return
        raw_text = self.reg_listbox.get(sel[0])
        name = raw_text.strip().rsplit("  (", 1)[0].strip()

        if not messagebox.askyesno("Confirm Deletion", f"Permanently delete '{name}' and remove all biometric embeddings?"):
            return

        if name in self.system.known_faces:
            del self.system.known_faces[name]
            self.system.save_database()
            self._refresh_registered_list()
            self._set_status(f"Profile deleted: {name}")

    def _on_close(self):
        self._stop_attendance()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PowerFaceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
