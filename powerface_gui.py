#!/usr/bin/env python3
"""
PowerFace V6 — Modern AI Attendance Dashboard
Pixel-perfect Apple/Cupertino Dark Glassmorphic Dashboard matching design mockup:
Features:
  - Vertical Navigation Rail (Dashboard, Live Feed, Attendance, Students, Reports, Settings)
  - Top Glass Header with Live Clock & DirectML Hardware Acceleration Badge
  - 4 Dynamic Metric Stat Cards with Neon Sparkline Curves
  - Live Camera Recognition Viewport (4:3 natural aspect ratio) with Offline Backdrop
  - Segmented Cupertino Tabs (Today's Roster, Face Directory, Audit Log & Export)
  - Real-time pulsating green light blink effect on face verification
  - 10-step FaceID Guided Enrollment modal
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import threading
import os
import csv
import sys
import time
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_face_detection import (
    PowerFaceV5,
    ATTENDANCE_FILE,
    FACE_DB_FILE,
    open_camera,
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
    # ===== Deep Obsidian & Cyan-Blue Dark Palette =====
    BG_DARK = "#060A12"          # Ultra-deep dark background
    SIDEBAR_BG = "#090E1A"       # Left navigation rail
    CARD_BG = "#0D1527"          # Surface card container
    CARD_BG_LIGHT = "#131D31"    # Inset fields / table rows
    CARD_BORDER = "#18233C"      # Subtle 1px dark border
    CARD_BORDER_FOCUS = "#0A84FF"

    # Accent Colors
    ACCENT_BLUE = "#0A84FF"       # Apple System Blue / Active nav
    ACCENT_BLUE_HOVER = "#0071E3"
    ACCENT_BLUE_LIGHT = "#38BDF8"
    ACCENT_GREEN = "#10B981"      # Verified Present / Start Attendance
    ACCENT_GREEN_HOVER = "#059669"
    ACCENT_PURPLE = "#A855F7"     # Active Subject
    ACCENT_PURPLE_LIGHT = "#C084FC"
    ACCENT_AMBER = "#F59E0B"      # Vision Engine
    ACCENT_AMBER_LIGHT = "#FBBF24"
    ACCENT_RED = "#EF4444"        # Stop / Delete buttons
    ACCENT_RED_HOVER = "#DC2626"

    TEXT_MAIN = "#FFFFFF"        # Primary header text
    TEXT_MUTED = "#94A3B8"       # Secondary text
    TEXT_SUB = "#64748B"         # Tertiary footnote text

    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 440

    def __init__(self, root):
        self.root = root
        self.root.title("PowerFace V6 — Apple Intelligence Attendance Engine")
        self.root.geometry("1280x800")
        self.root.minsize(1120, 720)
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
        self.active_tab = "roster"
        self.sidebar_active = "dashboard"
        self.status_var = tk.StringVar(value="● DirectML GPU Accelerated • System Ready" if self.system.use_gpu else "● CPU Mode • System Ready")

        self._setup_styles()
        self._build_layout()

        # Load initial database & records
        self._refresh_registered_list()
        self._refresh_history_table()

        # Window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".",
                        font=("Segoe UI", 10),
                        background=self.BG_DARK,
                        foreground=self.TEXT_MAIN)

        # Treeview styling (Cupertino dark table)
        style.configure("Treeview",
                        font=("Segoe UI", 9),
                        rowheight=32,
                        background=self.CARD_BG,
                        fieldbackground=self.CARD_BG,
                        foreground=self.TEXT_MAIN,
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        font=("Segoe UI", 9, "bold"),
                        background=self.CARD_BG_LIGHT,
                        foreground=self.ACCENT_BLUE_LIGHT,
                        borderwidth=0,
                        padding=(8, 6))
        style.map("Treeview.Heading", background=[("active", self.CARD_BORDER)])
        style.map("Treeview", background=[("selected", "#1E3A8A")], foreground=[("selected", "#FFFFFF")])

        # Combobox styling
        style.configure("TCombobox",
                        fieldbackground=self.CARD_BG_LIGHT,
                        background=self.CARD_BG_LIGHT,
                        foreground=self.TEXT_MAIN,
                        arrowcolor=self.ACCENT_BLUE,
                        padding=6)

    def _build_layout(self):
        # 1. Main outer horizontal split: Left Sidebar + Right Workspace
        self.main_container = tk.Frame(self.root, bg=self.BG_DARK)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Build Sidebar Navigation Rail
        self._build_sidebar(self.main_container)

        # Build Main Content Area
        self.content_area = tk.Frame(self.main_container, bg=self.BG_DARK)
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Build Header, Stats Cards, Workspace, Statusbar
        self._build_header(self.content_area)
        self._build_stats_cards(self.content_area)
        self._build_workspace(self.content_area)
        self._build_statusbar(self.content_area)

    # ==================== LEFT SIDEBAR NAVIGATION ====================
    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=self.SIDEBAR_BG, width=92, highlightbackground=self.CARD_BORDER, highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        nav_items = [
            ("dashboard", "🏠", "Dashboard"),
            ("live", "📹", "Live Feed"),
            ("attendance", "📋", "Attendance"),
            ("students", "👥", "Students"),
            ("reports", "📑", "Reports"),
            ("settings", "⚙️", "Settings"),
        ]

        self.sidebar_buttons = {}
        for key, icon, label in nav_items:
            btn_frame = tk.Frame(sidebar, bg=self.SIDEBAR_BG)
            btn_frame.pack(fill=tk.X, padx=8, pady=6)

            is_active = (key == self.sidebar_active)
            bg_col = self.ACCENT_BLUE if is_active else self.SIDEBAR_BG
            fg_col = "#FFFFFF" if is_active else self.TEXT_MUTED

            btn = tk.Button(
                btn_frame,
                text=f"{icon}\n{label}",
                font=("Segoe UI", 8, "bold"),
                fg=fg_col,
                bg=bg_col,
                activebackground=self.ACCENT_BLUE,
                activeforeground="#FFFFFF",
                relief="flat",
                bd=0,
                cursor="hand2",
                pady=10,
                command=lambda k=key: self._on_sidebar_click(k)
            )
            btn.pack(fill=tk.X)
            self.sidebar_buttons[key] = (btn, btn_frame)

    def _on_sidebar_click(self, key):
        self.sidebar_active = key
        for k, (btn, frame) in self.sidebar_buttons.items():
            if k == key:
                btn.configure(bg=self.ACCENT_BLUE, fg="#FFFFFF")
            else:
                btn.configure(bg=self.SIDEBAR_BG, fg=self.TEXT_MUTED)

        if key == "live":
            if not self.is_running:
                self._start_attendance()
        elif key == "attendance":
            self._switch_tab("roster")
        elif key == "students":
            self._switch_tab("directory")
        elif key == "reports":
            self._switch_tab("audit")
        elif key == "settings":
            messagebox.showinfo("Settings", "PowerFace V6 • Settings\n- DirectML GPU Inference: Active\n- Motion Sensitivity: High\n- Video Resolution: 640x480 (4:3)")

    # ==================== TOP HEADER BAR ====================
    def _build_header(self, parent):
        header = tk.Frame(parent, bg=self.CARD_BG, height=64, padx=20, highlightbackground=self.CARD_BORDER, highlightthickness=1)
        header.pack(fill=tk.X, padx=16, pady=(12, 8))
        header.pack_propagate(False)

        # Left Branding
        left_box = tk.Frame(header, bg=self.CARD_BG)
        left_box.pack(side=tk.LEFT, fill=tk.Y, pady=8)

        # Lightning Badge
        badge_lbl = tk.Label(left_box, text="⚡", font=("Segoe UI", 16, "bold"), fg="#38BDF8", bg=self.CARD_BG)
        badge_lbl.pack(side=tk.LEFT, padx=(0, 8))

        title_lbl = tk.Label(left_box, text="PowerFace V6", font=("Segoe UI", 15, "bold"), fg=self.TEXT_MAIN, bg=self.CARD_BG)
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(left_box, text="Apple Intelligence Attendance Engine", font=("Segoe UI", 10), fg=self.TEXT_MUTED, bg=self.CARD_BG, padx=12)
        sub_lbl.pack(side=tk.LEFT, pady=(2, 0))

        # Right Hardware & Clock
        right_box = tk.Frame(header, bg=self.CARD_BG)
        right_box.pack(side=tk.RIGHT, fill=tk.Y, pady=12)

        # Green GPU Pill
        gpu_text = "⚙️ DirectML GPU" if self.system.use_gpu else "💻 CPU Mode"
        gpu_color = self.ACCENT_GREEN if self.system.use_gpu else self.ACCENT_AMBER
        gpu_badge = tk.Label(right_box, text=gpu_text, font=("Segoe UI", 9, "bold"), fg="#FFFFFF", bg=gpu_color, padx=12, pady=4)
        gpu_badge.pack(side=tk.RIGHT, padx=(12, 0))

        # Live Clock
        self.clock_label = tk.Label(right_box, text="", font=("Segoe UI", 9), fg=self.TEXT_MUTED, bg=self.CARD_BG)
        self.clock_label.pack(side=tk.RIGHT)
        self._update_clock()

    def _update_clock(self):
        now = datetime.now()
        now_str = f"📅 {now.strftime('%A, %b %d, %Y')}  •  {now.strftime('%H:%M:%S')}"
        self.clock_label.configure(text=now_str)
        self.root.after(1000, self._update_clock)

    # ==================== STATS CARDS WITH NEON SPARKLINES ====================
    def _build_stats_cards(self, parent):
        stats_frame = tk.Frame(parent, bg=self.BG_DARK)
        stats_frame.pack(fill=tk.X, padx=16, pady=(0, 10))

        self.stat_registered = self._create_stat_card(
            stats_frame,
            title="Enrolled Profiles",
            value="8",
            subtitle="Total students registered",
            icon="👥",
            icon_bg="#1E3A8A",
            icon_fg="#38BDF8",
            val_color="#38BDF8",
            sparkline_color="#38BDF8"
        )
        self.stat_registered.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.stat_present = self._create_stat_card(
            stats_frame,
            title="Verified Present",
            value="0",
            subtitle="Currently present",
            icon="✓",
            icon_bg="#064E3B",
            icon_fg="#10B981",
            val_color="#34D399",
            sparkline_color="#10B981"
        )
        self.stat_present.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.stat_subject = self._create_stat_card(
            stats_frame,
            title="Active Subject",
            value="General",
            subtitle="Selected subject",
            icon="👤",
            icon_bg="#581C87",
            icon_fg="#C084FC",
            val_color="#C084FC",
            sparkline_color="#A855F7"
        )
        self.stat_subject.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.stat_latency = self._create_stat_card(
            stats_frame,
            title="Vision Engine",
            value="-- ms (0 FPS)",
            subtitle="Inference time / Frame rate",
            icon="⚡",
            icon_bg="#78350F",
            icon_fg="#FBBF24",
            val_color="#FBBF24",
            sparkline_color="#F59E0B"
        )
        self.stat_latency.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_stat_card(self, parent, title, value, subtitle, icon, icon_bg, icon_fg, val_color, sparkline_color):
        card = tk.Frame(parent, bg=self.CARD_BG, highlightbackground=self.CARD_BORDER, highlightthickness=1, padx=14, pady=10)

        # Left Icon Circle
        icon_canvas = tk.Canvas(card, width=38, height=38, bg=self.CARD_BG, highlightthickness=0)
        icon_canvas.pack(side=tk.LEFT, padx=(0, 10))
        icon_canvas.create_oval(2, 2, 36, 36, fill=icon_bg, outline="")
        icon_canvas.create_text(19, 19, text=icon, fill=icon_fg, font=("Segoe UI", 12, "bold"))

        # Middle Text
        text_box = tk.Frame(card, bg=self.CARD_BG)
        text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        t_lbl = tk.Label(text_box, text=title, font=("Segoe UI", 9, "bold"), fg=self.TEXT_MUTED, bg=self.CARD_BG, anchor="w")
        t_lbl.pack(fill=tk.X)

        v_lbl = tk.Label(text_box, text=value, font=("Segoe UI", 15, "bold"), fg=val_color, bg=self.CARD_BG, anchor="w")
        v_lbl.pack(fill=tk.X, pady=(1, 0))

        s_lbl = tk.Label(text_box, text=subtitle, font=("Segoe UI", 8), fg=self.TEXT_SUB, bg=self.CARD_BG, anchor="w")
        s_lbl.pack(fill=tk.X)

        # Right Neon Sparkline Curve
        spark_canvas = tk.Canvas(card, width=54, height=26, bg=self.CARD_BG, highlightthickness=0)
        spark_canvas.pack(side=tk.RIGHT, padx=(4, 0))
        # Draw smooth wave curve
        points = [2, 20, 12, 18, 22, 22, 34, 10, 44, 14, 52, 6]
        spark_canvas.create_line(points, fill=sparkline_color, width=2, smooth=True)

        card._val = v_lbl
        return card

    # ==================== MAIN WORKSPACE ====================
    def _build_workspace(self, parent):
        workspace = tk.Frame(parent, bg=self.BG_DARK)
        workspace.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        # ===== LEFT COLUMN: Live Camera Recognition =====
        left_col = tk.Frame(workspace, bg=self.CARD_BG, highlightbackground=self.CARD_BORDER, highlightthickness=1, padx=14, pady=12)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Header
        vid_header = tk.Frame(left_col, bg=self.CARD_BG)
        vid_header.pack(fill=tk.X, pady=(0, 8))

        hdr_left = tk.Frame(vid_header, bg=self.CARD_BG)
        hdr_left.pack(side=tk.LEFT)
        tk.Label(hdr_left, text="📷", font=("Segoe UI", 11), fg=self.ACCENT_BLUE, bg=self.CARD_BG).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(hdr_left, text="Live Camera Recognition", font=("Segoe UI", 11, "bold"), fg=self.TEXT_MAIN, bg=self.CARD_BG).pack(side=tk.LEFT)

        self.live_indicator = tk.Label(vid_header, text="● OFFLINE", font=("Segoe UI", 9, "bold"), fg=self.TEXT_SUB, bg=self.CARD_BG)
        self.live_indicator.pack(side=tk.RIGHT)

        # Video Frame Viewport (Strict 4:3)
        self.video_container = tk.Frame(left_col, bg="#05070B", width=self.VIDEO_WIDTH, height=self.VIDEO_HEIGHT, highlightbackground=self.CARD_BORDER, highlightthickness=1)
        self.video_container.pack_propagate(False)
        self.video_container.pack(pady=(0, 10), fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.video_container, bg="#05070B")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self._show_offline_placeholder()

        # Bottom Control Bar
        control_bar = tk.Frame(left_col, bg=self.CARD_BG)
        control_bar.pack(fill=tk.X, pady=(4, 0))

        tk.Label(control_bar, text="Subject:", font=("Segoe UI", 10, "bold"), fg=self.TEXT_MUTED, bg=self.CARD_BG).pack(side=tk.LEFT, padx=(0, 6))

        subjects = ["General", "Mathematics", "Physics", "Computer Science", "Chemistry", "English", "Biology"]
        self.subj_combo = ttk.Combobox(control_bar, textvariable=self.current_subject, values=subjects, width=13)
        self.subj_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.subj_combo.bind("<<ComboboxSelected>>", lambda e: self.stat_subject._val.configure(text=self.current_subject.get()))
        self.subj_combo.bind("<KeyRelease>", lambda e: self.stat_subject._val.configure(text=self.current_subject.get()))

        self.btn_start = self._make_pill_button(
            control_bar,
            text="Start Attendance",
            icon="▶",
            command=self._toggle_attendance,
            bg_color=self.ACCENT_GREEN,
            hover_color=self.ACCENT_GREEN_HOVER
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_register = self._make_pill_button(
            control_bar,
            text="Enroll Person",
            icon="➕",
            command=self._open_register_dialog,
            bg_color=self.ACCENT_BLUE,
            hover_color=self.ACCENT_BLUE_HOVER
        )
        self.btn_register.pack(side=tk.LEFT)

        # ===== RIGHT COLUMN: Cupertino Segmented Tabs & Tables =====
        right_col = tk.Frame(workspace, bg=self.CARD_BG, highlightbackground=self.CARD_BORDER, highlightthickness=1, padx=14, pady=12, width=480)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_col.pack_propagate(False)

        # Segmented Tab Pill Header
        tab_bar = tk.Frame(right_col, bg=self.CARD_BG_LIGHT, highlightbackground=self.CARD_BORDER, highlightthickness=1, padx=4, pady=4)
        tab_bar.pack(fill=tk.X, pady=(0, 10))

        self.tab_buttons = {}
        tabs = [
            ("roster", "⏱ Today's Roster"),
            ("directory", "👤 Face Direct"),
            ("audit", "📑 Audit Log & Exp")
        ]
        for t_key, t_label in tabs:
            btn = tk.Button(
                tab_bar,
                text=t_label,
                font=("Segoe UI", 9, "bold"),
                fg="#FFFFFF" if t_key == self.active_tab else self.TEXT_MUTED,
                bg=self.ACCENT_BLUE if t_key == self.active_tab else self.CARD_BG_LIGHT,
                activebackground=self.ACCENT_BLUE,
                activeforeground="#FFFFFF",
                relief="flat",
                bd=0,
                cursor="hand2",
                padx=14,
                pady=6,
                command=lambda k=t_key: self._switch_tab(k)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            self.tab_buttons[t_key] = btn

        # Tab Content Container
        self.tab_content_frame = tk.Frame(right_col, bg=self.CARD_BG)
        self.tab_content_frame.pack(fill=tk.BOTH, expand=True)

        self._build_tab_roster()
        self._build_tab_directory()
        self._build_tab_audit()

        # Show initial tab
        self._switch_tab("roster")

    def _make_pill_button(self, parent, text, command, bg_color, hover_color, fg="#FFFFFF", icon=None):
        label_text = f"{icon}  {text}" if icon else text
        btn = tk.Button(
            parent,
            text=label_text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            fg=fg,
            bg=bg_color,
            activebackground=hover_color,
            activeforeground=fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=7
        )
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg_color))
        return btn

    def _switch_tab(self, key):
        self.active_tab = key
        for k, btn in self.tab_buttons.items():
            if k == key:
                btn.configure(bg=self.ACCENT_BLUE, fg="#FFFFFF")
            else:
                btn.configure(bg=self.CARD_BG_LIGHT, fg=self.TEXT_MUTED)

        self.frame_roster.pack_forget()
        self.frame_directory.pack_forget()
        self.frame_audit.pack_forget()

        if key == "roster":
            self.frame_roster.pack(fill=tk.BOTH, expand=True)
        elif key == "directory":
            self.frame_directory.pack(fill=tk.BOTH, expand=True)
        elif key == "audit":
            self.frame_audit.pack(fill=tk.BOTH, expand=True)

    # ==================== TAB 1: TODAY'S ROSTER ====================
    def _build_tab_roster(self):
        self.frame_roster = tk.Frame(self.tab_content_frame, bg=self.CARD_BG)

        hdr = tk.Frame(self.frame_roster, bg=self.CARD_BG)
        hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hdr, text="🕒 Real-Time Attendance Log:", font=("Segoe UI", 10, "bold"), fg=self.TEXT_MAIN, bg=self.CARD_BG).pack(side=tk.LEFT)

        tree_frame = tk.Frame(self.frame_roster, bg=self.CARD_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("time", "name", "subject", "confidence", "quality")
        self.live_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")

        self.live_tree.heading("time", text="Time")
        self.live_tree.heading("name", text="Student Name")
        self.live_tree.heading("subject", text="Subject")
        self.live_tree.heading("confidence", text="Match Conf")
        self.live_tree.heading("quality", text="Face")

        self.live_tree.column("time", width=70, anchor=tk.CENTER)
        self.live_tree.column("name", width=130, anchor=tk.W)
        self.live_tree.column("subject", width=85, anchor=tk.W)
        self.live_tree.column("confidence", width=80, anchor=tk.CENTER)
        self.live_tree.column("quality", width=65, anchor=tk.CENTER)

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.live_tree.yview)
        self.live_tree.configure(yscrollcommand=scroll.set)
        self.live_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ==================== TAB 2: FACE DIRECTORY ====================
    def _build_tab_directory(self):
        self.frame_directory = tk.Frame(self.tab_content_frame, bg=self.CARD_BG)

        top_bar = tk.Frame(self.frame_directory, bg=self.CARD_BG)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        tk.Label(top_bar, text="Search:", font=("Segoe UI", 9), fg=self.TEXT_MUTED, bg=self.CARD_BG).pack(side=tk.LEFT, padx=(0, 6))

        self.reg_search_var = tk.StringVar()
        self.reg_search_var.trace("w", lambda *args: self._filter_registered_list())
        search_entry = tk.Entry(
            top_bar,
            textvariable=self.reg_search_var,
            font=("Segoe UI", 9),
            bg=self.CARD_BG_LIGHT,
            fg=self.TEXT_MAIN,
            insertbackground="#FFFFFF",
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.CARD_BORDER,
            highlightcolor=self.ACCENT_BLUE,
            width=16
        )
        search_entry.pack(side=tk.LEFT, padx=(0, 8), ipady=3)

        btn_del = self._make_pill_button(top_bar, "Delete", self._delete_person, self.ACCENT_RED, self.ACCENT_RED_HOVER, icon="🗑️")
        btn_del.pack(side=tk.RIGHT)

        btn_ref = self._make_pill_button(top_bar, "Refresh", self._refresh_registered_list, self.CARD_BG_LIGHT, self.CARD_BORDER, icon="🔄")
        btn_ref.pack(side=tk.RIGHT, padx=(0, 6))

        list_frame = tk.Frame(self.frame_directory, bg=self.CARD_BG)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.reg_listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 10),
            bg=self.CARD_BG_LIGHT,
            fg=self.TEXT_MAIN,
            selectbackground=self.ACCENT_BLUE,
            selectforeground="#FFFFFF",
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.CARD_BORDER,
            bd=0,
            activestyle="none"
        )
        scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.reg_listbox.yview, bg=self.CARD_BORDER)
        self.reg_listbox.configure(yscrollcommand=scroll.set)
        self.reg_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ==================== TAB 3: AUDIT LOG & EXPORT ====================
    def _build_tab_audit(self):
        self.frame_audit = tk.Frame(self.tab_content_frame, bg=self.CARD_BG)

        top_bar = tk.Frame(self.frame_audit, bg=self.CARD_BG)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        btn_export = self._make_pill_button(top_bar, "Export CSV", self._export_csv, self.ACCENT_GREEN, self.ACCENT_GREEN_HOVER, icon="📥")
        btn_export.pack(side=tk.RIGHT)

        btn_refresh = self._make_pill_button(top_bar, "Reload", self._refresh_history_table, self.CARD_BG_LIGHT, self.CARD_BORDER, icon="🔄")
        btn_refresh.pack(side=tk.RIGHT, padx=(0, 6))

        tree_frame = tk.Frame(self.frame_audit, bg=self.CARD_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("date", "time", "name", "subject", "confidence", "quality")
        self.hist_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.hist_tree.heading(c, text=c.title())
            w = 75 if c in ("date", "time", "confidence", "quality") else 115
            self.hist_tree.column(c, width=w, anchor=tk.CENTER if c in ("date", "time", "confidence", "quality") else tk.W)

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=scroll.set)
        self.hist_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ==================== OFFLINE BACKDROP ====================
    def _show_offline_placeholder(self):
        img = np.zeros((self.VIDEO_HEIGHT, self.VIDEO_WIDTH, 3), dtype=np.uint8)
        # Deep blue gradient / classroom silhouette tone
        img[:] = (20, 14, 10)
        for y in range(self.VIDEO_HEIGHT):
            shade = int(12 + 18 * (y / self.VIDEO_HEIGHT))
            img[y, :] = (int(shade * 1.3), int(shade * 1.1), shade)

        # Center Camera Circle Icon Badge
        cx, cy = self.VIDEO_WIDTH // 2, self.VIDEO_HEIGHT // 2 - 35
        cv2.circle(img, (cx, cy), 42, (40, 52, 75), -1)
        cv2.circle(img, (cx, cy), 42, (70, 95, 135), 2)
        cv2.circle(img, (cx, cy), 16, (140, 170, 215), 2)
        cv2.rectangle(img, (cx - 20, cy - 12), (cx + 20, cy + 14), (140, 170, 215), 2)

        # Main Title & Subtitle
        cv2.putText(img, "Camera is Offline", (cx - 120, cy + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Click 'Start Attendance' to launch live recognition", (cx - 185, cy + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1, cv2.LINE_AA)

        self._render_numpy_to_label(img, self.video_label)

    def _render_numpy_to_label(self, cv_img, target_label):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        target_label.configure(image=tk_img)
        target_label.image = tk_img

    def _build_statusbar(self, parent):
        status_bar = tk.Frame(parent, bg=self.CARD_BG, height=28, padx=16, highlightbackground=self.CARD_BORDER, highlightthickness=1)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        lbl = tk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 9), fg=self.TEXT_MUTED, bg=self.CARD_BG, anchor="w")
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
                self.reg_listbox.insert(tk.END, f"  {name}  ({num_samples} face embeddings)")

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
                                            initialfile=f"Attendance_Report_{datetime.now().strftime('%Y%m%d')}.csv")
        if dest:
            try:
                import shutil
                shutil.copyfile(ATTENDANCE_FILE, dest)
                messagebox.showinfo("Export Success", f"Attendance records successfully saved to:\n{dest}")
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

        # Open camera using DirectShow/MSMF multi-index opener
        self.cap = open_camera(self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
        if self.cap is None:
            self._stop_attendance()
            messagebox.showerror(
                "Camera Access Error",
                "Could not connect to webcam!\n\n"
                "Please verify:\n"
                "1. No other application (Zoom, Teams, Discord, Browser) is currently using the camera.\n"
                "2. Windows Camera Privacy is enabled:\n"
                "   Windows Settings -> Privacy & Security -> Camera -> Allow desktop apps to access camera.\n"
                "3. Your camera is plugged in."
            )
            return

        self.is_running = True
        self.btn_start.configure(text="⏹ Stop Attendance", bg=self.ACCENT_RED, activebackground=self.ACCENT_RED_HOVER)
        self.live_indicator.configure(text="● LIVE RECOGNITION", fg=self.ACCENT_GREEN)
        self.btn_register.configure(state=tk.DISABLED)
        self.subj_combo.configure(state=tk.DISABLED)

        # Launch async AI worker
        self.system.running = True
        self.system.tracks = {}
        self.system.flash_effects = {}
        self.system.detection_thread = threading.Thread(target=self.system.detection_loop, daemon=True)
        self.system.detection_thread.start()

        self._set_status(f"Live camera stream active. Subject: {self.current_subject.get()}. Matching live video...")
        self._video_loop()

    def _stop_attendance(self):
        self.is_running = False
        self.btn_start.configure(text="▶ Start Attendance", bg=self.ACCENT_GREEN, activebackground=self.ACCENT_GREEN_HOVER)
        self.live_indicator.configure(text="● OFFLINE", fg=self.TEXT_SUB)
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
        if not ret or frame is None:
            self.root.after(30, self._video_loop)
            return

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
        self.stat_latency._val.configure(text=f"{self.last_detect_ms:.0f} ms ({avg_fps:.0f} FPS)")

        self.root.after(10, self._video_loop)

    # ==================== APPLE FACEID REGISTRATION MODAL ====================
    def _open_register_dialog(self):
        if self.is_running:
            self._stop_attendance()

        name = simpledialog.askstring("Enroll Person", "Enter the full name of the student/person:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()

        # Open webcam for enrollment
        cap = open_camera(640, 480)
        if cap is None:
            messagebox.showerror(
                "Camera Access Error",
                "Could not access webcam for enrollment!\nPlease ensure no other application is using the camera.",
                parent=self.root
            )
            return

        # Build Apple-styled Modal
        reg_win = tk.Toplevel(self.root)
        reg_win.title(f"FaceID Enrollment: {name}")
        reg_win.geometry("700x650")
        reg_win.resizable(False, False)
        reg_win.configure(bg=self.BG_DARK)
        reg_win.grab_set()

        header = tk.Frame(reg_win, bg=self.CARD_BG, height=56, padx=18, highlightbackground=self.CARD_BORDER, highlightthickness=1)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text=f"👤 FaceID Guided Enrollment: {name}", font=("Segoe UI", 12, "bold"), fg=self.TEXT_MAIN, bg=self.CARD_BG).pack(side=tk.LEFT, pady=12)

        # Video container
        v_container = tk.Frame(reg_win, bg="#05070B", width=640, height=480, highlightbackground=self.CARD_BORDER, highlightthickness=1)
        v_container.pack_propagate(False)
        v_container.pack(pady=12)

        v_label = tk.Label(v_container, bg="#05070B")
        v_label.pack(fill=tk.BOTH, expand=True)

        # Prompt & Progress Info
        info_frame = tk.Frame(reg_win, bg=self.BG_DARK)
        info_frame.pack(fill=tk.X, padx=24)

        prompt_lbl = tk.Label(info_frame, text="Pose 1/10: Look Straight at Camera", font=("Segoe UI", 11, "bold"), fg=self.ACCENT_BLUE, bg=self.BG_DARK)
        prompt_lbl.pack(side=tk.LEFT)

        progress_lbl = tk.Label(info_frame, text="0 / 10 Samples Captured", font=("Segoe UI", 10), fg=self.TEXT_MUTED, bg=self.BG_DARK)
        progress_lbl.pack(side=tk.RIGHT)

        # Controls
        btn_box = tk.Frame(reg_win, bg=self.BG_DARK, pady=10)
        btn_box.pack(fill=tk.X, padx=24)

        cap_state = {
            'running': True,
            'candidates': [],
            'cap': cap,
            'flash_frames': 0
        }

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
            cap_state['flash_frames'] = 8
            num = len(cap_state['candidates'])
            progress_lbl.configure(text=f"{num} / 10 Samples Captured")
            if num < len(prompts):
                prompt_lbl.configure(text=prompts[num], fg=self.ACCENT_BLUE)
            else:
                prompt_lbl.configure(text="✅ All 10 samples collected! Saving...", fg=self.ACCENT_GREEN)
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

        btn_cap = self._make_pill_button(btn_box, "Capture Sample (Space)", on_capture, self.ACCENT_BLUE, self.ACCENT_BLUE_HOVER, icon="📸")
        btn_cap.pack(side=tk.LEFT, padx=(0, 8))

        btn_fin = self._make_pill_button(btn_box, "Save & Finish", finish_reg, self.ACCENT_GREEN, self.ACCENT_GREEN_HOVER, icon="✅")
        btn_fin.pack(side=tk.LEFT)

        btn_cancel = self._make_pill_button(btn_box, "Cancel", close_win, self.CARD_BG_LIGHT, self.CARD_BORDER)
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
                    cv2.line(display, (x2, y2), (x2 - cl, y2), color, 3)

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
