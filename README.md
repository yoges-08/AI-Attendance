# ⚡ PowerFace V6 — AI Face Recognition Attendance System

An advanced, real-time AI-powered attendance and biometric recognition application featuring DirectML GPU acceleration, motion-adaptive face tracking, an Apple Cupertino-inspired GUI dashboard, and Apple FaceID-style guided registration.

---

## ✨ Features

- **⚡ DirectML GPU Acceleration & Load Sharing**: Automatically leverages GPU via DirectML with intelligent CPU thread balancing to prevent CPU core saturation and thermal throttling.
- **🏃 Motion-Adaptive Face Tracking**: High-speed Hungarian matcher and Kalman/EMA trajectory tracking accurately recognizes moving and walking subjects with dynamic velocity-based confidence thresholds.
- **✨ Animated Green Light Blinking Feedback**: Visually highlights confirmed attendance with a pulsating neon green glow effect and Apple FaceID corner reticles.
- **🖥️ Apple Cupertino Dark Dashboard**: Clean dark titanium UI (`powerface_gui.py`) with 4:3 distortion-free live video feed, segmented navigation tabs, live roster table, audit log, and one-click CSV export.
- **👤 Apple FaceID-Style Guided Enrollment**: 10-step multi-angle posture enrollment with live face quality scoring and automated sample filtering.
- **📊 Real-time Attendance & Historical Audit**: Tracks daily attendance, confidence ratings, image quality metrics, timestamps, and subjects.

---

## 🛠️ Tech Stack & Dependencies

- **Python 3.11**
- **InsightFace 0.7.3** (ArcFace / MobileFaceNet embeddings with buffalo_s model)
- **ONNX Runtime DirectML 1.24.4** (Hardware-accelerated neural network inference)
- **OpenCV 4.11.0** (Real-time video processing & visual effects)
- **NumPy 1.26.4**
- **Tkinter & Pillow** (Cupertino desktop dashboard)
- **Scipy & Scikit-Learn** (Tracking, Hungarian association, similarity metrics)

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yoges-08/AI-Attendance.git
cd AI-Attendance
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for Windows**: If installing `insightface` prompts for C++ Build Tools, use the included precompiled wheel:
> ```bash
> pip install insightface-0.7.3-cp311-cp311-win_amd64.whl
> ```

---

## 🖥️ Usage

### Launch GUI Dashboard
Run the Apple Cupertino dark theme desktop application:
```bash
python powerface_gui.py
```
- Click **Start Attendance** to begin live recognition.
- Select your subject/session from the dropdown.
- Click **Enroll Person** to register a new user using the guided FaceID scanner.
- View real-time attendance in **Today's Roster**, manage enrolled profiles in **Face Directory**, and export records in **Audit Log & Export**.

### Launch CLI Recognition Loop
Run the standalone command-line video recognition loop:
```bash
python ai_face_detection.py
```
- Press `Q` to exit the recognition stream and automatically log attendance to CSV.

---

## 📁 Project Structure

```
AI-Attendance/
├── powerface_gui.py             # Apple-styled GUI desktop application
├── ai_face_detection.py         # Core AI detection, tracking & recognition engine
├── attendance_insightface.py    # Baseline InsightFace recognition module
├── face_database_v6.pkl         # Biometric face embedding database
├── attendance_v6.csv            # Attendance records audit log
├── registered_photos_v6/        # Enrolled profile reference images
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # Project documentation
```

---

## 📜 License
This project is open-source under the MIT License.
