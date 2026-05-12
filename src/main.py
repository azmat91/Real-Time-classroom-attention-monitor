"""SCB-05 Classroom Analyzer - Complete Working Version"""

import sys
import os

# Force numpy to use old version compatibility
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Import required modules
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import threading
import time
import json
import csv
from collections import defaultdict

# Import numpy and cv2 with error handling
try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import error: {e}")
    print("Please run: pip install opencv-python==4.8.1.78")
    sys.exit(1)

# Import PIL
from PIL import Image, ImageTk

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO imported successfully")
except ImportError as e:
    print(f"⚠️ YOLO not available: {e}")
    YOLO_AVAILABLE = False

# Import DeepSORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
    print("✅ DeepSORT imported successfully")
except ImportError as e:
    print(f"⚠️ DeepSORT not available: {e}")
    DEEPSORT_AVAILABLE = False

class SCB05App:
    def __init__(self, root):
        self.root = root
        self.root.title("SCB-05 Classroom Analyzer")
        self.root.geometry("1400x800")
        
        # Create directories
        for d in ["results/exports", "results/screenshots", "results/reports", "models/train/weights", "models/train2/weights"]:
            os.makedirs(d, exist_ok=True)
        
        # Initialize YOLO model
        self.model = None
        self.load_model()
        
        # Initialize DeepSORT tracker
        self.tracker = None
        self.init_tracker()
        
        # Statistics
        self.total_points = 0
        self.behavior_points = defaultdict(int)
        self.behavior_counts = defaultdict(int)
        self.track_points = defaultdict(int)
        self.detection_history = []
        self.current_frame = 0
        
        # Video settings
        self.cap = None
        self.playing = False
        self.monitoring = False
        self.video_path = None
        self.is_camera = False
        
        # Class names for SCB-05
        self.class_names = {
            0: 'hand-raising', 1: 'read', 2: 'write', 3: 'bow_head',
            4: 'turn_head', 5: 'talk', 6: 'guide', 7: 'board_writing',
            8: 'stand', 9: 'answer', 10: 'stage_interaction', 11: 'discuss',
            12: 'clap', 13: 'yawn', 14: 'screen', 15: 'blackboard',
            16: 'teacher', 17: 'leaning_on_desk', 18: 'using_phone', 19: 'using_computer'
        }
        
        # Setup GUI
        self.setup_gui()
        
        # Start update loop
        self.update_display()
        
        # Handle closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self):
        """Load YOLO model"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO not available, running in simulation mode")
            return
        
        # Try to find model in various locations
        model_paths = [
            "models/train/weights/best.pt",
            "models/train2/weights/best.pt",
            "best.pt",
            "yolov8n.pt"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    print(f"Loading model from {path}...")
                    self.model = YOLO(path)
                    print(f"✅ Model loaded successfully")
                    return
                except Exception as e:
                    print(f"Error loading model: {e}")
        
        print("⚠️ No model found, running in simulation mode")
        self.model = None
    
    def init_tracker(self):
        """Initialize DeepSORT tracker"""
        if not DEEPSORT_AVAILABLE:
            print("⚠️ DeepSORT not available")
            return
        
        try:
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.4,
                nn_budget=None,
                embedder="mobilenet",
                half=True,
                bgr=True
            )
            print("✅ DeepSORT tracker initialized")
        except Exception as e:
            print(f"⚠️ Error initializing DeepSORT: {e}")
            self.tracker = None
    
    def setup_gui(self):
        """Setup GUI"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header = tk.Frame(main_frame, bg='#2c3e50', height=70)
        header.pack(fill=tk.X, pady=(0, 10))
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🎓 SCB-05 Classroom Behavior Analyzer", 
                        font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white')
        title.pack(pady=15)
        
        # Status
        status_text = f"YOLO: {'✅' if self.model else '❌'} | DeepSORT: {'✅' if self.tracker else '❌'} | Threshold: 80%"
        status_label = tk.Label(header, text=status_text, font=('Arial', 9), 
                                bg='#2c3e50', fg='#1abc9c')
        status_label.pack()
        
        # Content
        content = tk.Frame(main_frame, bg='#f0f0f0')
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video
        left_panel = tk.Frame(content, bg='#333333')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        self.video_label = tk.Label(left_panel, 
                                   text="SCB-05 Analyzer\n\nLoad Video or Open Camera",
                                   bg='black', fg='white', font=('Arial', 14))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls
        controls = tk.Frame(left_panel, bg='#333333')
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        buttons = [
            ("📁 Load Video", self.load_video, '#3498db'),
            ("📷 Open Camera", self.open_camera, '#2ecc71'),
            ("▶️ Start", self.start_analysis, '#1abc9c'),
            ("⏸️ Pause", self.pause_analysis, '#f39c12'),
            ("⏹️ Stop", self.stop_analysis, '#e74c3c')
        ]
        
        for text, cmd, color in buttons:
            btn = tk.Button(controls, text=text, command=cmd, bg=color, fg='white',
                          font=('Arial', 10), padx=10, pady=5)
            btn.pack(side=tk.LEFT, padx=2)
        
        # Right panel
        right_panel = tk.Frame(content, bg='#f0f0f0', width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)
        
        # Statistics
        stats_frame = tk.LabelFrame(right_panel, text="Statistics", font=('Arial', 12, 'bold'),
                                   bg='white', padx=10, pady=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_vars = {}
        stats = [
            ("🏆 Points:", "points", "0"),
            ("🎯 80%+ Detections:", "hc", "0"),
            ("👥 Tracks:", "tracks", "0"),
            ("🎬 Frame:", "frame", "0"),
            ("⚡ FPS:", "fps", "0")
        ]
        
        for label, key, default in stats:
            frame = tk.Frame(stats_frame, bg='white')
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=label, font=('Arial', 10), bg='white', 
                    width=15, anchor='w').pack(side=tk.LEFT)
            self.stats_vars[key] = tk.StringVar(value=default)
            tk.Label(frame, textvariable=self.stats_vars[key], font=('Arial', 10, 'bold'),
                    bg='white', fg='#2c3e50').pack(side=tk.RIGHT)
        
        # Behavior points
        behavior_frame = tk.LabelFrame(right_panel, text="Behavior Points", font=('Arial', 12, 'bold'),
                                      bg='white', padx=10, pady=10)
        behavior_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.behavior_listbox = tk.Listbox(behavior_frame, font=('Courier', 9), height=10)
        scrollbar = ttk.Scrollbar(behavior_frame, command=self.behavior_listbox.yview)
        self.behavior_listbox.config(yscrollcommand=scrollbar.set)
        self.behavior_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Alerts
        alerts_frame = tk.LabelFrame(right_panel, text="Alerts", font=('Arial', 12, 'bold'),
                                    bg='white', padx=10, pady=10)
        alerts_frame.pack(fill=tk.BOTH, expand=True)
        
        self.alerts_text = tk.Text(alerts_frame, height=6, bg='#f8f9fa', font=('Courier', 8))
        scrollbar2 = ttk.Scrollbar(alerts_frame, command=self.alerts_text.yview)
        self.alerts_text.config(yscrollcommand=scrollbar2.set)
        self.alerts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export buttons
        export_frame = tk.Frame(right_panel, bg='white')
        export_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Button(export_frame, text="📊 Export", command=self.export_data,
                 bg='#9b59b6', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Button(export_frame, text="📈 Report", command=self.generate_report,
                 bg='#1abc9c', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        tk.Button(export_frame, text="🔄 Reset", command=self.reset_stats,
                 bg='#e74c3c', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
    
    def load_video(self):
        filename = filedialog.askopenfilename(filetypes=[('Video', '*.mp4 *.avi *.mov')])
        if filename:
            self.video_path = filename
            self.is_camera = False
            self.add_alert(f"Loaded: {os.path.basename(filename)}")
    
    def open_camera(self):
        self.video_path = 0
        self.is_camera = True
        self.add_alert("Camera opened")
    
    def start_analysis(self):
        if self.video_path is None:
            self.add_alert("Load video or open camera first")
            return
        if self.playing:
            return
        
        self.playing = True
        self.monitoring = True
        threading.Thread(target=self.analysis_thread, daemon=True).start()
        self.add_alert("Analysis started")
    
    def analysis_thread(self):
        try:
            if self.is_camera:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                raise Exception("Cannot open video source")
            
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = 0
            last_fps = time.time()
            
            while self.playing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_camera:
                        time.sleep(0.05)
                        continue
                    break
                
                self.current_frame += 1
                frame_count += 1
                
                # Process frame
                processed = self.process_frame(frame)
                
                # Update display
                self.update_display_frame(processed)
                
                # Update FPS
                now = time.time()
                if now - last_fps >= 1.0:
                    self.stats_vars['fps'].set(f"{frame_count:.1f}")
                    frame_count = 0
                    last_fps = now
                
                self.stats_vars['frame'].set(str(self.current_frame))
                time.sleep(max(0, 1/fps - 0.005))
            
            self.stop_analysis()
        except Exception as e:
            self.add_alert(f"Error: {str(e)}")
            self.stop_analysis()
    
    def process_frame(self, frame):
        processed = frame.copy()
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(processed, timestamp, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if self.monitoring:
            # Run detection
            detections = self.run_detection(frame)
            
            # Track detections
            tracked = self.run_tracking(detections, frame)
            
            # Draw results
            for obj in tracked:
                self.draw_detection(processed, obj)
                if obj['confidence'] >= 0.80:
                    self.award_points(obj)
            
            self.update_stats_display()
            cv2.putText(processed, "ANALYZING (80%+ threshold)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        return processed
    
    def run_detection(self, frame):
        """Run YOLO detection"""
        detections = []
        
        if self.model:
            try:
                results = self.model(frame, verbose=False)[0]
                if results.boxes:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Only keep 80%+ detections
                        if conf >= 0.80:
                            w, h = x2 - x1, y2 - y1
                            name = self.class_names.get(cls, f"class_{cls}")
                            detections.append({
                                'bbox': (x1, y1, w, h),
                                'confidence': conf,
                                'class_name': name,
                                'class_id': cls
                            })
            except Exception as e:
                print(f"Detection error: {e}")
        
        # Simulation mode (for testing)
        if not detections and not self.model:
            import random
            if random.random() < 0.2:  # 20% chance per frame
                behaviors = ['hand-raising', 'read', 'write', 'guide', 'answer']
                name = random.choice(behaviors)
                x = random.randint(100, 500)
                y = random.randint(100, 300)
                w = random.randint(80, 150)
                h = random.randint(80, 150)
                conf = random.uniform(0.80, 0.95)
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'class_name': name,
                    'class_id': 0
                })
        
        return detections
    
    def run_tracking(self, detections, frame):
        """Run DeepSORT tracking"""
        if self.tracker and detections:
            try:
                ds_dets = []
                for d in detections:
                    x, y, w, h = d['bbox']
                    ds_dets.append(([x, y, w, h], d['confidence'], d['class_name']))
                
                tracks = self.tracker.update_tracks(ds_dets, frame=frame)
                
                tracked = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    # Find matching detection
                    for d in detections:
                        dx, dy, dw, dh = d['bbox']
                        if abs(x1 - dx) < 50 and abs(y1 - dy) < 50:
                            tracked.append({
                                'track_id': track.track_id,
                                'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'confidence': d['confidence'],
                                'class_name': d['class_name']
                            })
                            break
                return tracked
            except Exception as e:
                print(f"Tracking error: {e}")
        
        # Fallback: no tracking
        tracked = []
        for i, d in enumerate(detections):
            tracked.append({
                'track_id': i + 1,
                'bbox': d['bbox'],
                'confidence': d['confidence'],
                'class_name': d['class_name']
            })
        return tracked
    
    def draw_detection(self, frame, obj):
        x, y, w, h = obj['bbox']
        track_id = obj['track_id']
        conf = obj['confidence']
        class_name = obj['class_name']
        
        # Color mapping
        colors = {
            'hand-raising': (0, 255, 0),
            'read': (255, 0, 0),
            'write': (0, 0, 255),
            'guide': (255, 255, 0),
            'answer': (255, 0, 255),
            'stage_interaction': (0, 255, 255),
            'board_writing': (128, 0, 128)
        }
        color = colors.get(class_name, (0, 255, 255))
        
        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"ID:{track_id} {class_name}: {conf:.0%}"
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Star for high confidence
        if conf >= 0.80:
            cv2.circle(frame, (x + w - 10, y + 10), 5, (0, 255, 255), -1)
    
    def award_points(self, obj):
        """Award points for 80%+ detection"""
        class_name = obj['class_name']
        track_id = obj['track_id']
        
        self.behavior_counts[class_name] += 1
        self.behavior_points[class_name] += 1
        self.track_points[track_id] += 1
        self.total_points += 1
        
        self.detection_history.append({
            'time': datetime.now().isoformat(),
            'frame': self.current_frame,
            'track': track_id,
            'behavior': class_name,
            'confidence': obj['confidence']
        })
        
        self.add_alert(f"🎯 {class_name} (ID:{track_id}) +1pt")
    
    def update_display_frame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        except:
            pass
    
    def update_stats_display(self):
        self.stats_vars['points'].set(str(self.total_points))
        self.stats_vars['hc'].set(str(sum(self.behavior_counts.values())))
        self.stats_vars['tracks'].set(str(len(self.track_points)))
        
        # Update behavior list
        self.behavior_listbox.delete(0, tk.END)
        for behavior, points in sorted(self.behavior_points.items(), key=lambda x: x[1], reverse=True):
            if points > 0:
                self.behavior_listbox.insert(tk.END, f"{behavior:20} {points:3} pts")
    
    def add_alert(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alerts_text.insert('1.0', f"[{timestamp}] {message}\n")
        if int(self.alerts_text.index('end-1c').split('.')[0]) > 15:
            self.alerts_text.delete('15.0', 'end')
    
    def pause_analysis(self):
        self.playing = False
        self.add_alert("Paused")
    
    def stop_analysis(self):
        self.playing = False
        self.monitoring = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.add_alert("Stopped")
    
    def export_data(self):
        if not self.detection_history:
            self.add_alert("No data to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/exports/data_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Frame', 'Track', 'Behavior', 'Confidence'])
            for d in self.detection_history:
                writer.writerow([d['time'], d['frame'], d['track'], d['behavior'], d['confidence']])
        
        self.add_alert(f"Exported to {filename}")
        messagebox.showinfo("Export Complete", f"Data saved to:\n{filename}")
    
    def generate_report(self):
        if not self.detection_history:
            self.add_alert("No data for report")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/reports/report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("SCB-05 ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Points: {self.total_points}\n")
            f.write(f"Total 80%+ Detections: {sum(self.behavior_counts.values())}\n\n")
            f.write("BEHAVIOR SCORES:\n")
            f.write("-" * 30 + "\n")
            for behavior, points in sorted(self.behavior_points.items(), key=lambda x: x[1], reverse=True):
                if points > 0:
                    f.write(f"  {behavior}: {points} points\n")
            f.write("\n" + "=" * 50 + "\n")
        
        self.add_alert(f"Report saved to {filename}")
        messagebox.showinfo("Report Generated", f"Report saved to:\n{filename}")
    
    def reset_stats(self):
        self.total_points = 0
        self.behavior_points.clear()
        self.behavior_counts.clear()
        self.track_points.clear()
        self.detection_history.clear()
        self.current_frame = 0
        self.update_stats_display()
        self.add_alert("Statistics reset")
    
    def update_display(self):
        if self.monitoring:
            self.update_stats_display()
        self.root.after(1000, self.update_display)
    
    def on_closing(self):
        self.stop_analysis()
        self.root.destroy()

if __name__ == "__main__":
    # Check if OpenCV works
    print("Testing imports...")
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
    
    # Create and run app
    root = tk.Tk()
    app = SCB05App(root)
    root.mainloop()