"""DeepSORT Tracking Module for SCB-05"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class SCB05Tracker:
    def __init__(self, max_age: int = 30, n_init: int = 3):
        self.max_age = max_age
        self.n_init = n_init
        self.tracker = None
        self.track_history = defaultdict(list)
        self.track_colors = {}
        self.next_color_idx = 0
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        self.init_tracker()
    
    def init_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                nms_max_overlap=1.0,
                max_cosine_distance=0.4,
                nn_budget=None,
                embedder="mobilenet",
                half=True,
                bgr=True
            )
            print("✅ DeepSORT tracker initialized")
            return True
        except ImportError:
            print("⚠️ DeepSORT not available. Install with: pip install deep-sort-realtime")
            self.tracker = None
            return False
    
    def update(self, detections: List[Dict], frame) -> List[Dict]:
        if not self.tracker:
            tracked = []
            for i, det in enumerate(detections):
                tracked.append({
                    'track_id': i + 1,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'color': self.colors[i % len(self.colors)]
                })
            return tracked
        
        if not detections:
            tracks = self.tracker.update_tracks([], frame=frame)
        else:
            ds_detections = []
            for det in detections:
                x, y, w, h = det['bbox']
                confidence = det['confidence']
                class_name = det['class_name']
                ds_detections.append(([x, y, w, h], confidence, class_name))
            
            tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            w, h = x2 - x1, y2 - y1
            
            best_det = None
            best_confidence = 0
            
            for det in detections:
                det_x, det_y, det_w, det_h = det['bbox']
                if abs(x1 - det_x) < 50 and abs(y1 - det_y) < 50:
                    if det['confidence'] > best_confidence:
                        best_confidence = det['confidence']
                        best_det = det
            
            if best_det:
                if track_id not in self.track_colors:
                    self.track_colors[track_id] = self.colors[self.next_color_idx % len(self.colors)]
                    self.next_color_idx += 1
                
                tracked_objects.append({
                    'track_id': track_id,
                    'bbox': (x1, y1, w, h),
                    'confidence': best_det['confidence'],
                    'class_id': best_det['class_id'],
                    'class_name': best_det['class_name'],
                    'color': self.track_colors[track_id]
                })
                
                self.track_history[track_id].append({
                    'frame': len(self.track_history[track_id]),
                    'bbox': (x1, y1, w, h),
                    'confidence': best_det['confidence'],
                    'class_name': best_det['class_name']
                })
        
        return tracked_objects
    
    def draw_track(self, frame, tracked_object):
        track_id = tracked_object['track_id']
        x, y, w, h = tracked_object['bbox']
        confidence = tracked_object['confidence']
        class_name = tracked_object['class_name']
        color = tracked_object.get('color', (0, 255, 0))
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        label = f"ID:{track_id} {class_name}: {confidence:.1%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        cv2.rectangle(frame, (x, y - label_h - 5), (x + label_w, y), color, -1)
        cv2.putText(frame, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if confidence >= 0.80:
            cv2.circle(frame, (x + w - 15, y + 15), 8, (255, 215, 0), -1)
            cv2.putText(frame, "★", (x + w - 18, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def get_track_info(self) -> Dict:
        return {
            'total_tracks': len(self.track_history),
            'active_tracks': len([t for t in self.track_history if len(self.track_history[t]) > 0])
        }
    
    def reset(self):
        if self.tracker:
            self.init_tracker()
        self.track_history.clear()
        self.track_colors.clear()
        self.next_color_idx = 0