"""YOLOv8 Detection Module for SCB-05"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import os

class SCB05Detector:
    def __init__(self, confidence_threshold: float = 0.80):
        self.confidence_threshold = confidence_threshold
        self.models = {}
        self.class_names = {}
        
    def load_model(self, model_path: str, model_name: str, 
                   class_ids: List[int], class_names: Dict):
        try:
            print(f"Loading {model_name} from {model_path}...")
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found: {model_path}")
                return False
            
            model = YOLO(model_path)
            self.models[model_name] = {
                'model': model,
                'class_ids': class_ids
            }
            self.class_names.update(class_names)
            print(f"✅ Loaded {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        all_detections = []
        
        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                results = model(frame, verbose=False)[0]
                
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        if confidence >= self.confidence_threshold:
                            w, h = x2 - x1, y2 - y1
                            class_name = self.class_names.get(class_id, f"class_{class_id}")
                            
                            all_detections.append({
                                'bbox': (x1, y1, w, h),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_name,
                                'model': model_name
                            })
            except Exception as e:
                print(f"Error in {model_name}: {e}")
        
        return self.remove_duplicates(all_detections)
    
    def remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        if not detections:
            return detections
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        for det in detections:
            keep_flag = True
            for kept in keep:
                iou = self.calculate_iou(det['bbox'], kept['bbox'])
                if iou > iou_threshold:
                    keep_flag = False
                    break
            if keep_flag:
                keep.append(det)
        
        return keep
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_model_info(self) -> Dict:
        return {
            'models_loaded': list(self.models.keys()),
            'total_classes': len(self.class_names),
            'confidence_threshold': self.confidence_threshold
        }