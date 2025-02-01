# app_models/model.py
import cv2
from load_model import InferenceModel

class Model:
    def __init__(self, model_name):
        super().__init__()
        # Basic properties
        self.model_name = model_name
        self.inference_model = InferenceModel(model_name)
        self.prev_frame_time = 0
        self.IMAGE_BOX_SIZE = 600
        self.flag_is_camera_thread_running = True

        # Camera properties
        self.camera = None
        self.work_thread_camera = None
        
        # Frame properties
        self.box_color = (251, 255, 12)
        self.box_thickness = 2
        self.text_color_conf = (251, 255, 12)
        self.text_color_class = (251, 255, 12)
        self.text_color_bg = (0, 0, 0)
        self.text_thickness = 1
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_font_scale = 0.5
        
        # Frame orientation
        self.frame_rotation = 0
        self.frame_orientation_vertical = 0
        self.frame_orientation_horizontal = 0
        self.bbox_mode = 1

    def process_frame(self, frame):
        results = self.inference_model.predict(frame)
        return self.draw_detections(frame, results)

    def draw_detections(self, frame, results):
        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                bbox = detection[:4].cpu().numpy()
                conf = float(detection[4])
                cls = int(detection[5])
                
                # Draw bounding box
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            self.box_color, 
                            self.box_thickness)
                
                # Draw label
                label = f"{'Good' if cls == 0 else 'Bad'} Posture ({conf:.2f})"
                cv2.putText(frame, 
                          label, 
                          (int(bbox[0]), int(bbox[1] - 10)), 
                          self.text_font, 
                          self.text_font_scale, 
                          self.text_color_class, 
                          self.text_thickness)
        return frame