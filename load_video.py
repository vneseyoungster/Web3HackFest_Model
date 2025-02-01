import cv2
import numpy as np
from datetime import timedelta

class VideoProcessor:
    def __init__(self):
        # Define duration thresholds in seconds
        self.one_minute = 60
        self.two_minutes = 120
        self.five_minutes = 300

    def get_speed_multiplier(self, duration):
        """
        Determine speed multiplier based on video duration
        """
        if duration <= self.one_minute:
            return 1  # Original speed
        elif duration <= self.two_minutes:
            return 2  # 2x speed
        elif duration <= self.five_minutes:
            return 3  # 3x speed
        else:
            return 4  # 4x speed

    def process_video(self, video_path):
        """
        Process the video with dynamic speed adjustment based on duration
        Returns: processed video path or original path if no processing needed
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps  # duration in seconds
        
        # Determine speed multiplier based on duration
        speed_multiplier = self.get_speed_multiplier(duration)
        
        # If no speed adjustment needed, return original
        if speed_multiplier == 1:
            cap.release()
            return video_path
            
        # Process videos that need speed adjustment
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output path
        output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'
        
        # Initialize video writer with original fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_counter = 0
        
        print(f"Processing video at {speed_multiplier}x speed...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frames based on dynamic speed multiplier
            if frame_counter % speed_multiplier == 0:
                out.write(frame)
            
            frame_counter += 1
        
        cap.release()
        out.release()
        
        return output_path

    def get_video_info(self, video_path):
        """
        Get video information with dynamic speed multiplier
        Returns: dict containing fps, frame_count, duration, and processing info
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Get appropriate speed multiplier
        speed_multiplier = self.get_speed_multiplier(duration)
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': str(timedelta(seconds=int(duration))),
            'is_long': duration > self.one_minute,
            'speed_multiplier': speed_multiplier,
            'processed_duration': str(timedelta(seconds=int(duration/speed_multiplier)))
        }