import cv2
import time
import os
from datetime import datetime
from load_model import InferenceModel
from load_video import VideoProcessor

class PostureDetectionApp:
    def __init__(self, video_path):
        print("Initializing Posture Detection System...")
        
        # Initialize model configurations
        self.model_name = 'small640.pt'
        self.inference_model = InferenceModel(self.model_name)
        self.video_processor = VideoProcessor()
        
        # Video and recording variables
        self.video_path = video_path
        self.output_folder = "recordings"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        # Statistics variables
        self.good_posture_frames = 0
        self.bad_posture_frames = 0
        self.total_frames = 0
        self.bad_posture_counter = 0
        self.BAD_POSTURE_THRESHOLD = 200
        self.is_bad_posture_active = False
        self.posture_timestamps = []  # To store posture change timestamps
        self.current_posture = None   # To track current posture state


    def format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
    
    def process_video(self):
        print(f"Starting video processing: {self.video_path}")
        start_time = time.time()

        # Get video info and process if needed
        video_info = self.video_processor.get_video_info(self.video_path)
        print(f"Original video duration: {video_info['duration']}")
        
        if video_info['is_long']:
            print("Video longer than 1 minute, processing at 4x speed...")
            self.video_path = self.video_processor.process_video(self.video_path)
            print("Speed processing completed")

        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_folder}/recording_{timestamp}.mp4"
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (frame_width, frame_height)
        )

        print("\nProcessing frames...")

        frame_count = 0
        bad_posture_counter = 0
        is_bad_posture_active = False
        current_posture = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            frame_count += 1
            current_time = frame_count / fps  # Calculate current time in seconds

            if frame_count % 30 == 0:  # Update progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

            # Model inference
            results = self.inference_model.predict(frame)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_name, confidence = \
                self.inference_model.get_results(results)

            if bbox_x1 is not None:
                self.total_frames += 1
                # Determine current posture
                new_posture = "Bad" if class_name == 1 else "Good"
            
                # Check if posture has changed
                if new_posture != current_posture:
                    timestamp = self.format_timestamp(current_time)
                    self.posture_timestamps.append({
                        'time': timestamp,
                        'posture': new_posture
                    })
                    current_posture = new_posture
                # Track bad posture
                if class_name == 1:  # Bad posture
                    bad_posture_counter += 1
                    if bad_posture_counter >= self.BAD_POSTURE_THRESHOLD:
                        is_bad_posture_active = True
                        self.bad_posture_frames += 1
                        color = (0, 0, 255)  # Red
                        status_text = f"Bad Posture! ({bad_posture_counter})"
                    else:
                        color = (255, 165, 0)  # Orange (warning)
                        status_text = f"Warning ({bad_posture_counter}/{self.BAD_POSTURE_THRESHOLD})"
                else:  # Good posture
                    bad_posture_counter = 0
                    is_bad_posture_active = False
                    self.good_posture_frames += 1
                    color = (0, 255, 0)  # Green
                    status_text = "Good Posture"

                        # After model inference and getting results

                # Draw bounding box and labels
                cv2.rectangle(frame, 
                            (bbox_x1, bbox_y1), 
                            (bbox_x2, bbox_y2), 
                            color, 
                            2)
                
                # Add status text
                label = f"{status_text} ({confidence:.2f})"
                cv2.putText(frame, 
                          label,
                          (bbox_x1, bbox_y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          color,
                          2)

                # Print status to console (optional)
                if frame_count % 30 == 0:  # Update every 30 frames
                    print(f"Current status: {status_text}")

            video_writer.write(frame)

        # Cleanup
        cap.release()
        video_writer.release()

        # Calculate processing time and save stats
        end_time = time.time()
        processing_time = end_time - start_time
        self.save_session_stats(processing_time)

        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
        print(f"Statistics saved to: {self.output_folder}/stats_{timestamp}.txt")


    def save_session_stats(self, processing_time):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_filename = f"{self.output_folder}/stats_{timestamp}.txt"
        
        # Get video FPS
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Convert frames to time
        def frames_to_time(frame_count):
            total_seconds = frame_count / fps
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        
        # Calculate percentages
        good_posture_percentage = (self.good_posture_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        bad_posture_percentage = (self.bad_posture_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        
        # Convert frame counts to time
        total_time = frames_to_time(self.total_frames)
        good_posture_time = frames_to_time(self.good_posture_frames)
        bad_posture_time = frames_to_time(self.bad_posture_frames)
        
        with open(stats_filename, 'w') as f:
            f.write(f"Session Statistics\n")
            f.write(f"================\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Video Source: {self.video_path}\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"\nTime Analysis:\n")
            f.write(f"Total Duration: {total_time}\n")
            f.write(f"Good Posture Duration: {good_posture_time} ({good_posture_percentage:.1f}%)\n")
            f.write(f"Bad Posture Duration: {bad_posture_time} ({bad_posture_percentage:.1f}%)\n")
            f.write(f"\nPosture Timeline:\n")
            f.write(f"================\n")
            for i, entry in enumerate(self.posture_timestamps):
                f.write(f"{entry['time']}: {entry['posture']} posture begins\n")
                
                # Calculate duration if not the last entry
                if i < len(self.posture_timestamps) - 1:
                    next_time = self.posture_timestamps[i + 1]['time']
                    f.write(f"Duration: {next_time}\n")
                
            f.write(f"\nDetailed Statistics:\n")
            f.write(f"==================\n")



if __name__ == '__main__':
    # Set your video path here
    VIDEO_PATH = "/Volumes/Extreme Pro/Personal/Web3HackFest_Functions/SittingPosture/videos/video9.mp4"
    
    # Initialize and run
    app = PostureDetectionApp(VIDEO_PATH)
    app.process_video()