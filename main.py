from datetime import datetime
from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
from service import PostureDetectionApp
from load_video import VideoProcessor
from load_model import InferenceModel, ModelLoadError
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to track model initialization
model_initialized = False
model_instance = None

@app.route("/")
def hello_world():
    """Return a message showing server is running with timestamp."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "status": "success",
        "message": "Server is running!",
        "timestamp": current_time,
        "server": "Flask on Google Cloud Run"
    }

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max-limit

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create necessary folders if they don't exist
for folder in [UPLOAD_FOLDER, 'recordings']:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_up_files(*files):
    """Helper function to clean up files"""
    for file in files:
        if file and os.path.exists(file):
            try:
                os.remove(file)
                logger.debug(f"Cleaned up file: {file}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file}: {str(e)}")

def initialize_model():
    """Initialize the model if not already initialized"""
    global model_initialized, model_instance
    if not model_initialized:
        try:
            logger.info("Initializing model...")
            model_instance = InferenceModel('small640.pt')
            model_initialized = True
            logger.info("Model initialized successfully")
        except ModelLoadError as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model initialization: {str(e)}")
            raise

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Try to initialize model if not already initialized
        if not model_initialized:
            initialize_model()
        return jsonify({
            "status": "healthy",
            "message": "Service is running",
            "model_status": "initialized" if model_initialized else "not initialized"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "message": f"Service error: {str(e)}",
            "model_status": "failed to initialize"
        }), 500

@app.route('/analyze-posture', methods=['POST'])
def analyze_posture():
    filepath = None
    processed_path = None
    
    try:
        # Ensure model is initialized
        if not model_initialized:
            initialize_model()
            
        # Check if video file is present in request
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        
        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.debug(f"File saved to: {filepath}")
        
        # Initialize video processor
        video_processor = VideoProcessor()
        
        try:
            # Get initial video info
            video_info = video_processor.get_video_info(filepath)
            logger.debug(f"Video info: {video_info}")
            
            # Process video (handles both conversion and speed adjustment)
            processed_path = video_processor.process_video(filepath)
            logger.debug(f"Processed video path: {processed_path}")
            
            # Process the video with posture detection
            detector = PostureDetectionApp(processed_path)
            detector.process_video()
            
            # Get the latest processed video and stats
            recordings_dir = "recordings"
            processed_files = [f for f in os.listdir(recordings_dir) if f.endswith('.mp4')]
            stats_files = [f for f in os.listdir(recordings_dir) if f.endswith('.txt')]
            
            logger.debug(f"Found processed files: {processed_files}")
            logger.debug(f"Found stats files: {stats_files}")
            
            if not processed_files or not stats_files:
                raise Exception("No output files generated")
            
            latest_video = sorted(processed_files)[-1]
            latest_stats = sorted(stats_files)[-1]
            
            logger.debug(f"Latest video: {latest_video}")
            logger.debug(f"Latest stats: {latest_stats}")
            
            # Read statistics
            with open(os.path.join(recordings_dir, latest_stats), 'r') as f:
                stats_content = f.read()
            
            # Prepare response
            response = {
                "message": "Video processed successfully",
                "original_video_info": {
                    "duration": video_info['duration'],
                    "fps": video_info['fps'],
                    "frame_count": video_info['frame_count']
                },
                "processed_video_path": f"/download-video/{latest_video}",
                "statistics": stats_content
            }
            
            # Clean up temporary files
            clean_up_files(filepath)
            if processed_path and processed_path != filepath:
                clean_up_files(processed_path)
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise Exception(f"Video processing failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in analyze_posture: {str(e)}")
        # Clean up any files in case of error
        clean_up_files(filepath, processed_path)
        return jsonify({"error": str(e)}), 500

@app.route('/download-video/<filename>', methods=['GET'])
def download_video(filename):
    try:
        video_path = os.path.join('recordings', filename)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {filename}")
            
        return send_file(
            video_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({"error": str(e)}), 404
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
