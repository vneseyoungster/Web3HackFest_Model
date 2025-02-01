# Web3HackFest_Model

# Posture Detection Web Service

A Flask-based web service that performs posture detection on uploaded videos using YOLOv5. The service processes videos, analyzes posture, and provides detailed statistics and processed output videos with annotations.

## Features

- Video upload and processing
- Real-time posture detection using YOLOv5
- Video frame analysis and annotation
- Statistical analysis of posture data
- REST API endpoints for integration
- Dockerized deployment support

## Prerequisites

- Python 3.11
- Docker (for containerized deployment)
- GPU support (optional, for faster processing)

## Installation

### Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r myenv/requirements.txt
```

4. Download the model file:
- Ensure `small640.pt` is present in the project directory
- This is the YOLOv5 model file required for posture detection

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t posture-detection-service .
```

2. Run the container:
```bash
docker run -p 8080:8080 posture-detection-service
```

## API Endpoints

### 1. Health Check
```
GET /health
```
Checks if the service and model are properly initialized.

### 2. Analyze Posture
```
POST /analyze-posture
```
Upload a video file for posture analysis.
- Request: Multipart form data with 'video' file
- Supported format: MP4
- Max file size: 16MB

### 3. Download Processed Video
```
GET /download-video/<filename>
```
Download the processed video with posture annotations.

## Environment Variables

- `PORT`: Server port (default: 8080)
- `PYTHONPATH`: Application path
- `PYTHONUNBUFFERED`: Python output buffering

## Project Structure

```
.
├── myenv/
│   ├── main.py              # Main Flask application
│   ├── load_model.py        # Model loading utilities
│   ├── model.py             # Model implementation
│   ├── service.py           # Posture detection service
│   ├── load_video.py        # Video processing utilities
│   └── requirements.txt     # Python dependencies
├── Dockerfile               # Docker configuration
├── uploads/                 # Temporary storage for uploads
└── recordings/             # Output directory for processed videos
```

## Technical Details

- Built with Flask and Flask-CORS
- Uses YOLOv5 for posture detection
- OpenCV for video processing
- Gunicorn as WSGI HTTP Server
- Containerized with Docker
- Supports concurrent processing with multiple workers

## Error Handling

The service includes comprehensive error handling for:
- Invalid file formats
- File size limits
- Processing errors
- Model initialization issues
- File system operations

## Performance Considerations

- Uses Gunicorn with 8 threads for concurrent processing
- Implements automatic cleanup of temporary files
- Configurable timeouts for long-running processes
- Supports both CPU and GPU inference

## Logging

The service implements detailed logging for:
- Request handling
- Video processing stages
- Error tracking
- Model initialization
- File operations

## Security Considerations

- Implements secure file handling
- Validates file types and sizes
- Sanitizes file names
- Uses secure file operations

## License

MIT License
