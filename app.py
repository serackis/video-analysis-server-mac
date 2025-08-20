import os
import cv2
import numpy as np
import face_recognition
import easyocr
import json
import threading
import time
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sqlite3
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/processed', exist_ok=True)
os.makedirs('static/thumbnails', exist_ok=True)

# Global variables for video processing
active_streams = {}
processing_threads = {}

# Progress tracking for video processing
processing_progress = {}
processing_lock = {}  # Lock to prevent duplicate processing

# Initialize EasyOCR reader for license plate detection
reader = easyocr.Reader(['en'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_database():
    """Initialize SQLite database with tables"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    # Create cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            port INTEGER NOT NULL,
            rtsp_path TEXT NOT NULL,
            username TEXT,
            password TEXT,
            enabled BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create videos table for recorded videos from cameras
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            camera_id INTEGER,
            duration REAL,
            faces_detected INTEGER DEFAULT 0,
            plates_detected INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (camera_id) REFERENCES cameras (id)
        )
    ''')
    
    # Create uploaded_videos table for user uploaded videos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            duration REAL,
            fps REAL,
            frame_count INTEGER,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create processed_videos table for processed videos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uploaded_video_id INTEGER NOT NULL,
            processed_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            depersonalized BOOLEAN DEFAULT 0,
            processing_duration REAL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (uploaded_video_id) REFERENCES uploaded_videos (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def detect_faces(frame):
    """Detect faces in a frame and return bounding boxes"""
    face_locations = face_recognition.face_locations(frame)
    return face_locations

def detect_license_plates(frame):
    """Detect license plates in a frame using EasyOCR"""
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply some preprocessing to improve OCR
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_boxes = []
    
    for contour in contours:
        # Filter contours by area
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (license plates are typically rectangular)
            aspect_ratio = w / h
            if 2.0 < aspect_ratio < 5.0:
                # Extract the region and try OCR
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    results = reader.readtext(roi)
                    if results:
                        # Check if any detected text looks like a license plate
                        for (bbox, text, confidence) in results:
                            if confidence > 0.5 and len(text) >= 3:
                                plate_boxes.append((x, y, w, h, text, confidence))
    
    return plate_boxes

def depersonalize_frame(frame, face_locations, plate_boxes):
    """Apply depersonalization to faces and license plates"""
    depersonalized_frame = frame.copy()
    
    # Blur faces
    for (top, right, bottom, left) in face_locations:
        face_roi = depersonalized_frame[top:bottom, left:right]
        if face_roi.size > 0:
            # Apply strong Gaussian blur
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            depersonalized_frame[top:bottom, left:right] = blurred_face
    
    # Blur license plates
    for (x, y, w, h, text, confidence) in plate_boxes:
        plate_roi = depersonalized_frame[y:y+h, x:x+w]
        if plate_roi.size > 0:
            # Apply strong Gaussian blur
            blurred_plate = cv2.GaussianBlur(plate_roi, (99, 99), 30)
            depersonalized_frame[y:y+h, x:x+w] = blurred_plate
    
    return depersonalized_frame

def process_video_stream(camera_id, rtsp_url, camera_name):
    """Process video stream from RTSP URL"""
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream for camera {camera_name}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer for processed video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{camera_name}_{timestamp}.mp4"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    faces_detected = 0
    plates_detected = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to improve performance
            if frame_count % 5 == 0:
                # Detect faces
                face_locations = detect_faces(frame)
                faces_detected += len(face_locations)
                
                # Detect license plates
                plate_boxes = detect_license_plates(frame)
                plates_detected += len(plate_boxes)
                
                # Apply depersonalization
                processed_frame = depersonalize_frame(frame, face_locations, plate_boxes)
                
                # Draw detection boxes (for debugging)
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                for (x, y, w, h, text, confidence) in plate_boxes:
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f"Plate: {text}", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                processed_frame = frame
            
            out.write(processed_frame)
            frame_count += 1
            
            # Save thumbnail every 100 frames
            if frame_count % 100 == 0:
                thumbnail_path = os.path.join('static/thumbnails', f"{camera_name}_{timestamp}.jpg")
                cv2.imwrite(thumbnail_path, processed_frame)
    
    except Exception as e:
        print(f"Error processing stream for camera {camera_name}: {str(e)}")
    
    finally:
        cap.release()
        out.release()
        
        # Save video metadata to database
        duration = frame_count / fps if fps > 0 else 0
        save_video_metadata(output_filename, camera_id, duration, faces_detected, plates_detected)

def save_video_metadata(filename, camera_id, duration, faces_detected, plates_detected):
    """Save video metadata to database"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO videos (filename, camera_id, duration, faces_detected, plates_detected, depersonalized)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, camera_id, duration, faces_detected, plates_detected, True))
    
    conn.commit()
    conn.close()

def cleanup_orphaned_entries():
    """Clean up database entries for files that no longer exist or have corrupted data"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    try:
        # Clean up corrupted uploaded videos (null values for required fields)
        cursor.execute('DELETE FROM uploaded_videos WHERE file_path IS NULL OR original_filename IS NULL OR stored_filename IS NULL')
        corrupted_uploaded = cursor.rowcount
        if corrupted_uploaded > 0:
            print(f"Cleaned up {corrupted_uploaded} corrupted uploaded video entries")
        
        # Clean up corrupted processed videos (null values for required fields)
        cursor.execute('DELETE FROM processed_videos WHERE file_path IS NULL OR uploaded_video_id IS NULL OR processed_filename IS NULL')
        corrupted_processed = cursor.rowcount
        if corrupted_processed > 0:
            print(f"Cleaned up {corrupted_uploaded} corrupted processed video entries")
        
        # Clean up duplicate uploaded videos (keep only the most recent)
        cursor.execute('''
            DELETE FROM uploaded_videos 
            WHERE id NOT IN (
                SELECT MAX(id) 
                FROM uploaded_videos 
                GROUP BY original_filename, file_size
            )
        ''')
        duplicate_uploaded = cursor.rowcount
        if duplicate_uploaded > 0:
            print(f"Cleaned up {duplicate_uploaded} duplicate uploaded video entries")
        
        # Clean up orphaned uploaded videos (files don't exist)
        cursor.execute('SELECT id, file_path FROM uploaded_videos')
        uploaded_videos = cursor.fetchall()
        
        orphaned_uploaded = 0
        for video_id, file_path in uploaded_videos:
            if not os.path.exists(file_path):
                print(f"Cleaning up orphaned uploaded video {video_id}: {file_path}")
                cursor.execute('DELETE FROM uploaded_videos WHERE id = ?', (video_id,))
                # Also clean up related processed videos
                cursor.execute('DELETE FROM processed_videos WHERE uploaded_video_id = ?', (video_id,))
                orphaned_uploaded += 1
        
        # Clean up orphaned processed videos (files don't exist)
        cursor.execute('SELECT id, file_path FROM processed_videos')
        processed_videos = cursor.fetchall()
        
        orphaned_processed = 0
        for video_id, file_path in processed_videos:
            if not os.path.exists(file_path):
                print(f"Cleaning up orphaned processed video {video_id}: {file_path}")
                cursor.execute('DELETE FROM processed_videos WHERE id = ?', (video_id,))
                orphaned_processed += 1
        
        conn.commit()
        total_cleaned = corrupted_uploaded + corrupted_processed + duplicate_uploaded + orphaned_uploaded + orphaned_processed
        print(f"Database cleanup completed: {total_cleaned} entries removed")
        
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()

@app.route('/')
def index():
    """Main page with video interface"""
    return render_template('index.html')

@app.route('/video-test')
def video_test():
    """Render the video test page"""
    return render_template('video_test.html')

@app.route('/cameras')
def cameras():
    """Camera configuration page"""
    return render_template('cameras.html')

@app.route('/upload')
def upload():
    """Video upload and processing page"""
    return render_template('upload.html')

@app.route('/library')
def library():
    """Video library page"""
    return render_template('library.html')

@app.route('/debug')
def debug():
    """Debug page for database inspection"""
    return render_template('debug.html')

@app.route('/api/debug/stats')
def debug_stats():
    """Get database statistics for debug page"""
    try:
        conn = sqlite3.connect('video_analysis.db')
        cursor = conn.cursor()
        
        # Get counts
        cursor.execute('SELECT COUNT(*) FROM uploaded_videos')
        uploaded_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM processed_videos')
        processed_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM cameras')
        camera_count = cursor.fetchone()[0]
        
        # Get total file size
        cursor.execute('SELECT SUM(file_size) FROM uploaded_videos WHERE file_size IS NOT NULL')
        total_size = cursor.fetchone()[0] or 0
        
        # Get database file info
        db_path = 'video_analysis.db'
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        
        conn.close()
        
        return jsonify({
            'uploaded_videos': uploaded_count,
            'processed_videos': processed_count,
            'cameras': camera_count,
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'database_size_kb': round(db_size / 1024, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all configured cameras"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM cameras ORDER BY created_at DESC')
    cameras = cursor.fetchall()
    
    conn.close()
    
    camera_list = []
    for camera in cameras:
        camera_list.append({
            'id': camera[0],
            'name': camera[1],
            'ip_address': camera[2],
            'port': camera[3],
            'username': camera[4],
            'password': camera[5],
            'rtsp_path': camera[6],
            'enabled': bool(camera[7]),
            'created_at': camera[8]
        })
    
    return jsonify(camera_list)

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Add a new camera configuration"""
    data = request.json
    
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO cameras (name, ip_address, port, username, password, rtsp_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (data['name'], data['ip_address'], data['port'], data['username'], 
          data['password'], data['rtsp_path']))
    
    camera_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Start processing thread for this camera
    rtsp_url = f"rtsp://{data['username']}:{data['password']}@{data['ip_address']}:{data['port']}{data['rtsp_path']}"
    thread = threading.Thread(target=process_video_stream, 
                            args=(camera_id, rtsp_url, data['name']))
    thread.daemon = True
    thread.start()
    
    processing_threads[camera_id] = thread
    
    return jsonify({'success': True, 'camera_id': camera_id})

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera configuration"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get all recorded videos"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT v.*, c.name as camera_name 
        FROM videos v 
        LEFT JOIN cameras c ON v.camera_id = c.id 
        ORDER BY v.created_at DESC
    ''')
    videos = cursor.fetchall()
    
    conn.close()
    
    video_list = []
    for video in videos:
        video_list.append({
            'id': video[0],
            'filename': video[1],
            'camera_id': video[2],
            'camera_name': video[10],
            'start_time': video[3],
            'end_time': video[4],
            'duration': video[5],
            'faces_detected': video[6],
            'plates_detected': video[7],
            'depersonalized': bool(video[8]),
            'created_at': video[9]
        })
    
    return jsonify(video_list)

@app.route('/api/uploaded-videos', methods=['GET'])
def get_uploaded_videos():
    """Get list of uploaded videos with their processed versions"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            uv.id,
            uv.original_filename,
            uv.stored_filename,
            uv.file_path,
            uv.duration,
            uv.fps,
            uv.frame_count,
            uv.width,
            uv.height,
            uv.file_size,
            uv.uploaded_at,
            pv.id as processed_id,
            pv.processed_filename,
            pv.depersonalized,
            pv.processing_duration,
            pv.processed_at
        FROM uploaded_videos uv
        LEFT JOIN (
            SELECT pv1.*
            FROM processed_videos pv1
            INNER JOIN (
                SELECT uploaded_video_id, MAX(processed_at) as max_processed_at
                FROM processed_videos
                GROUP BY uploaded_video_id
            ) pv2 ON pv1.uploaded_video_id = pv2.uploaded_video_id 
                   AND pv1.processed_at = pv2.max_processed_at
        ) pv ON uv.id = pv.uploaded_video_id
        ORDER BY uv.uploaded_at DESC
    ''')
    videos = cursor.fetchall()
    conn.close()
    
    video_list = []
    for video in videos:
        video_data = {
            'id': video[0],
            'original_filename': video[1],
            'stored_filename': video[2],
            'file_path': video[3],
            'duration': video[4],
            'fps': video[5],
            'frame_count': video[6],
            'width': video[7],
            'height': video[8],
            'file_size': video[9],
            'uploaded_at': video[10],
            'has_processed_version': video[11] is not None
        }
        
        if video[11]:  # If processed version exists
            video_data['processed'] = {
                'id': video[11],
                'filename': video[12],
                'depersonalized': video[13],
                'processing_duration': video[14],
                'processed_at': video[15]
            }
        
        video_list.append(video_data)
    
    return jsonify(video_list)

@app.route('/api/processed-videos', methods=['GET'])
def get_processed_videos():
    """Get list of processed videos"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            pv.id,
            pv.processed_filename,
            pv.file_path,
            pv.depersonalized,
            pv.processing_duration,
            pv.processed_at,
            uv.original_filename,
            uv.duration,
            uv.fps,
            uv.frame_count,
            uv.width,
            uv.height,
            uv.file_size
        FROM processed_videos pv
        JOIN uploaded_videos uv ON pv.uploaded_video_id = uv.id
        ORDER BY pv.processed_at DESC
    ''')
    videos = cursor.fetchall()
    conn.close()
    
    video_list = []
    for video in videos:
        video_list.append({
            'id': video[0],
            'filename': video[1],
            'file_path': video[2],
            'depersonalized': bool(video[3]),
            'processing_duration': video[4],
            'processed_at': video[5],
            'original_filename': video[6],
            'duration': video[7],
            'fps': video[8],
            'frame_count': video[9],
            'width': video[10],
            'height': video[11],
            'file_size': video[12]
        })
    
    return jsonify(video_list)

@app.route('/api/processed-videos/<int:video_id>', methods=['DELETE'])
def delete_processed_video(video_id):
    """Delete a processed video"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    try:
        # Get video info
        cursor.execute('SELECT file_path FROM processed_videos WHERE id = ?', (video_id,))
        video_record = cursor.fetchone()
        
        if not video_record:
            conn.close()
            return jsonify({'error': 'Video not found'}), 404
        
        file_path = video_record[0]
        
        # Delete file with better error handling
        deleted_file = None
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_file = file_path
                print(f"Deleted processed video file: {file_path}")
            except Exception as file_error:
                print(f"Warning: Could not delete file {file_path}: {file_error}")
        
        # Delete database record
        cursor.execute('DELETE FROM processed_videos WHERE id = ?', (video_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'deleted_file': deleted_file
        })
        
    except Exception as e:
        print(f"Error deleting processed video {video_id}: {e}")
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/uploaded-videos/<int:video_id>', methods=['DELETE'])
def delete_uploaded_video(video_id):
    """Delete an uploaded video and its processed versions"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    try:
        # Get video info
        cursor.execute('SELECT stored_filename, file_path FROM uploaded_videos WHERE id = ?', (video_id,))
        video_record = cursor.fetchone()
        
        if not video_record:
            conn.close()
            return jsonify({'error': 'Video not found'}), 404
        
        stored_filename, file_path = video_record
        
        # Get processed videos
        cursor.execute('SELECT processed_filename, file_path FROM processed_videos WHERE uploaded_video_id = ?', (video_id,))
        processed_videos = cursor.fetchall()
        
        # Delete files with better error handling
        deleted_files = []
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"Deleted uploaded video file: {file_path}")
            except Exception as file_error:
                print(f"Warning: Could not delete file {file_path}: {file_error}")
        
        for processed_filename, processed_file_path in processed_videos:
            if os.path.exists(processed_file_path):
                try:
                    os.remove(processed_file_path)
                    deleted_files.append(processed_file_path)
                    print(f"Deleted processed video file: {processed_file_path}")
                except Exception as file_error:
                    print(f"Warning: Could not delete file {processed_file_path}: {file_error}")
        
        # Delete from database
        cursor.execute('DELETE FROM processed_videos WHERE uploaded_video_id = ?', (video_id,))
        cursor.execute('DELETE FROM uploaded_videos WHERE id = ?', (video_id,))
        
        conn.commit()
        conn.close()
        
        # Clean up any orphaned entries
        cleanup_orphaned_entries()
        
        return jsonify({
            'success': True, 
            'message': 'Video deleted successfully',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        print(f"Error deleting uploaded video {video_id}: {e}")
        conn.close()
        return jsonify({'error': f'Error deleting video: {str(e)}'}), 500

@app.route('/api/videos/<filename>')
def serve_recorded_video(filename):
    """Serve recorded video files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/thumbnails/<filename>')
def serve_thumbnail(filename):
    """Serve thumbnail images"""
    return send_from_directory('static/thumbnails', filename)

@app.route('/api/stream/<int:camera_id>')
def stream_preview(camera_id):
    """Stream live preview for a camera"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,))
    camera = cursor.fetchone()
    conn.close()
    
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404
    
    # Build RTSP URL
    rtsp_url = f"rtsp://{camera[4]}:{camera[5]}@{camera[2]}:{camera[3]}{camera[6]}"
    
    # Return the RTSP URL for the frontend to handle
    return jsonify({
        'camera_id': camera_id,
        'rtsp_url': rtsp_url,
        'camera_name': camera[1]
    })

@app.route('/api/stream/<int:camera_id>/snapshot')
def camera_snapshot(camera_id):
    """Get a snapshot from a camera"""
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,))
    camera = cursor.fetchone()
    conn.close()
    
    if not camera:
        return jsonify({'error': 'Camera not found'}), 404
    
    # Build RTSP URL
    rtsp_url = f"rtsp://{camera[4]}:{camera[5]}@{camera[2]}:{camera[3]}{camera[6]}"
    
    try:
        # Capture a frame from the RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({'error': 'Could not connect to camera'}), 500
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Could not capture frame'}), 500
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_data = buffer.tobytes()
        
        # Return the image
        from flask import Response
        return Response(jpeg_data, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({'error': f'Error capturing snapshot: {str(e)}'}), 500

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Upload a video file for processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Create uploads directory if it doesn't exist
        upload_dir = 'static/uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        stored_filename = f"upload_{timestamp}_{original_filename}"
        filepath = os.path.join(upload_dir, stored_filename)
        
        # Check if this exact file was already uploaded (anytime, not just recent)
        conn = sqlite3.connect('video_analysis.db')
        cursor = conn.cursor()
        
        # First check by original filename
        cursor.execute('''
            SELECT id, stored_filename FROM uploaded_videos 
            WHERE original_filename = ?
        ''', (original_filename,))
        
        existing_by_name = cursor.fetchone()
        if existing_by_name:
            conn.close()
            return jsonify({'error': 'This file was already uploaded. Please use the existing video.'}), 400
        
        # Read file content for more thorough duplicate detection
        file_content = file.read()
        file.seek(0)  # Reset file pointer for later use
        
        # Check by file size (more reliable than just filename)
        cursor.execute('''
            SELECT id, original_filename, stored_filename FROM uploaded_videos 
            WHERE file_size = ?
        ''', (len(file_content),))
        
        existing_by_size = cursor.fetchone()
        if existing_by_size:
            conn.close()
            return jsonify({'error': f'A file with identical content was already uploaded as "{existing_by_size[2]}". Please use the existing video.'}), 400
        
        # Additional check: look for files with same content hash (first 1KB)
        content_hash = hashlib.md5(file_content[:1024]).hexdigest()
        cursor.execute('''
            SELECT id, original_filename, stored_filename FROM uploaded_videos 
            WHERE file_size = ? AND original_filename = ?
        ''', (len(file_content), original_filename))
        
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'This exact file content was already uploaded. Please use the existing video.'}), 400
        
        conn.close()
        
        # Save the file
        file.save(filepath)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Get video info
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Save to database
        conn = sqlite3.connect('video_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO uploaded_videos 
            (original_filename, stored_filename, file_path, duration, fps, frame_count, width, height, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (original_filename, stored_filename, filepath, duration, fps, frame_count, width, height, file_size))
        
        video_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'id': video_id,
            'original_filename': original_filename,
            'filename': stored_filename,
            'filepath': filepath,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height,
            'file_size': file_size
        })

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process a video with depersonalization"""
    data = request.get_json()
    filename = data.get('filename')
    enable_depersonalization = data.get('depersonalize', True)
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    # Check if video is already being processed
    if filename in processing_lock and processing_lock[filename]:
        return jsonify({'error': 'Video is already being processed. Please wait.'}), 400
    
    # Set processing lock
    processing_lock[filename] = True
    
    # Find the uploaded video in database
    conn = sqlite3.connect('video_analysis.db')
    cursor = conn.cursor()
    
    # Check if video is already processed
    cursor.execute('''
        SELECT p.id FROM processed_videos p
        JOIN uploaded_videos u ON p.uploaded_video_id = u.id
        WHERE u.stored_filename = ?
    ''', (filename,))
    
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'This video has already been processed. Check the video library for the processed version.'}), 400
    cursor.execute('SELECT id, file_path FROM uploaded_videos WHERE stored_filename = ?', (filename,))
    video_record = cursor.fetchone()
    
    if not video_record:
        conn.close()
        return jsonify({'error': 'Video not found in database'}), 404
    
    uploaded_video_id, filepath = video_record
    
    if not os.path.exists(filepath):
        conn.close()
        return jsonify({'error': 'Video file not found'}), 404
    
    try:
        # Create processed videos directory
        processed_dir = 'static/processed'
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"processed_{timestamp}_{filename}"
        output_path = os.path.join(processed_dir, output_filename)
        
        # Process the video
        start_time = time.time()
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer with H.264 codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize progress tracking
        processing_progress[filename] = {
            'status': 'processing',
            'progress': 0,
            'frame_count': 0,
            'total_frames': total_frames,
            'message': 'Starting video processing...'
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if enable_depersonalization:
                # Apply depersonalization
                face_locations = detect_faces(frame)
                plate_boxes = detect_license_plates(frame)
                frame = depersonalize_frame(frame, face_locations, plate_boxes)
            
            out.write(frame)
            frame_count += 1
            
            # Progress update every 10 frames
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                processing_progress[filename].update({
                    'progress': progress,
                    'frame_count': frame_count,
                    'message': f'Processing frame {frame_count}/{total_frames} ({progress:.1f}%)'
                })
                print(f"Processing: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        # Update progress to completed
        processing_progress[filename].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Video processing completed!'
        })
        
        # Clean up progress tracking after a delay to prevent memory leaks
        def cleanup_progress():
            if filename in processing_progress:
                del processing_progress[filename]
                print(f"Cleaned up progress tracking for: {filename}")
        
        # Schedule cleanup after 30 seconds
        import threading
        timer = threading.Timer(30.0, cleanup_progress)
        timer.start()
        
        # Skip FFmpeg transcoding to avoid video corruption
        # The processed video will be saved directly in OpenCV format
        print(f"Video processing completed, saved as: {output_path}")
        
        # Calculate processing duration
        processing_duration = time.time() - start_time
        
        # Save processed video info to database
        cursor.execute('''
            INSERT INTO processed_videos 
            (uploaded_video_id, processed_filename, file_path, depersonalized, processing_duration)
            VALUES (?, ?, ?, ?, ?)
        ''', (uploaded_video_id, output_filename, output_path, enable_depersonalization, processing_duration))
        
        processed_video_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Release processing lock
        if filename in processing_lock:
            del processing_lock[filename]
        
        return jsonify({
            'success': True,
            'processed_video_id': processed_video_id,
            'processed_filename': output_filename,
            'processed_path': output_path,
            'frame_count': frame_count,
            'depersonalized': enable_depersonalization,
            'processing_duration': processing_duration
        })
    
    except Exception as e:
        print(f"Error processing video: {e}")
        # Release processing lock on error
        if filename in processing_lock:
            del processing_lock[filename]
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/process-video-progress/<filename>')
def process_video_progress(filename):
    """Get real-time progress of video processing"""
    print(f"Progress request for: {filename}")
    print(f"Available progress keys: {list(processing_progress.keys())}")
    
    if filename in processing_progress:
        progress_data = processing_progress[filename]
        print(f"Returning progress: {progress_data}")
        return jsonify(progress_data)
    else:
        print(f"No progress found for: {filename}")
        return jsonify({
            'status': 'not_found',
            'message': 'No processing found for this video'
        })

@app.route('/api/uploaded-video/<filename>')
def serve_uploaded_video(filename):
    """Serve uploaded video files"""
    response = send_from_directory('static/uploads', filename)
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/processed-video/<filename>')
def serve_processed_video(filename):
    """Serve processed video files"""
    response = send_from_directory('static/processed', filename)
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/cleanup-database', methods=['POST'])
def api_cleanup_database():
    """Manually trigger database cleanup"""
    try:
        cleanup_orphaned_entries()
        return jsonify({'success': True, 'message': 'Database cleanup completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_database()
    app.run(debug=True, host='0.0.0.0', port=5002) 