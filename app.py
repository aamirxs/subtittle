from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import whisper
import tempfile
from datetime import timedelta
import json
from pathlib import Path
import ffmpeg
from werkzeug.utils import secure_filename
import uuid
from threading import Thread
import math
from threading import Timer

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2000MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['STATIC_FOLDER'] = 'static'
app.static_folder = app.config['STATIC_FOLDER']
app.secret_key = 'your-secret-key-here'  # Required for session

# Global task tracking
tasks = {}

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean'
}

# Create upload directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def format_ass_time(seconds):
    """Convert seconds to ASS time format (h:mm:ss.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    centiseconds = int((seconds % 1) * 100)
    seconds = int(seconds)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

def get_video_metadata(video_path):
    """Get video metadata using ffmpeg."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return {
            'width': int(video_info['width']),
            'height': int(video_info['height']),
            'duration': float(probe['format']['duration']),
            'format': probe['format']['format_name']
        }
    except Exception as e:
        return None

def get_time_estimate(file_size):
    """
    Estimate processing time based on file size.
    Uses a basic calculation assuming ~100MB per minute processing speed
    with some overhead for initialization.
    """
    # Base time for initialization (in minutes)
    base_time = 0.5
    
    # Convert file size to MB
    size_in_mb = file_size / (1024 * 1024)
    
    # Calculate processing time
    # Assuming ~100MB per minute processing speed
    processing_time = (size_in_mb / 100) + base_time
    
    # Round up to nearest 0.5 minute
    processing_time = math.ceil(processing_time * 2) / 2
    
    return processing_time

@app.route('/')
def index():
    return render_template('index.html', languages=SUPPORTED_LANGUAGES)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
            
        language = request.form.get('language', 'en')
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({'error': 'Language not supported'}), 400

        # Create unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Create task
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'file_path': file_path,
            'original_filename': filename,
            'output_files': {}
        }
        
        # Start processing in background
        thread = Thread(target=process_video_task, args=(task_id, file_path, language))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/estimate_time', methods=['POST'])
def estimate_time():
    if 'file_size' not in request.json:
        return jsonify({'error': 'No file size provided'}), 400
        
    file_size = request.json['file_size']
    estimate = get_time_estimate(file_size)
    
    # Convert to a more readable format
    if estimate < 1:
        time_str = f"{int(estimate * 60)} seconds"
    elif estimate == 1:
        time_str = "1 minute"
    elif estimate.is_integer():
        time_str = f"{int(estimate)} minutes"
    else:
        minutes = int(estimate)
        seconds = int((estimate - minutes) * 60)
        time_str = f"{minutes} minutes {seconds} seconds"
    
    return jsonify({
        'estimate_minutes': estimate,
        'estimate_readable': time_str
    })

def process_video_task(task_id, video_path, language):
    """Background task to process video and generate subtitles."""
    task = tasks[task_id]
    try:
        # Load the model
        model = whisper.load_model("base")
        
        # Update progress
        task['progress'] = 10
        
        # Transcribe
        result = model.transcribe(video_path, language=language)
        
        # Update progress
        task['progress'] = 50
        
        # Get original filename without extension
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(app.root_path, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate subtitles in different formats
        formats = {
            'srt': generate_srt,
            'vtt': generate_vtt,
            'ass': generate_ass
        }
        
        for format_name, generator_func in formats.items():
            output_path = os.path.join(output_dir, f"{base_name}.{format_name}")
            generator_func(result['segments'], output_path)
            task['output_files'][format_name] = output_path
        
        # Update task status
        task['status'] = 'completed'
        task['progress'] = 100
        
        # Clean up input video
        try:
            os.remove(video_path)
        except:
            pass
            
    except Exception as e:
        task['status'] = 'error'
        task['error'] = str(e)
        # Clean up on error
        try:
            os.remove(video_path)
        except:
            pass

@app.route('/download/<format>/<task_id>')
def download_subtitle(format, task_id):
    try:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
            
        task = tasks[task_id]
        if task['status'] != 'completed':
            return jsonify({'error': 'Subtitles not ready yet'}), 400
            
        if format not in task['output_files']:
            return jsonify({'error': f'No {format} subtitle available'}), 404
            
        subtitle_path = task['output_files'][format]
        if not os.path.exists(subtitle_path):
            return jsonify({'error': 'Subtitle file not found'}), 404
            
        # Get original filename without extension
        original_name = os.path.splitext(task['original_filename'])[0]
        
        # Create download filename
        download_name = f"{original_name}.{format}"
        
        # Schedule cleanup after a delay (5 minutes)
        timer = Timer(300, cleanup_files, args=[task_id])
        timer.daemon = True
        timer.start()
        
        return send_file(
            subtitle_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/plain'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def cleanup_files(task_id):
    """Clean up all files associated with a task after a delay."""
    try:
        task = tasks.get(task_id)
        if not task:
            return
            
        # Clean up video file
        if 'file_path' in task and os.path.exists(task['file_path']):
            try:
                os.remove(task['file_path'])
            except Exception as e:
                print(f"Error removing video file: {e}")
                
        # Clean up subtitle files
        if 'output_files' in task:
            for format_type, file_path in task['output_files'].items():
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing {format_type} subtitle file: {e}")
                        
        # Remove task from tasks dictionary
        tasks.pop(task_id, None)
        
    except Exception as e:
        print(f"Error in cleanup: {e}")

def generate_srt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = str(timedelta(seconds=segment['start'])).replace('.', ',')[:12]
            end_time = str(timedelta(seconds=segment['end'])).replace('.', ',')[:12]
            f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n")

def generate_vtt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, segment in enumerate(segments, 1):
            start_time = str(timedelta(seconds=segment['start']))[:11].replace(',', '.')
            end_time = str(timedelta(seconds=segment['end']))[:11].replace(',', '.')
            f.write(f"{start_time} --> {end_time}\n{segment['text'].strip()}\n\n")

def generate_ass(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write ASS header
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for segment in segments:
            start_time = format_ass_time(segment['start'])
            end_time = format_ass_time(segment['end'])
            text = segment['text'].strip().replace('\n', '\\N')
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    """Get the status of a background task."""
    task = tasks.get(task_id)
    if task is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)

if __name__ == '__main__':
    app.run(debug=True)
