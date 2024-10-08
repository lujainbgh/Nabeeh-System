from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time
from pymongo import MongoClient
from dotenv import load_dotenv
import googlemaps
from datetime import datetime, timedelta
import threading
from gridfs import GridFS
from io import BytesIO
import numpy as np
import os


app = Flask(__name__)

# Load YOLO models
accident_model = YOLO('models/bestlast.pt')
car_count_model = YOLO('models/count.pt')

# Connect to MongoDB
try:
    client = MongoClient(os.environ.get('MONGODB_URI'))    db = client["accident_db1"]
    accidents_collection = db["accidents"]
    fs = GridFS(db)  # Initialize GridFS
    print("Connected to MongoDB successfully!")
except Exception as e:
    print("Could not connect to MongoDB:", e)
    exit()

# Google Maps API setup
gmaps = googlemaps.Client(key=os.environ.get('GOOGLE_MAPS_API_KEY'))
# Global variables
accident_detected = False
accident_start_time = None
estimated_wait_time = 0
current_location = None
remaining_time = 0

# Lock for thread-safe access to shared variables
lock = threading.Lock()

def save_accident_data(frames):
    # Create a directory to store accident videos if it doesn't exist
    if not os.path.exists('accident_videos'):
        os.makedirs('accident_videos')

    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f'accident_{timestamp}.mp4'
    local_video_path = f'accident_videos/{video_filename}'

    # Get the frame size and FPS
    height, width, _ = frames[0].shape
    fps = 20.0  # Assuming 20 fps as mentioned earlier

    # Create a VideoWriter object for local storage
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(local_video_path, fourcc, fps, (width, height))

    # Write frames to the local video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter
    out.release()

    # Save the video to GridFS
    with open(local_video_path, 'rb') as video_file:
        file_id = fs.put(video_file, filename=video_filename)

    # Save accident details to MongoDB
    accident_data = {
        "timestamp": datetime.now(),
        "location": current_location,
        "local_video_path": local_video_path,
        "gridfs_video_id": file_id
    }
    db.accidents.insert_one(accident_data)

    print(f"Accident video saved locally at {local_video_path} and in MongoDB GridFS with id {file_id}")

def process_frames():
    global accident_detected, accident_start_time, estimated_wait_time, current_location, remaining_time
    # url="rtsp://admin:LHFFMW@192.168.8.4:554/H264_STREAM"
    cap = cv2.VideoCapture(0)  # Use 1 for webcam
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Ensure frame is a numpy array
            if isinstance(frame, list):
                frame = np.array(frame)

            # Perform accident detection
            results = accident_model(frame)
            
            with lock:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        conf = box.conf.item()
                        if conf > 0.50:
                            if not accident_detected:
                                accident_detected = True
                                accident_start_time = time.time()
                                current_location = get_current_location()
                                frames = [frame]
                                # Capture 5 seconds of video (assuming 20 fps)
                                for _ in range(99):  # 5 * 20 - 1
                                    ret, frame = cap.read()
                                    if ret:
                                        frames.append(frame)
                                save_accident_data(frames)
                                frames = []
                            
                            # Count cars and estimate wait time
                            car_count = count_cars(frame)
                            estimated_wait_time = estimate_wait_time(car_count)
                            remaining_time = estimated_wait_time * 60
                
                # Reset if accident duration has passed
                if accident_detected and remaining_time <= 0:
                    accident_detected = False
                    accident_start_time = None
                    estimated_wait_time = 0
                    current_location = None
                    remaining_time = 0
            
            # Yield the frame as bytes
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in process_frames: {e}")
            continue


    cap.release()

def count_cars(frame):
    results = car_count_model(frame)
    car_count = 0
    for r in results:
        car_count += len(r.boxes)
    return car_count

def estimate_wait_time(car_count):
    if car_count < 10:
        return 5
    elif car_count < 20:
        return 10
    else:
        return 15

def get_current_location():
    # Implement logic to get current location
    # This is a placeholder
    return {"lat": 24.7136, "lng": 46.6753}  # Coordinates for Riyadh, Saudi Arabia

def get_alternative_routes():
    if not current_location:
        return {}

    # Define routes
    main_road = "Makkah Al Mukarramah Road, Riyadh"
    left_exit = "King Fahad Branch Rd, Riyadh"
    right_exit = "Olaya Street, Riyadh"

    now = datetime.now()

    # Get traffic data for Left Exit
    left_exit_directions = gmaps.directions(
        main_road, left_exit, mode="driving", departure_time=now
    )
    
    # Get traffic data for Right Exit
    right_exit_directions = gmaps.directions(
        main_road, right_exit, mode="driving", departure_time=now
    )

    # Extract real-time traffic durations
    left_exit_time = left_exit_directions[0]['legs'][0].get('duration_in_traffic', {}).get('text', 'N/A') if left_exit_directions else "N/A"
    right_exit_time = right_exit_directions[0]['legs'][0].get('duration_in_traffic', {}).get('text', 'N/A') if right_exit_directions else "N/A"

    # Create routes dictionary
    routes = {
        "left": {
            "name": "King Fahad Branch Rd",
            "name_ar": "طريق الملك فهد",
            "time": left_exit_time
        },
        "right": {
            "name": "Olaya Street",
            "name_ar": "شارع العليا",
            "time": right_exit_time
        }
    }

    return routes

def update_remaining_time():
    global remaining_time
    while True:
        with lock:
            if accident_detected and remaining_time > 0:
                remaining_time -= 1
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    global accident_detected, estimated_wait_time, remaining_time
    routes = get_alternative_routes()
    with lock:
        return jsonify({
            "accident": accident_detected,
            "main_road": {
                "name": "Makkah Al Mukarramah Road",
                "name_ar": "طريق مكة المكرمة",
                "wait_time": f"{remaining_time // 60}:{remaining_time % 60:02d}"
            },
            "alternative_routes": routes
        })

if __name__ == '__main__':
    # Start the thread to update remaining time
    threading.Thread(target=update_remaining_time, daemon=True).start()
    
    # Start the Flask application
    app.run(debug=True, threaded=True)


