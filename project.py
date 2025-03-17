import cv2
import numpy as np
import os
import time

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the cascade classifier for face detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Initialize smile detection
smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Initialize eye detection
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Additional cascade classifiers for better detection
# Upper body detection can help with posture analysis
upper_body_cascade_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
upper_body_cascade = cv2.CascadeClassifier(upper_body_cascade_path)

# Load classifier for closed eyes - helps detect blinks and expressions
eye_closed_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
eye_closed_cascade = cv2.CascadeClassifier(eye_closed_cascade_path)

# History of emotions for temporal smoothing
emotion_history = []
MAX_HISTORY = 10

# Function to apply temporal smoothing to emotions
def smooth_emotion(current_emotion):
    global emotion_history
    
    # Add current emotion to history
    emotion_history.append(current_emotion)
    
    # Keep history at maximum length
    if len(emotion_history) > MAX_HISTORY:
        emotion_history.pop(0)
    
    # Count occurrences of each emotion
    emotion_counts = {}
    for emotion in emotion_history:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    
    # Find most common emotion
    most_common_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # If the most common emotion is very dominant, return it
    if emotion_counts[most_common_emotion] >= len(emotion_history) * 0.6:
        return most_common_emotion
    # Otherwise, return the current emotion (allows for quicker transitions)
    else:
        return current_emotion

# Function to calculate facial symmetry
def calculate_symmetry(face_roi):
    height, width = face_roi.shape
    
    # Split face into left and right halves
    left_half = face_roi[:, :width//2]
    right_half = face_roi[:, width//2:]
    
    # Flip right half to match left half orientation
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Resize to ensure both halves are the same size
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate absolute difference between halves
    diff = cv2.absdiff(left_half, right_half_flipped)
    
    # Calculate symmetry score (lower means more symmetrical)
    symmetry_score = np.mean(diff)
    
    return symmetry_score

# Function to detect emotions with improved accuracy
def detect_emotions(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram for better feature detection
    gray_eq = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    
    # Detect faces with improved parameters
    faces = face_cascade.detectMultiScale(
        blurred,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face region with some margin
        y_margin = int(0.1 * h)
        x_margin = int(0.05 * w)
        
        # Ensure coordinates are within frame boundaries
        y1 = max(0, y - y_margin)
        y2 = min(frame.shape[0], y + h + y_margin)
        x1 = max(0, x - x_margin)
        x2 = min(frame.shape[1], x + w + x_margin)
        
        face_roi = gray[y1:y2, x1:x2]
        face_roi_color = frame[y1:y2, x1:x2]
        
        # Skip if face_roi is empty
        if face_roi.size == 0:
            continue
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        try:
            # Calculate face region features
            
            # 1. Detect smile - indicator of happiness
            smile = smile_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.5,
                minNeighbors=15,
                minSize=(25, 25)
            )
            has_smile = len(smile) > 0
            
            # 2. Detect eyes - important for surprise/fear detection
            eyes = eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # 3. Detect closed eyes - helps with disgust/sadness
            closed_eyes = eye_closed_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # 4. Calculate average pixel intensity
            avg_intensity = np.mean(face_roi)
            
            # 5. Apply edge detection to find facial muscle tensions
            edges = cv2.Canny(face_roi, 50, 150)
            edge_intensity = np.mean(edges)
            
            # 6. Calculate the ratio of face width to height
            face_ratio = w / h
            
            # 7. Calculate facial symmetry
            symmetry = calculate_symmetry(face_roi)
            
            # 8. Calculate histogram of oriented gradients for texture analysis
            # Simplified version
            gradients_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            gradients_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradients_x**2 + gradients_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            
            # Enhanced emotion detection logic
            if has_smile and len(eyes) >= 2 and edge_intensity < 9:
                emotion = "Happy"
            elif len(eyes) >= 2 and edge_intensity > 10 and face_ratio < 0.85 and symmetry < 30:
                emotion = "Surprise"
            elif avg_intensity < 120 and edge_intensity > 9 and gradient_mean > 20 and symmetry > 40:
                emotion = "Angry"
            elif len(closed_eyes) > len(eyes) and avg_intensity < 130 and edge_intensity < 8:
                emotion = "Sad"
            elif edge_intensity > 11 and face_ratio > 0.9 and symmetry < 35:
                emotion = "Fear"
            elif avg_intensity < 110 and gradient_mean < 15 and edge_intensity < 7:
                emotion = "Disgust"
            else:
                emotion = "Neutral"
            
            # Apply temporal smoothing
            smoothed_emotion = smooth_emotion(emotion)
            
            # Draw facial features
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
            # Display emotion text
            cv2.putText(frame, smoothed_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Show confidence metrics
            metrics_text = f"Edge: {edge_intensity:.1f}, Sym: {symmetry:.1f}, Grad: {gradient_mean:.1f}"
            cv2.putText(frame, metrics_text, (x, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        except Exception as e:
            print(f"Error processing face: {e}")
    
    return frame

# Function to start webcam
def start_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam started. Press 'q' to quit.")
    
    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
            
        # Process frame
        output_frame = detect_emotions(frame)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS
        cv2.putText(output_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Enhanced Emotion Recognition', output_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Check if OpenCV cascade files exist
    required_files = [face_cascade_path, smile_cascade_path, eye_cascade_path]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Warning: Cascade file not found: {file_path}")
            print("Some features may not work properly.")
    
    # Start webcam
    start_webcam()