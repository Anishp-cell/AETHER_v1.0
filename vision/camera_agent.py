import cv2
import mediapipe as mp
import time

class VisionPerception:
    def __init__(self):
        """
        Initializes the AETHER Vision Module using OpenCV and MediaPipe.
        According to the system design, this is 'non-learning' signal detection.
        It converts real-world visual data into a structured text signal for the LLM.
        """
        print("[Vision Layer] Initializing Camera and Face Mesh tracking...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Open default laptop webcam
        self.cap = cv2.VideoCapture(0)
        # Briefly warm up the camera
        time.sleep(1)
        
    def capture_emotion_signal(self):
        ret, frame = self.cap.read()
        if not ret:
            self.last_frame = None
            return "Camera is not accessible."
        
        # Store for GUI
        self.last_frame = frame

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return "The user is not currently visible in the camera frame."
            
        # If a face is found, we extract structured signals. 
        # (For V1.0, we perform a basic landmark check. We can expand this mathematically to detect smiles/frowns later).
        landmarks = results.multi_face_landmarks[0]
        
        # Calculate a basic 'smile' ratio based on mouth corners (Landmarks 61 and 291) vs top/bottom lips.
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        top_lip = landmarks.landmark[13]
        bottom_lip = landmarks.landmark[14]

        mouth_width = abs(right_corner.x - left_corner.x)
        mouth_height = abs(top_lip.y - bottom_lip.y)
        
        # Simple heuristic for demonstration (can be tuned extensively)
        ratio = mouth_width / (mouth_height + 0.0001)

        emotion_signal = "The user is looking directly at you with a neutral expression."
        if ratio > 5.5:  # Wide mouth relative to height usually indicates a smile
            emotion_signal = "The user is looking directly at you and smiling."
        elif ratio < 2.0: # Tall mouth indicates surprise or talking widely
            emotion_signal = "The user's mouth is open, possibly talking or surprised."
            
        return emotion_signal
        
    def get_latest_frame(self):
        """Returns the most recent BGR frame from OpenCV"""
        if hasattr(self, 'last_frame'):
            return self.last_frame
        return None

    def close(self):
        self.cap.release()

if __name__ == "__main__":
    vision = VisionPerception()
    print("\n--- Testing Vision Perception ---")
    print("Looking at webcam...")
    
    signal = vision.capture_emotion_signal()
    print(f"Captured Signal: {signal}")
    
    vision.close()
