import cv2
import time
import os
import tempfile
import streamlit as st
import numpy as np
from ultralytics import YOLO

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GymGuard AI", layout="wide")

# Create evidence folder if it doesn't exist
if not os.path.exists("evidence"):
    os.makedirs("evidence")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Surveillance Settings")

# A. Input Source (Cloud Safe)
input_source = st.sidebar.radio("Select Source:", ["Upload Video File", "Live Webcam (Local Only)"])

# B. Sensitivity
conf_threshold = st.sidebar.slider("AI Confidence", 0.2, 1.0, 0.4)

st.sidebar.info("ℹ️ Note: For Cloud deployment, use 'Upload Video File'. Webcam only works on localhost.")

# --- 3. HELPER FUNCTIONS ---
def check_overlap(box_a, box_b):
    """Checks if the center of the Object is inside the Safe Zone."""
    center_x = int((box_a[0] + box_a[2]) / 2)
    center_y = int((box_a[1] + box_a[3]) / 2)
    return (box_b[0] < center_x < box_b[2]) and (box_b[1] < center_y < box_b[3])

# --- 4. MAIN LOGIC ---
def main():
    st.title("🏋️ GymGuard: Cloud Etiquette Monitor")

    col1, col2 = st.columns([0.7, 0.3])

    with col2:
        st.subheader("📋 Violation Log")
        log_placeholder = st.empty()
        
    with col1:
        st.subheader("Live Feed Analysis")
        # Load Model
        model = YOLO("yolov8n.pt") 
        cap = None

        # HANDLE INPUTS
        if input_source == "Upload Video File":
            uploaded_file = st.file_uploader("Upload a gym video (mp4/avi)...", type=['mp4', 'mov', 'avi'])
            if uploaded_file:
                # Save temp file for OpenCV (Crucial for Cloud)
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                
        elif input_source == "Live Webcam (Local Only)":
            if st.button("Start Webcam Feed"):
                cap = cv2.VideoCapture(0)

        # RUN PROCESSOR
        if cap:
            st_frame = st.empty()
            
            # Logic Variables
            violation_start_time = None
            violation_threshold = 3.0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.warning("End of video stream.")
                    break

                # Resize for consistent web performance
                frame = cv2.resize(frame, (720, 480))

                # Define Safe Zone (Fixed for now)
                safe_zone = [50, 100, 350, 400] 
                
                # State Flags
                weight_detected = False
                weight_is_safe = True 
                person_detected = False
                
                # Run AI
                results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    names = model.names 

                    for box, cls in zip(boxes, class_ids):
                        obj_name = names[cls]

                        if obj_name == "person":
                            person_detected = True
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"USER", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        else:
                            # Treat non-person objects as weights
                            weight_detected = True
                            if check_overlap(box, safe_zone):
                                weight_is_safe = True
                                color = (0, 255, 0)
                            else:
                                weight_is_safe = False
                                color = (0, 0, 255)
                                
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                            cv2.putText(frame, f"ITEM", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # --- JUDGE LOGIC ---
                if weight_detected and not weight_is_safe and not person_detected:
                    if violation_start_time is None:
                        violation_start_time = time.time()
                    
                    elapsed = time.time() - violation_start_time
                    countdown = violation_threshold - elapsed
                    
                    if countdown <= 0:
                        # VIOLATION CONFIRMED
                        cv2.putText(frame, "VIOLATION!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        log_placeholder.error("🚨 ALERT: Weight Abandoned!")
                    else:
                         cv2.putText(frame, f"TIMER: {countdown:.1f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                         log_placeholder.warning(f"⚠️ Determining Owner... {countdown:.1f}s")
                else:
                    violation_start_time = None
                    log_placeholder.success("✅ Sector Clear")

                # Draw Safe Zone
                cv2.rectangle(frame, (safe_zone[0], safe_zone[1]), (safe_zone[2], safe_zone[3]), (255, 255, 255), 2)
                cv2.putText(frame, "RACK ZONE", (safe_zone[0], safe_zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- DISPLAY ON WEB ---
                # Convert BGR (OpenCV) to RGB (Streamlit)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame, channels="RGB")

            cap.release()

if __name__ == "__main__":
    main()
