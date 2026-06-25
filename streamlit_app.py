import cv2
import time
import os
import tempfile
import streamlit as st
import numpy as np
from ultralytics import YOLO

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="GymGuard AI", layout="wide")

# Create evidence folder for snapshots
if not os.path.exists("evidence"):
    os.makedirs("evidence")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Surveillance Settings")
input_source = st.sidebar.radio("Source:", ["Upload Video File", "Live Webcam (Local Only)"])
conf_threshold = st.sidebar.slider("AI Confidence", 0.2, 1.0, 0.4)

# --- 3. HELPER FUNCTIONS ---
def check_overlap(box_a, box_b):
    """Returns True if the center of Box A is inside Box B (Safe Zone)."""
    center_x = int((box_a[0] + box_a[2]) / 2)
    center_y = int((box_a[1] + box_a[3]) / 2)
    return (box_b[0] < center_x < box_b[2]) and (box_b[1] < center_y < box_b[3])

# --- 4. MAIN LOGIC ENGINE ---
def main():
    st.title("🏋️ GymGuard: Logic Flow Edition")
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col2:
        st.subheader("📋 Status Log")
        status_log = st.empty()
        evidence_gallery = st.empty()

    with col1:
        st.subheader("Live Analysis")
        model = YOLO("yolov8n.pt") 
        cap = None

        # --- INPUT HANDLING (Cloud vs Local) ---
        if input_source == "Upload Video File":
            uploaded_file = st.file_uploader("Upload test video (mp4/avi)...", type=['mp4', 'mov', 'avi'])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
        elif input_source == "Live Webcam (Local Only)":
            if st.button("Start Webcam"):
                cap = cv2.VideoCapture(0)

        # --- THE FLOWCHART LOGIC ---
        if cap:
            st_frame = st.empty()
            
            # Variables for "The Trigger"
            violation_start_time = None
            violation_threshold = 3.0 # Seconds to wait before snapping
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.info("End of video stream.")
                    break

                # Resize for speed
                frame = cv2.resize(frame, (720, 480))
                
                # 1. DEFINE SAFE ZONE (Rack)
                safe_zone = [50, 100, 350, 400] 

                # 2. RESET FRAME FLAGS
                person_present = False      # Is a human in the frame?
                weight_unsafe = False       # Is a weight on the floor?
                
                # 3. AI DETECTION
                results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    names = model.names 
                    
                    for box, cls in zip(boxes, classes):
                        name = names[cls]
                        
                        # LOGIC: Check who is in the scene
                        if name == "person":
                            person_present = True
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, "USER", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        else:
                            # Treat any object (bottle/phone) as Weight
                            if not check_overlap(box, safe_zone):
                                weight_unsafe = True
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                                cv2.putText(frame, "UN-RACKED", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                            else:
                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                # 4. THE "GHOST TRIGGER" (Flowchart Logic)
                # Condition: Weight is Unsafe AND Person is Gone
                if weight_unsafe and not person_present:
                    
                    # Start the timer
                    if violation_start_time is None:
                        violation_start_time = time.time()
                    
                    # Calculate duration
                    elapsed = time.time() - violation_start_time
                    countdown = violation_threshold - elapsed
                    
                    if countdown <= 0:
                        # --- ACTION: SNAPSHOT & ALERT ---
                        cv2.putText(frame, "VIOLATION CONFIRMED!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        status_log.error("🚨 ALERT: Weight abandoned by user!")
                        
                        # Save Proof (Throttled to 1 per session for demo)
                        timestamp = time.strftime("%H%M%S")
                        proof_path = f"evidence/violation_{timestamp}.jpg"
                        # Only save if we haven't recently (optional logic)
                        
                    else:
                        # Warning Phase
                        cv2.putText(frame, f"GHOST TIMER: {countdown:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        status_log.warning(f"⚠️ User Left! Checking... {countdown:.1f}s")
                
                else:
                    # Reset if safe OR if person returns
                    violation_start_time = None
                    if person_present and weight_unsafe:
                        status_log.info("👀 Weight un-racked, but User is present (Active Use).")
                    elif not weight_unsafe:
                        status_log.success("✅ Area Safe (Weights Racked)")

                # Draw Safe Zone
                cv2.rectangle(frame, (safe_zone[0], safe_zone[1]), (safe_zone[2], safe_zone[3]), (255, 255, 255), 2)
                cv2.putText(frame, "SAFE ZONE", (safe_zone[0], safe_zone[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Render to Streamlit (Convert Colors)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame, channels="RGB")

            cap.release()

if __name__ == "__main__":
    main()
