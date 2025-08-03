import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import threading
import queue

# Page config
st.set_page_config(
    page_title="AI Face Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .alert-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .success-box {
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
</style>
""", unsafe_allow_html=True)

class FaceMaskDetector:
    def __init__(self, model_path):
        """Initialize the Face Mask Detector"""
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.detection_history = []
            self.session_stats = {
                'total_detections': 0,
                'with_mask': 0,
                'without_mask': 0,
                'incorrect_mask': 0,
                'start_time': datetime.now()
            }
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please make sure your YOLOv11 model file (.pt) is available")
    
    def detect_image(self, image, confidence_threshold=0.5):
        """Detect faces and masks in a single image"""
        try:
            results = self.model(image, conf=confidence_threshold)
            return self.process_results(results[0], image)
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return image, []
    
    def process_results(self, result, image):
        """Process YOLO results and draw bounding boxes"""
        detections = []
        annotated_image = image.copy()
        
        if result.boxes is not None:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                # Debug: Show detected class names
                st.write(f"🔍 Detected: {class_name} (ID: {class_id}) with confidence: {confidence:.2f}")
                
                # Show current stats before updating
                st.write(f"📊 Current Stats - Total: {self.session_stats['total_detections']}, With Mask: {self.session_stats['with_mask']}, Without: {self.session_stats['without_mask']}, Incorrect: {self.session_stats['incorrect_mask']}")
                
                # Color coding for different classes
                colors = {
                    'with_mask': (0, 255, 0),      # Green
                    'without_mask': (0, 0, 255),   # Red
                    'mask_weared_incorrect': (0, 165, 255)  # Orange
                }
                
                color = colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Store detection info
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'timestamp': datetime.now()
                })
                
                # Update session stats
                self.update_stats(class_name)
                
                # Show updated stats after counting
                st.write(f"📈 Updated Stats - Total: {self.session_stats['total_detections']}, With Mask: {self.session_stats['with_mask']}, Without: {self.session_stats['without_mask']}, Incorrect: {self.session_stats['incorrect_mask']}")
                st.write("---")
        
        return annotated_image, detections
    
    def update_stats(self, class_name):
        """Update detection statistics"""
        self.session_stats['total_detections'] += 1
        
        # More flexible class name matching
        class_name_lower = class_name.lower()
        
        # Check for various possible class name formats
        if any(keyword in class_name_lower for keyword in ['with_mask', 'mask_on', 'masked', 'wearing_mask', 'mask']):
            # Only count as "with_mask" if it's not "mask incorrect"
            if 'incorrect' not in class_name_lower:
                self.session_stats['with_mask'] += 1
                st.write(f"✅ Counted as WITH MASK: {class_name}")
            else:
                self.session_stats['incorrect_mask'] += 1
                st.write(f"⚠️ Counted as INCORRECT MASK: {class_name}")
        elif any(keyword in class_name_lower for keyword in ['without_mask', 'no_mask', 'unmasked', 'mask_off', 'no mask']):
            self.session_stats['without_mask'] += 1
            st.write(f"❌ Counted as WITHOUT MASK: {class_name}")
        elif any(keyword in class_name_lower for keyword in ['incorrect', 'wrong', 'improper', 'mask_weared_incorrect', 'mask incorrect']):
            self.session_stats['incorrect_mask'] += 1
            st.write(f"⚠️ Counted as INCORRECT MASK: {class_name}")
        else:
            # If no match found, log the unknown class name
            st.write(f"❓ Unknown class name: {class_name} - not counted in stats")
    
    def get_compliance_rate(self):
        """Calculate mask compliance rate"""
        total = self.session_stats['total_detections']
        if total == 0:
            return 0
        compliant = self.session_stats['with_mask']
        return (compliant / total) * 100
    
    def get_model_classes(self):
        """Get all available class names from the model"""
        return self.class_names

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'live_detection' not in st.session_state:
    st.session_state.live_detection = False

# Main App
def main():
    st.markdown('<h1 class="main-header">🎯 AI Face Mask Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar for model configuration
    with st.sidebar:
        st.markdown("## 🔧 Configuration")
        
        # Model upload
        model_file = st.file_uploader(
            "Upload YOLOv11 Model (.pt)", 
            type=['pt'],
            help="Upload your trained YOLOv11 face mask detection model"
        )
        
        if model_file:
            # Save uploaded model temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(model_file.read())
                model_path = tmp_file.name
            
            # Initialize detector
            if st.session_state.detector is None:
                with st.spinner("Loading model..."):
                    st.session_state.detector = FaceMaskDetector(model_path)
                st.success("✅ Model loaded successfully!")
                
                # Show model class names for debugging
                detector = st.session_state.detector
                if detector and detector.get_model_classes():
                    st.info("📋 **Model Class Names:**")
                    for class_id, class_name in detector.get_model_classes().items():
                        st.write(f"  - Class {class_id}: {class_name}")
        
        st.markdown("---")
        
        # Detection settings
        st.markdown("## ⚙️ Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05
        )
        
        # Analytics toggle
        st.markdown("---")
        show_analytics = st.checkbox("📊 Show Analytics", value=True)
        
        # Alert settings
        st.markdown("## 🚨 Alert Settings")
        enable_alerts = st.checkbox("Enable Compliance Alerts", value=True)
        compliance_threshold = st.slider(
            "Minimum Compliance Rate (%)", 
            min_value=50, 
            max_value=100, 
            value=80
        )
    
    if st.session_state.detector is None:
        st.warning("⚠️ Please upload your YOLOv11 model file (.pt) to get started!")
        st.info("""
        ### How to get started:
        1. Upload your trained YOLOv11 model file (.pt) in the sidebar
        2. Choose your input type (Image, Video, or Live Camera)
        3. Start detecting! 🚀
        """)
        return
    
    detector = st.session_state.detector
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📹 Live Camera", "📊 Analytics"])
    
    # Image Detection Tab
    with tab1:
        st.markdown("### Upload Image for Detection")
        uploaded_image = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )
        
        if uploaded_image:
            # Display original image
            image = Image.open(uploaded_image)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("#### Detection Results")
                with st.spinner("Detecting..."):
                    result_image, detections = detector.detect_image(image_np, confidence_threshold)
                
                st.image(result_image, use_column_width=True)
                
                # Display detection results
                if detections:
                    st.markdown("#### Detection Summary")
                    for i, detection in enumerate(detections):
                        st.markdown(f"**Person {i+1}:** {detection['class']} (Confidence: {detection['confidence']:.2f})")
                else:
                    st.info("No faces detected in the image")
    
    # Video Detection Tab
    with tab2:
        st.markdown("### Upload Video for Detection")
        uploaded_video = st.file_uploader(
            "Choose a video...", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )
        
        if uploaded_video:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Original Video")
                st.video(uploaded_video)
            
            with col2:
                st.markdown("#### Process Video")
                if st.button("🎬 Start Video Detection", key="process_video"):
                    process_video(detector, video_path, confidence_threshold)
    
    # Live Camera Tab
    with tab3:
        st.markdown("### Live Camera Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            camera_placeholder = st.empty()
            
        with col2:
            st.markdown("#### Live Controls")
            
            if st.button("📹 Start Live Detection", key="start_live"):
                st.session_state.live_detection = True
            
            if st.button("⏹️ Stop Detection", key="stop_live"):
                st.session_state.live_detection = False
            
            # Live stats placeholder
            live_stats_placeholder = st.empty()
        
        # Live detection logic
        if st.session_state.live_detection:
            run_live_detection(detector, camera_placeholder, live_stats_placeholder, confidence_threshold)
    
    # Analytics Tab
    with tab4:
        if show_analytics:
            display_analytics(detector, enable_alerts, compliance_threshold)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "### 🚀 **Built with YOLOv11 • Streamlit • OpenCV** | "
        "💡 **Advanced AI Computer Vision Project**"
    )

def process_video(detector, video_path, confidence_threshold):
    """Process uploaded video file"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    st.info(f"📹 Processing video: {total_frames} frames, {duration:.1f} seconds")
    
    # Progress tracking
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    frame_count = 0
    video_detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame for performance
        if frame_count % 5 == 0:
            result_frame, detections = detector.detect_image(frame, confidence_threshold)
            
            # Display current frame
            frame_placeholder.image(result_frame, channels="BGR", use_column_width=True)
            
            # Store detections
            video_detections.extend(detections)
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Update stats
            stats_placeholder.markdown(f"""
            **Processing Progress:** {progress*100:.1f}%  
            **Current Frame:** {frame_count}/{total_frames}  
            **Detections So Far:** {len(video_detections)}
            """)
        
        frame_count += 1
    
    cap.release()
    progress_bar.progress(1.0)
    st.success(f"✅ Video processing complete! Total detections: {len(video_detections)}")

def run_live_detection(detector, camera_placeholder, stats_placeholder, confidence_threshold):
    """Run live camera detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check your camera connection.")
        return
    
    # Performance optimization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    
    while st.session_state.live_detection:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to read from camera")
            break
        
        # Process every 3rd frame for better performance
        if frame_count % 3 == 0:
            result_frame, detections = detector.detect_image(frame, confidence_threshold)
            
            # Calculate FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
            else:
                fps = 0
            
            # Add FPS counter to frame
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            camera_placeholder.image(result_frame, channels="BGR", use_column_width=True)
            
            # Update live stats
            compliance_rate = detector.get_compliance_rate()
            stats_placeholder.markdown(f"""
            **🔴 LIVE DETECTION**  
            **FPS:** {fps:.1f}  
            **Total Detections:** {detector.session_stats['total_detections']}  
            **Compliance Rate:** {compliance_rate:.1f}%  
            **With Mask:** {detector.session_stats['with_mask']}  
            **Without Mask:** {detector.session_stats['without_mask']}
            """)
        
        frame_count += 1
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.01)
    
    cap.release()

def display_analytics(detector, enable_alerts, compliance_threshold):
    """Display analytics dashboard"""
    st.markdown("### 📊 Detection Analytics Dashboard")
    
    # Session statistics
    stats = detector.session_stats
    compliance_rate = detector.get_compliance_rate()
    
    # Alert system
    if enable_alerts and compliance_rate < compliance_threshold:
        st.markdown(f"""
        <div class="alert-box">
            🚨 <strong>COMPLIANCE ALERT!</strong><br>
            Current compliance rate ({compliance_rate:.1f}%) is below threshold ({compliance_threshold}%)
        </div>
        """, unsafe_allow_html=True)
    elif compliance_rate >= compliance_threshold and stats['total_detections'] > 0:
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>COMPLIANCE OK</strong><br>
            Current compliance rate: {compliance_rate:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{stats['total_detections']}</h3>
            <p>Total Detections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{stats['with_mask']}</h3>
            <p>With Mask</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{stats['without_mask']}</h3>
            <p>Without Mask</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{compliance_rate:.1f}%</h3>
            <p>Compliance Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    if stats['total_detections'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for mask distribution
            labels = ['With Mask', 'Without Mask', 'Incorrect Mask']
            values = [stats['with_mask'], stats['without_mask'], stats['incorrect_mask']]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                marker_colors=['#00b894', '#e17055', '#fdcb6e']
            )])
            fig_pie.update_layout(title="Mask Detection Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart for compliance
            fig_bar = go.Figure([
                go.Bar(x=['Compliant', 'Non-Compliant'], 
                       y=[stats['with_mask'], stats['without_mask'] + stats['incorrect_mask']],
                       marker_color=['#00b894', '#e17055'])
            ])
            fig_bar.update_layout(title="Compliance Overview")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Session info
        st.markdown("### 📅 Session Information")
        session_duration = datetime.now() - stats['start_time']
        st.markdown(f"""
        **Session Start:** {stats['start_time'].strftime('%H:%M:%S')}  
        **Duration:** {str(session_duration).split('.')[0]}  
        **Detection Rate:** {stats['total_detections'] / max(session_duration.total_seconds() / 60, 1):.1f} detections/minute
        """)
    else:
        st.info("📊 No detections yet. Start detecting to see analytics!")

if __name__ == "__main__":
    main()