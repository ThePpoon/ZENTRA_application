# ppe_detection.py
import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import RTSPSource, StreamConfig, VideoMetadata

print("🚀 Starting ZENTRA PPE Detection System...")

# Initialize client
client = InferenceHTTPClient.init(
    api_url="http://localhost:9001",
    api_key="jXY0KZy21ofMXdHz60n0"  # ใส่ API key ของคุณ
)

# เลือก source (เริ่มจาก demo ก่อน)
print("📹 Connecting to camera...")
# ทดสอบด้วย demo stream ก่อน
source = RTSPSource("rtsp://demo.roboflow.com:8554")

# หรือใช้ webcam MacBook (ง่ายกว่า เริ่มจากนี้ก่อน!)
# source = cv2.VideoCapture(0)

# หรือใช้ RTSP camera ของคุณ
# source = RTSPSource("rtsp://admin:password@192.168.1.100:554/stream1")

# Configure streaming
config = StreamConfig(
    stream_output=["output_image"],
    data_output=["predictions", "detection_predictions"],
    processing_timeout=3600
)

# Create session
print("🔗 Creating streaming session...")
session = client.webrtc.stream(
    source=source,
    workflow="detect-and-classify",
    workspace="pholawats-workspace",  # เปลี่ยนเป็น workspace ของคุณ
    image_input="image",
    config=config
)

# Statistics
frame_count = 0
violation_count = 0

# Handle video frames
@session.on_frame
def show_frame(frame, metadata):
    global frame_count
    frame_count += 1
    
    # แสดงสถิติบนภาพ
    cv2.putText(frame, f"Frames: {frame_count} | Violations: {violation_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("ZENTRA - PPE Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n⏹️  Stopping session...")
        session.close()

# Handle predictions
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    global violation_count
    
    if "predictions" in data and data["predictions"]:
        predictions = data["predictions"]
        
        # เช็ค violations
        detected_classes = [pred.get("class", "") for pred in predictions]
        violations = [cls.replace("no_", "") for cls in detected_classes if cls.startswith("no_")]
        
        if violations:
            violation_count += 1
            print(f"⚠️  Frame {metadata.frame_id}: Missing PPE - {', '.join(violations)}")
        
        # แสดง detections ทุก 100 frames
        if frame_count % 100 == 0:
            print(f"📊 Processed {frame_count} frames, {violation_count} violations")

# Run
print("✅ System ready! Press 'q' in video window to quit")
print("-" * 50)

try:
    session.run()
except KeyboardInterrupt:
    print("\n⏹️  Stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    print(f"\n📈 Session Summary:")
    print(f"   Total Frames: {frame_count}")
    print(f"   Total Violations: {violation_count}")