from flask import Flask, Response, render_template, request, jsonify
import cv2
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the latest detection data
latest_detection = {
    "timestamp": None,
    "location": None,
    "detections": [],
    "total_persons": 0,
    "total_packages": 0,
    "alert_type": None,
}

# Detection history for logging
detection_history = []
MAX_HISTORY_ENTRIES = 100


def generate_frames():
    cap = cv2.VideoCapture("udp://127.0.0.1:5010", cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open UDP stream")

    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("Unable to read frame.")
            # Small delay before retrying
            import time

            time.sleep(0.1)
            # Try to reopen the stream if connection lost
            cap = cv2.VideoCapture("udp://127.0.0.1:5010", cv2.CAP_FFMPEG)
            continue
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/update_detection", methods=["POST"])
def update_detection():
    global latest_detection, detection_history

    try:
        data = request.json

        # Update latest detection data
        latest_detection = data

        # Add to history with timestamp
        if data.get("timestamp"):
            history_entry = data.copy()
            detection_history.insert(0, history_entry)

            # Limit history size
            if len(detection_history) > MAX_HISTORY_ENTRIES:
                detection_history = detection_history[:MAX_HISTORY_ENTRIES]

        logger.info(
            f"Updated detection: {len(data.get('detections', []))} items detected"
        )
        return jsonify({"status": "success"})

    except Exception as e:
        logger.error(f"Error updating detection data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/get_detection")
def get_detection():
    return jsonify(latest_detection)


@app.route("/get_history")
def get_history():
    limit = request.args.get("limit", default=10, type=int)
    offset = request.args.get("offset", default=0, type=int)

    if limit > MAX_HISTORY_ENTRIES:
        limit = MAX_HISTORY_ENTRIES

    result = detection_history[offset : offset + limit]
    return jsonify(
        {
            "history": result,
            "total": len(detection_history),
            "limit": limit,
            "offset": offset,
        }
    )


@app.route("/system_status")
def system_status():
    # A simple endpoint to check if the system is running
    return jsonify(
        {
            "status": "active",
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_active": latest_detection["timestamp"] is not None,
            "last_detection": latest_detection["timestamp"],
        }
    )


if __name__ == "__main__":
    logger.info("Starting security monitoring web server on port 8080")
    app.run(host="0.0.0.0", port=8080, threaded=True)
