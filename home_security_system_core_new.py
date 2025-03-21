import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import cv2
import hailo
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import requests
import os
import face_recognition  # Added for face recognition
import numpy as np  # Added for face recognition operations
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntruderDetectionCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        # Email configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "dhairyaparikh67@gmail.com"
        self.sender_password = "kqie ptxn rymk nlzb"
        self.receiver_email = "dhairyaparikh1998@gmail.com"

        # Detection parameters
        self.last_alert_time = datetime.min
        self.last_package_alert_time = datetime.min  # Added for package alerts
        self.alert_cooldown = 180  # 3 minutes between alerts
        self.confidence_threshold = 0.6
        self.location_name = "Main Entrance"

        # Server configuration
        self.server_url = "http://localhost:8080"

        # Face recognition initialization - NEW
        self.known_face_dir = (
            "basic_pipelines/known_faces"  # Directory with known face images
        )
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # Use frame for face recognition
        self.use_frame = True

        # Track detected known persons to avoid duplicate prints
        self.detected_known_persons = set()

        # Track cooldown notifications to avoid duplicate prints
        self.intruder_cooldown_notified = False
        self.package_cooldown_notified = False

    def load_known_faces(self):
        """Load known faces from the specified directory"""
        logger.info(f"Loading known faces from {self.known_face_dir}")

        try:
            if not os.path.exists(self.known_face_dir):
                logger.warning(
                    f"Known faces directory '{self.known_face_dir}' does not exist"
                )
                return

            for filename in os.listdir(self.known_face_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    name = os.path.splitext(filename)[
                        0
                    ]  # Get the name from the filename
                    image_path = os.path.join(self.known_face_dir, filename)

                    try:
                        face_image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(face_image)

                        if face_encodings:
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(name)
                            logger.info(f"Loaded face for {name}")
                        else:
                            logger.warning(f"No face found in {filename}")

                    except Exception as e:
                        logger.error(f"Error loading face from {filename}: {str(e)}")

            logger.info(f"Loaded {len(self.known_face_names)} known faces")
        except Exception as e:
            logger.error(f"Error during face loading: {str(e)}")

    def recognize_faces(self, frame):
        """Recognize faces in the frame and return list of known faces found"""
        recognized_faces = []
        try:
            if not self.known_face_encodings:
                return recognized_faces

            # Resize frame for faster face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding
                )
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        recognized_faces.append(name)

                        # Log detection of known person only once
                        if name not in self.detected_known_persons:
                            logger.info(f"Known person detected: {name}")
                            self.detected_known_persons.add(name)

        except Exception as e:
            logger.error(f"Error during face recognition: {str(e)}")

        return recognized_faces

    def format_email_body(self, detections, current_time, alert_type="intruder"):
        """Create a formal, structured email body"""
        try:
            # Select appropriate alert title based on alert type
            alert_title = "SECURITY ALERT: Intruder Detection"
            if alert_type == "package":
                alert_title = "SECURITY ALERT: Package Detected"

            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="padding: 20px;">
                    <h2 style="color: #FF0000;">{alert_title}</h2>
                   
                    <div style="background-color: #f5f5f5; padding: 15px; margin: 10px 0;">
                        <h3>Incident Details</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Location:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{self.location_name}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Date:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{current_time.strftime("%B %d, %Y")}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Time:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{current_time.strftime("%H:%M:%S %Z")}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Total Items Detected:</strong></td>
                                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{len(detections)}</td>
                            </tr>
                        </table>
                    </div>

                    <div style="background-color: #f5f5f5; padding: 15px; margin: 10px 0;">
                        <h3>Detection Summary</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="background-color: #e0e0e0;">
                                <th style="padding: 8px; border: 1px solid #ddd;">Detection ID</th>
                                <th style="padding: 8px; border: 1px solid #ddd;">Type</th>
                                <th style="padding: 8px; border: 1px solid #ddd;">Confidence Score</th>
                            </tr>
                            {self._format_detection_table(detections)}
                        </table>
                    </div>

                    <div style="margin-top: 20px; font-size: 12px; color: #666;">
                        <p>This is an automated security alert. Please do not reply to this email.</p>
                        <p>If this alert requires immediate attention, please contact security personnel.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            return html_body
        except Exception as e:
            logger.error(f"Error formatting email body: {str(e)}")
            # Return a basic version if there's an error
            return f"<html><body><h2>{alert_type.capitalize()} Alert</h2><p>Detection occurred at {current_time}</p></body></html>"

    def _format_detection_table(self, detections):
        """Format the table of detection details"""
        try:
            return "".join(
                [
                    f'<tr><td style="padding: 8px; border: 1px solid #ddd;">{det["id"]}</td>'
                    f'<td style="padding: 8px; border: 1px solid #ddd;">{det["type"]}</td>'
                    f'<td style="padding: 8px; border: 1px solid #ddd;">{det["confidence"]:.2f}</td></tr>'
                    for det in detections
                ]
            )
        except Exception as e:
            logger.error(f"Error formatting detection table: {str(e)}")
            return "<tr><td colspan='3'>Error formatting detection data</td></tr>"

    def send_alert(self, frame, detections, alert_type="intruder"):
        """Send email alert and update server with detection information"""
        try:
            current_time = datetime.now()

            # Check the right cooldown based on alert type
            last_alert_time = (
                self.last_alert_time
                if alert_type == "intruder"
                else self.last_package_alert_time
            )

            if (current_time - last_alert_time).total_seconds() < self.alert_cooldown:
                # Only log the cooldown message once per detection period
                if alert_type == "intruder" and not self.intruder_cooldown_notified:
                    logger.info(f"Skipping intruder alert due to cooldown period")
                    self.intruder_cooldown_notified = True
                elif alert_type == "package" and not self.package_cooldown_notified:
                    logger.info(f"Skipping package alert due to cooldown period")
                    self.package_cooldown_notified = True
                return

            # Reset cooldown notification flags when sending a new alert
            if alert_type == "intruder":
                self.intruder_cooldown_notified = False
            else:
                self.package_cooldown_notified = False

            # Send email alert
            msg = MIMEMultipart()

            if alert_type == "intruder":
                msg["Subject"] = (
                    f"SECURITY ALERT - Intruder Detection at {self.location_name}"
                )
            else:
                msg["Subject"] = (
                    f"SECURITY ALERT - Package Detected at {self.location_name}"
                )

            msg["From"] = self.sender_email
            msg["To"] = self.receiver_email

            html_body = self.format_email_body(detections, current_time, alert_type)
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                logger.info(f"Email alert sent: {alert_type}")

            # Send detection data to web server
            alert_data = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "location": self.location_name,
                "detections": detections,
                "total_persons": sum(1 for d in detections if d["type"] == "person"),
                "total_packages": sum(1 for d in detections if d["type"] == "package"),
                "alert_type": alert_type,
            }

            try:
                response = requests.post(
                    f"{self.server_url}/update_detection", json=alert_data, timeout=5
                )
                response.raise_for_status()
                logger.info(f"Detection data sent to server: {alert_type}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send detection data to server: {str(e)}")

            # Update the appropriate last alert time
            if alert_type == "intruder":
                self.last_alert_time = current_time
            else:
                self.last_package_alert_time = current_time

            logger.info(
                f"{alert_type.capitalize()} alert email sent and detection data updated"
            )

        except smtplib.SMTPException as e:
            logger.error(f"SMTP error when sending {alert_type} alert: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to send {alert_type} alert: {str(e)}")

    def reset_detection_tracking(self):
        """Reset the detection tracking when scene changes significantly"""
        # Reset detected known persons
        self.detected_known_persons.clear()
        # Reset cooldown notification flags
        self.intruder_cooldown_notified = False
        self.package_cooldown_notified = False
        logger.debug("Detection tracking reset")


def app_callback(pad, info, user_data):
    """Callback function for processing video frames"""
    try:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        format, width, height = get_caps_from_pad(pad)
        recognized_faces = []
        frame = None

        # Process frame for face recognition if format is available
        if format is not None and width is not None and height is not None:
            try:
                frame = get_numpy_from_buffer(buffer, format, width, height)
                # Recognize faces in the frame
                recognized_faces = (
                    user_data.recognize_faces(frame) if frame is not None else []
                )
            except Exception as e:
                logger.error(f"Error processing frame for face recognition: {str(e)}")
                recognized_faces = []

        # Get object detections from the buffer
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        except Exception as e:
            logger.error(f"Error getting detections from buffer: {str(e)}")
            return Gst.PadProbeReturn.OK

        person_detection_info = []
        package_detection_info = []

        # Process each detection
        for detection in detections:
            try:
                confidence = detection.get_confidence()
                label = detection.get_label()

                if confidence > user_data.confidence_threshold:
                    bbox = detection.get_bbox()
                    track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                    track_id = track[0].get_id() if track else 0

                    # Person detection
                    if label == "person":
                        # Only add to detection info if not a known face
                        if not recognized_faces:
                            person_detection_info.append(
                                {
                                    "id": track_id,
                                    "type": "person",
                                    "confidence": confidence,
                                }
                            )

                        # Draw bounding box if frame is available
                        if frame is not None:
                            x1, y1, x2, y2 = map(
                                int,
                                [
                                    bbox.xmin() * width,
                                    bbox.ymin() * height,
                                    bbox.xmax() * width,
                                    bbox.ymax() * height,
                                ],
                            )
                            # Use green for known persons, red for unknown persons (intruders)
                            color = (0, 255, 0) if recognized_faces else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            label_text = f"ID: {track_id}"
                            if recognized_faces:
                                label_text = f"ID: {track_id} - Known: {', '.join(recognized_faces)}"

                            cv2.putText(
                                frame,
                                label_text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                    # Package detection (using bottle as proxy)
                    elif label == "bottle":  # Using bottle detection as package proxy
                        package_detection_info.append(
                            {
                                "id": track_id,
                                "type": "package",
                                "confidence": confidence,
                            }
                        )

                        # Draw bounding box if frame is available
                        if frame is not None:
                            x1, y1, x2, y2 = map(
                                int,
                                [
                                    bbox.xmin() * width,
                                    bbox.ymin() * height,
                                    bbox.xmax() * width,
                                    bbox.ymax() * height,
                                ],
                            )
                            # Use blue for packages
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(
                                frame,
                                f"Package {track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                2,
                            )
            except Exception as e:
                logger.error(f"Error processing detection: {str(e)}")
                continue

        # Send intruder alerts if unknown persons detected
        if person_detection_info:
            user_data.send_alert(frame, person_detection_info, "intruder")

        # Send package alerts
        if package_detection_info:
            user_data.send_alert(frame, package_detection_info, "package")

        # Update the frame if available
        if frame is not None:
            try:
                user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logger.error(f"Error setting frame: {str(e)}")

    except Exception as e:
        logger.error(f"Error in app_callback: {str(e)}")

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    try:
        logger.info("Initializing Intruder Detection System")
        user_data = IntruderDetectionCallback()
        app = GStreamerDetectionApp(app_callback, user_data)
        logger.info("Starting Intruder Detection System")
        app.run()
    except Exception as e:
        logger.critical(f"Failed to start Intruder Detection System: {str(e)}")
