# Raspberry Pi AI Home Security System (Part 3)

This repository contains the third part of a series on building an AI-powered home security system using the Raspberry Pi 5 and Hailo AI Kit. This part adds face recognition for known individuals and package detection features to the existing system.

## New Features

- **Face Recognition**: Identifies known individuals (family/friends) to reduce unnecessary alerts
- **Package Detection**: Detects when a package is delivered and sends a separate notification
- **Enhanced UI**: Updated web interface with separate tabs for person and package detection
- **System Logs**: Live logging directly in the web interface

## Getting Started

### Prerequisites

- Raspberry Pi 5
- Hailo AI Kit
- USB Camera
- Completed setup from [Part 2](https://github.com/Dhairya1007/Raspberry-Pi-AI-Home-Security-System-Part-2)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/Raspberry-Pi-AI-Home-Security-System-Part-3.git
   ```

2. Install dependencies:

   ```sh
   cd Raspberry-Pi-AI-Home-Security-System-Part-3
   pip install -r requirements.txt
   ```

3. Setting up known faces:
   - Create a folder named `known_faces` in the project directory
   - Add photos of known individuals (one face per image)
   - Name each file with the person's name (e.g., `john.jpg`, `emma.png`)

### Running the System

1. Start the core detection system:

   ```sh
   python home_security_system_core.py --input rpi
   ```

2. In another terminal, start the web server:

   ```sh
   python stream_security_system_new.py
   ```

3. Access the web interface at `http://[raspberry-pi-ip]:8080`

## How It Works

- The system uses the YOLO model for person detection
- Face recognition is implemented with the `face_recognition` library
- Bottles are used as proxies for package detection (for demonstration)
- Email alerts are sent for both unknown persons and packages
- The web interface provides real-time monitoring of all detections

## Files Description

- `home_security_system_core_new.py`: Main detection script with face recognition and package detection
- `stream_security_system_new.py`: Web server for streaming video and detection data
- `index.html`: Updated UI with tabs for different detection types
- `requirements.txt`: List of required Python packages

## References

- [Hailo Raspberry Pi 5 Examples Repository](https://github.com/hailo-ai/hailo-rpi5-examples)
- [Part 2 Repository](https://github.com/Dhairya1007/Raspberry-Pi-AI-Home-Security-System-Part-2)

## Author

- Dhairya Parikh
