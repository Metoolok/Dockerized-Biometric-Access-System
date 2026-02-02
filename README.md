



https://github.com/user-attachments/assets/e5533b2a-7a7d-4173-b5fd-93cb0884c81a



https://github.com/user-attachments/assets/962665de-7181-4261-8823-ca62d5722598

https://github.com/user-attachments/assets/ad3a7fed-faba-41fd-9968-e944b1481330








🔐 Dockerized Biometric Access Control System


This repository contains an end-to-end Biometric Face Recognition and Authorization System developed as part of my AI Engineering studies. The project bridges the gap between Deep Learning models and DevOps by utilizing Docker for seamless, environment-independent deployment.

🧠 Technical Overview


The system is designed to provide high-precision facial recognition by following a multi-stage pipeline:

Face Detection & Alignment: Utilizes Dlib's 68-point landmarks to detect and geometrically normalize faces. This ensures that the model remains robust even if the subject is at an angle.

Feature Extraction: Implements a ResNet-29 based architecture to extract 128-D unique vector embeddings from each face.

Vector Matching: Performs real-time comparison using Euclidean Distance and Cosine Similarity against a local biometric database.

Containerization: The entire stack—including OpenCV GUI and hardware (camera) passthrough—is containerized via Docker, eliminating "it works on my machine" issues.

🛠 Tech Stack
Language: Python 3.10+

Computer Vision: OpenCV, Dlib, Face_Recognition

Models: ResNet-29 (Deep Learning Architecture)

DevOps: Docker & Docker-Compose

OS Environment: Developed on Ubuntu (Acer Nitro 5)

📂 Project Structure

Plaintext
face_access_system/
├── vision/            # Facial landmark and alignment logic
├── recognition/       # ResNet-29 embedding & matching algorithms
├── database/          # User enrollment and log storage
├── Dockerfile         # Container configuration
├── docker-compose.yml # Hardware & GUI orchestration
└── main.py            # System entry point
🚀 Getting Started
Ensure you have Docker and X11 (for GUI) configured on your system.

Bash
# Clone the repository
git clone https://github.com/Metoolok/docker-face-access-control.git

# Navigate to project

cd docker-face-access-control

# Build and Run

docker-compose up --build
👨‍💻 Author
Metin Mert Turan Artificial Intelligence Engineering Studen
