# 🚀 Drowsiness Detection System  

A real-time **Drowsiness Detection System** using **OpenCV, MediaPipe, and Raspberry Pi** to analyze eye movement, mouth patterns, and head position. Triggers **audio alerts**, **SMS notifications**, and logs events for improved safety.  

---

## 🎯 Features  
✅ **Real-Time Face Tracking** – Detects eyes, mouth, and head position using **MediaPipe**  
✅ **Eye & Mouth Aspect Ratio Analysis** – Identifies drowsiness based on **blink rate and mouth openness**  
✅ **Head Pose Estimation** – Detects abnormal head tilts for enhanced accuracy  
✅ **Alarm System** – Plays an **alert sound** when drowsiness is detected  
✅ **SMS Alerts** – Notifies emergency contacts via **Twilio API**  
✅ **Data Logging** – Stores detection events in a CSV file for analysis  
✅ **Optimized for Raspberry Pi** – Runs efficiently on **Raspberry Pi with OpenCV & TFLite**  

---

## 🛠️ Installation  

### 📌 Prerequisites  
- Raspberry Pi (or any system with a webcam)  
- Python 3.x  
- OpenCV, MediaPipe, NumPy, Pygame  
- Twilio account for SMS alerts  

### 📥 Clone the Repository  
```bash
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
```

### ⚙️ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 📝 Configure Twilio (For SMS Alerts)  
Create a `.env` file and add your Twilio credentials:  
```
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number
ALERT_PHONE_NUMBER=your_alert_number
```

---

## ▶️ Usage  

Run the system:  
```bash
python src/main.py
```
Press **'q'** to exit.  

---

## 🎯 How It Works  

1️⃣ Captures live video and detects **face landmarks**.  
2️⃣ Calculates **eye aspect ratio (EAR)** and **mouth aspect ratio (MAR)**.  
3️⃣ Estimates **head pose (yaw, pitch, roll)** using SolvePnP.  
4️⃣ Triggers an **alarm and SMS alert** if drowsiness is detected.  
5️⃣ Logs events with timestamps in a **CSV file**.  

---

## 🖥️ Raspberry Pi Integration  
To run on a Raspberry Pi, install dependencies:  
```bash
sudo apt update
sudo apt install python3-opencv
pip install mediapipe numpy pygame twilio
```
Enable the camera and run:  
```bash
python src/main.py
```

---

## 📊 Data Logging  
Detection logs are stored in `drowsiness_log.csv` with:  
- **Timestamp**  
- **Drowsiness status**  
- **Blink Rate**  
- **Head Pose (Yaw, Pitch, Roll)**  

---

## 🛡️ Future Enhancements  
🔹 **Machine Learning Integration** – Improve accuracy with a trained model  
🔹 **Mobile Notifications** – Send alerts via WhatsApp/Telegram  
🔹 **Web Dashboard** – Real-time analytics with Flask & React  

---

## 🤝 Contributing  
Pull requests are welcome! For major changes, open an issue first.  

---

🚀 **Stay Alert. Stay Safe.**
