from dotenv import load_dotenv
import os

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ALERT_PHONE_NUMBER = os.getenv("ALERT_PHONE_NUMBER")
ALARM_PATH = os.getenv("ALARM_PATH")
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 3))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.25))
EAR_CONSEC_FRAMES = int(os.getenv("EAR_CONSEC_FRAMES", 20))
MAR_THRESHOLD = float(os.getenv("MAR_THRESHOLD", 0.5))
HEAD_POSE_THRESHOLD = float(os.getenv("HEAD_POSE_THRESHOLD", 20))