import os
import requests
import time
import json
from jose import jwt
from langchain.tools import tool
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import tempfile
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SID = os.getenv("TWILIO_VERIFY_SID")

# Global variable to cache DeepFace after first import
_deepface_module = None

def _get_deepface():
    """Lazy load DeepFace module to avoid startup delays"""
    global _deepface_module
    if _deepface_module is None:
        print("--- Loading DeepFace module (first time only) ---")
        # Suppress TensorFlow warnings before importing DeepFace
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
        
        # Import DeepFace after setting up suppression
        from deepface import DeepFace
        _deepface_module = DeepFace
        print("--- DeepFace module loaded successfully ---")
    
    return _deepface_module

def _get_twilio_client():
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_VERIFY_SID]):
        raise ValueError("Twilio credentials (ACCOUNT_SID, AUTH_TOKEN, VERIFY_SID) are not fully configured in environment variables.")
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@tool("send_otp")
def send_otp(mobile_number: str) -> str:
    """Sends a verification OTP to the given mobile number using the Twilio Verify API."""
    print(f"--- Attempting to send Twilio OTP to {mobile_number} ---")
    try:
        client = _get_twilio_client()
        verification = client.verify.v2.services(TWILIO_VERIFY_SID).verifications.create(to=mobile_number, channel='sms')
        print(f"--- Twilio verification status: {verification.status} ---")
        return f"An OTP has been sent to {mobile_number}. Please check your messages."
    except TwilioRestException as e:
        return f"Twilio Error: {e.msg}"
    except Exception as e:
        return f"An unexpected error occurred while sending OTP: {e}"

@tool("verify_otp")
def verify_otp(mobile_number: str, otp_code: str) -> str:
    """Verifies the OTP for the given mobile number using the Twilio Verify API."""
    print(f"--- Attempting to verify Twilio OTP for {mobile_number} ---")
    try:
        client = _get_twilio_client()
        verification_check = client.verify.v2.services(TWILIO_VERIFY_SID).verification_checks.create(to=mobile_number, code=otp_code)
        if verification_check.status == "approved":
            return "OTP verification successful."
        else:
            return "OTP verification failed. The code is incorrect or has expired."
    except TwilioRestException as e:
        return f"Twilio Error: {e.msg}"
    except Exception as e:
        return f"An unexpected error occurred during OTP verification: {e}"

@tool("verify_face")
def verify_face(aadhaar_image_data: bytes, live_image_data: bytes) -> str:
    """
    Compares the face from an Aadhaar card image with a live selfie using the DeepFace AI library.
    It expects image data as bytes for both images.
    """
    print("--- Starting AI Face Verification with DeepFace ---")
    
    tmp_aadhaar_file = None
    tmp_live_file = None
    
    try:
        # Lazy load DeepFace only when face verification is needed
        DeepFace = _get_deepface()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_aadhaar_file:
            tmp_aadhaar_file.write(aadhaar_image_data)
            aadhaar_path = tmp_aadhaar_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_live_file:
            tmp_live_file.write(live_image_data)
            live_path = tmp_live_file.name

        print(f"--- Comparing images: {aadhaar_path} and {live_path} ---")
        
        # Use DeepFace with suppressed output
        result = DeepFace.verify(
            img1_path=aadhaar_path,
            img2_path=live_path,
            model_name="SFace",
            enforce_detection=False,
            silent=True  # This suppresses DeepFace logs
        )
        
        print(f"--- DeepFace Result: {result} ---")

        is_verified = result.get("verified", False)
        distance = result.get("distance", 1.0)
        if is_verified:
            message = "Face verification successful. The faces are a match."
            return json.dumps({"status": "success", "match": True, "distance": distance, "message": message})
        else:
            message = "Face verification failed. The faces do not match or a face could not be detected."
            return json.dumps({"status": "success", "match": False, "distance": distance, "message": message})

    except Exception as e:
        error_message = f"An unexpected error occurred during face verification: {e}"
        print(f"--- {error_message} ---")
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        if tmp_aadhaar_file and os.path.exists(tmp_aadhaar_file.name):
            os.remove(tmp_aadhaar_file.name)
        if tmp_live_file and os.path.exists(tmp_live_file.name):
            os.remove(tmp_live_file.name)