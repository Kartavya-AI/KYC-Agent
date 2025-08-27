import os
import io
import logging
import json
from typing import Optional
from PIL import Image
import numpy as np

# Suppress TensorFlow warnings before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Twilio imports (lightweight)
from twilio.rest import Client

# Global variables for lazy loading
_deepface_module = None
_twilio_client = None

def _get_twilio_client():
    """Lazy load Twilio client"""
    global _twilio_client
    if _twilio_client is None:
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        if not account_sid or not auth_token:
            raise ValueError("Twilio credentials not found in environment variables")
        _twilio_client = Client(account_sid, auth_token)
    return _twilio_client

def _get_deepface():
    """Lazy load DeepFace module"""
    global _deepface_module
    if _deepface_module is None:
        print("Loading DeepFace module (this may take a moment)...")
        try:
            # Import with suppressed output
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                from deepface import DeepFace
            
            _deepface_module = DeepFace
            print("DeepFace loaded successfully")
        except Exception as e:
            print(f"Failed to load DeepFace: {e}")
            raise
    return _deepface_module

def send_otp(mobile_number: str) -> str:
    """
    Send OTP to the provided mobile number using Twilio Verify API
    
    Args:
        mobile_number (str): Mobile number with country code (e.g., +919876543210)
    
    Returns:
        str: Success or error message
    """
    try:
        client = _get_twilio_client()
        verify_sid = os.getenv('TWILIO_VERIFY_SID')
        
        if not verify_sid:
            return "Error: Twilio Verify SID not configured"
        
        # Send OTP verification
        verification = client.verify.v2.services(verify_sid).verifications.create(
            to=mobile_number,
            channel='sms'
        )
        
        if verification.status == 'pending':
            return f"OTP has been sent to {mobile_number}. Please check your messages."
        else:
            return f"Failed to send OTP. Status: {verification.status}"
            
    except Exception as e:
        return f"Error sending OTP: {str(e)}"

def verify_otp(mobile_number: str, otp_code: str) -> str:
    """
    Verify the OTP code for the given mobile number
    
    Args:
        mobile_number (str): Mobile number with country code
        otp_code (str): 6-digit OTP code
    
    Returns:
        str: Verification result message
    """
    try:
        client = _get_twilio_client()
        verify_sid = os.getenv('TWILIO_VERIFY_SID')
        
        if not verify_sid:
            return "Error: Twilio Verify SID not configured"
        
        # Verify OTP
        verification_check = client.verify.v2.services(verify_sid).verification_checks.create(
            to=mobile_number,
            code=otp_code
        )
        
        if verification_check.status == 'approved':
            return "OTP verification successful"
        else:
            return f"OTP verification failed. Status: {verification_check.status}"
            
    except Exception as e:
        return f"Error verifying OTP: {str(e)}"

def verify_face(aadhaar_image_data: bytes, live_image_data: bytes) -> str:
    """
    Verify if the face in the Aadhaar card matches the live selfie
    
    Args:
        aadhaar_image_data (bytes): Image data from Aadhaar card
        live_image_data (bytes): Image data from live selfie
    
    Returns:
        str: JSON string with verification result
    """
    try:
        # Lazy load DeepFace
        DeepFace = _get_deepface()
        
        # Convert bytes to PIL Images
        aadhaar_image = Image.open(io.BytesIO(aadhaar_image_data))
        live_image = Image.open(io.BytesIO(live_image_data))
        
        # Convert to RGB if necessary
        if aadhaar_image.mode != 'RGB':
            aadhaar_image = aadhaar_image.convert('RGB')
        if live_image.mode != 'RGB':
            live_image = live_image.convert('RGB')
        
        # Convert PIL images to numpy arrays
        aadhaar_array = np.array(aadhaar_image)
        live_array = np.array(live_image)
        
        # Perform face verification using DeepFace
        result = DeepFace.verify(
            img1_path=aadhaar_array,
            img2_path=live_array,
            model_name='VGG-Face',
            distance_metric='cosine',
            enforce_detection=False  # Allow processing even if face detection is uncertain
        )
        
        # Return JSON result
        return json.dumps({
            "status": "success",
            "match": bool(result["verified"]),
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"]),
            "confidence": 1.0 - float(result["distance"]),
            "model": result["model"],
            "similarity_metric": result["distance_metric"]
        })
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific DeepFace errors
        if "Face could not be detected" in error_msg:
            return json.dumps({
                "status": "error",
                "message": "Could not detect face in one or both images. Please ensure the images clearly show faces.",
                "error_type": "face_detection_failed"
            })
        elif "No face detected" in error_msg:
            return json.dumps({
                "status": "error", 
                "message": "No face detected in the provided images. Please upload clear photos with visible faces.",
                "error_type": "no_face_found"
            })
        else:
            return json.dumps({
                "status": "error",
                "message": f"Face verification failed: {error_msg}",
                "error_type": "verification_error"
            })

# Health check function for the face verification module
def check_face_verification_health() -> dict:
    """
    Check if face verification is working properly
    
    Returns:
        dict: Health status information
    """
    try:
        # Try to load DeepFace
        DeepFace = _get_deepface()
        
        return {
            "status": "healthy",
            "deepface_loaded": True,
            "available_models": ["VGG-Face", "Facenet", "OpenFace", "DeepID"],
            "message": "Face verification service is ready"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "deepface_loaded": False,
            "error": str(e),
            "message": "Face verification service is not available"
        }