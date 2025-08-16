import os
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import structlog
from datetime import datetime
import json

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistent results

# Suppress TensorFlow warnings before importing
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import httpx

from tool import send_otp, verify_otp

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global variable to store verification results
verification_results = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    logger.info("Starting KYC Agent API - Fast startup mode")
    
    # Validate required environment variables
    required_env_vars = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN", 
        "TWILIO_VERIFY_SID"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error("Missing required environment variables", missing_vars=missing_vars)
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("KYC Agent API initialized successfully - DeepFace will load on first face verification request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down KYC Agent API")

# Initialize FastAPI app
app = FastAPI(
    title="KYC Agent API",
    description="AI-powered Know Your Customer (KYC) verification system with OTP and face verification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class KYCInitiateRequest(BaseModel):
    aadhaar_number: str = Field(..., description="12-digit Aadhaar number")
    mobile_number: str = Field(..., description="Mobile number with country code (e.g., +919876543210)")
    
    @validator('aadhaar_number')
    def validate_aadhaar(cls, v):
        # Remove any spaces or hyphens
        clean_aadhaar = v.replace(' ', '').replace('-', '')
        if not clean_aadhaar.isdigit() or len(clean_aadhaar) != 12:
            raise ValueError('Aadhaar number must be 12 digits')
        return v
    
    @validator('mobile_number')
    def validate_mobile(cls, v):
        if not v.startswith('+'):
            raise ValueError('Mobile number must include country code (e.g., +91)')
        return v

class OTPVerificationRequest(BaseModel):
    mobile_number: str = Field(..., description="Mobile number with country code")
    otp_code: str = Field(..., description="6-digit OTP code")
    
    @validator('otp_code')
    def validate_otp(cls, v):
        if not v.isdigit() or len(v) != 6:
            raise ValueError('OTP must be 6 digits')
        return v

class VideoVerificationRequest(BaseModel):
    aadhaar_number: str = Field(..., description="12-digit Aadhaar number")
    mobile_number: str = Field(..., description="Mobile number with country code")

class KYCResponse(BaseModel):
    status: str = Field(..., description="Response status: success, error, or pending")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Dict[Any, Any]] = Field(None, description="Additional response data")
    request_id: str = Field(..., description="Unique request identifier")

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str

# In-memory storage for session tracking (use Redis in production)
active_sessions = {}

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "KYC Agent API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "production")
    )

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service status"""
    try:
        # Check if DeepFace module is loaded
        from tool import _deepface_module
        face_verification_loaded = _deepface_module is not None
    except:
        face_verification_loaded = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "services": {
            "api": "ready",
            "otp_service": "ready",
            "face_verification": "loaded" if face_verification_loaded else "ready_to_load"
        }
    }

@app.post("/api/v1/kyc/initiate", response_model=KYCResponse)
async def initiate_kyc(request: KYCInitiateRequest, background_tasks: BackgroundTasks):
    """
    Initiate KYC process by sending OTP to mobile number
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "KYC initiation requested",
            request_id=request_id,
            aadhaar_number=request.aadhaar_number[:4] + "****" + request.aadhaar_number[-4:],
            mobile_number=request.mobile_number[:3] + "****" + request.mobile_number[-4:]
        )
        
        # Send OTP using the custom tool
        otp_result = send_otp(request.mobile_number)
        
        # Store session data
        active_sessions[request_id] = {
            "aadhaar_number": request.aadhaar_number,
            "mobile_number": request.mobile_number,
            "otp_sent": True,
            "otp_verified": False,
            "video_verified": False,
            "created_at": datetime.utcnow(),
            "status": "otp_pending"
        }
        
        if "OTP has been sent" in otp_result:
            return KYCResponse(
                status="success",
                message="OTP sent successfully. Please verify to continue.",
                data={"next_step": "verify_otp"},
                request_id=request_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to send OTP: {otp_result}"
            )
            
    except Exception as e:
        logger.error("Error during KYC initiation", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during KYC initiation: {str(e)}"
        )

@app.post("/api/v1/kyc/verify-otp", response_model=KYCResponse)
async def verify_otp_endpoint(request: OTPVerificationRequest, request_id: str):
    """
    Verify OTP for KYC process
    """
    try:
        # Find session
        if request_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid or expired request ID"
            )
        
        session = active_sessions[request_id]
        
        if session["mobile_number"] != request.mobile_number:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mobile number mismatch"
            )
        
        logger.info("OTP verification requested", request_id=request_id)
        
        # Verify OTP using custom tool
        verification_result = verify_otp(request.mobile_number, request.otp_code)
        
        if "successful" in verification_result.lower():
            # Update session
            session["otp_verified"] = True
            session["status"] = "otp_verified"
            
            return KYCResponse(
                status="success",
                message="OTP verified successfully. You can now proceed to video verification.",
                data={
                    "next_step": "video_verification",
                    "otp_verified": True
                },
                request_id=request_id
            )
        else:
            return KYCResponse(
                status="error",
                message="OTP verification failed. Please check the code and try again.",
                data={"otp_verified": False},
                request_id=request_id
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during OTP verification", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during OTP verification: {str(e)}"
        )

@app.post("/api/v1/kyc/verify-face", response_model=KYCResponse)
async def verify_face_endpoint(
    request_id: str,
    aadhaar_photo: UploadFile = File(..., description="Aadhaar card photo"),
    live_photo: UploadFile = File(..., description="Live selfie photo")
):
    """
    Perform face verification between Aadhaar photo and live selfie
    """
    try:
        # Find session
        if request_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid or expired request ID"
            )
        
        session = active_sessions[request_id]
        
        if not session.get("otp_verified", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP verification required before face verification"
            )
        
        logger.info("Face verification requested", request_id=request_id)
        
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        if aadhaar_photo.content_type not in allowed_types or live_photo.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG and PNG images are supported"
            )
        
        # Read image data
        aadhaar_data = await aadhaar_photo.read()
        live_data = await live_photo.read()
        
        # Lazy import face verification to avoid startup delays
        try:
            from tool import verify_face
        except ImportError as e:
            logger.error("Failed to import face verification module", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Face verification service unavailable"
            )
        
        # Verify faces using custom tool
        verification_result = verify_face(aadhaar_data, live_data)
        result_data = json.loads(verification_result)
        
        # Update session
        session["video_verified"] = result_data.get("match", False)
        session["status"] = "completed" if result_data.get("match", False) else "face_verification_failed"
        session["verification_result"] = result_data
        
        if result_data.get("status") == "success" and result_data.get("match"):
            return KYCResponse(
                status="success",
                message="Face verification successful. KYC process completed.",
                data={
                    "face_match": True,
                    "confidence_score": 1.0 - result_data.get("distance", 1.0),
                    "kyc_completed": True
                },
                request_id=request_id
            )
        elif result_data.get("status") == "success":
            return KYCResponse(
                status="error",
                message="Face verification failed. The faces do not match sufficiently.",
                data={
                    "face_match": False,
                    "confidence_score": 1.0 - result_data.get("distance", 1.0),
                    "kyc_completed": False
                },
                request_id=request_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Face verification error: {result_data.get('message', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during face verification", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during face verification: {str(e)}"
        )

@app.post("/api/v1/kyc/complete-verification", response_model=KYCResponse)
async def complete_kyc_verification(request_id: str):
    """
    Complete the KYC verification process by checking all verification steps
    """
    try:
        # Find session
        if request_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid or expired request ID"
            )
        
        session = active_sessions[request_id]
        
        logger.info("Complete KYC verification requested", request_id=request_id)
        
        # Check if all required verifications are complete
        otp_verified = session.get("otp_verified", False)
        face_verified = session.get("video_verified", False)
        
        if not otp_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP verification required before completing KYC"
            )
        
        if not face_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Face verification required before completing KYC"
            )
        
        # Mark KYC as fully completed
        session["kyc_completed"] = True
        session["status"] = "kyc_completed"
        session["completed_at"] = datetime.utcnow()
        
        # Calculate overall confidence score
        verification_result = session.get("verification_result", {})
        face_confidence = 1.0 - verification_result.get("distance", 1.0) if verification_result.get("distance") is not None else 0.8
        
        return KYCResponse(
            status="success",
            message="KYC verification completed successfully. All checks passed.",
            data={
                "kyc_status": "completed",
                "otp_verified": True,
                "face_verified": True,
                "overall_confidence": face_confidence,
                "completed_at": session["completed_at"].isoformat(),
                "verification_summary": {
                    "mobile_verification": "passed",
                    "identity_verification": "passed",
                    "face_match_confidence": face_confidence
                }
            },
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during KYC completion", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during KYC completion: {str(e)}"
        )

@app.get("/api/v1/kyc/status/{request_id}", response_model=KYCResponse)
async def get_kyc_status(request_id: str):
    """
    Get the current status of a KYC process
    """
    try:
        if request_id not in active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Request ID not found"
            )
        
        session = active_sessions[request_id]
        
        return KYCResponse(
            status="success",
            message="KYC status retrieved successfully",
            data={
                "current_status": session["status"],
                "otp_sent": session.get("otp_sent", False),
                "otp_verified": session.get("otp_verified", False),
                "face_verified": session.get("video_verified", False),
                "kyc_completed": session.get("kyc_completed", False),
                "created_at": session["created_at"].isoformat(),
                "completed_at": session.get("completed_at", {}).isoformat() if session.get("completed_at") else None,
                "aadhaar_masked": session["aadhaar_number"][:4] + "****" + session["aadhaar_number"][-4:],
                "mobile_masked": session["mobile_number"][:3] + "****" + session["mobile_number"][-4:]
            },
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving KYC status", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error retrieving status: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(
        "HTTP exception occurred",
        path=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        detail=exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(
        "Unexpected error occurred",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)