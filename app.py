import streamlit as st
import json
import cv2
import numpy as np
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tool import send_otp, verify_otp, verify_face

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.frame_lock:
            self.latest_frame = img
        return frame

st.set_page_config(layout="wide")

st.title("Customer KYC Verification Portal")
st.write("""
Welcome to the automated KYC verification system. This demonstration uses live APIs
for SMS and AI-powered face matching from your webcam.
""")

# --- Session State Initialization ---
if 'step' not in st.session_state:
    st.session_state.step = "initial_input"
if 'aadhaar_image' not in st.session_state:
    st.session_state.aadhaar_image = None
if 'verification_result' not in st.session_state:
    st.session_state.verification_result = None

# --- Step 1: Initial Data Input ---
if st.session_state.step == "initial_input":
    st.header("Step 1: Provide Your Details")
    with st.form("kyc_form"):
        aadhaar_number = st.text_input("Enter your 12-digit Aadhaar Number", key="aadhaar_input")
        mobile_number = st.text_input("Enter your Mobile Number in E.164 format (e.g., +919876543210)", key="mobile_input")
        uploaded_file = st.file_uploader("Upload a clear image of the front of your Aadhaar Card", type=['jpg', 'jpeg', 'png'])

        submitted = st.form_submit_button("Start OTP Verification")
        if submitted:
            if aadhaar_number and mobile_number and mobile_number.startswith('+') and len(aadhaar_number) == 12 and uploaded_file is not None:
                st.session_state.aadhaar_number = aadhaar_number
                st.session_state.mobile_number = mobile_number
                st.session_state.aadhaar_image = uploaded_file.getvalue()
                st.session_state.step = "otp_verification"
                st.rerun()
            else:
                st.error("Please fill all fields: a valid 12-digit Aadhaar, a mobile number in E.164 format, and upload your Aadhaar image.")

# --- Step 2: OTP Verification ---
if st.session_state.step == "otp_verification":
    st.header("Step 2: One-Time Password (OTP) Verification")
    # This section remains functionally the same
    st.info("Since Aadhaar numbers are not linked to a live OTP system in this demo, we will only perform live OTP verification for your mobile number.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mobile Number Verification")
        if st.button("Send OTP to my Mobile"):
            with st.spinner("Sending OTP via SMS..."):
                response = send_otp.func(st.session_state.mobile_number)
                if "sent" in response: st.success(response)
                else: st.error(response)
        mobile_otp = st.text_input("Enter the OTP you received via SMS", key="mobile_otp_input", max_chars=6)
        if st.button("Verify Mobile OTP"):
            if mobile_otp:
                with st.spinner("Verifying..."):
                    result = verify_otp.func(st.session_state.mobile_number, mobile_otp)
                    if "successful" in result:
                        st.success(f"Mobile Verification: {result}")
                        st.session_state.mobile_verified = True
                    else: st.error(f"Mobile Verification: {result}")
            else: st.warning("Please enter the OTP.")
    with col2:
        st.subheader("Aadhaar Verification (Simulated)")
        st.write("This step is simulated as we cannot send a real OTP to an Aadhaar-linked number.")
        aadhaar_verified = st.checkbox("Manually approve Aadhaar verification for this demo.")
        if aadhaar_verified:
            st.session_state.aadhaar_verified = True
            st.success("Aadhaar verification marked as complete.")
    if st.session_state.get('mobile_verified') and st.session_state.get('aadhaar_verified'):
        st.success("Both Mobile and Aadhaar have been verified!")
        if st.button("Proceed to Liveness & Face Verification"):
            st.session_state.step = "video_verification"
            st.rerun()

# --- Step 3: Live Face Verification ---
if st.session_state.step == "video_verification":
    st.header("Step 3: Live Face Verification")
    st.info("Your webcam will now activate. Position your face in the frame and click 'Capture & Verify'.")

    col1, col2 = st.columns(2)
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="kyc-video",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.write("Aadhaar Card on File:")
        st.image(st.session_state.aadhaar_image, use_container_width=True)

    if st.button("Capture & Verify Face"):
        if webrtc_ctx.video_processor:
            with webrtc_ctx.video_processor.frame_lock:
                live_image = webrtc_ctx.video_processor.latest_frame
            
            if live_image is not None:
                is_success, buffer = cv2.imencode(".jpg", live_image)
                if not is_success:
                    st.error("Could not encode captured frame.")
                else:
                    live_image_data = buffer.tobytes()
                    aadhaar_image_data = st.session_state.aadhaar_image

                    with st.spinner("Analyzing faces... This may take a moment."):
                        result_str = verify_face.func(aadhaar_image_data, live_image_data)
                        st.session_state.verification_result = json.loads(result_str)
            else:
                st.warning("No frame captured from webcam. Please make sure your face is visible and try again.")
        else:
            st.warning("Webcam is not ready. Please wait a moment for the video to start and try again.")
    
    if st.session_state.verification_result:
        st.subheader("Verification Result:")
        result_json = st.session_state.verification_result
        if result_json.get("status") == "success":
            if result_json.get("match"):
                st.success(f"✅ KYC COMPLETE! {result_json.get('message')}")
            else:
                st.error(f"❌ VERIFICATION FAILED! {result_json.get('message')}")
        else:
            st.error(f"An error occurred: {result_json.get('message')}")

    st.divider()
    if st.button("Reset Full Process"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.step = "initial_input"
        st.rerun()