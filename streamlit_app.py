import streamlit as st
import subprocess
import os
import signal

# Function to run a script
def run_script(script_name):
    return subprocess.Popen(["python3", script_name])

# Function to stop a running process
def stop_process(process):
    if process:
        os.kill(process.pid, signal.SIGTERM)

# Initialize process variables
gesture_process = None
face_mesh_process = None

st.set_page_config(page_title="AI GUI", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFE0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: black;
        margin-bottom: 50px;
    }
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .button-group {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 20px;
    }
    .button {
        background-color: #FF8C00;
        color: black;
        font-size: 32px;
        padding: 20px 40px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        width: 300px;
        text-align: center;
        transition: background-color 0.3s ease;
    }
    .button:hover {
        background-color: #FFA500;
    }
    .button-text {
        font-size: 18px;
        color: black;
        margin-top: 10px; /* Adjust as needed to align perfectly */
    }
    .tooltip {
        position: absolute;
        background-color: white;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 14px;
        visibility: hidden;
    }
    .button:hover .tooltip {
        visibility: visible;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for project description
st.sidebar.title("Project Description")
st.sidebar.write("Project description text")

# Main title
st.markdown('<div class="main-title">COMPUTER VISION PROJECT</div>', unsafe_allow_html=True)

# Centered button container
st.markdown('<div class="button-container">', unsafe_allow_html=True)

# Button group 1
st.markdown('<div class="button-group">', unsafe_allow_html=True)
st.markdown('<div class="button-text" style="color: black;">Gesture Recognition</div>', unsafe_allow_html=True)
if st.button("LAUNCH", key="gesture_button"):
    if face_mesh_process:
        stop_process(face_mesh_process)
        face_mesh_process = None
    gesture_process = run_script("gesture_rec.py")
st.markdown('<div class="button-text">Press \'q\' to exit window</div>', unsafe_allow_html=True)
st.markdown('<div class="tooltip">Launch Gesture Recognition</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Button group 2
st.markdown('<div class="button-group">', unsafe_allow_html=True)
st.markdown('<div class="button-text" style="color: black;">Face Mesh</div>', unsafe_allow_html=True)
if st.button("LAUNCH", key="face_mesh_button"):
    if gesture_process:
        stop_process(gesture_process)
        gesture_process = None
    face_mesh_process = run_script("face_mesh.py")
st.markdown('<div class="button-text">Press \'q\' to exit window</div>', unsafe_allow_html=True)
st.markdown('<div class="tooltip">Launch Face Mesh</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Pause button
if st.sidebar.button("Pause"):
    if gesture_process:
        stop_process(gesture_process)
        gesture_process = None
    if face_mesh_process:
        stop_process(face_mesh_process)
        face_mesh_process = None
