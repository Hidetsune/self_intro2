import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os
import base64
import fitz  # pip install pymupdf
from PIL import Image
import io
import cv2
import mediapipe as mp

st.set_page_config(
    page_title="Hidetsune.T",
    page_icon=":mortar_board:",
    layout="centered"
)

st.title("Hidetsune Takahashi")

mp_face_detection = mp.solutions.face_detection

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector = mp_face_detection.FaceDetection(
            min_detection_confidence=0.6,
            model_selection=0
        )

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                score = detection.score
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame_bgr.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                text = f"Face: {score[0]*100:.1f}%"

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(frame_bgr, format="bgr24")

@st.cache_data
def get_encoded_paper_images():
    data_dir = os.path.join(".", "data")
    encoded_images = {}
    image_names = {
        "task1": "task1.png",
        "task3": "task3.png",
        "task4": "task4.png",
        "task10": "task10.png"
    }

    for key, name in image_names.items():
        image_path = os.path.join(data_dir, name)
        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as image_file:
                    encoded_images[key] = base64.b64encode(image_file.read()).decode()
            except Exception as e:
                st.error(f"Error reading image {name}: {e}")
                encoded_images[key] = None
        else:
            encoded_images[key] = None
    return encoded_images

@st.cache_data
def get_lib_detection_video_bytes():
    video_path = os.path.join(".", "data", "lib_detection_video.mp4")
    if os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                return f.read()
        except Exception as e:
            st.error(f"Video error: {e}")
    return None

def create_clickable_image_html(encoded_image, url, description):
    html = f"<p style='font-size: 1.1em; margin:8px 0;'><strong>{description}</strong></p>"
    if encoded_image:
        html += f"<a href='{url}' target='_blank'><img src='data:image/png;base64,{encoded_image}' style='width:100%;border-radius:8px;'/></a>"
    else:
        html += "<p style='color:red;'>Image not found</p>"
    return f"<div style='margin-bottom: 20px;'>{html}</div>"

with st.sidebar:
    st.title("Academic Contributions")
    st.header("📄 SemEval 2024")
    images_data = get_encoded_paper_images()

    for task, url, desc in [
        ("task1", "https://aclanthology.org/2024.semeval-1.2/", "Textual Relatedness Evaluation System"),
        ("task10", "https://aclanthology.org/2024.semeval-1.58/", "Emotion Detection in Complex Contexts"),
        ("task4", "https://aclanthology.org/2024.semeval-1.57/", "Multilingual Propaganda Memes Detection"),
        ("task3", "https://aclanthology.org/2024.semeval-1.55/", "Emotion Classification & Cause Analysis")
    ]:
        st.markdown(create_clickable_image_html(images_data.get(task), url, desc), unsafe_allow_html=True)
        st.markdown("---")

st.image("data/photo_me.jpg", width=300)

st.subheader("🎓 Waseda University")
st.markdown("""
- School of Creative Science and Engineering  
- Dept. Modern Mechanical Engineering (4th year)  
- Tokyo, Japan
""")
st.markdown("Contact: [takahashi78h@toki.waseda.jp](mailto:takahashi78h@toki.waseda.jp)")

st.header("Demo: Real-time Face Detection")
webrtc_streamer(key="demo", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})

st.header("Research: Lithium-ion Battery Sorting")
st.markdown("AI-based system for detecting LIBs in products using Transformer models.")
video_bytes = get_lib_detection_video_bytes()
if video_bytes:
    st.video(video_bytes)

st.header("PDF Preview: Fuel Cell Lab Report")
pdf_path = os.path.join(".", "data", "fuel_cell.pdf")
if os.path.exists(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_index = st.slider("Page", 1, total_pages, 1) - 1
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes()))
    st.image(img, caption=f"Page {page_index + 1} of {total_pages}")
else:
    st.warning("PDF not found")

st.header("Images from Lecture")
cols = st.columns(2)
cols[0].image("data/transformer_architecture.png", caption="Transformer Architecture")
cols[1].image("data/lecture.jpg", caption="Lecture Example")
