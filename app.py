import streamlit as st
import cv2
import numpy as np
import pandas as pd
import math

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="Shape and Contour Analyser", layout="wide")

# =====================================================
# SIDEBAR: Learning Outcomes, Legend & Filters
# =====================================================
st.sidebar.markdown("## 📘 Learning Outcomes")
st.sidebar.markdown("""
- **Contour Detection** using OpenCV  
- **Geometric Shape Recognition**  
- **Object Counting** from contours  
- **Feature Extraction** (Area & Perimeter)  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧩 Shape Legend")
st.sidebar.markdown("""
- **Triangle** – 3 contour vertices  
- **Square** – 4 equal sides  
- **Rectangle** – 4 sides, unequal aspect ratio  
- **Pentagon** – 5 vertices  
- **Hexagon** – 6 vertices  
- **Circle** – More than 6 vertices (smooth contour)  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Shape Filter")

# =====================================================
# Main Title
# =====================================================
st.markdown("## Shape and Contour Analyser")
st.caption(
    "Interactive dashboard demonstrating contour detection and geometric feature extraction"
)

# =====================================================
# Shape Detection Function
# =====================================================
def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        return "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    elif vertices > 6:
        return "Circle"
    else:
        return "Unknown"

# =====================================================
# Image Selection
# =====================================================
mode = st.radio(
    "Select input mode:",
    ["Use sample image (demo)", "Upload your own image"]
)

image = None

if mode == "Use sample image (demo)":
    image = cv2.imread("sample images\cat.jpg")
    st.info("Using preloaded sample image for demonstration.")
    if image is None:
        st.error("❌ Sample image not found. Check folder path.")
        st.stop()
else:
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# =====================================================
# Processing Pipeline
# =====================================================
if image is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_val = st.slider("Threshold Sensitivity", 0, 255, 130)
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

    with col2:
        st.subheader("Binary Image (Contours Source)")
        st.image(thresh, use_container_width=True)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 150]

    output = image.copy()
    records = []

    # Detect shapes first (for filter)
    detected_shapes = [detect_shape(cnt) for cnt in contours]
    unique_shapes = sorted(set(detected_shapes))

    selected_shapes = st.sidebar.multiselect(
        "Display shapes:",
        options=unique_shapes,
        default=unique_shapes
    )

    # Draw & extract features
    for cnt in contours:
        shape = detect_shape(cnt)
        if shape not in selected_shapes:
            continue

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        compactness = (
            (4 * math.pi * area) / (perimeter ** 2)
            if perimeter != 0 else 0
        )

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(
            output, shape,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 0, 0), 2
        )

        records.append([
            shape,
            round(area, 2),
            round(perimeter, 2),
            round(compactness, 3)
        ])

    df = pd.DataFrame(
        records,
        columns=["Shape", "Area (px²)", "Perimeter (px)", "Compactness"]
    )

    # =====================================================
    # Metrics
    # =====================================================
    shape_counts = df["Shape"].value_counts()
    diversity = len(shape_counts)

    m1, m2, m3 = st.columns(3)
    m1.metric("Objects Detected", len(df))
    m2.metric("Unique Shapes", diversity)
    m3.metric("Contours Extracted", len(contours))

    # =====================================================
    # Results
    # =====================================================
    st.subheader("Detected Shapes & Contours")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)


    st.subheader("Shape Distribution")
    st.bar_chart(shape_counts)

    st.subheader("Extracted Geometric Features")
    st.dataframe(df)

else:
    st.warning("Please select a mode and provide an image.")
