# Shape and Contour Analyzer

## Overview
The **Shape and Contour Analyzer** is an interactive web application built using **Streamlit** and **OpenCV**. It allows users to detect, classify, and analyze 2D geometric shapes in images. The system provides real-time visualization of contours, geometric features, and shape distributions, making it an educational and exploratory tool for computer vision and image processing.

This project demonstrates contour-based shape detection, feature extraction (area, perimeter, compactness), and interactive thresholding, highlighting the impact of preprocessing and segmentation on object detection.

---

## Features

- Detect basic 2D geometric shapes:  
  Triangle, Square, Rectangle, Pentagon, Hexagon, Circle  
- Count objects in an image
- Display geometric features: Area, Perimeter, Compactness
- Interactive **Threshold Sensitivity** slider for segmentation control
- **Shape filter** to display selected shape categories
- Visualize:
  - Input image
  - Binary thresholded image
  - Annotated shapes with labels
  - Shape distribution chart
  - Detailed feature table
- Sidebar for **learning outcomes** and **shape legend**

---

## Demo

![Sample Demo](sample_images/cat.jpg)  

> **Live Demo:** [[Streamlit Deployment Link](#)](https://shape-and-contour-analyser-da1-22mia1063.streamlit.app/)  

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adlibalibi/shape-contour-analyser.git
cd shape-contour-analyzer
