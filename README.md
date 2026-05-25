# UAV Optical Tracking System

A real-time, hardware-accelerated object detection and target tracking pipeline built in C++ for Unmanned Aerial Vehicle (UAV) camera systems. 

## Features

* **AI Inference Pipeline**: Executes a YOLOv8n ONNX model using OpenCV's DNN module, featuring custom output parsing strictly filtered to detect humans (Class 0) and cars (Class 2).
* **Hardware Acceleration**: Utilizes `DNN_BACKEND_OPENCV` and `DNN_TARGET_OPENCL_FP16` for high-frame-rate performance natively optimized for OpenCL-compatible hardware, including the MacBook Air M3.
* **Predictive Tracking**: Implements a 4-state constant velocity Kalman Filter (position and velocity in 2D space) to predict trajectories and maintain target lock even through brief occlusions.
* **Data Association & Gating**: Associates raw detections with the Kalman state using a Nearest Neighbor algorithm based on Euclidean distance, applying a 100-pixel spatial gating threshold to reject false positives.
* **Ground Station Telemetry**: Includes a modular interface simulating a UDP communication channel to transmit live target coordinates and system lock states to a Ground Control Station.
* **Advanced HUD**: Renders real-time bounding boxes, crosshairs, Kalman-predicted trajectory markers, and system status overlays directly onto the video feed.

## Architecture

* `CMakeLists.txt`: Build configuration supporting C++17 and OpenCV.
* `src/main.cpp`: Core application loop, dataset parsing, AI inference pipeline, and HUD rendering.
* `src/UavTracker.hpp` & `src/UavTracker.cpp`: Kalman filter logic and nearest neighbor target association implementation.
* `src/Telemetry.hpp`: Distributed communication architecture mockups for UAV-to-GCS links.

## Prerequisites

* **Compiler**: C++17 compatible compiler.
* **Build System**: CMake 3.16+.
* **Dependencies**: OpenCV compiled with OpenCL and DNN module support.

## Build Instructions

1.  Clone the repository and initialize the build directory:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
    This compiles the source and generates the `tracker` executable.

## Usage & Configuration

Prior to executing the tracking pipeline, update the absolute paths in `src/main.cpp` to target your local data:

1.  **Image Sequence:** Specify the path and zero-padding format for the sequential dataset (e.g., Kaggle/VisDrone sequences):
    ```cpp
    std::string kaggleSequencePath = "/Users/alidai/Downloads/KaggleDataset/images/frame_%06d.jpg";
    ```
2.  **Model File:** Set the correct path to your downloaded `yolov8n.onnx` file:
    ```cpp
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/Users/alidai/Desktop/opencv-gpu-tracker/models/yolov8n.onnx");
    ```

Once configured and built, run the executable:
```bash
./tracker
