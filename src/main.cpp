#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "UavTracker.hpp"
#include "Telemetry.hpp"

// In src/main.cpp
void parseYoloOutput(const cv::Mat& output, float confThreshold, std::vector<cv::Rect>& detections, int frameWidth, int frameHeight) {
    float* data = (float*)output.data;
    int dimensions = output.size[1]; // Number of columns (e.g., 84 for YOLOv8/v10 with 80 classes)
    int rows = output.size[2];       // Number of bounding box predictions

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4 * rows + i]; // Objectness score
        
        if (confidence >= confThreshold) {
            float maxClassScore = 0;
            int classId = -1;
            
            // Loop through all 80 COCO class scores (starting at tensor index 5)
            for (int c = 5; c < dimensions; ++c) {
                float score = data[c * rows + i];
                if (score > maxClassScore) {
                    maxClassScore = score;
                    classId = c - 5; 
                }
            }

            // Filter: Accept ONLY Class 0 (Human) and Class 2 (Car)
            if (maxClassScore >= confThreshold && (classId == 0 || classId == 2)) {
                float cx = data[0 * rows + i];
                float cy = data[1 * rows + i];
                float w = data[2 * rows + i];
                float h = data[3 * rows + i];

                int left = int((cx - 0.5 * w) * frameWidth);
                int top = int((cy - 0.5 * h) * frameHeight);
                int width = int(w * frameWidth);
                int height = int(h * frameHeight);
                
                detections.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
}

int main(){
    // 1. Initialize VisDrone Dataset Sequence (Adjust path to your local VisDrone dataset directory)
    std::string kaggleSequencePath = "/Users/alidai/Downloads/KaggleDataset/images/frame_%06d.jpg";
    cap.open(kaggleSequencePath, cv::CAP_IMAGES);

    if(!cap.isOpened()){
        std::cout << "Kaggle sequence not opened properly!! Check absolute path and zero-padding format.\n";
        return -1;
    }

    // Initialize YOLO Model
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/Users/alidai/Desktop/opencv-gpu-tracker/models/yolov8n.onnx");
    
    // Apple M3 Acceleration: OpenCL FP16 configuration
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);

    while(true){
        bool ok = cap.read(frame);
        if (!ok || frame.empty()) break;

        // --- 1. AI INFERENCE PIPELINE ---
        cv::Mat blob;
        // Standard YOLO scaling and target resolution (e.g., 640x640)
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);
        
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        std::vector<cv::Rect> detections;
        // Example threshold of 0.4. YOLO output shape logic depends on model generation.
        parseYoloOutput(outputs[0], 0.4f, detections, frame.cols, frame.rows);

        // --- 2. CORE TRACKING (OOP DELEGATION) ---
        cv::Point kalmanPt;
        cv::Rect sensorBox;
        
        bool isLocked = tracker.updateTracker(detections, kalmanPt, sensorBox);

        // --- 3. DATA VISUALIZATION & TELEMETRY ---
        if(isLocked){
            int center_x = sensorBox.x + (sensorBox.width / 2);
            int center_y = sensorBox.y + (sensorBox.height / 2);

            int lineLen = 15;
            cv::line(frame, cv::Point(center_x - lineLen, center_y), cv::Point(center_x + lineLen, center_y), cv::Scalar(0, 255, 0), 2);
            cv::line(frame, cv::Point(center_x, center_y - lineLen), cv::Point(center_x, center_y + lineLen), cv::Scalar(0, 255, 0), 2);
            cv::rectangle(frame, sensorBox, cv::Scalar(0, 255, 0), 1);
            
            groundStationLink.sendDataToGroundStation(center_x, center_y, true);
        } else {
            groundStationLink.sendDataToGroundStation(-1, -1, false);
        }

        if (tracker.getStatus()) {
            cv::circle(frame, kalmanPt, 4, cv::Scalar(0, 0, 255), -1);
            cv::Rect predictRect(kalmanPt.x - 25, kalmanPt.y - 25, 50, 50);
            cv::rectangle(frame, predictRect, cv::Scalar(0, 0, 255), 1);
        }

        // --- 4. ADVANCED HUD ---
        cv::putText(frame, "UAV OPTICAL TRACKING SYS v2.0 (YOLO/M3 ACCEL)", cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2);
        
        std::string statusText = tracker.getStatus() ? "SYS: TRK LOCKED" : "SYS: SEARCHING";
        cv::Scalar statusColor = tracker.getStatus() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::putText(frame, statusText, cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1.2, statusColor, 2);
        
        cv::imshow("Hardware Accelerated UAV Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}