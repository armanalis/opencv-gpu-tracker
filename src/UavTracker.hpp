#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class UavTracker {
private:
    cv::KalmanFilter KF;
    cv::Mat measurement;
    bool isTracking;
    int lostFrames;

public:
    UavTracker();
    
    // Evaluates YOLO detections, associates the best target via Euclidean distance,
    // updates the Kalman Filter, and outputs the prediction and matched sensor box.
    bool updateTracker(const std::vector<cv::Rect>& detections, cv::Point& outKalmanPt, cv::Rect& outSensorBox);
    
    bool getStatus() const;
};