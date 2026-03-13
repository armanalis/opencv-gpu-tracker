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
    UavTracker(); // Constructor
    
    // Takes the binary mask, updates Kalman, and returns true if target is locked.
    // Outputs the predicted Kalman point and the actual Sensor bounding box.
    bool updateTracker(const cv::Mat& mask, cv::Point& outKalmanPt, cv::Rect& outSensorBox);
    
    // Returns the current status of the tracking system
    bool getStatus() const;
};