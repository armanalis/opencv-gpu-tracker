#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>

class UavTracker {
private:
    cv::Ptr<cv::Tracker> tracker;
    cv::KalmanFilter KF;
    bool isTracking;
    int lostFrames;

public:
    UavTracker(); // Constructor
    
    // Initializes CSRT and Kalman when the target is found for the first time
    void initTracker(const cv::Mat& frame, cv::Rect initialBox);
    
    // Tracks the target in each frame and updates the Kalman prediction
    bool updateTracker(const cv::Mat& frame, cv::Rect& outputBox, cv::Point& kalmanPoint);
    
    bool getStatus() const { return isTracking; }
    void reset();
};