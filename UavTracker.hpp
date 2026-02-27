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
    
    // Hedefi ilk kez bulduğumuzda CSRT ve Kalman'ı başlatır
    void initTracker(const cv::Mat& frame, cv::Rect initialBox);
    
    // Her karede hedefi takip eder ve Kalman tahminini günceller
    bool updateTracker(const cv::Mat& frame, cv::Rect& outputBox, cv::Point& kalmanPoint);
    
    bool getStatus() const { return isTracking; }
    void reset();
};