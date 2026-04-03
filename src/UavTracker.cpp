#include "UavTracker.hpp"

UavTracker::UavTracker() {
    KF.init(4, 2, 0);
    measurement = cv::Mat::zeros(2, 1, CV_32F);

    KF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0, 
        0, 0, 0, 1);

    KF.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

    isTracking = false;
    lostFrames = 0;
}

bool UavTracker::getStatus() const {
    return isTracking;
}

bool UavTracker::updateTracker(const std::vector<cv::Rect>& detections, cv::Point& outKalmanPt, cv::Rect& outSensorBox) {
    // 1. Predict Phase
    cv::Mat prediction = KF.predict();
    outKalmanPt = cv::Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));

    double minDistance = 1e9;
    int bestMatchIdx = -1;

    // 2. Data Association Phase (Nearest Neighbor)
    for (size_t i = 0; i < detections.size(); i++) {
        int temp_center_x = detections[i].x + (detections[i].width / 2);
        int temp_center_y = detections[i].y + (detections[i].height / 2);
        
        if (!isTracking) {
            // Initial acquisition: Lock onto the first valid detection (can be modified with heuristics)
            bestMatchIdx = (int)i;
            break;
        } else {
            // Calculate Euclidean distance between detection and Kalman prediction
            double dist = cv::norm(cv::Point(temp_center_x, temp_center_y) - outKalmanPt);
            
            // Gating threshold to reject false positives (e.g., 100 pixels)
            if (dist < 100.0 && dist < minDistance) {
                minDistance = dist;
                bestMatchIdx = (int)i; 
            }
        }
    }

    // 3. Sensor Update Phase
    if (bestMatchIdx != -1) {
        lostFrames = 0;
        outSensorBox = detections[bestMatchIdx];
        
        int center_x = outSensorBox.x + (outSensorBox.width / 2);
        int center_y = outSensorBox.y + (outSensorBox.height / 2);

        measurement.at<float>(0) = (float)center_x;
        measurement.at<float>(1) = (float)center_y;

        if (!isTracking) {
            KF.statePre.at<float>(0) = (float)center_x;
            KF.statePre.at<float>(1) = (float)center_y;
            KF.statePre.at<float>(2) = 0; 
            KF.statePre.at<float>(3) = 0;

            KF.statePost.at<float>(0) = (float)center_x;
            KF.statePost.at<float>(1) = (float)center_y;
            KF.statePost.at<float>(2) = 0; 
            KF.statePost.at<float>(3) = 0;
            isTracking = true;
        } else {
            KF.correct(measurement);
        }
        return true; 
    } else {
        if (isTracking) {
            lostFrames++; 
            if (lostFrames > 15) {
                isTracking = false; 
            }
        }
        return false; 
    }
}