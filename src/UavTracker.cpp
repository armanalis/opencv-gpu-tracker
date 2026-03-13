#include "UavTracker.hpp"

UavTracker::UavTracker() {
    // Initialize Kalman Filter: 4 states (x, y, dx, dy), 2 measurements (x, y)
    KF.init(4, 2, 0);
    measurement = cv::Mat::zeros(2, 1, CV_32F);

    // Transition Matrix
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0, 
        0, 0, 0, 1);

    // Measurement Matrix
    KF.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Set noise covariances
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

    isTracking = false;
    lostFrames = 0;
}

bool UavTracker::getStatus() const {
    return isTracking;
}

bool UavTracker::updateTracker(const cv::Mat& mask, cv::Point& outKalmanPt, cv::Rect& outSensorBox) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    // Find contours from the binary mask
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    int maxAreaIDx = -1;
    double minDistance = 1e9; 

    // 1. Predict Phase
    cv::Mat prediction = KF.predict();
    outKalmanPt = cv::Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));

    // 2. Target Identification Phase
    for(size_t i = 0; i < contours.size(); i++){
        double area = cv::contourArea(contours[i]);

        if (!isTracking) {
            // Strict bounds for initial global search
            if(area > 8000 && area < 10000){ 
                if (area > maxArea) {
                    maxArea = area;
                    maxAreaIDx = (int)i;
                }
            }
        } else {
            // Loose bounds and distance calculation for active tracking mode
            if(area > 8000 && area < 10000){ 
                cv::Rect temp_box = cv::boundingRect(contours[i]);
                int temp_center_x = temp_box.x + (temp_box.width / 2);
                int temp_center_y = temp_box.y + (temp_box.height / 2);
                
                double dist = cv::norm(cv::Point(temp_center_x, temp_center_y) - outKalmanPt);
                
                // Prioritize the contour closest to the Kalman prediction
                if (dist < 150 && dist < minDistance) {
                    minDistance = dist;
                    maxAreaIDx = (int)i; 
                }
            }
        }
    }

    // 3. Sensor Update Phase
    if(maxAreaIDx != -1){
        lostFrames = 0;
        outSensorBox = cv::boundingRect(contours[maxAreaIDx]);
        
        int center_x = outSensorBox.x + (outSensorBox.width / 2);
        int center_y = outSensorBox.y + (outSensorBox.height / 2);

        measurement.at<float>(0) = (float)center_x;
        measurement.at<float>(1) = (float)center_y;

        if (!isTracking) {
            // Initialize Kalman states if locking for the first time
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
            // Correct the prediction with actual sensor data
            KF.correct(measurement);
        }
        return true; // Target locked
    } else {
        if (isTracking) {
            lostFrames++; 
            // Drop track if target is lost for more than 15 frames
            if (lostFrames > 15) {
                isTracking = false; 
            }
        }
        return false; // Target lost or still searching
    }
}