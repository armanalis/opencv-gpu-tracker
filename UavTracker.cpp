#include "UavTracker.hpp"

UavTracker::UavTracker() {
    isTracking = false;
    lostFrames = 0;
    
    // Initialize Kalman Filter (4 States: x, y, vx, vy | 2 Measurements: x, y)
    KF.init(4, 2, 0);
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1, 0, 1, 0,  
        0, 1, 0, 1,  
        0, 0, 1, 0,  
        0, 0, 0, 1);
    
    cv::setIdentity(KF.measurementMatrix);
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-2)); // Agile Kalman setup
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));
}

void UavTracker::initTracker(const cv::Mat& frame, cv::Rect initialBox) {
    // 1. Initialize CSRT Tracker (Tracks texture and shape, highly resistant to illumination)
    // Note: If CSRT fails on Mac, cv::TrackerKCF::create() can be used as an alternative.
    tracker = cv::TrackerCSRT::create(); 
    tracker->init(frame, initialBox);

    // 2. Set the initial position for the Kalman Filter
    int center_x = initialBox.x + initialBox.width / 2;//our box starts from the leftmost point
    //and we want the center of x's so we add half of the width

    int center_y = initialBox.y + initialBox.height / 2;//our box starts from the topmost point
    //and we want the center of y's so we add half of the height
    
    KF.statePre.at<float>(0) = center_x;
    KF.statePre.at<float>(1) = center_y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    
    KF.statePost = KF.statePre.clone();
    
    isTracking = true;
    lostFrames = 0;
    std::cout << "[SYSTEM] Target Locked. CSRT Tracker Activated!" << std::endl;
}

bool UavTracker::updateTracker(const cv::Mat& frame, cv::Rect& outputBox, cv::Point& kalmanPoint) {
    if (!isTracking) return false;

    // 1. Kalman Prediction Phase
    cv::Mat prediction = KF.predict();
    kalmanPoint = cv::Point(prediction.at<float>(0), prediction.at<float>(1));

    // 2. CSRT Sensor Update Phase
    bool ok = tracker->update(frame, outputBox);

    if (ok) {
        // --- ANTI-COLLAPSE GUARD ---
        // If the bounding box shrinks to an impossibly small size, CSRT has failed (Scale Drift)!
        // We force a reset if width/height is less than 30 pixels or area is too small.
        if (outputBox.width < 30 || outputBox.height < 30 || outputBox.area() < 1000) {
            std::cout << "[WARNING] Bounding box collapsed! Forcing reset..." << std::endl;
            reset();
            return false; // Force failure to go back to global search
        }

        lostFrames = 0; // Target is secured
        int center_x = outputBox.x + outputBox.width / 2;
        int center_y = outputBox.y + outputBox.height / 2;

        // Correct Kalman with real sensor data
        cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
        measurement.at<float>(0) = center_x;
        measurement.at<float>(1) = center_y;
        KF.correct(measurement);
        return true;
    } else {
        lostFrames++;
        // Reset system if CSRT loses the target for 30 consecutive frames (approx. 1 sec)
        if (lostFrames > 30) {
            reset();
        }
        return false;
    }
}

void UavTracker::reset() {
    isTracking = false;
    lostFrames = 0;
    std::cout << "[SYSTEM] Target Lost! Returning to Global Search Mode..." << std::endl;
}