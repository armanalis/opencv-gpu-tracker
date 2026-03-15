#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "UavTracker.hpp"
#include "Telemetry.hpp"

int main(){

    cv::VideoCapture cap;
    cap.open("/Users/alidai/Desktop/opencv-gpu-tracker/data/input.mp4", cv::CAP_ANY);

    if(!cap.isOpened()){
        std::cout << "Video is not opened properly!!\n";
        return -1;
    }

    cv::Mat frame, hsvFrame, mask, enhancedFrame;

    // Instantiate OOP modules
    UavTracker tracker;
    TelemetrySender groundStationLink;

    // CLAHE Initialization for Image Enhancement
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    while(true){

        bool ok = cap.read(frame);
        if (!ok || frame.empty()) break;

        // --- 1. IMAGE ENHANCEMENT PIPELINE ---
        cv::Mat labFrame;
        cv::cvtColor(frame, labFrame, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> labPlanes;
        cv::split(labFrame, labPlanes);
        clahe->apply(labPlanes[0], labPlanes[0]); 
        cv::merge(labPlanes, labFrame);
        cv::cvtColor(labFrame, enhancedFrame, cv::COLOR_Lab2BGR);

        cv::cvtColor(enhancedFrame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::Scalar lower_black(0, 0, 0);
        cv::Scalar upper_black(180, 255, 100);
        cv::inRange(hsvFrame, lower_black, upper_black, mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::erode(mask, mask, kernel);
        cv::dilate(mask, mask, kernel);

        // --- 2. CORE TRACKING (OOP DELEGATION) ---
        cv::Point kalmanPt;
        cv::Rect sensorBox;
        
        // Pass the mask to the tracker class and get lock status
        bool isLocked = tracker.updateTracker(mask, kalmanPt, sensorBox);

        // --- 3. DATA VISUALIZATION & TELEMETRY ---
        if(isLocked){
            int center_x = sensorBox.x + (sensorBox.width / 2);
            int center_y = sensorBox.y + (sensorBox.height / 2);

            // Draw AR Crosshair (Target Lock)
            int lineLen = 15;
            cv::line(frame, cv::Point(center_x - lineLen, center_y), cv::Point(center_x + lineLen, center_y), cv::Scalar(0, 255, 0), 2);
            cv::line(frame, cv::Point(center_x, center_y - lineLen), cv::Point(center_x, center_y + lineLen), cv::Scalar(0, 255, 0), 2);
            cv::rectangle(frame, sensorBox, cv::Scalar(0, 255, 0), 1);
            
            // Dispatch telemetry
            groundStationLink.sendDataToGroundStation(center_x, center_y, true);
        } else {
            groundStationLink.sendDataToGroundStation(-1, -1, false);
        }

        // Always draw Kalman Prediction if track is initiated
        if (tracker.getStatus()) {
            cv::circle(frame, kalmanPt, 4, cv::Scalar(0, 0, 255), -1);
            cv::Rect predictRect(kalmanPt.x - 25, kalmanPt.y - 25, 50, 50);
            cv::rectangle(frame, predictRect, cv::Scalar(0, 0, 255), 1);
        }

        // --- 4. ADVANCED HUD (Heads Up Display) ---
        cv::putText(frame, "UAV OPTICAL TRACKING SYS v1.0", cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2);
        
        std::string statusText = tracker.getStatus() ? "SYS: TRK LOCKED" : "SYS: SEARCHING";
        cv::Scalar statusColor = tracker.getStatus() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::putText(frame, statusText, cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1.2, statusColor, 2);
        
        cv::putText(frame, "ALT: 1200m AGL", cv::Point(10, frame.rows - 50), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255), 1);
        cv::putText(frame, "LINK: UDP ESTABLISHED", cv::Point(10, frame.rows - 30), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255), 1);

        cv::imshow("Hardware Accelerated UAV Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}