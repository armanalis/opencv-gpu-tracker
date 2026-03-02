#include <opencv2/opencv.hpp>
#include <iostream>
#include "UavTracker.hpp" // Including our custom tracking class

int main() {
    cv::VideoCapture cap("/Users/alidai/Desktop/opencv-gpu-tracker/data/input.mp4", cv::CAP_ANY);
    if(!cap.isOpened()) return -1;

    // CSRT works more reliably with cv::Mat instead of cv::UMat
    cv::Mat frame, hsvFrame, mask; 
    UavTracker ihaTracker; // Instantiate the tracking object

    while(true) {
        cap >> frame;
        if (frame.empty()) break;

        // IF NOT LOCKED ON TARGET: Global search with HSV
        if (!ihaTracker.getStatus()) {
            cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
            cv::Scalar lower_black(0, 0, 0);
            cv::Scalar upper_black(180, 255, 100); // Flexible black range for illumination changes
            
            cv::inRange(hsvFrame, lower_black, upper_black, mask);
            
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::erode(mask, mask, kernel);
            cv::dilate(mask, mask, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            double maxArea = 0;
            int maxAreaIDx = -1;

            for(size_t i = 0; i < contours.size(); i++){
                double area = cv::contourArea(contours[i]);
                // Relaxed the minimum area constraint from 6000 to 4000 for better re-detection
                if(area > 4000 && area < 15000) { 
                    if (area > maxArea) {
                        maxArea = area;
                        maxAreaIDx = i;
                    }
                }
            }

            // Once the optimal target is found, delegate the tracking to the UavTracker class!
            if (maxAreaIDx != -1) {
                cv::Rect initialBox = cv::boundingRect(contours[maxAreaIDx]);
                ihaTracker.initTracker(frame, initialBox);
            }
        } 
        // IF LOCKED ON TARGET: HSV search is disabled, CSRT and Kalman take over
        else {
            cv::Rect trackedBox;
            cv::Point kalmanPoint;
            
            bool ok = ihaTracker.updateTracker(frame, trackedBox, kalmanPoint);

            if (ok) {
                // Draw the Sensor (CSRT) bounding box in GREEN
                cv::rectangle(frame, trackedBox, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "CSRT Tracker", cv::Point(trackedBox.x, trackedBox.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            // Draw the Kalman filter prediction in RED
            cv::circle(frame, kalmanPoint, 4, cv::Scalar(0, 0, 255), -1);
            cv::Rect predictRect(kalmanPoint.x - 25, kalmanPoint.y - 25, 50, 50);
            cv::rectangle(frame, predictRect, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Hardware Accelerated UAV Tracking", frame);
        if (cv::waitKey(1) == 27) break; // Exit loop on ESC key press
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}