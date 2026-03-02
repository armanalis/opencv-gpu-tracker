#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include "UavTracker.hpp"
#include "ThreadSafeQueue.hpp"

// Global Queues for Inter-Thread Communication
ThreadSafeQueue<cv::Mat> rawFrameQueue;       // Stores raw frames directly from the camera/video
ThreadSafeQueue<cv::Mat> processedFrameQueue; // Stores frames after AI/Kalman processing and drawing

// Safe flag to stop all threads cleanly. Atomic ensures thread-safe read/write operations.
std::atomic<bool> isRunning(true);

// ---------------------------------------------------------
// THREAD 1: VIDEO CAPTURE (Producer)
// ---------------------------------------------------------
void captureThreadFunc() {
    cv::VideoCapture cap("/Users/alidai/Desktop/opencv-gpu-tracker/data/input.mp4", cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open video source!" << std::endl;
        isRunning = false;
        return;
    }

    cv::Mat frame;
    while (isRunning) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "[SYSTEM] Video stream ended." << std::endl;
            isRunning = false; 
            break;
        }
        
        // Push the raw frame to the queue for the processing thread
        rawFrameQueue.push(frame.clone()); 
        
        // Simulate a real-time camera feed (approx. 30 FPS) to prevent queue overflow
        std::this_thread::sleep_for(std::chrono::milliseconds(30)); 
    }
    cap.release();
}

// ---------------------------------------------------------
// THREAD 2: TRACKING & PROCESSING (Brain)
// ---------------------------------------------------------
void processingThreadFunc() {
    UavTracker ihaTracker;
    cv::Mat frame, hsvFrame, mask;

    while (isRunning) {
        // Pop a frame from the queue. If empty, the thread safely sleeps until a frame arrives.
        rawFrameQueue.pop(frame);
        if (frame.empty()) continue;

        // IF NOT LOCKED ON TARGET: Global search with HSV
        if (!ihaTracker.getStatus()) {
            cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
            cv::Scalar lower_black(0, 0, 0);
            cv::Scalar upper_black(180, 255, 100);
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
                if(area > 4000 && area < 15000) { 
                    if (area > maxArea) {
                        maxArea = area;
                        maxAreaIDx = i;
                    }
                }
            }

            // Initialize tracker if a valid target is found
            if (maxAreaIDx != -1) {
                cv::Rect initialBox = cv::boundingRect(contours[maxAreaIDx]);
                ihaTracker.initTracker(frame, initialBox);
            }
        } 
        // IF LOCKED ON TARGET: Update CSRT and Kalman
        else {
            cv::Rect trackedBox;
            cv::Point kalmanPoint;
            
            bool ok = ihaTracker.updateTracker(frame, trackedBox, kalmanPoint);

            if (ok) {
                // Draw Sensor bounding box
                cv::rectangle(frame, trackedBox, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "CSRT Tracker", cv::Point(trackedBox.x, trackedBox.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            // Draw Kalman prediction
            cv::circle(frame, kalmanPoint, 4, cv::Scalar(0, 0, 255), -1);
            cv::Rect predictRect(kalmanPoint.x - 25, kalmanPoint.y - 25, 50, 50);
            cv::rectangle(frame, predictRect, cv::Scalar(0, 0, 255), 2);
        }

        // Push the fully processed and drawn frame to the display queue
        processedFrameQueue.push(frame);
    }
}

// ---------------------------------------------------------
// THREAD 3: DISPLAY CONTROLLER (UI)
// ---------------------------------------------------------
void displayThreadFunc() {
    cv::Mat frame;
    while (isRunning) {
        // Retrieve the processed frame
        processedFrameQueue.pop(frame);
        if (frame.empty()) continue;

        cv::imshow("Multi-Threaded UAV Target Tracker", frame);
        
        // Listen for ESC key to terminate the program safely
        if (cv::waitKey(1) == 27) { 
            std::cout << "[SYSTEM] Termination signal received (ESC)." << std::endl;
            isRunning = false; 
            break;
        }
    }
    cv::destroyAllWindows();
}

// ---------------------------------------------------------
// MAIN ORCHESTRATOR
// ---------------------------------------------------------
int main() {
    std::cout << "[SYSTEM] Initializing Multi-Threaded Architecture..." << std::endl;

    // Spawn the threads concurrently
    std::thread captureThread(captureThreadFunc);
    std::thread processingThread(processingThreadFunc);
    std::thread displayThread(displayThreadFunc);

    // Wait for all threads to finish their execution before closing the main process
    if(captureThread.joinable()) captureThread.join();
    if(processingThread.joinable()) processingThread.join();
    if(displayThread.joinable()) displayThread.join();

    std::cout << "[SYSTEM] System shutdown completed safely." << std::endl;
    return 0;
}