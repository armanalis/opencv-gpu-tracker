#include <opencv2/opencv.hpp>
#include <iostream>

int main(){

    cv::VideoCapture cap;
    cap.open("/Users/alidai/Desktop/opencv-gpu-tracker/data/input.mp4", cv::CAP_ANY);

    if(!cap.isOpened()){
        std::cout << "Video is not opened properly!!\n";
        return -1;
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::UMat frame, grayFrame, enhancedFrame;

    while(true){

        bool ok = cap.read(frame);

        if (!ok)//checking if the read has failed or not
        {
            std::cout << "read() failed\n";
            break;
        }

        if (frame.empty())//checking if the frame is empty or not
        {
            std::cout << "frame empty\n";
            break;
        }

        //changing to gray tone
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        //usage of CLAHE
        clahe->apply(grayFrame, enhancedFrame);

        //Show the original and processed image side by side
        cv::imshow("Original IHA cam", frame);
        cv::imshow("T-API Enchanced Frame (CLAHE)", enhancedFrame);

        int key = cv::waitKey(1);

        if (key == 27)//ESC key to quit
            break;
        }

    std::cout << "Loop ended\n";

    cap.release();
    cv::destroyAllWindows();
    return 0;
    


}