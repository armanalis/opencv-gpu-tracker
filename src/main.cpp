#include <opencv2/opencv.hpp>
#include <iostream>

int main(){

    cv::VideoCapture cap;
    cap.open("/Users/alidai/Desktop/opencv-gpu-tracker/data/input.mp4", cv::CAP_ANY);

    if(!cap.isOpened()){
        std::cout << "Video is not opened properly!!\n";
        return -1;
    }

    cv::UMat frame, hsvFrame, mask;

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

        //convert original BGR fram to HSV(Hue, Saturation, Value)
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

        //we are keeping the boundaries for the black objects
        cv::Scalar lower_black(0, 0, 0);
        cv::Scalar upper_black(180, 255, 100);

        //Masking process only black frames will survive.
        cv::inRange(hsvFrame, lower_black, upper_black, mask);

        //Cleaning the noise.
        /*
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

        cv::erode(mask, mask, kernel);//erase some shadows
        cv::dilate(mask, mask, kernel);//restore the size of the actual target
        */

        //Show the original and processed image side by side
        cv::imshow("Original IHA cam", frame);
        cv::imshow("Black cam", mask);

        int key = cv::waitKey(1);

        if (key == 27)//ESC key to quit
            break;
        }

    std::cout << "Loop ended\n";

    cap.release();
    cv::destroyAllWindows();
    return 0;
    


}