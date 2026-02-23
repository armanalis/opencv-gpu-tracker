#include <opencv2/opencv.hpp>
#include <iostream>

int main(){

    cv::VideoCapture cap;
    cap.open("/Users/alidai/Desktop/opencv-gpu-tracker/data/input.mp4", cv::CAP_ANY);

    if(!cap.isOpened()){
        std::cout << "Video is not opened properly!!\n";
        return -1;
    }

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::Mat frame;

    while(true){

        bool ok = cap.read(frame);

        if (!ok)
        {
            std::cout << "read() failed\n";
            break;
        }

        if (frame.empty())
        {
            std::cout << "frame empty\n";
            break;
        }

        std::cout << "Frame OK: "
                  << frame.cols << " x "
                  << frame.rows << std::endl;

        cv::imshow("Video", frame);

        // macOS için minimum 1 ms gerekli
        int key = cv::waitKey(1);

        if (key == 'q')
            break;
        }

    std::cout << "Loop ended\n";

    cap.release();
    cv::destroyAllWindows();
    return 0;
    


}