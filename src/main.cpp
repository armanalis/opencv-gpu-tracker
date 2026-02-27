#include <opencv2/opencv.hpp>
#include <iostream>
#include "UavTracker.hpp" // Yazdığımız sınıfı dahil ettik

int main() {
    cv::VideoCapture cap("/Users/alidai/Desktop/opencv-gpu-tracker/data/input.mp4", cv::CAP_ANY);
    if(!cap.isOpened()) return -1;

    cv::Mat frame, hsvFrame, mask; // CSRT UMat ile değil Mat ile daha stabil çalışır
    UavTracker ihaTracker; // Sınıfımızdan bir nesne (obje) ürettik!

    while(true) {
        cap >> frame;
        if (frame.empty()) break;

        // EĞER HEDEFE KİLİTLİ DEĞİLSEK: HSV ile bul (Eski yöntemin sadece ilk anı)
        if (!ihaTracker.getStatus()) {
            cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
            cv::Scalar lower_black(0, 0, 0);
            cv::Scalar upper_black(180, 255, 100); // Esnek bir siyah aralığı
            
            cv::inRange(hsvFrame, lower_black, upper_black, mask);
            
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
            cv::erode(mask, mask, kernel);
            cv::dilate(mask, mask, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            double maxArea = 0;
            int maxAreaIDx = -1;

            for(size_t i = 0; i < contours.size(); i++){
                double area = cv::contourArea(contours[i]);
                if(area > 6000 && area < 15000) { // Sıkı alan sınırıyla adamın kamerasını bul
                    if (area > maxArea) {
                        maxArea = area;
                        maxAreaIDx = i;
                    }
                }
            }

            // İlk uygun nesneyi bulduğumuzda Tracker'a ver ve işi ona bırak!
            if (maxAreaIDx != -1) {
                cv::Rect initialBox = cv::boundingRect(contours[maxAreaIDx]);
                ihaTracker.initTracker(frame, initialBox);
            }
        } 
        // EĞER HEDEFE KİLİTLİYSEK: Artık HSV yok, CSRT ve Kalman çalışıyor!
        else {
            cv::Rect trackedBox;
            cv::Point kalmanPoint;
            
            bool ok = ihaTracker.updateTracker(frame, trackedBox, kalmanPoint);

            if (ok) {
                // Sensör (CSRT) kutusunu YEŞİL çiz
                cv::rectangle(frame, trackedBox, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "CSRT Tracker", cv::Point(trackedBox.x, trackedBox.y - 10), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            // Kalman tahminini KIRMIZI çiz
            cv::circle(frame, kalmanPoint, 4, cv::Scalar(0, 0, 255), -1);
            cv::Rect predictRect(kalmanPoint.x - 25, kalmanPoint.y - 25, 50, 50);
            cv::rectangle(frame, predictRect, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Donanim Hizlandirmali IHA Takip", frame);
        if (cv::waitKey(1) == 27) break; // ESC ile çıkış
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}