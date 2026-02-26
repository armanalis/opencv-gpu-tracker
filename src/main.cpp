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

    cv::KalmanFilter KF(4, 2, 0);//initializing Kalman Filter
    cv::Mat state(4, 1, CV_32F);
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);

    //Transition Matrix
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0, 
        0, 0, 0, 1);

    //Measurement Matrix
    KF.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

    bool isTracking = false;
    int lostFrames = 0;//Counts  how many frames that the frame is lost

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
        cv::Scalar upper_black(180, 255, 60);

        //Masking process only black frames will survive.
        cv::inRange(hsvFrame, lower_black, upper_black, mask);

        //Cleaning the noise.
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

        cv::erode(mask, mask, kernel);//erase small white shadows
        cv::dilate(mask, mask, kernel);//restore the size of the actual target
        

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        //finding contours
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // ... (findContours satırı aynı kalıyor)

        double maxArea = 0;
        int maxAreaIDx = -1;
        
        // Mesafe takibi için yeni değişkenler (Başlangıçta sonsuz büyük bir mesafe veriyoruz)
        double minDistance = 1e9; 

                // ---------------------------------------------------------
        // KALMAN FILTER: PREDICT PHASE (Happens in every frame)
        // ---------------------------------------------------------
        // Predict the next position based on previous velocity and position
        cv::Mat prediction = KF.predict();
        // Extract the predicted X and Y coordinates
        cv::Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

        for(int i = 0; i < contours.size(); i++){
            double area = cv::contourArea(contours[i]);

            if (!isTracking) {
                // SADECE KAMERANIN BOYUTLARINI KABUL ET (Sıkı Sınırlar)
                if(area > 8000 && area < 10000){ 
                    if (area > maxArea) {
                        maxArea = area;
                        maxAreaIDx = i;
                    }
                }
            } 
            // 2. DURUM: EĞER HEDEFE ZATEN KİLİTLİYSEK (TAKİP MODU)
            else {
                // HEDEF UZAKLAŞABİLİR/YAKLAŞABİLİR, SINIRLARI ESNET (Gevşek Sınırlar)
                if(area > 8000 && area < 10000){ 
                    
                    cv::Rect temp_box = cv::boundingRect(contours[i]);
                    int temp_center_x = temp_box.x + (temp_box.width / 2);
                    int temp_center_y = temp_box.y + (temp_box.height / 2);
                    
                    // Kırmızı kutuya olan mesafesini ölç
                    double dist = cv::norm(cv::Point(temp_center_x, temp_center_y) - predictPt);
                    
                    // Mesafe 150 pikselden yakınsa EN YAKIN olanı seç
                    if (dist < 150 && dist < minDistance) {
                        minDistance = dist;
                        maxAreaIDx = i; 
                    }
                }
            }
        }

        // ---------------------------------------------------------
        // SENSOR & KALMAN UPDATE PHASE
        // ---------------------------------------------------------
        if(maxAreaIDx != -1){

            lostFrames = 0;

            // OpenCV (Sensor) FOUND the target!
            cv::Rect bounding_box = cv::boundingRect(contours[maxAreaIDx]);
            int center_x = bounding_box.x + (bounding_box.width / 2);
            int center_y = bounding_box.y + (bounding_box.height / 2);

            // 1. Give the real sensor data to Kalman measurement matrix
            measurement.at<float>(0) = center_x;
            measurement.at<float>(1) = center_y;

            if (!isTracking) {
                // If it's the FIRST TIME we found the target, initialize Kalman state
                KF.statePre.at<float>(0) = center_x;
                KF.statePre.at<float>(1) = center_y;
                KF.statePre.at<float>(2) = 0; // Initial X velocity is 0
                KF.statePre.at<float>(3) = 0; // Initial Y velocity is 0

                KF.statePost.at<float>(0) = center_x;
                KF.statePost.at<float>(1) = center_y;
                KF.statePost.at<float>(2) = 0; 
                KF.statePost.at<float>(3) = 0;
                isTracking = true;
            } else {
                // If we are already tracking, CORRECT the prediction with real sensor data
                KF.correct(measurement);
            }

            // --- VISUALIZE SENSOR DATA (GREEN BOX) ---
            // Draw a GREEN box representing the RAW CAMERA SENSOR
            cv::rectangle(frame, bounding_box, cv::Scalar(0, 255, 0), 2);
            std::string text = "Sensor: Locked";
            cv::putText(frame, text, cv::Point(bounding_box.x, bounding_box.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            // OpenCV (Sensor) LOST the target!
            if (isTracking) {
                lostFrames++; // Kayıp sayacını artır
                
                // Eğer hedef 15 kare (yaklaşık yarım saniye) boyunca bulunamazsa TAKİBİ DÜŞÜR!
                if (lostFrames > 15) {
                    isTracking = false; // Sistemi sıfırla, tekrar tüm ekranda aramaya başlasın
                    std::cout << "Target Lost! Resetting tracker to global search...\n";
                }
            }
        }

        // ---------------------------------------------------------
        // VISUALIZE KALMAN PREDICTION (RED BOX)
        // ---------------------------------------------------------
        if (isTracking) {
            // Draw a RED point at the center of the Kalman prediction
            cv::circle(frame, predictPt, 4, cv::Scalar(0, 0, 255), -1);
            
            // Draw a RED box representing the AI/Kalman prediction
            // We draw a fixed 50x50 box around the predicted center
            cv::Rect predictRect(predictPt.x - 25, predictPt.y - 25, 50, 50);
            cv::rectangle(frame, predictRect, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Kalman", cv::Point(predictPt.x - 25, predictPt.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        //Show the original and processed image side by side
        cv::imshow("Original IHA cam", frame);
        //cv::imshow("Black cam", mask);

        int key = cv::waitKey(1);

        if (key == 27)//ESC key to quit
            break;
        }

    std::cout << "Loop ended\n";

    cap.release();
    cv::destroyAllWindows();
    return 0;
    


}