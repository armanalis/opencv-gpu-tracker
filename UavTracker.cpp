#include "UavTracker.hpp"

UavTracker::UavTracker() {
    isTracking = false;
    lostFrames = 0;
    
    // Kalman Filtresi Başlatma (4 Durum: x, y, vx, vy | 2 Ölçüm: x, y)
    KF.init(4, 2, 0);
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 
        1, 0, 1, 0,  
        0, 1, 0, 1,  
        0, 0, 1, 0,  
        0, 0, 0, 1);
    
    cv::setIdentity(KF.measurementMatrix);
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-2)); // Çevik bir Kalman
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));
}

void UavTracker::initTracker(const cv::Mat& frame, cv::Rect initialBox) {
    // 1. CSRT Tracker'ı başlat (Doku ve şekil takibi yapar, ışığa çok dirençlidir)
    // Eğer Mac'inde CSRT hata verirse cv::TrackerKCF::create() kullanabilirsin.
    tracker = cv::TrackerCSRT::create(); 
    tracker->init(frame, initialBox);

    // 2. Kalman Filtresinin ilk konumunu ayarla
    int center_x = initialBox.x + initialBox.width / 2;
    int center_y = initialBox.y + initialBox.height / 2;
    
    KF.statePre.at<float>(0) = center_x;
    KF.statePre.at<float>(1) = center_y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    
    KF.statePost = KF.statePre.clone();
    
    isTracking = true;
    lostFrames = 0;
    std::cout << "[SISTEM] Hedefe Kilitlenildi. CSRT Tracker Devrede!" << std::endl;
}

bool UavTracker::updateTracker(const cv::Mat& frame, cv::Rect& outputBox, cv::Point& kalmanPoint) {
    if (!isTracking) return false;

    // 1. Kalman Tahmini (Predict)
    cv::Mat prediction = KF.predict();
    kalmanPoint = cv::Point(prediction.at<float>(0), prediction.at<float>(1));

    // 2. CSRT Sensör Güncellemesi
    bool ok = tracker->update(frame, outputBox);

    if (ok) {
        lostFrames = 0; // Hedef güvende
        int center_x = outputBox.x + outputBox.width / 2;
        int center_y = outputBox.y + outputBox.height / 2;

        // Kalman'ı gerçek sensör verisiyle düzelt (Correct)
        cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
        measurement.at<float>(0) = center_x;
        measurement.at<float>(1) = center_y;
        KF.correct(measurement);
        return true;
    } else {
        lostFrames++;
        // 30 kare (1 saniye) boyunca CSRT hedefi kaybederse sistemi sıfırla
        if (lostFrames > 30) {
            reset();
        }
        return false;
    }
}

void UavTracker::reset() {
    isTracking = false;
    lostFrames = 0;
    std::cout << "[SISTEM] Hedef Kaybedildi! Arama Moduna Donuluyor..." << std::endl;
}