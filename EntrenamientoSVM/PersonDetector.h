#ifndef PERSON_DETECTOR_H
#define PERSON_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

class PersonDetector {
private:
    cv::HOGDescriptor hog;
    cv::Ptr<cv::ml::SVM> svm;
    
    std::vector<float> getSVMDetector();

public:
    PersonDetector();
    cv::Mat extractHOG( cv::Mat img, cv::HOGDescriptor hog);
    bool loadModel(const std::string& modelPath);
    void detect(const cv::Mat& frame, std::vector<cv::Rect>& detections);
};

#endif
