#include "PersonDetector.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

PersonDetector::PersonDetector()
{
    hog = HOGDescriptor(
        Size(64, 128),
        Size(16, 16),
        Size(8, 8),
        Size(8, 8),
        9);
}

bool PersonDetector::loadModel(const string &modelPath)
{
    svm = SVM::load(modelPath);
    if (svm.empty())
    {
        cerr << "❌ No se pudo cargar el modelo SVM\n";
        return false;
    }

    vector<float> detector = getSVMDetector();
    hog.setSVMDetector(detector);

    cout << "✅ Modelo cargado correctamente\n";
    return true;
}

vector<float> PersonDetector::getSVMDetector()
{
    Mat sv = svm->getSupportVectors();
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    vector<float> detector(sv.cols + 1);
    memcpy(detector.data(), sv.ptr(), sv.cols * sizeof(float));
    detector[sv.cols] = (float)-rho;

    return detector;
}
Mat PersonDetector::extractHOG( Mat img, HOGDescriptor hog)
{
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Mat sobelX, sobelY;
    // Sobel(gray, sobelX, CV_64F, 1, 0, 3);
    // Sobel(gray, sobelY, CV_64F, 0, 1, 3);

    // Mat magnitude;
    // magnitude = abs(sobelX) + abs(sobelY);

    // normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);
    // magnitude.convertTo(magnitude, CV_8U);

    // resize(gray, gray, Size(64, 128));

    vector<float> descriptors;
    hog.compute(gray, descriptors);

    return Mat(descriptors).reshape(1, 1);
}
void PersonDetector::detect(const Mat& frame, vector<Rect>& detections)
{
    detections.clear();
    hog.detectMultiScale(
        frame,
        detections,
        0,
        Size(8, 8),
        Size(32, 32),
        1.05,
        2
    );
}
