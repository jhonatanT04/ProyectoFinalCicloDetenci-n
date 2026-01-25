#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
   
    
    
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    
    
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("svm_personas.yml");
    
    if (svm.empty()) {
        std::cerr << "❌ Error: No se pudo cargar svm_personas.yml" << std::endl;
        return -1;
    }
    std::cout << "✅ SVM cargada correctamente" << std::endl;
    
    // 3. Configurar HOG para GPU
    cv::Ptr<cv::cuda::HOG> hog_gpu = cv::cuda::HOG::create(
        cv::Size(64, 128),  // winSize
        cv::Size(16, 16),   // blockSize
        cv::Size(8, 8),     // blockStride
        cv::Size(8, 8)      // cellSize
    );
    
    
    hog_gpu->setNumLevels(13);
    hog_gpu->setHitThreshold(0.0);
    hog_gpu->setWinStride(cv::Size(8, 8));
    hog_gpu->setScaleFactor(1.05);
    hog_gpu->setGroupThreshold(2);
    
    
    std::vector<float> detector;
    cv::Mat sv = svm->getSupportVectors();
    double rho = svm->getDecisionFunction(0, detector, detector);
    
    detector.assign(sv.begin<float>(), sv.end<float>());
    detector.push_back(-rho);
    
    hog_gpu->setSVMDetector(detector);
    
    

    cv::VideoCapture cap(0);
    
    
    
    

    
    

    
    cv::Mat frame;
    cv::cuda::GpuMat gpu_frame, gpu_gray;
    int frame_count = 0;
    double total_time = 0.0;
    

    while (true) {
        auto start = cv::getTickCount();
        
        
        cap >> frame;
        if (frame.empty()) {
            std::cout << "\n✅ Fin del video" << std::endl;
            break;
        }
        
        frame_count++;
        
        
        gpu_frame.upload(frame);
        
        
        cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);
        
        
        std::vector<cv::Rect> detections;
        hog_gpu->detectMultiScale(gpu_gray, detections);
        
        
        auto end = cv::getTickCount();
        double frame_time = (end - start) / cv::getTickFrequency();
        total_time += frame_time;
        double current_fps = 1.0 / frame_time;
        
       
        for (const auto& rect : detections) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 3);
            
            cv::putText(frame, "Persona", 
                       cv::Point(rect.x, rect.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                       cv::Scalar(0, 255, 0), 2);
        }
        
        
        
        
        // 12. Mostrar resultado
        cv::imshow("Deteccion GPU", frame);
        
        
        
        // 13. Control de teclado
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            std::cout << "\n⏹️  Detenido por usuario" << std::endl;
            break;
        } 
        
        
        
    }
    
    
    cv::destroyAllWindows();
    

    
    return 0;
}