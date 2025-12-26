#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/dnn.hpp>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>
#include <itkImageRegionConstIterator.h>

#include <iostream>
#include <filesystem>
#include <string>
#include "ImageProcessor.h"
#include <filesystem>

using namespace itk;
using PixelType = signed short;
using ImageType = itk::Image<PixelType, 2>;
using namespace cv;
using namespace std;
namespace fs = filesystem;
ImageProcessor::ImageProcessor() {
    
}

ImageProcessor::~ImageProcessor() {}

bool ImageProcessor::loadImage(const string& filePath) {
    cout << "Loading: " << filePath << endl;
    
    // Detectar si es DICOM/IMA o imagen estándar
    string ext = fs::path(filePath).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".ima" || ext == ".dcm") {
        // Cargar con ITK
        try {
            using ReaderType = itk::ImageFileReader<ImageType>;
            using ImageIOType = itk::GDCMImageIO;
            
            auto dicomIO = ImageIOType::New();
            auto reader = ReaderType::New();
            reader->SetFileName(filePath);
            reader->SetImageIO(dicomIO);
            reader->Update();
            
            auto itkImage = reader->GetOutput();
            auto region = itkImage->GetLargestPossibleRegion();
            auto size = region.GetSize();
            
            int width = size[0];
            int height = size[1];
            
            m_rawImage = Mat(height, width, CV_16SC1);
            
            itk::ImageRegionConstIterator<ImageType> it(itkImage, region);
            int idx = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++idx) {
                int y = idx / width;
                int x = idx % width;
                m_rawImage.at<short>(y, x) = it.Get();
            }
            
            
            normalize(m_rawImage, m_originalImage, 0, 255, NORM_MINMAX, CV_8UC1);
            
            cout << "DICOM/IMA loaded: " << width << "x" << height << endl;
            return true;
            
        } catch (const itk::ExceptionObject& ex) {
            cerr << "ITK Error: " << ex << endl;
            return false;
        }
    } else {
        // Cargar imagen estándar con OpenCV
        m_originalImage = imread(filePath, IMREAD_GRAYSCALE);
        if (m_originalImage.empty()) {
            cerr << "Error: Could not load image with OpenCV" << endl;
            return false;
        }
        
        // Convertir a 16 bits para compatibilidad
        m_originalImage.convertTo(m_rawImage, CV_16SC1);
        
        cout << "Standard image loaded: " << m_originalImage.cols 
                  << "x" << m_originalImage.rows << endl;
        //mas informacion de la imagen
        cout << "Type: " << m_originalImage.type() << endl;
        cout << "Channels: " << m_originalImage.channels() << endl;
        return true;
    }
}



Mat ImageProcessor::eliminarCamilla(Mat img) {
    Mat binaria;
    threshold(img, binaria, 30, 255, THRESH_BINARY);
    
    // 2. Eliminar ruido con morfología
    Mat limpia = morphClosing(binaria, 10);
    
    // 3. Encontrar todos los contornos
    vector<vector<cv::Point>> contours;
    findContours(limpia.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 4. Encontrar el contorno más grande (el cuerpo)
    if (contours.empty()) {
        cout << "No se encontraron contornos" << endl;
        return img;
    }
    
    int indiceMax = 0;
    double areaMax = 0;
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > areaMax) {
            areaMax = area;
            indiceMax = i;
        }
    }
    
    // 5. Crear máscara solo con el contorno del cuerpo
    Mat maskCuerpo = Mat::zeros(img.size(), CV_8U);
    drawContours(maskCuerpo, contours, indiceMax, Scalar(255), -1);
    
    // 6. Erosionar un poco para eliminar piel/grasa superficial
    Mat maskErosionada = morphErosion(maskCuerpo, 10);
    
    // 7. Aplicar máscara
    Mat resultado = Mat::zeros(img.size(), CV_8U);
    img.copyTo(resultado, maskErosionada);
    
    return resultado;
}

// ============ NORMALIZACIÓN Y CONTRASTE ============


Mat ImageProcessor::applyCLAHE( Mat input, double clipLimit) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    auto clahe = createCLAHE(clipLimit, cv::Size(8, 8));
    clahe->apply(gray, output);
    return output;
}

// ============ THRESHOLDING ============
Mat ImageProcessor::thresholdIMG( Mat input, int threshValue) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    threshold(gray, output, threshValue, 255, THRESH_BINARY);
    return output;
}

Mat ImageProcessor::segmentByIntensity( Mat input, int lower, int upper) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    inRange(gray, Scalar(lower), Scalar(upper), output);
    return output;
}
// ============ OPERACIONES LÓGICAS ============
Mat ImageProcessor::applyNOT( Mat input) {
    Mat output;
    bitwise_not(input, output);
    return output;
}

Mat ImageProcessor::applyAND( Mat input1,  Mat input2) {
    Mat output;
    bitwise_and(input1, input2, output);
    return output;
}

Mat ImageProcessor::applyOR( Mat input1,  Mat input2) {
    Mat output;
    bitwise_or(input1, input2, output);
    return output;
}

Mat ImageProcessor::applyXOR( Mat input1,  Mat input2) {
    Mat output;
    bitwise_xor(input1, input2, output);
    return output;
}



// ============ FILTROS DE SUAVIZADO ============
Mat ImageProcessor::filterGaussian( Mat input, int ksize) {
    if (ksize % 2 == 0) ksize++;
    Mat output;
    GaussianBlur(input, output, cv::Size(ksize, ksize), 0);
    return output;
}

Mat ImageProcessor::filterMedian( Mat input, int ksize) {
    if (ksize % 2 == 0) ksize++;
    Mat output;
    medianBlur(input, output, ksize);
    return output;
}


Mat ImageProcessor::filterNLMeans( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    fastNlMeansDenoising(gray, output, 10, 7, 21);
    return output;
}

// ============ MORFOLOGÍA ============
Mat ImageProcessor::morphErosion( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    erode(input, output, kernel);
    return output;
}

Mat ImageProcessor::morphDilation( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    dilate(input, output, kernel);
    return output;
}

Mat ImageProcessor::morphOpening( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(ksize, ksize));
    morphologyEx(input, output, MORPH_OPEN, kernel);
    return output;
}

Mat ImageProcessor::morphClosing( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    morphologyEx(input, output, MORPH_CLOSE, kernel);
    return output;
}


std::vector<Mat> ImageProcessor::deteccionPulmones(Mat img,int a, int b,int tamanio) {
    
    std::vector<Mat> capas;
    

    if (img.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen." << std::endl;
        return Mat();
    }

    
    Mat binary;
    threshold(img, binary, a, b, THRESH_BINARY_INV);
    capas.push_back(binary);
    Mat mask = binary.clone();

    floodFill(mask, cv::Point(0, 0), Scalar(0));

    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(tamanio, tamanio));
    
    
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    
    capas.push_back(mask);
    
    Mat resultado;
    
    bitwise_and(img, img, resultado, mask);
    capas.push_back(resultado);

    return capas;
}
// ============ VISUALIZACIÓN ============
Mat ImageProcessor::createColorOverlay( Mat original,  Mat mask, 
                                              Scalar color, double alpha) {
    Mat output, bgr;
    if (original.channels() == 1) cvtColor(original, bgr, COLOR_GRAY2BGR);
    else bgr = original.clone();
    
    Mat overlay = bgr.clone();
    overlay.setTo(color, mask);
    addWeighted(bgr, 1 - alpha, overlay, alpha, 0, output);
    return output;
}

Mat ImageProcessor::createMultiMaskOverlay(const Mat& original, 
                                           const Mat& mask1, 
                                           const Mat& mask2,
                                           const Scalar& color1, 
                                           const Scalar& color2, 
                                           double alpha) {
    Mat output, bgr;
    if (original.channels() == 1) 
        cvtColor(original, bgr, COLOR_GRAY2BGR);
    else 
        bgr = original.clone();
    
    Mat overlay = bgr.clone();
    
    overlay.setTo(color1, mask1);
    
    overlay.setTo(color2, mask2);
    
    addWeighted(bgr, 1 - alpha, overlay, alpha, 0, output);
    return output;
}

Mat ImageProcessor::createHeatmap( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    applyColorMap(gray, output, COLORMAP_JET);
    return output;
}


// ============ RESALTAR REGIÓN  ============
Mat ImageProcessor::highlightRegion(String name,Mat mask,  Mat background, Scalar color) {
    vector<vector<cv::Point>> contours;
    Mat maskCopy = mask.clone();

    findContours(maskCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Mat output;
    if (background.empty()) {
        if (mask.channels() == 1) {
            cvtColor(mask, output, COLOR_GRAY2BGR);
        } else {
            output = mask.clone();
        }
    } else {
        if (background.channels() == 1) {
            cvtColor(background, output, COLOR_GRAY2BGR);
        } else {
            output = background.clone();
        }
    }
    
    
    drawContours(output, contours, -1, color, 2);
    
    // Dibujar rectángulos y etiquetas
    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundRect = boundingRect(contours[i]);
        double area = contourArea(contours[i]);
        
        if (area > 100) {
            rectangle(output, boundRect, Scalar(0, 255, 0), 2);
            
            std::string label = name +" : " + std::to_string((int)area);
            putText(output, label, cv::Point(boundRect.x, boundRect.y - 5),
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        }
    }
    
    return output;
}


Mat ImageProcessor::createMultiColorOverlay(const Mat& imgOriginal, 
                                            const Mat& mask1, 
                                            const Mat& mask2, 
                                            const Mat& mask3,
                                            const Mat& mask4,
                                            const Scalar& color1,
                                            const Scalar& color2,
                                            const Scalar& color3,
                                            const Scalar& color4,
                                            double alpha)
{
    if (imgOriginal.empty()) {
        return Mat();
    }
    
    Mat resultado;
    if (imgOriginal.channels() == 1) {
        cvtColor(imgOriginal, resultado, COLOR_GRAY2BGR);
    } else {
        resultado = imgOriginal.clone();
    }
    
    if (!mask1.empty()) {
        Mat overlay1 = resultado.clone();
        overlay1.setTo(color1, mask1);
        addWeighted(resultado, 1.0 - alpha, overlay1, alpha, 0, resultado);
    }
    
    if (!mask2.empty()) {
        Mat overlay2 = resultado.clone();
        overlay2.setTo(color2, mask2);
        addWeighted(resultado, 1.0 - alpha, overlay2, alpha, 0, resultado);
    }
    
    if (!mask3.empty()) {
        Mat overlay3 = resultado.clone();
        overlay3.setTo(color3, mask3);
        addWeighted(resultado, 1.0 - alpha, overlay3, alpha, 0, resultado);
    }
    
    if (!mask4.empty()) {
        Mat overlay4 = resultado.clone();
        overlay4.setTo(color4, mask4);
        addWeighted(resultado, 1.0 - alpha, overlay4, alpha, 0, resultado);
    }
    
    return resultado;
}