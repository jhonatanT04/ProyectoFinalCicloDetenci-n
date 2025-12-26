#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// ITK
#include <itkImage.h>
using namespace cv;
class ImageProcessor {
public:
    using PixelType = signed short;
    using ImageType = itk::Image<PixelType, 2>;

    ImageProcessor();
    ~ImageProcessor();

    bool loadImage(const std::string& filePath);
    std::vector<Mat> deteccionPulmones(Mat img,int a, int b, int tamanio);
    
        // ================ NORMALIZACIÓN Y CONTRASTE ========
    Mat applyCLAHE( Mat input, double clipLimit = 2.0);
   
    // ================ THRESHOLDING =====================
    Mat thresholdIMG( Mat input, int threshValue);
    

    // ================ OPERADORES LÓGICOS ===============
    Mat applyNOT( Mat input);
    Mat applyAND( Mat input1,  Mat input2);
    Mat applyOR( Mat input1,  Mat input2);
    Mat applyXOR( Mat input1,  Mat input2);

    
    
    // ================ FILTROS DE SUAVIZADO =============
    Mat filterGaussian( Mat input, int ksize = 5);
    Mat filterMedian( Mat input, int ksize = 5);
    Mat filterNLMeans( Mat input);

    // ================ MORFOLOGÍA =======================
    Mat morphErosion( Mat input, int ksize = 5);
    Mat morphDilation( Mat input, int ksize = 5);
    Mat morphOpening( Mat input, int ksize = 5);
    Mat morphClosing( Mat input, int ksize = 5);
    

    // ================ SEGMENTACIÓN =====================
    Mat segmentByIntensity( Mat input, int lower, int upper);

     // ================ VISUALIZACIÓN ====================
    Mat createColorOverlay( Mat original,  Mat mask,
                               Scalar color, double alpha = 0.5);
    Mat createHeatmap( Mat input);
    
    Mat highlightRegion(String name, Mat mask,  Mat background,Scalar color);
    
    Mat createMultiMaskOverlay(const Mat& original, 
                                           const Mat& mask1, 
                                           const Mat& mask2,
                                           const Scalar& color1, 
                                           const Scalar& color2, 
                                           double alpha);
    

    Mat eliminarCamilla(Mat img);
    Mat createMultiColorOverlay(const Mat& imgOriginal, 
                            const Mat& mask1, 
                            const Mat& mask2, 
                            const Mat& mask3,
                            const Mat& mask4,
                            const Scalar& color1,
                            const Scalar& color2,
                            const Scalar& color3,
                            const Scalar& color4,
                            double alpha);


   
    const Mat& getOriginalImage() const { return m_originalImage; }
    const Mat& getRawImage() const { return m_rawImage; }
    
private:
    Mat m_originalImage;  // Imagen lista para visualizar (8 bits)
    Mat m_rawImage;       // Imagen cruda cargada (16 bits)
};

