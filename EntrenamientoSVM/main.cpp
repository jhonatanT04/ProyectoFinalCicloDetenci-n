#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <filesystem>
#include "PersonDetector.h"
namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;
using namespace std;

HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

Mat extractHOG(const Mat &img, HOGDescriptor &hog)
{
    Mat gray;

    // Convertir a escala de grises si es necesario
    if (img.channels() == 3)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img.clone();

    // Ya NO redimensionar aquí porque las imágenes ya son 64x128
    // Verificar tamaño
    if (gray.cols != 64 || gray.rows != 128)
    {
        cout << "Error: Imagen debe ser 64x128, es "
             << gray.cols << "x" << gray.rows << endl;
        resize(gray, gray, Size(64, 128));
    }

    // Calcular descriptores HOG
    vector<float> descriptors;
    hog.compute(gray, descriptors);

    // Convertir a Mat (1 fila)
    return Mat(descriptors).reshape(1, 1);
}

vector<float> getSVMDetector(const Ptr<SVM> &svm)
{
    Mat sv = svm->getSupportVectors();
    Mat alpha, svidx;

    double rho = svm->getDecisionFunction(0, alpha, svidx);

    vector<float> detector(sv.cols + 1);
    memcpy(detector.data(), sv.ptr(), sv.cols * sizeof(float));
    detector[sv.cols] = (float)-rho;

    return detector;
}

void entrenarModelo()
{
    vector<Mat> features;
    vector<int> labels;

    // POSITIVOS: Imágenes de personas (64x128)
    string posDir = "Ania/";
    int posCount = 0;

    for (const auto &entry : fs::directory_iterator(posDir))
    {
        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png")
            continue;

        Mat img = imread(entry.path().string());
        if (img.empty())
            continue;

        // Verificar que sea 64x128
        if (img.cols != 64 || img.rows != 128)
        {
            cout << "Advertencia: " << entry.path().filename()
                 << " no es 64x128, redimensionando..." << endl;
            resize(img, img, Size(64, 128));
        }

        Mat feat = extractHOG(img, hog);
        features.push_back(feat);
        labels.push_back(+1); // Clase positiva
        posCount++;
    }
    cout << "Muestras positivas (personas): " << posCount << endl;

    // NEGATIVOS: Extraer ventanas de fondos sin personas
    string negDir = "street-objects-3/train";
    int negCount = 0;
    int negTarget = posCount * 2; // Queremos 2x negativos que positivos

    for (const auto &entry : fs::directory_iterator(negDir))
    {
        if (negCount >= negTarget)
            break;

        if (entry.path().extension() != ".jpg" &&
            entry.path().extension() != ".png")
            continue;

        Mat img = imread(entry.path().string());
        if (img.empty())
            continue;

        // Extraer ventanas aleatorias de 64x128 de las imágenes de fondo
        for (int i = 0; i < 3 && negCount < negTarget; i++) // 3 ventanas por imagen
        {
            if (img.cols < 64 || img.rows < 128)
            {
                resize(img, img, Size(max(64, img.cols), max(128, img.rows)));
            }

            // Posición aleatoria
            int x = rand() % (img.cols - 64);
            int y = rand() % (img.rows - 128);

            Rect roi(x, y, 64, 128);
            Mat window = img(roi).clone();

            Mat feat = extractHOG(window, hog);
            features.push_back(feat);
            labels.push_back(-1); // Clase negativa
            negCount++;
        }
    }
    cout << "Muestras negativas (fondos): " << negCount << endl;

    if (posCount < 50 || negCount < 50)
    {
        cout << "ERROR: Necesitas al menos 50 muestras de cada clase" << endl;
        return;
    }

    // Preparar datos
    Mat trainData;
    vconcat(features, trainData);
    trainData.convertTo(trainData, CV_32F);

    Mat labelsMat(labels);
    labelsMat.convertTo(labelsMat, CV_32S);

    cout << "\nEntrenando SVM con " << trainData.rows << " muestras..." << endl;
    cout << "Dimensión de características: " << trainData.cols << endl;

    // Configurar SVM para clasificación binaria
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);    // Clasificación
    svm->setKernel(SVM::LINEAR); // Kernel lineal (necesario para HOG)
    svm->setC(0.1);              // Parámetro de regularización
    svm->setTermCriteria(
        TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-6));

    // Entrenar
    auto start = getTickCount();
    svm->train(trainData, ROW_SAMPLE, labelsMat);
    auto end = getTickCount();

    cout << "Entrenamiento completado en "
         << (end - start) / getTickFrequency() << " segundos" << endl;

    // Guardar modelo
    svm->save("svm.yml");
    cout << "Modelo guardado: svm.yml" << endl;

    // Evaluar en datos de entrenamiento
    int correct = 0;
    for (int i = 0; i < trainData.rows; i++)
    {
        float prediction = svm->predict(trainData.row(i));
        if ((prediction > 0 && labels[i] == 1) ||
            (prediction < 0 && labels[i] == -1))
            correct++;
    }

    float accuracy = 100.0f * correct / trainData.rows;
    cout << "Precisión en entrenamiento: " << accuracy << "%" << endl;

    // Probar el detector
    vector<float> detector = getSVMDetector(svm);
    cout << "Vector detector tiene " << detector.size() << " elementos" << endl;

    cout << "\nEntrenamiento finalizado correctamente" << endl;
}
void runModel()
{
    // Opción 1: Usar el detector por defecto de OpenCV (para probar)
    cout << "¿Usar detector por defecto de OpenCV? (1=Sí, 0=No): ";
    int useDefault;
    cin >> useDefault;

    if (useDefault == 1)
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        cout << "Usando detector por defecto" << endl;
    }
    else
    {
        // Cargar modelo personalizado
        Ptr<SVM> svm = SVM::load("svm.yml");
        if (svm.empty())
        {
            cout << "Error: No se pudo cargar svm.yml" << endl;
            return;
        }

        vector<float> detector = getSVMDetector(svm);
        hog.setSVMDetector(detector);
        cout << "Usando modelo personalizado" << endl;
    }

    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Error: No se pudo abrir la cámara" << endl;
        return;
    }

    // Ajustar resolución para mejor rendimiento
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    cout << "Presiona 'q' para salir" << endl;

    Mat frame;
    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        vector<Rect> detections;
        vector<double> weights;

        // Parámetros ajustados para mejor detección
        hog.detectMultiScale(frame, detections, weights,
                             0,       // hitThreshold (más bajo = más detecciones)
                             Size(4, 4), // winStride (más pequeño = más preciso pero lento)
                             Size(8, 8), // padding
                             1.05,       // scale
                             1.5);       // finalThreshold

        // Dibujar detecciones
        for (size_t i = 0; i < detections.size(); i++)
        {
            rectangle(frame, detections[i], Scalar(0, 255, 0), 2);

            string label = "Conf: " + to_string(weights[i]).substr(0, 4);
            putText(frame, label,
                    Point(detections[i].x, detections[i].y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        string info = "Detecciones: " + to_string(detections.size());
        putText(frame, info, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

        imshow("Deteccion en Camara", frame);

        if (waitKey(1) == 'q')
            break;
    }

    cap.release();
    destroyAllWindows();
}


int main()
{
    // entrenarModelo();
       runModel();
    // probarConCamara();
    return 0;
}
