#define ASIO_STANDALONE

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <thread>
#include <set>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::connection_hdl;
using namespace cv;
using namespace std;

// Variables globales para WebSocket
server ws_server;
std::set<connection_hdl, std::owner_less<connection_hdl>> connections;

// Non-Maximum Suppression
vector<int> applyNMS(const vector<Rect>& boxes, 
                     const vector<float>& scores,
                     float iouThreshold = 0.35)
{
    vector<int> indices;
    if (boxes.empty()) return indices;

    vector<pair<float, int>> scoreIndex;
    for (size_t i = 0; i < scores.size(); i++)
        scoreIndex.push_back({scores[i], i});
    
    sort(scoreIndex.begin(), scoreIndex.end(), 
         [](const pair<float,int>& a, const pair<float,int>& b) {
             return a.first > b.first;
         });

    vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < scoreIndex.size(); i++)
    {
        int idx = scoreIndex[i].second;
        if (suppressed[idx]) continue;

        indices.push_back(idx);

        for (size_t j = i + 1; j < scoreIndex.size(); j++)
        {
            int idx2 = scoreIndex[j].second;
            if (suppressed[idx2]) continue;

            Rect inter = boxes[idx] & boxes[idx2];
            float iou = (float)inter.area() / 
                       (boxes[idx].area() + boxes[idx2].area() - inter.area());

            if (iou > iouThreshold)
                suppressed[idx2] = true;
        }
    }

    return indices;
}

// Validación de tamaño BALANCEADA
bool isValidSize(const Rect& bbox, const Size& frameSize) {
    if (bbox.width < 20 || bbox.height < 45)
        return false;
    
    if (bbox.width > frameSize.width * 0.9 || 
        bbox.height > frameSize.height * 0.9)
        return false;
    
    float aspectRatio = (float)bbox.height / bbox.width;
    if (aspectRatio < 1.3 || aspectRatio > 5.0)
        return false;
    
    return true;
}

// Análisis de varianza MODERADO
bool hasGoodVariance(const Mat& roi) {
    if (roi.empty()) return false;
    
    Mat gray;
    if (roi.channels() == 3) {
        cvtColor(roi, gray, COLOR_BGR2GRAY);
    } else {
        gray = roi.clone();
    }
    
    Scalar mean, stddev;
    meanStdDev(gray, mean, stddev);
    
    return stddev[0] > 15.0 && stddev[0] < 80.0;
}

// Análisis de bordes MODERADO
bool hasValidEdges(const Mat& roi) {
    if (roi.empty()) return false;
    
    Mat gray;
    if (roi.channels() == 3) {
        cvtColor(roi, gray, COLOR_BGR2GRAY);
    } else {
        gray = roi.clone();
    }
    
    GaussianBlur(gray, gray, Size(3, 3), 0);
    
    Mat edges;
    Canny(gray, edges, 35, 115);
    
    double edgeDensity = countNonZero(edges) / (double)(edges.rows * edges.cols);
    
    return edgeDensity > 0.05 && edgeDensity < 0.60;
}

// Resultado de verificación
struct VerificationResult {
    bool isValid;
    float score;
};

// Verificación multi-modelo BALANCEADA
VerificationResult verifyWithAllModels(
    HOGDescriptor& hogCustom,
    Ptr<ml::SVM>& svm1,
    Ptr<ml::SVM>& svm2,
    Ptr<ml::SVM>& svm3,
    const Mat& roi,
    float hogScore)
{
    VerificationResult result;
    result.isValid = false;
    result.score = 0.0f;

    Mat roiResized;
    resize(roi, roiResized, Size(64, 128));
    
    Mat gray;
    if (roiResized.channels() == 3) {
        cvtColor(roiResized, gray, COLOR_BGR2GRAY);
    } else {
        gray = roiResized.clone();
    }
    
    // Extraer features HOG
    vector<float> descriptors;
    hogCustom.compute(gray, descriptors);
    
    Mat sample(1, descriptors.size(), CV_32F);
    for (size_t i = 0; i < descriptors.size(); i++)
        sample.at<float>(0, i) = descriptors[i];
    
    // Verificar con los 3 modelos SVM
    Mat rawOutput1, rawOutput2, rawOutput3;
    svm1->predict(sample, rawOutput1, ml::SVM::RAW_OUTPUT);
    svm2->predict(sample, rawOutput2, ml::SVM::RAW_OUTPUT);
    svm3->predict(sample, rawOutput3, ml::SVM::RAW_OUTPUT);
    
    float score1 = rawOutput1.at<float>(0, 0);
    float score2 = rawOutput2.at<float>(0, 0);
    float score3 = rawOutput3.at<float>(0, 0);
    
    // Votación moderada
    int positiveVotes = 0;
    if (score1 > 0.0) positiveVotes++;
    if (score2 > 0.0) positiveVotes++;
    if (score3 > 0.0) positiveVotes++;
    
    float maxSvmScore = max({score1, score2, score3});
    float avgSvmScore = (score1 + score2 + score3) / 3.0f;
    
    // Criterio BALANCEADO: 2 de 3 modelos O 1 muy seguro
    if (positiveVotes >= 2) {
        // Al menos 2 modelos dicen que sí
        float combined = (0.5 * hogScore) + (0.5 * maxSvmScore);
        if (combined > -0.6) {
            result.isValid = true;
            result.score = abs(combined);
        }
    }
    else if (maxSvmScore > 0.8) {
        // Un modelo muy seguro
        float combined = (0.5 * hogScore) + (0.5 * maxSvmScore);
        if (combined > -0.4) {
            result.isValid = true;
            result.score = abs(combined);
        }
    }
    else if (hogScore > 0.6) {
        // HOG default con buena confianza
        result.isValid = true;
        result.score = abs(hogScore);
    }
    
    return result;
}

// Temporal Tracking BALANCEADO
class TemporalTracker {
private:
    struct Detection {
        Rect bbox;
        int framesSeen;
        int framesNotSeen;
        int lastFrame;
        float avgScore;
    };
    
    vector<Detection> tracked;
    int minFrames = 2;
    int maxGapFrames = 8;
    float distThreshold = 60.0;
    
public:
    bool shouldDisplay(const Rect& bbox, int currentFrame, float score) {
        Point center(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
        
        int bestIdx = -1;
        float minDist = distThreshold;
        
        for (size_t i = 0; i < tracked.size(); i++) {
            if (currentFrame - tracked[i].lastFrame > maxGapFrames)
                continue;
            
            Point trackedCenter(
                tracked[i].bbox.x + tracked[i].bbox.width/2,
                tracked[i].bbox.y + tracked[i].bbox.height/2
            );
            
            float dist = sqrt(
                pow(center.x - trackedCenter.x, 2) + 
                pow(center.y - trackedCenter.y, 2)
            );
            
            if (dist < minDist) {
                minDist = dist;
                bestIdx = i;
            }
        }
        
        if (bestIdx >= 0) {
            tracked[bestIdx].bbox = bbox;
            tracked[bestIdx].framesSeen++;
            tracked[bestIdx].framesNotSeen = 0;
            tracked[bestIdx].lastFrame = currentFrame;
            tracked[bestIdx].avgScore = (tracked[bestIdx].avgScore * 0.7) + (score * 0.3);
            
            return tracked[bestIdx].framesSeen >= minFrames;
        } else {
            Detection newDet;
            newDet.bbox = bbox;
            newDet.framesSeen = 1;
            newDet.framesNotSeen = 0;
            newDet.lastFrame = currentFrame;
            newDet.avgScore = score;
            tracked.push_back(newDet);
            
            return false;
        }
    }
    
    void cleanup(int currentFrame) {
        tracked.erase(
            remove_if(tracked.begin(), tracked.end(),
                [currentFrame, this](const Detection& d) {
                    return currentFrame - d.lastFrame > this->maxGapFrames + 5;
                }),
            tracked.end()
        );
    }
};

int main(int argc, char** argv)
{
    string videoPath = (argc > 1) ? argv[1] : "Moscow.mp4";

    HOGDescriptor hogDefault;
    hogDefault.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    HOGDescriptor hogCustom(
        Size(64, 128),
        Size(16, 16),
        Size(8, 8),
        Size(8, 8),
        9
    );

    Ptr<ml::SVM> svm1 = ml::SVM::load("models_improved/svm_pose_t.yml");
    if (svm1.empty()) {
        cerr << "No se pudo cargar modelo 1 (svm_pose_t.yml)\n";
        return -1;
    }

    Ptr<ml::SVM> svm2 = ml::SVM::load("models_improved/svm_sentado.yml");
    if (svm2.empty()) {
        cerr << "No se pudo cargar modelo 2 (svm_sentado.yml)\n";
        return -1;
    }

    Ptr<ml::SVM> svm3 = ml::SVM::load("models_improved/svm_rodillas.yml");
    if (svm3.empty()) {
        cerr << "No se pudo cargar modelo 3 (svm_rodillas.yml)\n";
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "No se pudo abrir la cámara\n";
        return -1;
    }

    // ============ CONFIGURACIÓN WEBSOCKET ============
    ws_server.init_asio();

    ws_server.set_open_handler([](connection_hdl hdl) {
        connections.insert(hdl);
        cout << "Cliente WebSocket conectado\n";
    });

    ws_server.set_close_handler([](connection_hdl hdl) {
        connections.erase(hdl);
        cout << "Cliente WebSocket desconectado\n";
    });

    ws_server.listen(9092);
    ws_server.start_accept();

    // Iniciar servidor WebSocket en thread separado
    thread ws_thread([&]() {
        cout << "Servidor WebSocket iniciado en puerto 9092\n";
        ws_server.run();
    });
    // ================================================

    TemporalTracker tracker;
    Mat frame;
    int frameCount = 0;
    double prev_time = (double)getTickCount() / getTickFrequency();
    
    // Buffer para codificar imagen
    vector<uchar> buffer;
    
    while (cap.read(frame))
    {
        resize(frame, frame, Size(), 0.75, 0.75);
        frameCount++;

        vector<Rect> detections;
        vector<double> hogWeights;

        // Parámetros BALANCEADOS
        hogDefault.detectMultiScale(
            frame,
            detections,
            hogWeights,
            -0.3,           // Umbral moderado
            Size(6, 6),     // Stride medio
            Size(16, 16),   
            1.04,           // Escala moderada
            1.7,            // Threshold moderado
            false
        );

        vector<Rect> validDetections;
        vector<float> validScores;

        for (size_t i = 0; i < detections.size(); i++)
        {
            Rect bbox = detections[i];
            bbox = bbox & Rect(0, 0, frame.cols, frame.rows);
            if (bbox.area() == 0) continue;

            // Validar tamaño
            if (!isValidSize(bbox, frame.size()))
                continue;

            Mat roi = frame(bbox);
            
            // Validar varianza
            if (!hasGoodVariance(roi))
                continue;
            
            // Validar bordes
            if (!hasValidEdges(roi))
                continue;
            
            // Verificar con modelos
            VerificationResult result = verifyWithAllModels(
                hogCustom, svm1, svm2, svm3, roi, hogWeights[i]
            );
            
            if (result.isValid) {
                validDetections.push_back(bbox);
                validScores.push_back(result.score);
            }
        }

        // NMS
        vector<int> nmsIndices = applyNMS(validDetections, validScores, 0.35);

        // Temporal Tracking
        vector<Rect> finalDetections;
        vector<float> finalScores;

        for (int idx : nmsIndices)
        {
            if (tracker.shouldDisplay(validDetections[idx], frameCount, validScores[idx])) {
                finalDetections.push_back(validDetections[idx]);
                finalScores.push_back(validScores[idx]);
            }
        }

        tracker.cleanup(frameCount);

        // ============ ENVIAR DATOS POR WEBSOCKET ============
        if (!connections.empty()) {
            // Codificar frame a JPEG
            imencode(".jpg", frame, buffer);
            
            string img_base64 = websocketpp::base64_encode(
                reinterpret_cast<const unsigned char*>(buffer.data()),
                buffer.size()
            );
            
            // Crear JSON con detecciones
            json msg;
            msg["type"] = "frame";
            msg["width"] = frame.cols;
            msg["height"] = frame.rows;
            
            json detections_json = json::array();
            
            for (size_t i = 0; i < finalDetections.size(); i++)
            {
                // Convertir score a porcentaje
                int confidence = (int)(finalScores[i] * 100);
                if (confidence > 100) confidence = 100;
                
                detections_json.push_back({
                    {"label", "Persona"},
                    {"x", finalDetections[i].x},
                    {"y", finalDetections[i].y},
                    {"w", finalDetections[i].width},
                    {"h", finalDetections[i].height},
                    {"score", finalScores[i]},
                    {"confidence", confidence}
                });
            }
            
            msg["detections"] = detections_json;
            msg["image"] = img_base64;
            
            string payload = msg.dump();
            
            // Enviar a todos los clientes conectados
            for (auto& con : connections) {
                ws_server.send(con, payload, websocketpp::frame::opcode::text);
            }
        }
        // ===================================================

        // Visualización LOCAL
        Scalar color = Scalar(0, 255, 0);
        
        for (size_t i = 0; i < finalDetections.size(); i++)
        {
            Rect box = finalDetections[i];
            float score = finalScores[i];
            
            // Convertir score a porcentaje (0-100%)
            int confidence = (int)(score * 100);
            if (confidence > 100) confidence = 100;

            rectangle(frame, box, color, 2);
            
            // Etiqueta arriba del cuadro con porcentaje
            putText(frame,
                    format("Persona %d%%", confidence),
                    Point(box.x, box.y - 8),
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2);
        }

        // Info FPS
        double curr_time = (double)getTickCount() / getTickFrequency();
        double fps = 1.0 / (curr_time - prev_time);
        prev_time = curr_time;

        putText(frame, format("FPS: %d", (int)fps), 
               Point(12, 32), FONT_HERSHEY_SIMPLEX, 0.7, 
               Scalar(0, 0, 0), 3);
        putText(frame, format("FPS: %d", (int)fps), 
               Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, 
               Scalar(0, 255, 0), 2);

        putText(frame, format("Personas: %d", (int)finalDetections.size()),
               Point(12, 62), FONT_HERSHEY_SIMPLEX, 0.6, 
               Scalar(0, 0, 0), 3);
        putText(frame, format("Personas: %d", (int)finalDetections.size()),
               Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, 
               Scalar(255, 255, 255), 2);

        // Info WebSocket
        putText(frame, format("WS Clientes: %d", (int)connections.size()),
               Point(12, 92), FONT_HERSHEY_SIMPLEX, 0.6, 
               Scalar(0, 0, 0), 3);
        putText(frame, format("WS Clientes: %d", (int)connections.size()),
               Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, 
               Scalar(255, 255, 0), 2);

        imshow("Multi-Model Person Detection", frame);
        
        int key = waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    cap.release();
    destroyAllWindows();
    
    // Detener servidor WebSocket
    ws_server.stop();
    ws_thread.join();
    
    return 0;
}