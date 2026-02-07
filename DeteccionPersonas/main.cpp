#define ASIO_STANDALONE

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <thread>
#include <vector>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::connection_hdl;
using namespace cv;
using namespace std;

server ws_server;
std::set<connection_hdl, std::owner_less<connection_hdl>> connections;

// Non-Maximum Suppression
vector<int> applyNMS(const vector<Rect>& boxes, 
                     const vector<float>& scores,
                     float iouThreshold = 0.3)
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

bool isValidAspectRatio(const Rect& bbox) {
    double aspect = (double)bbox.height / bbox.width;
    return aspect >= 1.3 && aspect <= 4.0;
}

bool isValidSize(const Rect& bbox, const Size& frameSize) {
    if (bbox.width < 25 || bbox.height < 50)
        return false;
    
    if (bbox.width > frameSize.width * 0.8 || 
        bbox.height > frameSize.height * 0.8)
        return false;
    
    return true;
}

float getCombinedScore(HOGDescriptor& customHog, 
                       Ptr<ml::SVM>& customSVM,
                       const Mat& roi,
                       float hogScore)
{
    Mat roiResized;
    resize(roi, roiResized, Size(64, 128));
    
    Mat gray;
    if (roiResized.channels() == 3) {
        cvtColor(roiResized, gray, COLOR_BGR2GRAY);
    } else {
        gray = roiResized.clone();
    }
    
    vector<float> descriptors;
    customHog.compute(gray, descriptors);
    
    Mat sample(1, descriptors.size(), CV_32F);
    for (size_t i = 0; i < descriptors.size(); i++)
        sample.at<float>(0, i) = descriptors[i];
    
    Mat rawOutput;
    customSVM->predict(sample, rawOutput, ml::SVM::RAW_OUTPUT);
    
    float svmScore = rawOutput.at<float>(0, 0);
    
    return (0.6 * hogScore) + (0.4 * svmScore);
}

class TemporalTracker {
private:
    struct Detection {
        Rect bbox;
        int framesSeen;
        int lastFrame;
    };
    
    vector<Detection> tracked;
    int minFrames = 2;
    float distThreshold = 50.0;
    
public:
    bool shouldDisplay(const Rect& bbox, int currentFrame) {
        Point center(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
        
        int bestIdx = -1;
        float minDist = distThreshold;
        
        for (size_t i = 0; i < tracked.size(); i++) {
            if (currentFrame - tracked[i].lastFrame > 5)
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
            tracked[bestIdx].lastFrame = currentFrame;
            
            return tracked[bestIdx].framesSeen >= minFrames;
        } else {
            Detection newDet;
            newDet.bbox = bbox;
            newDet.framesSeen = 1;
            newDet.lastFrame = currentFrame;
            tracked.push_back(newDet);
            
            return false;
        }
    }
    
    void cleanup(int currentFrame) {
        tracked.erase(
            remove_if(tracked.begin(), tracked.end(),
                [currentFrame](const Detection& d) {
                    return currentFrame - d.lastFrame > 10;
                }),
            tracked.end()
        );
    }
};

bool hasValidEdges(const Mat& roi) {
    if (roi.empty()) return false;
    
    Mat gray;
    if (roi.channels() == 3) {
        cvtColor(roi, gray, COLOR_BGR2GRAY);
    } else {
        gray = roi.clone();
    }
    
    Mat edges;
    Canny(gray, edges, 50, 150);
    
    double edgeDensity = countNonZero(edges) / (double)(edges.rows * edges.cols);
    
    return edgeDensity > 0.07 && edgeDensity < 0.50;
}

int main()
{
    // Verificar GPU disponible
    int deviceCount = cuda::getCudaEnabledDeviceCount();
    if (deviceCount == 0) {
        cerr << "No se detectó GPU CUDA compatible\n";
        return -1;
    }

    cout << "GPU detectada - Aceleración activada\n";
    cuda::printShortCudaDeviceInfo(cuda::getDevice());

    string customModelPath = "hog_svm.yml";

    // HOG GPU (Detector principal)
    Ptr<cuda::HOG> hogGPU = cuda::HOG::create(
        Size(64, 128),
        Size(16, 16),
        Size(8, 8),
        Size(8, 8),
        9
    );
    hogGPU->setSVMDetector(hogGPU->getDefaultPeopleDetector());
    hogGPU->setNumLevels(13);
    hogGPU->setHitThreshold(0.0);
    hogGPU->setWinStride(Size(8, 8));
    hogGPU->setScaleFactor(1.05);
    hogGPU->setGroupThreshold(0);

    // HOG CPU (Para SVM custom)
    HOGDescriptor hogCustom(
        Size(64, 128),
        Size(16, 16),
        Size(8, 8),
        Size(8, 8),
        9
    );

    Ptr<ml::SVM> customSVM = ml::SVM::load(customModelPath);
    if (customSVM.empty())
    {
        cerr << "No se pudo cargar el modelo\n";
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "No se pudo abrir la cámara\n";
        return -1;
    }

    TemporalTracker tracker;
    int frameCount = 0;

    // Buffers GPU
    cuda::GpuMat d_frame, d_gray, d_resized;
    
    // WebSocket Server
    ws_server.init_asio();

    ws_server.set_open_handler([](connection_hdl hdl)
                               {
        connections.insert(hdl);
        cout << "Cliente conectado\n"; });

    ws_server.set_close_handler([](connection_hdl hdl)
                                {
        connections.erase(hdl);
        cout << "Cliente desconectado\n"; });

    ws_server.listen(9092);
    ws_server.start_accept();

    thread ws_thread([&]()
                     { ws_server.run(); });

    Mat frame;
    vector<uchar> buffer;

    while (cap.read(frame))
    {
        resize(frame, frame, Size(), 0.75, 0.75);
        frameCount++;

        // Pipeline GPU - Detección inicial
        d_frame.upload(frame);
        cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);
        cuda::resize(d_gray, d_resized, Size(320, 240));

        vector<Rect> detections;
        vector<double> hogWeights;
        
        hogGPU->detectMultiScale(d_resized, detections, &hogWeights);

        // Escalar detecciones a tamaño original
        vector<Rect> scaledDetections;
        vector<double> scaledWeights;
        
        for (size_t i = 0; i < detections.size(); i++) {
            Rect r = detections[i];
            r.x = r.x * 2;
            r.y = r.y * 2;
            r.width = r.width * 2;
            r.height = r.height * 2;
            
            double aspect = (double)r.height / r.width;
            if (aspect >= 1.5 && aspect <= 3.0) {
                scaledDetections.push_back(r);
                scaledWeights.push_back(hogWeights[i]);
            }
        }

        // Aplicar filtros en CPU
        vector<Rect> validDetections;
        vector<float> validScores;

        for (size_t i = 0; i < scaledDetections.size(); i++)
        {
            Rect bbox = scaledDetections[i];
            bbox = bbox & Rect(0, 0, frame.cols, frame.rows);
            if (bbox.area() == 0) continue;

            if (!isValidSize(bbox, frame.size()))
                continue;
            
            if (!isValidAspectRatio(bbox))
                continue;

            Mat roi = frame(bbox);
            
            if (!hasValidEdges(roi))
                continue;
            
            float combinedScore = getCombinedScore(
                hogCustom, customSVM, roi, scaledWeights[i]
            );
            
            if (combinedScore < -0.5)
                continue;

            validDetections.push_back(bbox);
            validScores.push_back(abs(combinedScore));
        }

        // NMS
        vector<int> nmsIndices = applyNMS(validDetections, validScores, 0.35);

        // Temporal Smoothing
        vector<Rect> finalDetections;
        vector<float> finalScores;

        for (int idx : nmsIndices)
        {
            if (tracker.shouldDisplay(validDetections[idx], frameCount)) {
                finalDetections.push_back(validDetections[idx]);
                finalScores.push_back(validScores[idx]);
            }
        }

        tracker.cleanup(frameCount);

        // Codificar frame a JPEG
        imencode(".jpg", frame, buffer);

        string img_base64 = websocketpp::base64_encode(
            reinterpret_cast<const unsigned char *>(buffer.data()),
            buffer.size());

        json msg;
        msg["type"] = "frame";
        msg["width"] = frame.cols;
        msg["height"] = frame.rows;

        json detections_json = json::array();

        for (size_t i = 0; i < finalDetections.size(); i++)
        {
            detections_json.push_back({{"label", "persona"},
                                  {"x", finalDetections[i].x},
                                  {"y", finalDetections[i].y},
                                  {"w", finalDetections[i].width},
                                  {"h", finalDetections[i].height},
                                  {"score", finalScores[i]}});
        }

        msg["detections"] = detections_json;
        msg["image"] = img_base64;

        string payload = msg.dump();

        for (auto &con : connections)
        {
            ws_server.send(con, payload, websocketpp::frame::opcode::text);
        }

        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    ws_server.stop();
    ws_thread.join();

    return 0;
}