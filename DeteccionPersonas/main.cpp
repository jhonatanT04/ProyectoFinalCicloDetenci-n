#define ASIO_STANDALONE

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::connection_hdl;
using namespace cv;
using namespace std;

server ws_server;
std::set<connection_hdl, std::owner_less<connection_hdl>> connections;

void broadcast_frame(const std::vector<uchar> &data)
{
    for (auto &con : connections)
    {
        ws_server.send(con, data.data(), data.size(),
                       websocketpp::frame::opcode::binary);
    }
}

int main()
{
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    vector<Rect> rects;
    vector<double> weights;

    VideoCapture video(0);
    Mat frame;
    // namedWindow("Video", WINDOW_AUTOSIZE);
    // namedWindow("Deteccion", WINDOW_AUTOSIZE);

    ws_server.init_asio();

    ws_server.set_open_handler([](connection_hdl hdl)
                               {
        connections.insert(hdl);
        cout << "Cliente conectado\n"; });

    ws_server.set_close_handler([](connection_hdl hdl)
                                {
        connections.erase(hdl);
        cout << "Cliente desconectado\n"; });

    ws_server.listen(9002);
    ws_server.start_accept();

    thread ws_thread([&]()
                     { ws_server.run(); });

    vector<uchar> buffer;

    while (true)
    {
        video >> frame;
        if (frame.empty())
        {
            cerr << "No se pudo capturar el frame" << endl;
            break;
        }

        hog.detectMultiScale(
            frame,
            rects,
            weights,
            0,
            Size(4, 4),
            Size(8, 8),
            1.5);

        // imshow("Video", frame);

        // ======== Se muestra la deteccion individual ==========
        //  if (!rects.empty())
        //  {
        //      for (int i = rects.size() - 1; i >= 0; i--)
        //      {
        //          //imshow("Deteccion"+to_string(i),frame(rects[i]));
        //         imencode(".jpg", frame(rects[i]), buffer);
        //      }
        // }

        // ======== Se muestra la deteccion en el video ==========
        // for (const auto &r : rects)
        // {
        //     rectangle(frame, r, Scalar(0, 255, 0), 2);
        // }
        // imshow("Deteccion", frame);
        // for (auto &&i : weights)
        // {
        //     cout << "Porcetaje de certeza: " << i << endl;
        // }

        imencode(".jpg", frame, buffer);

        string img_base64 = websocketpp::base64_encode(
            reinterpret_cast<const unsigned char *>(buffer.data()),
            buffer.size());

        json msg;
        msg["type"] = "frame";
        msg["width"] = frame.cols;
        msg["height"] = frame.rows;

        json detections = json::array();

        for (size_t i = 0; i < rects.size(); i++)
        {
            detections.push_back({{"label", "persona"},
                                  {"x", rects[i].x},
                                  {"y", rects[i].y},
                                  {"w", rects[i].width},
                                  {"h", rects[i].height},
                                  {"score", weights[i]}});
        }

        msg["detections"] = detections;
        msg["image"] = img_base64;

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        string payload = msg.dump();

        for (auto &con : connections)
        {
            ws_server.send(con, payload, websocketpp::frame::opcode::text);
        }

        if (waitKey(23) == 27)
            break;
    }

    destroyAllWindows();

    return 0;
}
