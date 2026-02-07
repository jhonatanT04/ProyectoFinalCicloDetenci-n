import websocket
import json
import base64
import cv2
import numpy as np
import pygame
# from telegram import enviar_video
import time
from datetime import datetime
import mediapipe as mp

pygame.init()
clock = pygame.time.Clock()
person_detected_since = None
recording = False
video_writer = None


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Crear instancia global (más eficiente)
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def on_message(ws, message):
    global person_detected_since, recording, video_writer, filename
    data = json.loads(message)
    
    if data.get("type") != "frame":
        return
    
    img_base64 = data["image"]
    img_bytes = base64.b64decode(img_base64)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for det in data["detections"]:
        x = det["x"]
        y = det["y"]
        w = det["w"]
        h = det["h"]
        score = det["score"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'{det["label"]} {score:.2f}',
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        person_roi = rgb_frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            continue
        
        # Detectar pose
        results = pose_detector.process(person_roi)
        
        if results.pose_landmarks:
            # Ajustar coordenadas
            
            
            for landmark in results.pose_landmarks.landmark:
                landmark.x = (landmark.x * (x2 - x1) + x1) / frame.shape[1]
                landmark.y = (landmark.y * (y2 - y1) + y1) / frame.shape[0]
            
            
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        # print(detec)
    
    cv2.imshow("Video", frame)
    
    cv2.waitKey(1) 


    
    # detections = data.get("detections", [])
    # current_time = time.time()
    
    # if len(detections) > 0:
    #     if person_detected_since is None:
    #         person_detected_since = current_time
    #     elapsed = current_time - person_detected_since
    #     if elapsed >= 5 and not recording:
    #         print("Grabando video")
    #         filename = datetime.now().strftime("videos/persona_%Y%m%d_%H%M%S.mp4")
    #         h, w, _ = frame.shape
    #         video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"XVID"), 25.0, (w, h))
    #         video_writer.write(frame)
    #         recording = True
            
    #     elif recording:
    #         video_writer.write(frame)
    # else:
    #     person_detected_since = None
    #     if recording:
    #         recording = False
    #         video_writer.release()
    #         print("Enviando...")
    #         enviar_video(filename, caption="Video de persona detectada")
    #         video_writer = None       
    
    clock.tick(30)


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        "ws://localhost:9092",
        on_message=on_message,
    )
    ws.run_forever()
    cv2.destroyAllWindows()  # Limpieza al salir