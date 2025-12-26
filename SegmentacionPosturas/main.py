import asyncio
import websockets

import cv2
import numpy as np
import pygame
import json
import base64
import viewFrame

import time
import os
from datetime import datetime
from telegram import enviar_video
person_detected_since = None
recording = False
video_writer = None

os.makedirs("videos", exist_ok=True)

WS_URL = "ws://localhost:9002"
clock = pygame.time.Clock()

async def receive_video():
    async with websockets.connect(WS_URL, max_size=2**25) as ws:
        global person_detected_since, recording, video_writer
        print("Conectado al servidor WebSocket")

        while True:
            # Manejo eventos pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            msg = await ws.recv()

            # 🔴 Ahora es JSON (str)
            if not isinstance(msg, str):
                print("Mensaje no texto recibido")
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                print("JSON inválido")
                continue

            if "image" not in data:
                print("JSON sin imagen")
                continue

            # Decodificar imagen
            img_bytes = base64.b64decode(data["image"])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error decodificando imagen")
                continue
            detections = data.get("detections", [])
            
            viewFrame.mostrar_frame(frame, detections)

            current_time = time.time()
            
            if len(detections) > 0:
                if person_detected_since is None:
                    person_detected_since = current_time

                elapsed = current_time - person_detected_since
                if elapsed >= 5 and not recording:
                    filename = datetime.now().strftime("videos/persona_%Y%m%d_%H%M%S.mp4")
                    h, w, _ = frame.shape
                    video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"XVID"), 25.0, (w, h))
                    video_writer.write(viewFrame.img_procesada(frame, detections))
                    recording = True
                    
                elif recording:
                    
                    video_writer.write(viewFrame.img_procesada(frame, detections))
            else:
                person_detected_since = None
                if recording:
                    recording = False
                    video_writer.release()
                    # enviar_video(filename, caption="Video de persona detectada")
                    video_writer = None       
            

            clock.tick(30)


asyncio.run(receive_video())
