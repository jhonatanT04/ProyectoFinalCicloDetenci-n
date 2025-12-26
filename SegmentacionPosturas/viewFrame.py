import cv2
import numpy as np
import pygame

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Streaming WebSocket + Detecciones")
clock = pygame.time.Clock()

def mostrar_frame(frame, detections):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dibujar detecciones
    for d in detections:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        label = d["label"]
        score = d["score"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    frame = np.swapaxes(frame, 0, 1)
    surface = pygame.surfarray.make_surface(frame)

    screen.blit(surface, (0, 0))
    pygame.display.update()
    
def img_procesada(frame, detections):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dibujar detecciones
    for d in detections:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        label = d["label"]
        score = d["score"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    return frame