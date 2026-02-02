import cv2
from telegram import enviar_video
captura = cv2.VideoCapture(0)

ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_writer = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (ancho, alto))

while True:
    ret, frame = captura.read() 
    
    if not ret:  
        break
        
    video_writer.write(frame)
    cv2.imshow('video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


captura.release()
video_writer.release()
cv2.destroyAllWindows()

enviar_video("video.mp4", caption="Video de persona detectada")
