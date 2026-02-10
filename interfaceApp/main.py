import tkinter as tk
from tkinter import ttk, messagebox
import threading
import websocket
import json
import base64
import cv2
import numpy as np
import time
from datetime import datetime
import mediapipe as mp
from telegram import enviar_video, lista_contactos, cargar_contactos
from PIL import Image, ImageTk
import os

class VideoDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección y Telegram Bot")
        self.root.geometry("1200x700")
        
        # Variables
        self.ws = None
        self.running = False
        self.person_detected_since = None
        self.recording = False
        self.video_writer = None
        self.filename = None
        self.current_frame = None
        
        # MediaPipe
        mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp_pose
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.setup_ui()
        self.actualizar_lista_contactos()
        
    def setup_ui(self):
        # Contenedor principal
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel Izquierdo - Contactos
        left_panel = tk.Frame(main_container, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Sección de Contactos
        contactos_frame = tk.LabelFrame(left_panel, text="Lista de Contactos", font=("Arial", 12, "bold"))
        contactos_frame.pack(fill=tk.BOTH, expand=True)
        
        # Lista de contactos con scrollbar
        scrollbar_contactos = tk.Scrollbar(contactos_frame)
        scrollbar_contactos.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.lista_contactos_widget = tk.Listbox(
            contactos_frame,
            yscrollcommand=scrollbar_contactos.set,
            font=("Courier", 10),
            selectmode=tk.SINGLE
        )
        self.lista_contactos_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar_contactos.config(command=self.lista_contactos_widget.yview)
        
        # Botón refrescar contactos
        btn_refrescar = tk.Button(
            contactos_frame,
            text="Refrescar",
            command=self.actualizar_lista_contactos,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10)
        )
        btn_refrescar.pack(pady=5)
        
        # Panel Derecho
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Frame de video
        video_frame = tk.LabelFrame(right_panel, text="Video en Vivo", font=("Arial", 12, "bold"))
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de control
        control_frame = tk.LabelFrame(right_panel, text="Control del Sistema", font=("Arial", 12, "bold"))
        control_frame.pack(fill=tk.X)
        
        # Estado
        self.estado_label = tk.Label(
            control_frame,
            text="Estado: Detenido",
            font=("Arial", 11, "bold"),
            fg="red"
        )
        self.estado_label.pack(pady=10)
        
        # Información de grabación
        self.info_label = tk.Label(
            control_frame,
            text="Sin detecciones",
            font=("Arial", 10),
            fg="gray"
        )
        self.info_label.pack(pady=5)
        
        # Botones de control
        btn_control_frame = tk.Frame(control_frame)
        btn_control_frame.pack(pady=10)
        
        self.btn_iniciar = tk.Button(
            btn_control_frame,
            text="Iniciar Detección",
            command=self.iniciar_websocket,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            width=20
        )
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        
        self.btn_detener = tk.Button(
            btn_control_frame,
            text="Detener",
            command=self.detener_websocket,
            bg="#f44336",
            fg="white",
            font=("Arial", 11, "bold"),
            width=20,
            state=tk.DISABLED
        )
        self.btn_detener.pack(side=tk.LEFT, padx=5)
        
    def actualizar_lista_contactos(self):
        """Actualiza la lista de contactos en la interfaz"""
        self.lista_contactos_widget.delete(0, tk.END)
        contactos = cargar_contactos()
        
        if not contactos:
            self.lista_contactos_widget.insert(tk.END, "  No hay contactos registrados")
        else:
            for i, chat_id in enumerate(contactos, 1):
                self.lista_contactos_widget.insert(tk.END, f"  {i}. Chat ID: {chat_id}")
        
        self.lista_contactos_widget.insert(0, f"  Total: {len(contactos)} contacto(s)")
        self.lista_contactos_widget.itemconfig(0, {'bg': '#E3F2FD'})
    
    def iniciar_websocket(self):
        """Inicia la conexión WebSocket"""
        if self.running:
            return
        
        self.running = True
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_detener.config(state=tk.NORMAL)
        self.estado_label.config(text="Estado: Conectando...", fg="orange")
        
        threading.Thread(target=self.conectar_websocket, daemon=True).start()
    
    def conectar_websocket(self):
        """Conecta al WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                "ws://localhost:9092",
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.ws.run_forever()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error de conexión: {e}"))
            self.detener_websocket()
    
    def on_open(self, ws):
        """Callback cuando se abre la conexión"""
        self.root.after(0, lambda: self.estado_label.config(text="Estado: Conectado", fg="green"))
    
    def on_error(self, ws, error):
        """Callback de error"""
        print(f"Error WebSocket: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Callback al cerrar conexión"""
        self.root.after(0, lambda: self.estado_label.config(text="Estado: Desconectado", fg="red"))
    
    def on_message(self, ws, message):
        """Procesa mensajes del WebSocket"""
        data = json.loads(message)
        
        if data.get("type") != "frame":
            return
        
        # Decodificar imagen
        img_base64 = data["image"]
        img_bytes = base64.b64decode(img_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar detecciones
        for det in data["detections"]:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            score = det["score"]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{det["label"]} {score:.2f}', (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Detección de pose
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            person_roi = rgb_frame[y1:y2, x1:x2]
            
            if person_roi.size > 0:
                results = self.pose_detector.process(person_roi)
                
                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        landmark.x = (landmark.x * (x2 - x1) + x1) / frame.shape[1]
                        landmark.y = (landmark.y * (y2 - y1) + y1) / frame.shape[0]
                    
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
        
        # Lógica de grabación
        detections = data.get("detections", [])
        current_time = time.time()
        
        if len(detections) > 0:
            if self.person_detected_since is None:
                self.person_detected_since = current_time
            
            elapsed = current_time - self.person_detected_since
            self.root.after(0, lambda: self.info_label.config(
                text=f"Persona detectada - Tiempo: {elapsed:.1f}s",
                fg="orange"
            ))
            
            if elapsed >= 5 and not self.recording:
                if not os.path.exists("videos"):
                    os.makedirs("videos")
                    
                self.filename = datetime.now().strftime("videos/persona_%Y%m%d_%H%M%S.mp4")
                h, w, _ = frame.shape
                self.video_writer = cv2.VideoWriter(
                    self.filename,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25.0,
                    (w, h)
                )
                self.video_writer.write(frame)
                self.recording = True
                self.root.after(0, lambda: self.info_label.config(
                    text="GRABANDO VIDEO...",
                    fg="red"
                ))
            elif self.recording:
                self.video_writer.write(frame)
        else:
            self.person_detected_since = None
            if self.recording:
                self.recording = False
                self.video_writer.release()
                self.video_writer = None
                
                # Enviar video
                threading.Thread(target=enviar_video, args=(self.filename, "Video de persona detectada"), daemon=True).start()
                
                self.root.after(0, lambda: self.info_label.config(
                    text="Video enviado a contactos",
                    fg="green"
                ))
            else:
                self.root.after(0, lambda: self.info_label.config(
                    text="Sin detecciones",
                    fg="gray"
                ))
        
        # Mostrar frame
        self.mostrar_frame(frame)
    
    def mostrar_frame(self, frame):
        """Muestra el frame en la interfaz"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar manteniendo aspecto
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 800, 600
        
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def detener_websocket(self):
        """Detiene la conexión WebSocket"""
        self.running = False
        
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.recording = False
        
        if self.ws:
            self.ws.close()
        
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)
        self.estado_label.config(text="Estado: Detenido", fg="red")
        self.info_label.config(text="Sin detecciones", fg="gray")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        self.detener_websocket()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()