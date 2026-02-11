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
from TelegramBot.funcioneTelegram import enviar_video, cargar_contactos, guardar_contactos
from PIL import Image, ImageTk
import os
import psutil
import GPUtil

class VideoDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección y Telegram Bot")
        self.root.geometry("1200x750")
        
        # Variables
        self.ws = None
        self.running = False
        self.person_detected_since = None
        self.recording = False
        
        # Dos writers: uno con keypoints y otro sin
        self.video_writer_kp = None      # Con keypoints
        self.video_writer_clean = None   # Sin keypoints
        self.filename_kp = None
        self.filename_clean = None
        self.current_frame = None
        
        # Variables para FPS
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
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
        
        # Proceso para memoria
        self.process = psutil.Process(os.getpid())
        
        self.setup_ui()
        self.actualizar_lista_contactos()
        self.actualizar_estadisticas_sistema()
        
    def setup_ui(self):
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel Izquierdo
        left_panel = tk.Frame(main_container, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Sección de Contactos
        contactos_frame = tk.LabelFrame(left_panel, text="Lista de Contactos", font=("Arial", 12, "bold"))
        contactos_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
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
        
        btn_contactos_frame = tk.Frame(contactos_frame)
        btn_contactos_frame.pack(pady=5)
        
        tk.Button(
            btn_contactos_frame, text="Refrescar",
            command=self.actualizar_lista_contactos,
            bg="#4CAF50", fg="white", font=("Arial", 10), width=12
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_contactos_frame, text="Eliminar",
            command=self.eliminar_contacto,
            bg="#f44336", fg="white", font=("Arial", 10), width=12
        ).pack(side=tk.LEFT, padx=2)
        
        # Sección de Estadísticas del Sistema
        stats_frame = tk.LabelFrame(left_panel, text="Estadísticas del Sistema", font=("Arial", 12, "bold"))
        stats_frame.pack(fill=tk.BOTH, expand=False)
        
        stats_inner = tk.Frame(stats_frame)
        stats_inner.pack(fill=tk.BOTH, padx=10, pady=10)
        
        def stat_row(parent, label, color):
            frame = tk.Frame(parent)
            frame.pack(fill=tk.X, pady=3)
            tk.Label(frame, text=label, font=("Arial", 10, "bold"), width=14, anchor='w').pack(side=tk.LEFT)
            lbl = tk.Label(frame, text="--", font=("Arial", 10), fg=color)
            lbl.pack(side=tk.LEFT)
            return lbl
        
        self.fps_label       = stat_row(stats_inner, "FPS:",          "green")
        self.mem_proc_label  = stat_row(stats_inner, "RAM Proceso:",  "blue")
        self.mem_sys_label   = stat_row(stats_inner, "RAM Sistema:",  "blue")
        self.gpu_label       = stat_row(stats_inner, "GPU:",          "purple")
        self.gpu_mem_label   = stat_row(stats_inner, "GPU Memoria:",  "purple")
        
        # Panel Derecho
        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        video_frame = tk.LabelFrame(right_panel, text="Video en Vivo", font=("Arial", 12, "bold"))
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        control_frame = tk.LabelFrame(right_panel, text="Control del Sistema", font=("Arial", 12, "bold"))
        control_frame.pack(fill=tk.X)
        
        self.estado_label = tk.Label(
            control_frame, text="Estado: Detenido",
            font=("Arial", 11, "bold"), fg="red"
        )
        self.estado_label.pack(pady=10)
        
        self.info_label = tk.Label(
            control_frame, text="Sin detecciones",
            font=("Arial", 10), fg="gray"
        )
        self.info_label.pack(pady=5)
        
        btn_control_frame = tk.Frame(control_frame)
        btn_control_frame.pack(pady=10)
        
        self.btn_iniciar = tk.Button(
            btn_control_frame, text="Iniciar Detección",
            command=self.iniciar_websocket,
            bg="#4CAF50", fg="white",
            font=("Arial", 11, "bold"), width=20
        )
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        
        self.btn_detener = tk.Button(
            btn_control_frame, text="Detener",
            command=self.detener_websocket,
            bg="#f44336", fg="white",
            font=("Arial", 11, "bold"), width=20,
            state=tk.DISABLED
        )
        self.btn_detener.pack(side=tk.LEFT, padx=5)

    # ─── Contactos ────────────────────────────────────────────────────────────

    def actualizar_lista_contactos(self):
        self.lista_contactos_widget.delete(0, tk.END)
        contactos = cargar_contactos()
        
        if not contactos:
            self.lista_contactos_widget.insert(tk.END, "  No hay contactos registrados")
        else:
            for i, chat_id in enumerate(contactos, 1):
                self.lista_contactos_widget.insert(tk.END, f"  {i}. Chat ID: {chat_id}")
        
        self.lista_contactos_widget.insert(0, f"  Total: {len(contactos)} contacto(s)")
        self.lista_contactos_widget.itemconfig(0, {'bg': '#E3F2FD'})
    
    def eliminar_contacto(self):
        seleccion = self.lista_contactos_widget.curselection()
        
        if not seleccion:
            messagebox.showwarning("Advertencia", "Selecciona un contacto para eliminar")
            return
        
        index = seleccion[0]
        
        if index == 0:
            messagebox.showwarning("Advertencia", "Selecciona un contacto válido para eliminar")
            return
        
        texto = self.lista_contactos_widget.get(index)
        
        if "No hay contactos" in texto:
            return
        
        try:
            chat_id = texto.split("Chat ID: ")[1].strip()
        except:
            messagebox.showerror("Error", "No se pudo obtener el Chat ID")
            return
        
        if not messagebox.askyesno("Confirmar", f"Eliminar contacto?\n\nChat ID: {chat_id}"):
            return
        
        contactos = cargar_contactos()
        if chat_id in contactos:
            contactos.remove(chat_id)
            guardar_contactos(contactos)
            messagebox.showinfo("Exito", f"Contacto {chat_id} eliminado")
            self.actualizar_lista_contactos()
        else:
            messagebox.showerror("Error", "El contacto no existe")

    # ─── Estadísticas ─────────────────────────────────────────────────────────

    def calcular_fps(self):
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
        return self.current_fps
    
    def obtener_estadisticas_sistema(self):
        stats = {}
        mem_info = self.process.memory_info()
        stats['mem_proceso'] = mem_info.rss / (1024 * 1024)
        
        mem_sys = psutil.virtual_memory()
        stats['mem_sistema_percent'] = mem_sys.percent
        stats['mem_sistema_used'] = mem_sys.used / (1024 ** 3)
        stats['mem_sistema_total'] = mem_sys.total / (1024 ** 3)
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats['gpu_load'] = gpu.load * 100
                stats['gpu_mem_used'] = gpu.memoryUsed
                stats['gpu_mem_total'] = gpu.memoryTotal
                stats['gpu_mem_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                stats['gpu_temp'] = gpu.temperature
            else:
                stats['gpu_available'] = False
        except:
            stats['gpu_available'] = False
        
        return stats
    
    def actualizar_estadisticas_sistema(self):
        stats = self.obtener_estadisticas_sistema()
        
        self.fps_label.config(text=f"{self.current_fps:.1f}")
        self.mem_proc_label.config(text=f"{stats['mem_proceso']:.1f} MB")
        self.mem_sys_label.config(
            text=f"{stats['mem_sistema_percent']:.1f}% "
                 f"({stats['mem_sistema_used']:.1f}/{stats['mem_sistema_total']:.1f} GB)"
        )
        
        if stats.get('gpu_available', True) and 'gpu_load' in stats:
            self.gpu_label.config(text=f"{stats['gpu_load']:.1f}% | {stats['gpu_temp']:.0f}C")
            self.gpu_mem_label.config(
                text=f"{stats['gpu_mem_percent']:.1f}% "
                     f"({stats['gpu_mem_used']:.0f}/{stats['gpu_mem_total']:.0f} MB)"
            )
        else:
            self.gpu_label.config(text="No disponible")
            self.gpu_mem_label.config(text="No disponible")
        
        self.root.after(1000, self.actualizar_estadisticas_sistema)

    # ─── WebSocket ────────────────────────────────────────────────────────────

    def iniciar_websocket(self):
        if self.running:
            return
        self.running = True
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_detener.config(state=tk.NORMAL)
        self.estado_label.config(text="Estado: Conectando...", fg="orange")
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        threading.Thread(target=self.conectar_websocket, daemon=True).start()
    
    def conectar_websocket(self):
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
        self.root.after(0, lambda: self.estado_label.config(text="Estado: Conectado", fg="green"))
    
    def on_error(self, ws, error):
        print(f"Error WebSocket: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        self.root.after(0, lambda: self.estado_label.config(text="Estado: Desconectado", fg="red"))

    # ─── Procesamiento de frames ──────────────────────────────────────────────

    def on_message(self, ws, message):
        data = json.loads(message)
        if data.get("type") != "frame":
            return
        
        img_bytes = base64.b64decode(data["image"])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # Frame limpio (sin keypoints) para grabar
        frame_clean = frame.copy()
        
        # Frame con anotaciones para mostrar y grabar
        frame_kp = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for det in data["detections"]:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            score = det["score"]
            
            # Dibujar bounding box en ambos frames
            cv2.rectangle(frame_clean, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_clean, f'{det["label"]} {score:.2f}', (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.rectangle(frame_kp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_kp, f'{det["label"]} {score:.2f}', (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Keypoints solo en frame_kp
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
                    
                    # Solo en frame con keypoints
                    self.mp_drawing.draw_landmarks(
                        frame_kp,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
        
        # Estadísticas en pantalla solo en frame_kp
        frame_kp = self.dibujar_estadisticas(frame_kp)
        
        # Lógica de grabación
        self.procesar_grabacion(frame_kp, frame_clean, data.get("detections", []))
        
        # Mostrar frame con keypoints en la interfaz
        self.mostrar_frame(frame_kp)
    
    def procesar_grabacion(self, frame_kp, frame_clean, detections):
        """Maneja la lógica de grabación de dos videos"""
        current_time = time.time()
        
        if len(detections) > 0:
            if self.person_detected_since is None:
                self.person_detected_since = current_time
            
            elapsed = current_time - self.person_detected_since
            self.root.after(0, lambda: self.info_label.config(
                text=f"Persona detectada - Tiempo: {elapsed:.1f}s",
                fg="orange"
            ))
            
            if elapsed >= 2 and not self.recording:
                if not os.path.exists("videos"):
                    os.makedirs("videos")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Rutas para los dos videos
                self.filename_kp    = f"videos/persona_{timestamp}_keypoints.mp4"
                self.filename_clean = f"videos/persona_{timestamp}_limpio.mp4"
                
                h, w, _ = frame_kp.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                
                self.video_writer_kp    = cv2.VideoWriter(self.filename_kp,    fourcc, 25.0, (w, h))
                self.video_writer_clean = cv2.VideoWriter(self.filename_clean,  fourcc, 25.0, (w, h))
                
                self.video_writer_kp.write(frame_kp)
                self.video_writer_clean.write(frame_clean)
                self.recording = True
                
                self.root.after(0, lambda: self.info_label.config(
                    text="GRABANDO VIDEO...",
                    fg="red"
                ))
            
            elif self.recording:
                self.video_writer_kp.write(frame_kp)
                self.video_writer_clean.write(frame_clean)
        
        else:
            self.person_detected_since = None
            if self.recording:
                self.recording = False
                self.video_writer_kp.release()
                self.video_writer_clean.release()
                self.video_writer_kp = None
                self.video_writer_clean = None
                
                fn_kp    = self.filename_kp
                fn_clean = self.filename_clean
                
                # Enviar los dos videos en hilos separados
                threading.Thread(
                    target=enviar_video,
                    args=(fn_kp, "Video con keypoints - persona detectada"),
                    daemon=True
                ).start()
                
                threading.Thread(
                    target=enviar_video,
                    args=(fn_clean, "Video - persona detectada"),
                    daemon=True
                ).start()
                
                self.root.after(0, lambda: self.info_label.config(
                    text="Videos enviados a contactos (keypoints + limpio)",
                    fg="green"
                ))
            else:
                self.root.after(0, lambda: self.info_label.config(
                    text="Sin detecciones",
                    fg="gray"
                ))

    def dibujar_estadisticas(self, frame):
        fps = self.calcular_fps()
        memoria = self.process.memory_info().rss / (1024 * 1024)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"FPS: {fps:.1f}",          (20, 35), font, 0.6, (0, 255, 0),   2)
        cv2.putText(frame, f"Memoria: {memoria:.1f} MB", (20, 65), font, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def mostrar_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(800 / w, 600 / h)
        frame_resized = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
        
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def detener_websocket(self):
        self.running = False
        
        if self.recording:
            if self.video_writer_kp:
                self.video_writer_kp.release()
            if self.video_writer_clean:
                self.video_writer_clean.release()
            self.recording = False
            self.video_writer_kp = None
            self.video_writer_clean = None
        
        if self.ws:
            self.ws.close()
        
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)
        self.estado_label.config(text="Estado: Detenido", fg="red")
        self.info_label.config(text="Sin detecciones", fg="gray")
    
    def on_closing(self):
        self.detener_websocket()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()