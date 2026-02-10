import telebot
import os
import json

bot = telebot.TeleBot("7897863184:AAFhiDp4oEHp0OaLU0Cfz2IqdLSw47J6iCs")


CONTACTOS_FILE = "contactos.json"


def cargar_contactos():
    if os.path.exists(CONTACTOS_FILE):
        with open(CONTACTOS_FILE, 'r') as f:
            return json.load(f)
    return []

def guardar_contactos(contactos):
    with open(CONTACTOS_FILE, 'w') as f:
        json.dump(contactos, f, indent=2)

def enviar_video(ruta_video, caption="Video detectado"):
    if not os.path.exists(ruta_video):
        print(f"Video no encontrado: {ruta_video}")
        return False

    enviados = 0
    errores = 0

    try:
        with open(ruta_video, "rb") as video:
            lista_contactos = cargar_contactos()
            for chat_id in lista_contactos:
                try:
                    video.seek(0)  # Reiniciar el puntero del archivo
                    bot.send_video(
                        chat_id,
                        video,
                        caption=caption,
                        supports_streaming=True
                    )
                    enviados += 1
                    print(f"Video enviado a {chat_id}")
                except Exception as e:
                    errores += 1
                    print(f"Error enviando a {chat_id}: {e}")

        print(f"Resumen: {enviados} enviados, {errores} errores")
        return enviados > 0

    except Exception as e:
        print(f"Error general: {e}")
        return False

