import telebot
import os
bot = telebot.TeleBot("7897863184:AAFhiDp4oEHp0OaLU0Cfz2IqdLSw47J6iCs")
chat_id = "1670126488"

def enviar_video(ruta_video, caption="Video detectado"):

    if not os.path.exists(ruta_video):
        print(f"Video no encontrado: {ruta_video}")
        return False

    try:
        with open(ruta_video, "rb") as video:
            bot.send_video(
                chat_id,
                video,
                caption=caption,
                supports_streaming=True
            )
        print("Video enviado")
        return True

    except Exception as e:
        print(f"Error enviando video: {e}")
        return False


bot.polling()