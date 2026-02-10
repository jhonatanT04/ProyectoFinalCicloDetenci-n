import telebot
import os
import json

bot = telebot.TeleBot("7897863184:AAFhiDp4oEHp0OaLU0Cfz2IqdLSw47J6iCs")

# Archivo para guardar la lista de contactos
CONTACTOS_FILE = "contactos.json"

# Cargar lista de contactos desde el archivo
def cargar_contactos():
    if os.path.exists(CONTACTOS_FILE):
        with open(CONTACTOS_FILE, 'r') as f:
            return json.load(f)
    return []

# Guardar lista de contactos en el archivo
def guardar_contactos(contactos):
    with open(CONTACTOS_FILE, 'w') as f:
        json.dump(contactos, f, indent=2)

# Lista global de contactos
lista_contactos = cargar_contactos()

# Comando para registrarse
@bot.message_handler(commands=['registrar', 'start'])
def registrar_usuario(message):
    chat_id = str(message.chat.id)
    
    if chat_id not in lista_contactos:
        lista_contactos.append(chat_id)
        guardar_contactos(lista_contactos)
        bot.reply_to(message, "✅ Te has registrado correctamente. Recibirás los videos cuando se detecten.")
    else:
        bot.reply_to(message, "ℹ️ Ya estás registrado en la lista.")

# Comando para desregistrarse
@bot.message_handler(commands=['desregistrar'])
def desregistrar_usuario(message):
    chat_id = str(message.chat.id)
    
    if chat_id in lista_contactos:
        lista_contactos.remove(chat_id)
        guardar_contactos(lista_contactos)
        bot.reply_to(message, "❌ Te has eliminado de la lista de contactos.")
    else:
        bot.reply_to(message, "ℹ️ No estabas registrado en la lista.")

# Comando para ver cuántos contactos hay
@bot.message_handler(commands=['total'])
def total_contactos(message):
    bot.reply_to(message, f"📊 Total de contactos registrados: {len(lista_contactos)}")

# Función para enviar video a todos los contactos
def enviar_video(ruta_video, caption="Video detectado"):
    if not os.path.exists(ruta_video):
        print(f"Video no encontrado: {ruta_video}")
        return False

    enviados = 0
    errores = 0

    try:
        with open(ruta_video, "rb") as video:
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

# Iniciar bot
print("Bot iniciado...")
bot.polling()