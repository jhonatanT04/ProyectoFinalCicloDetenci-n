import telebot
import os
import json

bot = telebot.TeleBot("7897863184:AAFhiDp4oEHp0OaLU0Cfz2IqdLSw47J6iCs")

from funcioneTelegram import cargar_contactos,guardar_contactos

# Comando para registrarse
@bot.message_handler(commands=['registrar', 'start'])
def registrar_usuario(message):
    chat_id = str(message.chat.id)
    lista_contactos = cargar_contactos()
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
    lista_contactos = cargar_contactos()
    if chat_id in lista_contactos:
        lista_contactos.remove(chat_id)
        guardar_contactos(lista_contactos)
        bot.reply_to(message, "❌ Te has eliminado de la lista de contactos.")
    else:
        bot.reply_to(message, "ℹ️ No estabas registrado en la lista.")

# Comando para ver cuántos contactos hay
@bot.message_handler(commands=['total'])
def total_contactos(message):
    lista_contactos = cargar_contactos()
    bot.reply_to(message, f"📊 Total de contactos registrados: {len(lista_contactos)}")



print("Bot iniciado...")
bot.polling()