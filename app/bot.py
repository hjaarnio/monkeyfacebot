import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from io import BytesIO
import PIL.Image
import os
from get_monkey import get_monkey
from face_finder import swap_faces


TOKEN = os.environ.get('TOKEN')
print('token', TOKEN)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="""Hello! To use this bot please send me a picture with a face and you will get some great times!""")


def help(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="Take a selfie or send other pic with face on it! The bot will make some adjustments to your pic and send it right back to ya!")


def image_process(bot, update):
    try:
        file_id = update.message.photo[-1]
        newFile = bot.get_file(file_id)
        file1 = newFile.download_as_bytearray()
        bio = BytesIO(file1)
        image = PIL.Image.open(BytesIO(file1))

        # Here should be done the changes to the PIL image created abow
        result_image = swap_faces(image, get_monkey())

        bio.name = 'image.jpeg'
        result_image.save(bio, 'JPEG')
        bio.seek(0)
        bot.send_photo(update.message.chat_id, photo=bio)
    except Exception as e:
        logger.warning('Update "%s" caused error "%s"', update, e)


def text_process(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="You were supposed to upload an image, not to chat with me you moron!")


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():

    updater = Updater(token=TOKEN)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help))
    dispatcher.add_handler(MessageHandler(Filters.photo, image_process))
    dispatcher.add_handler(MessageHandler(Filters.text, text_process))

    # log all errors
    dispatcher.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
