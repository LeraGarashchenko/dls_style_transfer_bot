import asyncio
from collections import defaultdict
import functools
import io
from typing import List

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters import MediaGroupFilter
from aiogram.types import ContentType
from aiogram_media_group import media_group_handler

from style_transfer_bot.config import TOKEN, TRANSFER_PARAMS
from style_transfer_bot.model.transfer_model import StyleTransfer
from style_transfer_bot.utils import get_logger


bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())

user_photos = defaultdict(list)
logger = get_logger("style_transfer_bot")


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply(
        "Бот по переносу стиля.\n"
        "Отправьте 2 фотографии:\n"
        "• 1 - контент:  фото, которое требуется изменить;\n"
        "• 2 - стиль: фото, с которого нужно взять стиль.\n"
        "В результате получится первое фото с перенесенным со второго фото стилем."
    )


async def process_photo(messages: List[types.Message]):
    chat_id = messages[0].chat.id
    last_message = messages[-1]

    for message in messages:
        file = await bot.get_file(message.photo[-1].file_id)
        downloaded_file = await bot.download_file(file.file_path)
        user_photos[chat_id].append(downloaded_file.read())

    if len(user_photos[chat_id]) == 1:
        await last_message.answer("Загружен контент.")
    elif len(user_photos[chat_id]) == 2:
        await last_message.answer("Контент и стиль загружены. Ожидание обработки изображений...")
        processed_image = await do_transfer(user_photos[chat_id][0], user_photos[chat_id][1])
        processed_image = types.InputFile(io.BytesIO(processed_image))
        await bot.send_photo(chat_id, processed_image)
        del user_photos[chat_id]
    elif len(user_photos[chat_id]) > 2:
        await last_message.answer("Дождитесь обработки предыдущих изображений и попробуйте снова.")


@dp.message_handler(MediaGroupFilter(is_media_group=False), content_types=ContentType.PHOTO)
async def process_photo_by_one(message: types.Message):
    await process_photo([message])


@dp.message_handler(MediaGroupFilter(is_media_group=True), content_types=ContentType.PHOTO)
@media_group_handler
async def process_photo_group(messages: List[types.Message]):
    await process_photo(messages)


async def do_transfer(content_image: bytes, style_image: bytes):
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                model.transfer_style,
                TRANSFER_PARAMS["n_epochs"],
                content_image,
                style_image,
                style_weight=TRANSFER_PARAMS["style_weight"],
                content_weight=TRANSFER_PARAMS["content_weight"],
                lr=TRANSFER_PARAMS["lr"]
            )
        )
    except Exception as e:
        logger.error(repr(e))


if __name__ == "__main__":
    model = StyleTransfer(TRANSFER_PARAMS["img_size"])
    executor.start_polling(dp, skip_updates=True)
