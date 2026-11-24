import io
import sys
import time
from PIL import Image
from rembg import remove
import telebot
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

# رفع کامل مشکل چاپ کاراکترها در ترمینال ویندوز و VS Code
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# توکن ربات خود را بین دو علامت " " زیر قرار دهید:
TOKEN = "7864602097:AAE2zKyruzspmtLj0-FcnaZVa3B_NLs-xTs"
# ==========================================

if ":" not in TOKEN or TOKEN == "YOUR_BOT_TOKEN":
    print("ERROR: Please put your real Telegram Bot Token in line 17!")
    print("Then press Ctrl+S to save the file and run again.")
    sys.exit(1)

bot = telebot.TeleBot(TOKEN)

# لیست فرمت‌های پشتیبانی شده
FORMATS = [
    "PNG",
    "JPEG",
    "WEBP",
    "PDF",
    "BMP",
    "GIF",
    "TIFF",
    "ICO",
    "EPS",
    "TGA",
    "PCX",
    "PPM",
]


# ساخت کیبورد شیشه‌ای
def get_format_keyboard():
    markup = InlineKeyboardMarkup(row_width=3)
    buttons = []
    for fmt in FORMATS:
        buttons.append(
            InlineKeyboardButton(text=f"🖼 {fmt}", callback_data=f"convert_{fmt}")
        )
    # اضافه کردن دکمه‌های فرمت (۳ تایی در هر ردیف)
    markup.add(*buttons)

    # اضافه کردن دکمه حذف پس‌زمینه به صورت تکی در پایین‌ترین ردیف
    markup.add(
        InlineKeyboardButton(
            text="✂️ حذف پس‌زمینه (با هوش مصنوعی)", callback_data="remove_bg"
        )
    )
    return markup


# هندلر دستور /start
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    welcome_text = (
        "سلام! 👋\n"
        "من ربات پیشرفته تغییر فرمت و ویرایش عکس هستم.\n\n"
        "📸 **کافیه یک عکس (معمولی یا فایل) بفرستی** تا فرمتش رو عوض کنم یا پس‌زمینه‌اش رو با هوش مصنوعی حذف کنم!"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")


# هندلر دریافت عکس (از گالری)
@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    file_id = message.photo[-1].file_id
    bot.send_photo(
        chat_id=message.chat.id,
        photo=file_id,
        caption="✅ عکس شما دریافت شد! (فرمت فعلی در تلگرام: JPEG)\nاز منوی زیر عملیات دلخواه را انتخاب کنید: 👇",
        reply_markup=get_format_keyboard(),
        reply_to_message_id=message.message_id,
    )


# هندلر دریافت عکس به صورت فایل (Document)
@bot.message_handler(content_types=["document"])
def handle_document(message):
    if message.document.mime_type and message.document.mime_type.startswith("image/"):
        file_id = message.document.file_id
        bot.send_document(
            chat_id=message.chat.id,
            document=file_id,
            caption="✅ فایل تصویر شما با کیفیت اصلی دریافت شد!\nاز منوی زیر عملیات دلخواه را انتخاب کنید: 👇",
            reply_markup=get_format_keyboard(),
            reply_to_message_id=message.message_id,
        )
    else:
        bot.reply_to(
            message,
            "⚠️ لطفاً فقط فایل‌های تصویری (عکس) ارسال کنید!",
        )


# هندلر تغییر فرمت عکس‌ها
@bot.callback_query_handler(func=lambda call: call.data.startswith("convert_"))
def callback_convert_image(call):
    target_format = call.data.split("_")[1]
    chat_id = call.message.chat.id

    try:
        # پیدا کردن آیدی فایل (عکس یا سند)
        if call.message.photo:
            file_id = call.message.photo[-1].file_id
        elif call.message.document:
            file_id = call.message.document.file_id
        else:
            bot.answer_callback_query(
                call.id, "❌ خطا در پیدا کردن تصویر!", show_alert=True
            )
            return

        # دانلود عکس در حافظه رم
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        image_stream = io.BytesIO(downloaded_file)
        img = Image.open(image_stream)

        # بررسی فرمت فعلی عکس
        current_format = img.format
        if current_format == target_format:
            bot.answer_callback_query(
                call.id,
                f"⚠️ فرمت فعلی این عکس همین الان هم {target_format} است! لطفاً فرمت دیگری انتخاب کنید.",
                show_alert=True,
            )
            return

        bot.answer_callback_query(
            call.id, f"در حال تبدیل از {current_format} به {target_format}... ⏳"
        )
        bot.send_chat_action(chat_id, "upload_document")

        # تبدیل مود رنگی برای فرمت‌هایی که از شفافیت پشتیبانی نمی‌کنند
        if target_format in ["JPEG", "PDF", "BMP", "EPS", "PCX", "PPM"] and img.mode in (
            "RGBA",
            "P",
            "LA",
        ):
            img = img.convert("RGB")

        # تغییر سایز استاندارد برای فرمت آیکون (ICO)
        if target_format == "ICO":
            img.thumbnail((256, 256))

        # ذخیره عکس با فرمت جدید در حافظه رم
        output_stream = io.BytesIO()
        img.save(output_stream, format=target_format)
        output_stream.seek(0)

        file_extension = "jpg" if target_format == "JPEG" else target_format.lower()
        output_stream.name = f"converted.{file_extension}"

        # ارسال عکس تبدیل شده
        bot.send_document(
            chat_id=chat_id,
            document=output_stream,
            caption=f"🎉 تصویر شما از **{current_format}** به **{target_format}** تبدیل شد!",
            reply_to_message_id=call.message.message_id,
            parse_mode="Markdown",
        )

    except Exception as e:
        bot.send_message(
            chat_id, f"❌ متاسفانه در تبدیل به فرمت {target_format} خطایی رخ داد."
        )
        print(f"Error converting image: {e}")


# هندلر حذف پس‌زمینه با هوش مصنوعی
@bot.callback_query_handler(func=lambda call: call.data == "remove_bg")
def callback_remove_bg(call):
    chat_id = call.message.chat.id
    wait_msg = None

    try:
        if call.message.photo:
            file_id = call.message.photo[-1].file_id
        elif call.message.document:
            file_id = call.message.document.file_id
        else:
            bot.answer_callback_query(
                call.id, "❌ خطا در پیدا کردن تصویر!", show_alert=True
            )
            return

        bot.answer_callback_query(call.id, "در حال پردازش با هوش مصنوعی... 🤖")

        # ارسال پیام موقت برای اطلاع‌رسانی به کاربر
        wait_msg = bot.send_message(
            chat_id,
            "🤖 **هوش مصنوعی در حال حذف پس‌زمینه است...**\n⏳ لطفاً چند لحظه صبر کنید (در اولین استفاده ممکن است دانلود مدل ۱ دقیقه طول بکشد).",
            reply_to_message_id=call.message.message_id,
            parse_mode="Markdown",
        )
        bot.send_chat_action(chat_id, "upload_document")

        # دانلود عکس
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # حذف پس‌زمینه توسط کتابخانه rembg
        output_bytes = remove(downloaded_file)

        # آماده‌سازی فایل خروجی در حافظه رم
        output_stream = io.BytesIO(output_bytes)
        output_stream.name = "no_background.png"

        # ارسال تصویر بدون پس‌زمینه به صورت فایل PNG
        bot.send_document(
            chat_id=chat_id,
            document=output_stream,
            caption="✨ **پس‌زمینه تصویر با موفقیت حذف شد!** (فرمت: PNG بدون بک‌گراند)",
            reply_to_message_id=call.message.message_id,
            parse_mode="Markdown",
        )

        # پاک کردن پیام انتظار بعد از اتمام کار
        if wait_msg:
            try:
                bot.delete_message(chat_id, wait_msg.message_id)
            except Exception:
                pass

    except Exception as e:
        if wait_msg:
            try:
                bot.delete_message(chat_id, wait_msg.message_id)
            except Exception:
                pass
        bot.send_message(chat_id, "❌ متاسفانه در حذف پس‌زمینه خطایی رخ داد.")
        print(f"Error removing background: {e}")


# اجرای ربات
if __name__ == "__main__":
    print("Cleaning up old webhooks...")
    bot.remove_webhook()
    time.sleep(1)

    print("Bot is running successfully! Waiting for images...")
    bot.infinity_polling(skip_pending=True)