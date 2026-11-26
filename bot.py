import io
import os
import sys
import json
import time
import threading
import requests
import numpy as np
from PIL import Image, ImageOps
from rembg import remove, new_session
import telebot
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# یک وب‌سرور ساده که فقط برای Railway اجرا می‌شود
def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Bot is alive")
        def log_message(self, format, *args):
            pass  # لاگ‌های HTTP را خاموش کن
    server = HTTPServer(('0.0.0.0', port), Handler)
    server.serve_forever()

# اجرای وب‌سرور در یک ترد جداگانه
threading.Thread(target=run_dummy_server, daemon=True).start()

# ---------- Swin2SR (افزایش کیفیت) ----------
try:
    import torch
    import torch.nn.functional as F
    from transformers import Swin2SRForImageSuperResolution
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# رفع مشکل چاپ کاراکترها در ترمینال ویندوز
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==========================================
# توکن ربات خود را بین دو علامت " " زیر قرار دهید:
TOKEN = "7864602097:AAE2zKyruzspmtLj0-FcnaZVa3B_NLs-xTs"
# ==========================================

if ":" not in TOKEN or TOKEN == "توکن_خودت_اینجا":
    print("ERROR: Please put your real Telegram Bot Token in the TOKEN variable!")
    sys.exit(1)

bot = telebot.TeleBot(TOKEN)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# مدل حذف پس‌زمینه
#   silueta ~43MB | u2netp ~4.7MB | u2net ~176MB | isnet-general-use ~179MB
BG_MODEL = "silueta"
# ==========================================

print(f"Loading background removal model: {BG_MODEL} ...")
bg_session = new_session(BG_MODEL)
bg_lock = threading.Semaphore(1)
print("Background model loaded.")


# ==========================================
# مدل افزایش کیفیت Swin2SR (Classical)
#   x2: caidas/swin2SR-classical-sr-x2-64  (سبک‌تر و سریع‌تر - پیشنهادی)
#   x4: caidas/swin2SR-classical-sr-x4-64
SR_REPO = "caidas/swin2SR-classical-sr-x2-64"

# آدرس‌های دانلود به ترتیب اولویت (اگر یکی کار نکرد، بعدی امتحان می‌شود)
# هر آدرس دیگری که ساختار مخزن Hugging Face را داشته باشد می‌توانی اضافه کنی
SR_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]

# پوشه ذخیره مدل (کنار bot.py)
SR_DIR = os.path.join(BASE_DIR, "models", SR_REPO.split("/")[-1])

SR_MAX_INPUT_SIDE = 1024   # بزرگ‌ترین ضلع ورودی
SR_TILE = 256              # اندازه تکه‌های پردازش (مضرب ۸)
SR_OVERLAP = 16            # هم‌پوشانی تکه‌ها برای جلوگیری از خط درز
# ==========================================


def _download_file(url, dest, min_size=1):
    """دانلود یک فایل با نمایش درصد پیشرفت. در صورت موفقیت True برمی‌گرداند."""
    tmp = dest + ".part"
    try:
        with requests.get(url, stream=True, timeout=(15, 60), allow_redirects=True) as r:
            if r.status_code != 200:
                print(f"  -> HTTP {r.status_code}")
                return False
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            last_print = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if total and time.time() - last_print > 1:
                        last_print = time.time()
                        print(f"\r  -> {done / 1048576:.1f} / {total / 1048576:.1f} MB "
                              f"({done * 100 // total}%)", end="", flush=True)
            print()
        if os.path.getsize(tmp) < min_size:
            os.remove(tmp)
            return False
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print(f"  -> failed: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def _has_weights(folder):
    return any(
        os.path.exists(os.path.join(folder, n))
        for n in ("model.safetensors", "pytorch_model.bin")
    )


def ensure_sr_model():
    """اگر مدل روی دیسک نبود، از میرورها دانلود می‌کند. مسیر پوشه مدل را برمی‌گرداند."""
    os.makedirs(SR_DIR, exist_ok=True)

    # فایل تنظیمات
    cfg_path = os.path.join(SR_DIR, "config.json")
    if not os.path.exists(cfg_path):
        ok = False
        for base in SR_MIRRORS:
            url = f"{base}/{SR_REPO}/resolve/main/config.json"
            print(f"Downloading config.json from {base} ...")
            if _download_file(url, cfg_path, min_size=50):
                ok = True
                break
        if not ok:
            raise RuntimeError("could not download config.json from any mirror")

    # فایل وزن‌ها
    if not _has_weights(SR_DIR):
        ok = False
        for base in SR_MIRRORS:
            for fname in ("model.safetensors", "pytorch_model.bin"):
                url = f"{base}/{SR_REPO}/resolve/main/{fname}"
                print(f"Downloading {fname} from {base} ...")
                if _download_file(url, os.path.join(SR_DIR, fname), min_size=1_000_000):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            raise RuntimeError("could not download model weights from any mirror")

    return SR_DIR


sr_model = None
sr_device = "cpu"
sr_lock = threading.Semaphore(1)

if TORCH_OK:
    try:
        model_dir = ensure_sr_model()
        print(f"Loading Swin2SR model from: {model_dir}")
        sr_device = "cuda" if torch.cuda.is_available() else "cpu"
        sr_model = Swin2SRForImageSuperResolution.from_pretrained(model_dir).to(sr_device).eval()
        print(f"Swin2SR loaded on {sr_device}.")
    except Exception as e:
        print(f"WARNING: could not load Swin2SR model: {e}")
        print("Manual fix: download 'config.json' and 'model.safetensors' (or 'pytorch_model.bin')")
        print(f"of the repo '{SR_REPO}' and put them in this folder:\n  {SR_DIR}")
else:
    print("WARNING: torch/transformers not installed. Upscale feature is disabled.")


# ==========================================
# دکمه شیشه‌ای با پارامتر رسمی style (Bot API 9.4+)
# مقادیر مجاز: "primary" (آبی) | "success" (سبز) | "danger" (قرمز)
# ==========================================
class StyledInlineKeyboardButton(InlineKeyboardButton):
    def __init__(self, *args, style=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.style = style

    def to_dict(self):
        data = super().to_dict()
        if self.style:
            data["style"] = self.style
        return data

    def to_json(self):
        return json.dumps(self.to_dict())


FORMATS = [
    "PNG", "JPEG", "WEBP", "PDF", "BMP", "GIF",
    "TIFF", "ICO", "EPS", "TGA",
]

# پالت ۹ رنگی: کلید -> (نام فارسی, RGB)
BG_COLORS = {
    "white":  ("سفید",    (255, 255, 255)),
    "black":  ("مشکی",    (0, 0, 0)),
    "red":    ("قرمز",    (220, 53, 69)),
    "orange": ("نارنجی",  (253, 126, 20)),
    "yellow": ("زرد",     (255, 205, 0)),
    "green":  ("سبز",     (40, 167, 69)),
    "blue":   ("آبی",     (13, 110, 253)),
    "purple": ("بنفش",    (111, 66, 193)),
    "brown":  ("قهوه‌ای",  (139, 90, 43)),
}


# ==========================================
# کیبوردها
# ==========================================
def get_format_keyboard():
    markup = InlineKeyboardMarkup(row_width=3)

    # دکمه‌های فرمت: سبز (success)
    buttons = [
        StyledInlineKeyboardButton(
            text=f"🖼 {fmt}",
            callback_data=f"convert_{fmt}",
            style="success",
        )
        for fmt in FORMATS
    ]
    markup.add(*buttons)

    # دکمه حذف پس‌زمینه: آبی (primary)
    markup.add(
        StyledInlineKeyboardButton(
            text=" حذف پس‌زمینه عکس ",
            callback_data="remove_bg",
            style="primary",
        )
    )

    # دکمه افزایش کیفیت (زیر دکمه حذف پس‌زمینه)
    markup.add(
        StyledInlineKeyboardButton(
            text=" افزایش کیفیت عکس ",
            callback_data="upscale",
            style="primary",
        )
    )
    return markup


def get_result_keyboard():
    """کیبورد زیر عکس بدون پس‌زمینه"""
    markup = InlineKeyboardMarkup()
    markup.add(
        StyledInlineKeyboardButton(
            text=" تغییر رنگ پس‌زمینه ",
            callback_data="bgcolor_menu",
            style="primary",
        )
    )
    return markup


def get_palette_keyboard():
    """۹ دکمه رنگ (۳×۳) فقط با نام رنگ + دکمه بازگشت"""
    markup = InlineKeyboardMarkup(row_width=3)
    buttons = [
        InlineKeyboardButton(text=name, callback_data=f"setbg_{key}")
        for key, (name, _rgb) in BG_COLORS.items()
    ]
    markup.add(*buttons)
    markup.add(InlineKeyboardButton(text="بازگشت", callback_data="bgcolor_back"))
    return markup


# ==========================================
# توابع کمکی
# ==========================================
def get_file_id(message):
    if message.photo:
        return message.photo[-1].file_id
    if message.document:
        return message.document.file_id
    return None


def apply_background_color(png_bytes, rgb):
    """قرار دادن تصویر شفاف روی یک رنگ ثابت"""
    fg = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", fg.size, rgb + (255,))
    return Image.alpha_composite(bg, fg).convert("RGB")


def _tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def upscale_image(img):
    """افزایش کیفیت با Swin2SR به‌صورت تکه‌تکه (کم‌مصرف برای رم)"""
    img = ImageOps.exif_transpose(img)

    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")
    alpha = img.getchannel("A") if has_alpha else None
    rgb = img.convert("RGB")

    # کنترل اندازه ورودی
    w, h = rgb.size
    longest = max(w, h)
    if longest > SR_MAX_INPUT_SIDE:
        r = SR_MAX_INPUT_SIDE / longest
        w, h = max(1, round(w * r)), max(1, round(h * r))
        rgb = rgb.resize((w, h), Image.LANCZOS)
        if alpha is not None:
            alpha = alpha.resize((w, h), Image.LANCZOS)

    # تبدیل به تنسور [0,1] و پدینگ به مضرب ۸ (window size مدل)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    pad_h, pad_w = (-h) % 8, (-w) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    scale = int(sr_model.config.upscale)
    _, _, H, W = tensor.shape
    th, tw = min(SR_TILE, H), min(SR_TILE, W)
    stride = SR_TILE - SR_OVERLAP

    out = torch.zeros((1, 3, H * scale, W * scale))
    weight = torch.zeros((1, 1, H * scale, W * scale))

    with torch.no_grad():
        for y in _tile_starts(H, th, stride):
            for x in _tile_starts(W, tw, stride):
                patch = tensor[:, :, y:y + th, x:x + tw].to(sr_device)
                res = sr_model(pixel_values=patch).reconstruction.float().cpu()
                out[:, :, y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += res
                weight[:, :, y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += 1

    out = (out / weight)[:, :, :h * scale, :w * scale].clamp_(0, 1)
    out_arr = (out.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    result = Image.fromarray(out_arr)

    if alpha is not None:
        result.putalpha(alpha.resize(result.size, Image.LANCZOS))
    return result


# ==========================================
# هندلرها
# ==========================================
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    welcome_text = (
        "سلام! 👋\n"
        "ربات ویرایش عکس و تغییر فرمت.\n\n"
        "📸 یک عکس بفرستید تا فرمت آن را تغییر دهم، پس‌زمینه‌اش را با هوش مصنوعی حذف کنم یا کیفیتش را افزایش دهم."
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")


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
        bot.reply_to(message, "⚠️ لطفاً فقط فایل‌های تصویری (عکس) ارسال کنید!")


# ---------- تغییر فرمت ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith("convert_"))
def callback_convert_image(call):
    target_format = call.data.split("_")[1]
    chat_id = call.message.chat.id

    try:
        file_id = get_file_id(call.message)
        if not file_id:
            bot.answer_callback_query(call.id, "❌ خطا در پیدا کردن تصویر!", show_alert=True)
            return

        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file))

        current_format = img.format
        if current_format == target_format:
            bot.answer_callback_query(
                call.id,
                f"⚠️ فرمت فعلی این عکس همین الان هم {target_format} است! لطفاً فرمت دیگری انتخاب کنید.",
                show_alert=True,
            )
            return

        bot.answer_callback_query(call.id, f"در حال تبدیل از {current_format} به {target_format}... ⏳")
        bot.send_chat_action(chat_id, "upload_document")

        if target_format in ["JPEG", "PDF", "BMP", "EPS", "PCX", "PPM"] and img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        if target_format == "ICO":
            img.thumbnail((256, 256))

        output_stream = io.BytesIO()
        img.save(output_stream, format=target_format)
        output_stream.seek(0)

        file_extension = "jpg" if target_format == "JPEG" else target_format.lower()
        output_stream.name = f"converted.{file_extension}"

        bot.send_document(
            chat_id=chat_id,
            document=output_stream,
            caption=f"🎉 تصویر شما از **{current_format}** به **{target_format}** تبدیل شد!",
            reply_to_message_id=call.message.message_id,
            parse_mode="Markdown",
        )

    except Exception as e:
        bot.send_message(chat_id, f"❌ متاسفانه در تبدیل به فرمت {target_format} خطایی رخ داد.")
        print(f"Error converting image: {e}")


# ---------- حذف پس‌زمینه ----------
@bot.callback_query_handler(func=lambda call: call.data == "remove_bg")
def callback_remove_bg(call):
    chat_id = call.message.chat.id
    wait_msg = None

    try:
        file_id = get_file_id(call.message)
        if not file_id:
            bot.answer_callback_query(call.id, "❌ خطا در پیدا کردن تصویر!", show_alert=True)
            return

        bot.answer_callback_query(call.id, "در حال پردازش با هوش مصنوعی... 🤖")

        wait_msg = bot.send_message(
            chat_id, "⏳", reply_to_message_id=call.message.message_id
        )
        bot.send_chat_action(chat_id, "upload_document")

        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with bg_lock:
            output_bytes = remove(downloaded_file, session=bg_session)

        output_stream = io.BytesIO(output_bytes)
        output_stream.name = "no_background.png"

        # دکمه «تغییر رنگ پس‌زمینه» زیر عکس خروجی
        bot.send_document(
            chat_id=chat_id,
            document=output_stream,
            caption="✅ **پس‌زمینه تصویر با موفقیت حذف شد!**",
            reply_to_message_id=call.message.message_id,
            parse_mode="Markdown",
            reply_markup=get_result_keyboard(),
        )

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


# ---------- نمایش پالت رنگ ----------
@bot.callback_query_handler(func=lambda call: call.data == "bgcolor_menu")
def callback_bgcolor_menu(call):
    bot.answer_callback_query(call.id, "یک رنگ انتخاب کنید 🎨")
    try:
        bot.edit_message_reply_markup(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=get_palette_keyboard(),
        )
    except Exception as e:
        print(f"Error showing palette: {e}")


# ---------- بازگشت از پالت ----------
@bot.callback_query_handler(func=lambda call: call.data == "bgcolor_back")
def callback_bgcolor_back(call):
    bot.answer_callback_query(call.id)
    try:
        bot.edit_message_reply_markup(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            reply_markup=get_result_keyboard(),
        )
    except Exception as e:
        print(f"Error hiding palette: {e}")


# ---------- ساخت تصویر با رنگ انتخابی ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith("setbg_"))
def callback_set_bg_color(call):
    chat_id = call.message.chat.id
    key = call.data.split("_", 1)[1]

    if key not in BG_COLORS:
        bot.answer_callback_query(call.id, "❌ رنگ نامعتبر است!", show_alert=True)
        return

    name, rgb = BG_COLORS[key]

    try:
        if not call.message.document:
            bot.answer_callback_query(call.id, "❌ خطا در پیدا کردن تصویر!", show_alert=True)
            return

        bot.answer_callback_query(call.id, f"در حال ساخت تصویر با پس‌زمینه {name}... ⏳")
        bot.send_chat_action(chat_id, "upload_document")

        # تصویر شفاف از خود پیام خوانده می‌شود (بدون نیاز به ذخیره‌سازی)
        file_info = bot.get_file(call.message.document.file_id)
        png_bytes = bot.download_file(file_info.file_path)

        result = apply_background_color(png_bytes, rgb)

        output_stream = io.BytesIO()
        result.save(output_stream, format="PNG")
        output_stream.seek(0)
        output_stream.name = f"background_{key}.png"

        bot.send_document(
            chat_id=chat_id,
            document=output_stream,
            caption=f"🎨 تصویر با پس‌زمینه **{name}** ساخته شد!",
            reply_to_message_id=call.message.message_id,
            parse_mode="Markdown",
        )

    except Exception as e:
        bot.send_message(chat_id, "❌ متاسفانه در تغییر رنگ پس‌زمینه خطایی رخ داد.")
        print(f"Error changing background color: {e}")


# ---------- افزایش کیفیت ----------
@bot.callback_query_handler(func=lambda call: call.data == "upscale")
def callback_upscale(call):
    chat_id = call.message.chat.id
    wait_msg = None

    try:
        if sr_model is None:
            bot.answer_callback_query(
                call.id, "❌ مدل افزایش کیفیت روی سرور فعال نیست!", show_alert=True
            )
            return

        file_id = get_file_id(call.message)
        if not file_id:
            bot.answer_callback_query(call.id, "❌ خطا در پیدا کردن تصویر!", show_alert=True)
            return

        bot.answer_callback_query(call.id, "در حال افزایش کیفیت با هوش مصنوعی... ✨")

        wait_msg = bot.send_message(
            chat_id,
            "⏳ در حال افزایش کیفیت... (ممکن است کمی طول بکشد)",
            reply_to_message_id=call.message.message_id,
        )
        bot.send_chat_action(chat_id, "upload_document")

        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        img = Image.open(io.BytesIO(downloaded_file))
        img.load()
        orig_w, orig_h = img.size

        with sr_lock:
            result = upscale_image(img)

        output_stream = io.BytesIO()
        result.save(output_stream, format="PNG")
        output_stream.seek(0)
        output_stream.name = "upscaled.png"

        bot.send_document(
            chat_id=chat_id,
            document=output_stream,
            caption=(
                "✨ **کیفیت تصویر افزایش یافت!**\n"
                f"📐 {orig_w}×{orig_h} ← {result.width}×{result.height}"
            ),
            reply_to_message_id=call.message.message_id,
            parse_mode="Markdown",
        )

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
        bot.send_message(chat_id, "❌ متاسفانه در افزایش کیفیت خطایی رخ داد.")
        print(f"Error upscaling image: {e}")


if __name__ == "__main__":
    print(f"pyTelegramBotAPI version: {getattr(telebot, '__version__', 'unknown')}")
    print("Cleaning up old webhooks...")
    bot.remove_webhook()
    time.sleep(1)

    print("Bot is running successfully! Waiting for images...")
    bot.infinity_polling(skip_pending=True)
