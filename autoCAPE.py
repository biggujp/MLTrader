import time
import subprocess
import cv2
import numpy as np
import pytesseract
import mss

from pywinauto import Application

# ---------------------------
# ⚙️ CONFIG
# ---------------------------
EXE_PATH = r"C:\CAPE2004\CAPE.EXE"
WINDOW_TITLE = "プロテクタ解除"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# ---------------------------
# 🚀 เปิดโปรแกรม
# ---------------------------
def launch_app():
    subprocess.Popen(EXE_PATH)
    print("🚀 Launching CAPE...")
    time.sleep(5)

# ---------------------------
# 🪟 connect window
# ---------------------------
def connect_app():
    app = Application(backend="uia").connect(title_re=WINDOW_TITLE)
    dlg = app.window(title_re=WINDOW_TITLE)
    dlg.wait('visible', timeout=10)
    return dlg

# ---------------------------
# 📸 capture window
# ---------------------------
def capture_window(dlg):
    rect = dlg.rectangle()
    with mss.mss() as sct:
        monitor = {
            "top": rect.top,
            "left": rect.left,
            "width": rect.width(),
            "height": rect.height()
        }
        img = np.array(sct.grab(monitor))
        return img

# ---------------------------
# ✂️ crop เฉพาะเลข (จูนจากภาพคุณ)
# ---------------------------
def crop_number_area(img):
    h, w = img.shape[:2]
    return img[int(h*0.35):int(h*0.55), int(w*0.1):int(w*0.9)]

# ---------------------------
# 🧠 preprocess (ทำให้ OCR แม่น)
# ---------------------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 ขยายภาพ
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # เพิ่ม contrast
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)

    # blur ลด noise
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # threshold auto
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh

# ---------------------------
# 🔍 OCR อ่านเลข 20 หลัก
# ---------------------------
def read_number(img):
    processed = preprocess(img)

    text = pytesseract.image_to_string(
        processed,
        config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    )

    number = ''.join(filter(str.isdigit, text))

    # ต้อง 20 หลักเท่านั้น
    if len(number) == 20:
        return number
    return None

# ---------------------------
# 🔒 verify อ่านซ้ำกันพลาด
# ---------------------------
def get_verified_number(img):
    for _ in range(5):
        n1 = read_number(img)
        if not n1:
            continue

        time.sleep(0.1)

        n2 = read_number(img)

        if n1 == n2:
            return n1

    return None

# ---------------------------
# 🔄 แปลงเลข
# ---------------------------
def transform(n):
    return ''.join(str((int(d)+1)%10) for d in n)

# ---------------------------
# ✍️ input text (ชัวร์ 100%)
# ---------------------------
def input_text(dlg, text):
    dlg.Edit.click_input()
    time.sleep(0.1)
    dlg.type_keys("^a{BACKSPACE}")
    time.sleep(0.1)
    dlg.type_keys(text, with_spaces=True)

# ---------------------------
# 🔘 click OK
# ---------------------------
def click_button(dlg):
    try:
        dlg.Button.click()
    except:
        dlg.type_keys("{ENTER}")

# ---------------------------
# 🔁 LOOP
# ---------------------------
def run_bot():
    launch_app()
    dlg = connect_app()

    print("🤖 BOT START")

    while True:
        img = capture_window(dlg)
        img = crop_number_area(img)

        number = get_verified_number(img)

        if not number:
            print("❌ OCR FAIL")
            continue

        new_number = transform(number)

        print(f"✅ {number} → {new_number}")

        input_text(dlg, new_number)
        time.sleep(0.2)

        click_button(dlg)

        time.sleep(1)

# ---------------------------
# ▶️ RUN
# ---------------------------
if __name__ == "__main__":
    run_bot()