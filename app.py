import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import requests
import streamlink
from ultralytics import YOLO

# 1. KONFIGURACJA I KATALOGI
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Czas działania skryptu w sekundach (13 minut = 780 sekund)
MAX_RUN_DURATION = 780

# Prawidłowy link eksportowy CSV z Google Sheets:
SHEETS_CSV_URL = "https://docs.google.com/spreadsheets/d/1zGjO7LvDWbewwL5vvmtSL8EFm0wTrfiKniH-a02aTjo/export?format=csv&gid=1919540486"

# Ładowanie lekkiego modelu YOLO na CPU
model = YOLO("yolov8n.pt")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
previous_frames = {}


def fetch_camera_list(csv_url):
    try:
        # dodano on_bad_lines='skip' żeby ignorować uszkodzone/nierówne wiersze
        df = pd.read_csv(csv_url, on_bad_lines='skip')
        if "stream_url" in df.columns:
            return df["stream_url"].dropna().unique().tolist()
        elif not df.empty:
            return df.iloc[:, 0].dropna().unique().tolist()
        return []
    except Exception as e:
        print(f"[BŁĄD] Nie udało się pobrać listy z Sheets: {e}")
        return []


def resolve_real_stream_url(url):
    """Przekształca linki ze stron WWW, YouTube lub wygasające tokeny w aktywny strumień HLS."""
    url = str(url).strip()

    # 1. Obsługa YouTube Live
    if "youtube.com" in url or "youtu.be" in url:
        try:
            streams = streamlink.streams(url)
            if "360p" in streams:
                return streams["360p"].url
            elif "720p" in streams:
                return streams["720p"].url
            elif "best" in streams:
                return streams["best"].url
        except Exception as e:
            print(f"[BŁĄD] Nie udało się wyciągnąć streamu z YouTube ({url}): {e}")
            return None

    # 2. Obsługa stron z dynamicznymi tokenami (np. SkylineWebcams, portale z odtwarzaczami)
    if not url.endswith(".m3u8") or "hd-auth" in url:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            res = requests.get(url, headers=headers, timeout=5)
            # Szukamy aktywnego linku .m3u8 w kodzie HTML strony
            match = re.search(
                r'https?://[^\s"\']+\.m3u8[^\s"\']*', res.text
            )
            if match:
                return match.group(0)
        except Exception:
            pass

    # 3. Zwykły, stały link HLS
    return url


def grab_frame_from_stream(raw_url):
    stream_url = resolve_real_stream_url(raw_url)
    if not stream_url:
        return None

    try:
        cap = cv2.VideoCapture(stream_url)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    except Exception:
        pass
    return None


def process_motion_and_detect(frame, camera_id):
    global previous_frames
    resized = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    if camera_id not in previous_frames:
        previous_frames[camera_id] = blurred
        return

    frame_delta = cv2.absdiff(previous_frames[camera_id], blurred)
    previous_frames[camera_id] = blurred

    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = cv2.countNonZero(thresh)

    if non_zero_count > 150:  # Próg wykrycia ruchu
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{SCREENSHOT_DIR}/cam_{camera_id}_{timestamp}.jpg"
        cv2.imwrite(filename, resized)
        print(
            f"[ANOMALIA] Wykryto ruch na kamerze #{camera_id}! Zapisano: {filename}"
        )

        results = model(filename, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(
                    f"    └─ YOLO wykrył obiekt ID: {cls_id} (Pewność: {conf:.2f})"
                )


if __name__ == "__main__":
    print("=== START SKY MONITOR NODE (GITHUB ACTIONS) ===")
    camera_list = fetch_camera_list(SHEETS_CSV_URL)
    print(f"[INFO] Załadowano {len(camera_list)} źródeł z arkusza.")

    start_time = time.time()
    
    # Pętla działa przez 13 minut (780 sekund)
    while time.time() - start_time < MAX_RUN_DURATION:
        loop_start = time.time()
        for idx, raw_url in enumerate(camera_list):
            # Przerwij wykonywanie pętli wewnątrz, jeśli przekroczono limit czasu
            if time.time() - start_time >= MAX_RUN_DURATION:
                break

            frame = grab_frame_from_stream(raw_url)
            if frame is not None:
                process_motion_and_detect(frame, camera_id=idx)

        elapsed = time.time() - loop_start
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    print(f"[INFO] Zakończono sesję analizy po {int(time.time() - start_time)} sekundach.")
    cv2.destroyAllWindows()
