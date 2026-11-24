import os
import sys

import cv2
import numpy as np


CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def load_face_detector():
    """Завантажує каскад Хаара для виявлення облич."""
    candidates = [
        os.path.join(os.path.dirname(__file__), CASCADE_FILENAME),
        CASCADE_FILENAME,
        os.path.join("python", CASCADE_FILENAME),
    ]

    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, CASCADE_FILENAME))

    for path in candidates:
        if os.path.isfile(path):
            detector = cv2.CascadeClassifier(path)
            if not detector.empty():
                return detector

    raise FileNotFoundError(
        f"Не вдалося завантажити каскад {CASCADE_FILENAME}. "
        "Перевірте наявність файлу або встановлення OpenCV."
    )


def detect_faces(image, detector):
    """Знаходить обличчя на зображенні та повертає копію з рамками."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    result = image.copy()
    for x, y, width, height in faces:
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return len(faces), result


def run_camera():
    """Запускає виявлення облич у відеопотоці вебкамери."""
    detector = load_face_detector()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Не вдалося відкрити камеру. Перевірте її підключення та доступ.")

    print("Камера запущена. Натисніть Q у вікні відео, щоб завершити.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Не вдалося отримати кадр із камери.")

            count, result = detect_faces(frame, detector)
            cv2.putText(
                result,
                f"Faces: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("OpenCV - face detection", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_streamlit():
    """Вебінтерфейс для виявлення облич на знімку з камери."""
    import streamlit as st

    st.set_page_config(page_title="Виявлення облич — OpenCV", page_icon="📷")
    st.title("📷 Виявлення облич за допомогою OpenCV")
    st.write(
        "Зробіть знімок камерою — OpenCV знайде обличчя та позначить їх рамками. "
        "Програма виявляє обличчя, але не визначає особу людини."
    )

    photo = st.camera_input("Зробити знімок")
    if photo is None:
        st.info("Дозвольте доступ до камери та зробіть знімок для аналізу.")
        return

    image_bytes = np.frombuffer(photo.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Не вдалося прочитати знімок із камери.")
        return

    detector = load_face_detector()
    count, result = detect_faces(image, detector)
    st.metric("Знайдено облич", count)
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Результат OpenCV")


if __name__ == "__main__":
    is_streamlit = (
        any("streamlit" in str(arg).lower() for arg in sys.argv[:2])
        or sys.platform == "emscripten"
        or "pyodide" in sys.modules
        or "streamlit" in sys.modules
    )

    if is_streamlit:
        run_streamlit()
    else:
        run_camera()
