import os
import sys
import cv2
import numpy as np


def get_image_path():
    """Знаходить шлях до figures.png."""
    for p in ["figures.png", "python/figures.png", "../python/figures.png"]:
        if os.path.isfile(p):
            return p
    return "figures.png"


def count_figures(image, thresh1=100, thresh2=200):
    """Рахує кількість фігур на зображенні за допомогою Canny та повертає етапи обробки."""
    # 1. Відтінки сірого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Згладжування Гауса для усунення шуму
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Межі Canny
    edges = cv2.Canny(blurred, thresh1, thresh2)
    
    # 4. Пошук зовнішніх контурів
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Відсікання дрібного шуму за площею
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    
    # 6. Малювання знайдених контурів
    result_img = image.copy()
    cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 2)
    
    stages = {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "result": result_img
    }
    
    return len(valid_contours), stages


def run_cli():
    """Консольний запуск."""
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_image_path()
    img = cv2.imread(img_path)
    if img is None:
        print(f"Помилка: не вдалося відкрити файл {img_path}")
        return

    count, stages = count_figures(img)
    print(f"Кількість знайдених фігур: {count}")

    out_path = os.path.join(os.path.dirname(img_path) or ".", "figures_detected.png")
    cv2.imwrite(out_path, stages["result"])
    print(f"Збережено результат у {out_path}")


def run_streamlit():
    """Веб-інтерфейс на базі Streamlit / stlite."""
    import streamlit as st

    st.set_page_config(page_title="Підрахунок фігур (Canny)", layout="wide")
    st.title("🔍 Підрахунок фігур за допомогою Canny (OpenCV)")

    img_path = get_image_path()
    img = cv2.imread(img_path)

    if img is None:
        st.error(f"Не вдалося завантажити зображення: {img_path}")
        return

    st.sidebar.header("Параметри Canny")
    thresh1 = st.sidebar.slider("Нижній поріг (Threshold 1)", 0, 255, 50, step=5)
    thresh2 = st.sidebar.slider("Верхній поріг (Threshold 2)", 0, 255, 150, step=5)

    count, stages = count_figures(img, thresh1, thresh2)

    # Вкладки: Результат та Етапи обробки
    tab_result, tab_stages = st.tabs(["🎯 Результат", "⚙️ Етапи обробки count_figures"])

    with tab_result:
        st.metric("Кількість знайдених фігур", count)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Вхідне зображення")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader(f"Знайдено {count} фігур")
            st.image(cv2.cvtColor(stages["result"], cv2.COLOR_BGR2RGB), use_container_width=True)

    with tab_stages:
        st.subheader("Послідовні етапи функції count_figures:")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**1. Відтінки сірого**")
            st.caption("cv2.cvtColor(..., COLOR_BGR2GRAY)")
            st.image(stages["gray"], use_container_width=True)
        with c2:
            st.markdown("**2. Розмиття Гауса**")
            st.caption("cv2.GaussianBlur(..., (5, 5), 0)")
            st.image(stages["blurred"], use_container_width=True)
        with c3:
            st.markdown("**3. Межі Canny**")
            st.caption(f"cv2.Canny(..., {thresh1}, {thresh2})")
            st.image(stages["edges"], use_container_width=True)
        with c4:
            st.markdown("**4. Виділені контури**")
            st.caption(f"cv2.drawContours(...) -> {count} фігур")
            st.image(cv2.cvtColor(stages["result"], cv2.COLOR_BGR2RGB), use_container_width=True)


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
        run_cli()
