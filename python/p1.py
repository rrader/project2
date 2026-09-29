import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Налаштування сторінки Streamlit
st.set_page_config(
    page_title="California Housing - Лінійна Регресія",
    page_icon="🏠",
    layout="wide"
)

# Кешування завантаження даних
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    # Перейменуємо MedHouseVal для зручності
    df = df.rename(columns={'MedHouseVal': 'Price ($100k)'})
    return df, housing.DESCR

df, dataset_description = load_data()

st.title("🏠 Аналіз та прогнозування вартості житла у Каліфорнії")
st.markdown("Інтерактивний веб-додаток на **Streamlit** для дослідження даних та демонстрації роботи моделі **Лінійної Регресії** (`scikit-learn`).")

# Бокова панель для налаштувань моделі
st.sidebar.header("⚙️ Налаштування моделі")
test_size = st.sidebar.slider("Розмір тестової вибірки (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
random_state = st.sidebar.number_input("Random State (зерно генератора)", value=42, step=1)

# Вкладки інтерфейсу
tab1, tab2 = st.tabs(["📊 Набір даних (Попередній перегляд)", "🤖 Як працює модель"])

# ---------------------------------------------------------
# Вкладка 1: Попередній перегляд даних
# ---------------------------------------------------------
with tab1:
    st.header("📋 Перегляд даних California Housing")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Кількість записів (рядків)", f"{df.shape[0]:,}")
    col2.metric("Кількість ознак (колонок)", f"{df.shape[1] - 1}")
    col3.metric("Цільова змінна", "Ціна ($100k)")
    
    st.subheader("🔍 Перші рядки таблиці")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📈 Описова статистика")
    st.dataframe(df.describe().T, use_container_width=True)
    
    with st.expander("ℹ️ Опис ознак датасету"):
        st.markdown("""
        * **MedInc**: Середній дохід у районі (у сотнях тисяч доларів США)
        * **HouseAge**: Середній вік будинків у районі
        * **AveRooms**: Середня кількість кімнат на будинок
        * **AveBedrms**: Середня кількість спалень на будинок
        * **Population**: Населення району
        * **AveOccup**: Середня кількість мешканців на будинок
        * **Latitude**: Географічна широта
        * **Longitude**: Географічна довгота
        * **Price ($100k)**: Середня вартість житла (у сотнях тисяч доларів США) — *цільова змінна*
        """)

# ---------------------------------------------------------
# Вкладка 2: Як працює модель
# ---------------------------------------------------------
with tab2:
    st.header("🎯 Оцінка роботи моделі Лінійної Регресії")
    
    # Підготовка даних
    X = df.drop(columns=['Price ($100k)'])
    y = df['Price ($100k)']
    
    # Розбиття на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Тренування моделі
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Прогнозування
    y_pred = model.predict(X_test)
    
    # Обчислення метрик
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Відображення метрик
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MSE (Mean Squared Error)", f"{mse:.4f}")
    m_col2.metric("RMSE (Середня помилка в $)", f"${rmse * 100000:,.2f}")
    m_col3.metric("R² Score (Точність моделі)", f"{r2 * 100:.2f}%")
    
    st.divider()
    
    # Графік: Фактичні vs Прогнозовані значення
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📊 Графік: Фактичні vs Прогнозовані ціни")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.3, color='#1f77b4', label='Прогнози')
        # Ідеальна пряма y = x
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ідеальний прогноз ($y=x$)')
        ax.set_xlabel("Фактична ціна ($100k)")
        ax.set_ylabel("Прогнозована ціна ($100k)")
        ax.set_title("Порівняння реальних цін із прогнозом моделі")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
    with col_right:
        st.subheader("🧮 Інтерактивний прогноз ціни")
        st.write("Змініть параметри будинку, щоб побачити прогноз вартості:")
        
        user_input = {}
        user_input['MedInc'] = st.slider("Середній дохід (MedInc, $100k)", float(X['MedInc'].min()), float(X['MedInc'].max()), float(X['MedInc'].median()))
        user_input['HouseAge'] = st.slider("Вік будинку (HouseAge, роки)", float(X['HouseAge'].min()), float(X['HouseAge'].max()), float(X['HouseAge'].median()))
        user_input['AveRooms'] = st.slider("Середня к-сть кімнат (AveRooms)", float(X['AveRooms'].min()), float(10.0), float(X['AveRooms'].median()))
        user_input['AveBedrms'] = st.slider("Середня к-сть спалень (AveBedrms)", float(X['AveBedrms'].min()), float(5.0), float(X['AveBedrms'].median()))
        user_input['Population'] = st.slider("Населеність району (Population)", float(X['Population'].min()), float(5000.0), float(X['Population'].median()))
        user_input['AveOccup'] = st.slider("К-сть мешканців на будинок (AveOccup)", float(X['AveOccup'].min()), float(6.0), float(X['AveOccup'].median()))
        user_input['Latitude'] = st.slider("Широта (Latitude)", float(X['Latitude'].min()), float(X['Latitude'].max()), float(X['Latitude'].median()))
        user_input['Longitude'] = st.slider("Довгота (Longitude)", float(X['Longitude'].min()), float(X['Longitude'].max()), float(X['Longitude'].median()))
        
        # Прогноз для введених даних
        input_df = pd.DataFrame([user_input])
        predicted_price_100k = model.predict(input_df)[0]
        predicted_price_dollars = max(0, predicted_price_100k * 100000)
        
        st.success(f"🏡 **Прогнозована вартість житла:** **${predicted_price_dollars:,.2f}**")
