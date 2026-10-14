import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# Налаштування сторінки Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="California Housing — Порівняння Моделей Регресії",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------------
# Кешування завантаження даних
# ---------------------------------------------------------
@st.cache_data
def load_data():
    import os
    # Перевірка наявності локального CSV (для stlite / Pyodide)
    for path in ['california_housing.csv', './california_housing.csv', './python/california_housing.csv', 'python/california_housing.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'MedHouseVal' in df.columns:
                df = df.rename(columns={'MedHouseVal': 'Price ($100k)'})
            return df
            
    # Якщо CSV немає, завантажуємо через sklearn
    try:
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df = df.rename(columns={'MedHouseVal': 'Price ($100k)'})
        return df
    except Exception as e:
        st.error(f"Помилка завантаження даних: {e}")
        raise e

df = load_data()

# ---------------------------------------------------------
# Бокова панель: Незалежний вибір ступеня для кожної моделі
# ---------------------------------------------------------
st.sidebar.header("⚙️ Вибір моделей для порівняння")
st.sidebar.markdown("Оберіть ступінь полінома для кожної моделі незалежно:")

deg1 = st.sidebar.slider(
    "Ступінь Моделі 1 (d₁)",
    min_value=1,
    max_value=3,
    value=1,
    step=1,
    help="Ступінь 1 відповідає звичайній лінійній регресії"
)

deg2 = st.sidebar.slider(
    "Ступінь Моделі 2 (d₂)",
    min_value=1,
    max_value=3,
    value=2,
    step=1,
    help="Оберіть ступінь для другої моделі (наприклад, 2 для квадратичної або 3 для кубічної)"
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Параметри навчання")
test_size = st.sidebar.slider("Розмір тестової вибірки (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
random_state = st.sidebar.number_input("Random State (зерно генератора)", value=42, step=1)

# ---------------------------------------------------------
# Заголовок сторінки
# ---------------------------------------------------------
st.title("🏠 California Housing: Дослідження та Порівняння Моделей Регресії")
st.markdown(
    "Інтерактивний додаток для дослідження **Лінійної** та **Поліноміальних** регресій, розрахунку **MSE на відомих (Train)** та **тестових (Test)** даних, демонстрації формул з **конкретними коефіцієнтами** та виявлення **перенавчання (Overfitting)**."
)

# ---------------------------------------------------------
# Підготовка даних для навчання
# ---------------------------------------------------------
X = df.drop(columns=['Price ($100k)'])
y = df['Price ($100k)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# ---------------------------------------------------------
# Функція навчання та оцінки моделі будь-якого ступеня
# ---------------------------------------------------------
def train_and_eval(deg):
    if deg == 1:
        poly = None
        X_tr = X_train
        X_te = X_test
        feat_names = list(X.columns)
        type_name = "Лінійна регресія"
    elif deg == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_tr = poly.fit_transform(X_train)
        X_te = poly.transform(X_test)
        feat_names = list(poly.get_feature_names_out(X.columns))
        type_name = "Квадратична поліноміальна регресія"
    else:
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_tr = poly.fit_transform(X_train)
        X_te = poly.transform(X_test)
        feat_names = list(poly.get_feature_names_out(X.columns))
        type_name = "Кубічна поліноміальна регресія"
        
    model = LinearRegression()
    model.fit(X_tr, y_train)
    
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    
    mse_tr = float(mean_squared_error(y_train, pred_tr))
    mse_te = float(mean_squared_error(y_test, pred_te))
    diff_mse = mse_te - mse_tr
    ratio_mse = (mse_te / mse_tr) if mse_tr > 0 else 0
    pct_growth = (diff_mse / mse_tr) * 100 if mse_tr > 0 else 0
    
    lse_tr = float(np.sum((y_train - pred_tr) ** 2))
    lse_te = float(np.sum((y_test - pred_te) ** 2))
    rmse_te = float(np.sqrt(mse_te))
    r2_tr = float(r2_score(y_train, pred_tr))
    r2_te = float(r2_score(y_test, pred_te))
    
    return {
        'deg': deg,
        'type_name': type_name,
        'label': f"{type_name} (d={deg})",
        'model': model,
        'poly': poly,
        'feat_names': feat_names,
        'n_features': len(feat_names),
        'intercept': float(model.intercept_),
        'coefs': model.coef_,
        'mse_tr': mse_tr,
        'mse_te': mse_te,
        'diff_mse': diff_mse,
        'ratio_mse': ratio_mse,
        'pct_growth': pct_growth,
        'lse_tr': lse_tr,
        'lse_te': lse_te,
        'rmse_te': rmse_te,
        'r2_tr': r2_tr,
        'r2_te': r2_te,
        'pred_te': pred_te
    }

m1 = train_and_eval(deg1)
m2 = train_and_eval(deg2)

# ---------------------------------------------------------
# Вкладки інтерфейсу
# ---------------------------------------------------------
tab_compare, tab_predict, tab_theory, tab_data = st.tabs([
    "⚖️ Порівняння Моделі 1 та Моделі 2 (MSE, Формули, LSE)",
    "🧮 Інтерактивний прогноз ціни для обох моделей",
    "🧠 Теорія: Чому Test MSE кращий за Train MSE та Overfitting",
    "📋 Набір даних California Housing"
])

# =========================================================
# Вкладка 1: Порівняння Моделі 1 та Моделі 2
# =========================================================
with tab_compare:
    st.header(f"⚖️ Порівняння: {m1['label']} проти {m2['label']}")
    st.markdown("""
    Нижче наведено порівняння двох обраних моделей. Ви можете будь-коли змінити ступінь $d_1$ та $d_2$ на боковій панелі ліворуч.
    """)
    
    # Порівняльні картки моделей у двох колонках
    col_m1, col_m2 = st.columns(2)
    
    # ------------------ МОДЕЛЬ 1 ------------------
    with col_m1:
        st.subheader(f"🔹 Модель 1: {m1['type_name']} (d = {m1['deg']})")
        st.write(f"**Кількість ознак у моделі:** `{m1['n_features']}`")
        
        # Метрики MSE Train vs Test
        met_col1, met_col2 = st.columns(2)
        met_col1.metric("MSE Train (Відомі дані)", f"{m1['mse_tr']:.4f}")
        delta_label1 = f"Δ: {m1['diff_mse']:+.4f} ({m1['pct_growth']:+.1f}%)"
        met_col2.metric("MSE Test (Тестові дані)", f"{m1['mse_te']:.4f}", delta_label1, delta_color="inverse")
        
        met_col3, met_col4 = st.columns(2)
        met_col3.metric("R² Test (Точність)", f"{m1['r2_te'] * 100:.2f}%")
        met_col4.metric("LSE Test (Помилка)", f"{m1['lse_te']:,.1f}")
        
        # Формула з конкретними коефіцієнтами
        st.markdown("##### 📐 Формула Моделі 1 з конкретними коефіцієнтами:")
        
        if m1['deg'] == 1:
            # Повна лінійна формула
            terms = [f"{m1['intercept']:.4f}"]
            for name, c in zip(m1['feat_names'], m1['coefs']):
                sign = "+" if c >= 0 else "-"
                c_val = f"{abs(c):.6f}" if abs(c) < 0.001 else f"{abs(c):.4f}"
                terms.append(rf"{sign} ({c_val} \cdot \text{{{name}}})")
            st.latex(r"\text{Price (\$100k)} = " + " ".join(terms[:5]))
            st.latex(r"& \quad " + " ".join(terms[5:]))
        else:
            # Поліноміальна формула (перші головні члени + таблиця)
            sample_terms = [f"{m1['intercept']:.4f}"]
            for name, c in list(zip(m1['feat_names'], m1['coefs']))[:5]:
                sign = "+" if c >= 0 else "-"
                c_val = f"{abs(c):.6f}" if abs(c) < 0.001 else f"{abs(c):.4f}"
                clean_name = name.replace('^2', '^{2}').replace('^3', '^{3}').replace(' ', r' \cdot ')
                sample_terms.append(rf"{sign} ({c_val} \cdot \text{{{clean_name}}})")
            sample_terms.append(rf"+ \dots \text{{ (разом {m1['n_features']} членів)}}")
            st.latex(r"\text{Price (\$100k)} = " + " ".join(sample_terms))
            
        st.caption(f"ℹ️ Вільний член (w₀ / Intercept): `{m1['intercept']:.6f}`")
        
        with st.expander(f"📋 Переглянути всі коефіцієнти Моделі 1 ({m1['n_features']} шт.)", expanded=False):
            coef_df1 = pd.DataFrame([
                {"Член рівняння": "Вільний член (w₀ / Intercept)", "Коефіцієнт (w)": f"{m1['intercept']:.6f}"}
            ] + [
                {"Член рівняння": name, "Коефіцієнт (w)": f"{c:.6f}" if abs(c) < 0.001 else f"{c:.4f}"}
                for name, c in zip(m1['feat_names'], m1['coefs'])
            ])
            st.dataframe(coef_df1, use_container_width=True, hide_index=True)

    # ------------------ МОДЕЛЬ 2 ------------------
    with col_m2:
        st.subheader(f"🔸 Модель 2: {m2['type_name']} (d = {m2['deg']})")
        st.write(f"**Кількість ознак у моделі:** `{m2['n_features']}`")
        
        # Метрики MSE Train vs Test
        met_col1, met_col2 = st.columns(2)
        met_col1.metric("MSE Train (Відомі дані)", f"{m2['mse_tr']:.4f}")
        delta_label2 = f"Δ: {m2['diff_mse']:+.4f} ({m2['pct_growth']:+.1f}%)"
        met_col2.metric("MSE Test (Тестові дані)", f"{m2['mse_te']:.4f}", delta_label2, delta_color="inverse")
        
        met_col3, met_col4 = st.columns(2)
        met_col3.metric("R² Test (Точність)", f"{m2['r2_te'] * 100:.2f}%")
        met_col4.metric("LSE Test (Помилка)", f"{m2['lse_te']:,.1f}")
        
        # Формула з конкретними коефіцієнтами
        st.markdown("##### 📐 Формула Моделі 2 з конкретними коефіцієнтами:")
        
        if m2['deg'] == 1:
            # Повна лінійна формула
            terms2 = [f"{m2['intercept']:.4f}"]
            for name, c in zip(m2['feat_names'], m2['coefs']):
                sign = "+" if c >= 0 else "-"
                c_val = f"{abs(c):.6f}" if abs(c) < 0.001 else f"{abs(c):.4f}"
                terms2.append(rf"{sign} ({c_val} \cdot \text{{{name}}})")
            st.latex(r"\text{Price (\$100k)} = " + " ".join(terms2[:5]))
            st.latex(r"& \quad " + " ".join(terms2[5:]))
        else:
            # Поліноміальна формула (перші головні члени + таблиця)
            sample_terms2 = [f"{m2['intercept']:.4f}"]
            for name, c in list(zip(m2['feat_names'], m2['coefs']))[:5]:
                sign = "+" if c >= 0 else "-"
                c_val = f"{abs(c):.6f}" if abs(c) < 0.001 else f"{abs(c):.4f}"
                clean_name = name.replace('^2', '^{2}').replace('^3', '^{3}').replace(' ', r' \cdot ')
                sample_terms2.append(rf"{sign} ({c_val} \cdot \text{{{clean_name}}})")
            sample_terms2.append(rf"+ \dots \text{{ (разом {m2['n_features']} членів)}}")
            st.latex(r"\text{Price (\$100k)} = " + " ".join(sample_terms2))
            
        st.caption(f"ℹ️ Вільний член (w₀ / Intercept): `{m2['intercept']:.6f}`")
        
        with st.expander(f"📋 Переглянути всі коефіцієнти Моделі 2 ({m2['n_features']} шт.)", expanded=False):
            coef_df2 = pd.DataFrame([
                {"Член рівняння": "Вільний член (w₀ / Intercept)", "Коефіцієнт (w)": f"{m2['intercept']:.6f}"}
            ] + [
                {"Член рівняння": name, "Коефіцієнт (w)": f"{c:.6f}" if abs(c) < 0.001 else f"{c:.4f}"}
                for name, c in zip(m2['feat_names'], m2['coefs'])
            ])
            st.dataframe(coef_df2, use_container_width=True, hide_index=True)

    st.divider()

    # ------------------ ЗВЕДЕНА ТАБЛИЦЯ ПОРІВНЯННЯ ------------------
    st.subheader("📊 Зведена порівняльна таблиця метрик")
    
    summary_data = []
    for m in [m1, m2]:
        verdict = ""
        if m['deg'] == 1:
            verdict = "🔹 Базова лінійна модель (Стабільна)"
        elif m['deg'] == 2:
            verdict = "🏆 Оптимальний баланс (Мінімальний Test MSE)"
        else:
            verdict = "⚠️ Перенавчання (Overfitting)"
            
        summary_data.append({
            "Модель": m['label'],
            "Ступінь (d)": m['deg'],
            "К-сть ознак": m['n_features'],
            "MSE Train (Відомі)": f"{m['mse_tr']:.4f}",
            "MSE Test (Тестові)": f"{m['mse_te']:.4f}",
            "Різниця Δ MSE (Test − Train)": f"{m['diff_mse']:+.4f}",
            "Зростання похибки": f"+{m['pct_growth']:.1f}% ({m['ratio_mse']:.1f}x)",
            "LSE Test": f"{m['lse_te']:,.1f}",
            "R² Train": f"{m['r2_tr'] * 100:.2f}%",
            "R² Test": f"{m['r2_te'] * 100:.2f}%",
            "Діагноз моделі": verdict
        })
        
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.divider()

    # ------------------ ГРАФІЧНЕ ПОРІВНЯННЯ ------------------
    st.subheader("📉 Графічне порівняння помилок та виявлення Overfitting")
    
    fig_comp, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(13, 4.8))
    
    mod_labels = [m1['label'], m2['label']]
    x_pos = np.arange(len(mod_labels))
    width = 0.35
    
    # 1. Стовпчаста діаграма: MSE Train vs Test
    rects_train = ax_bar1.bar(x_pos - width/2, [m1['mse_tr'], m2['mse_tr']], width, label="MSE Train (Відомі дані)", color="#2ca02c", alpha=0.85)
    rects_test = ax_bar1.bar(x_pos + width/2, [m1['mse_te'], m2['mse_te']], width, label="MSE Test (Тестові дані)", color="#d62728", alpha=0.85)
    
    ax_bar1.set_ylabel("MSE (Mean Squared Error)")
    ax_bar1.set_title("Порівняння MSE: Train (Відомі) vs Test (Тестові)")
    ax_bar1.set_xticks(x_pos)
    ax_bar1.set_xticklabels(mod_labels)
    ax_bar1.legend(loc="upper left")
    ax_bar1.grid(True, linestyle="--", alpha=0.5)
    
    for r in list(rects_train) + list(rects_test):
        h = r.get_height()
        ax_bar1.annotate(f"{h:.4f}",
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
                        
    # 2. Графік розриву Δ MSE
    bar_diffs = ax_bar2.bar(mod_labels, [m1['diff_mse'], m2['diff_mse']], color=["#1f77b4", "#ff7f0e" if m2['diff_mse'] < 1 else "#d62728"], alpha=0.85)
    ax_bar2.set_ylabel("Δ MSE = MSE(Test) − MSE(Train)")
    ax_bar2.set_title("Розрив помилки (Δ MSE) — Індикатор Overfitting")
    ax_bar2.grid(True, linestyle="--", alpha=0.5)
    
    for b, m_info in zip(bar_diffs, [m1, m2]):
        h = b.get_height()
        ax_bar2.annotate(f"{h:+.4f}\n(+{m_info['pct_growth']:.0f}%)",
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
                        
    plt.tight_layout()
    st.pyplot(fig_comp)
    
    # 3. Графіки розсіювання Фактичні vs Прогнозовані
    st.subheader("🎯 Графіки розсіювання: Фактичні vs Прогнозовані ціни (Test Set)")
    fig_sc, (sc_ax1, sc_ax2) = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    
    for ax_cur, m_cur, color_cur in zip([sc_ax1, sc_ax2], [m1, m2], ["#1f77b4", "#2ca02c" if m2['deg'] == 2 else "#d62728"]):
        ax_cur.scatter(y_test, m_cur['pred_te'], alpha=0.25, s=15, color=color_cur, label="Прогнози")
        ax_cur.plot([0, 5], [0, 5], "r--", lw=1.5, label="y = x (Ідеал)")
        ax_cur.set_xlim(-0.5, 5.5)
        ax_cur.set_ylim(-1, 7)  # Обмеження осі Y для читабельності
        ax_cur.set_xlabel("Фактична ціна ($100k)")
        ax_cur.set_ylabel("Прогнозована ціна ($100k)")
        ax_cur.set_title(f"{m_cur['label']}\nTest MSE: {m_cur['mse_te']:.4f} (Δ: {m_cur['diff_mse']:+.2f}) | R²: {m_cur['r2_te']*100:.1f}%")
        ax_cur.legend(loc="upper left")
        ax_cur.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    st.pyplot(fig_sc)
    st.caption("ℹ️ *Примітка: вісь Y на графіках розсіювання обмежено діапазоном [-1, 7] для порівнянності. У кубічної моделі (d=3) окремі прогнози можуть досягати екстремальних значень від -120 до +50 через перенавчання.*")

# =========================================================
# Вкладка 2: Інтерактивний прогноз ціни для обох моделей
# =========================================================
with tab_predict:
    st.header("🧮 Інтерактивний розрахунок прогнозу вартості для обох моделей")
    st.markdown("Змініть параметри будинку за допомогою повзунків та порівняйте прогнози обох моделей:")
    
    # Слайдери параметрів будинку у двох колонках
    pred_col_left, pred_col_right = st.columns(2)
    
    user_input = {}
    with pred_col_left:
        user_input['MedInc'] = st.slider("Середній дохід у районі (MedInc, $100k)", float(X['MedInc'].min()), float(X['MedInc'].max()), float(X['MedInc'].median()))
        user_input['HouseAge'] = st.slider("Вік будинку (HouseAge, роки)", float(X['HouseAge'].min()), float(X['HouseAge'].max()), float(X['HouseAge'].median()))
        user_input['AveRooms'] = st.slider("Середня к-сть кімнат (AveRooms)", float(X['AveRooms'].min()), float(10.0), float(X['AveRooms'].median()))
        user_input['AveBedrms'] = st.slider("Середня к-сть спалень (AveBedrms)", float(X['AveBedrms'].min()), float(5.0), float(X['AveBedrms'].median()))
        
    with pred_col_right:
        user_input['Population'] = st.slider("Населення району (Population)", float(X['Population'].min()), float(5000.0), float(X['Population'].median()))
        user_input['AveOccup'] = st.slider("К-сть мешканців на будинок (AveOccup)", float(X['AveOccup'].min()), float(6.0), float(X['AveOccup'].median()))
        user_input['Latitude'] = st.slider("Географічна широта (Latitude)", float(X['Latitude'].min()), float(X['Latitude'].max()), float(X['Latitude'].median()))
        user_input['Longitude'] = st.slider("Географічна довгота (Longitude)", float(X['Longitude'].min()), float(X['Longitude'].max()), float(X['Longitude'].median()))
        
    input_df = pd.DataFrame([user_input])
    
    # Розрахунок прогнозів для обох моделей
    def compute_pred(m_dict):
        if m_dict['poly'] is None:
            inp = input_df
            trans_vals = input_df.iloc[0].values
        else:
            inp = m_dict['poly'].transform(input_df)
            trans_vals = inp[0]
            
        p_100k = m_dict['model'].predict(inp)[0]
        p_dollars = max(0, p_100k * 100000)
        return p_100k, p_dollars, trans_vals

    p1_100k, p1_dollars, trans1 = compute_pred(m1)
    p2_100k, p2_dollars, trans2 = compute_pred(m2)
    diff_pred_dollars = p2_dollars - p1_dollars
    
    st.divider()
    
    # Відображення карток з фінальними прогнозами
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric(f"🏡 Прогноз {m1['label']}", f"${p1_dollars:,.2f}", f"Test MSE: {m1['mse_te']:.4f}")
    res_c2.metric(f"🏡 Прогноз {m2['label']}", f"${p2_dollars:,.2f}", f"Test MSE: {m2['mse_te']:.4f}")
    res_c3.metric("Різниця між прогнозами (M2 − M1)", f"${diff_pred_dollars:+,.2f}", delta_color="off")
    
    st.divider()
    
    # Деталізація внеску коефіцієнтів
    st.subheader("🔍 Детальна підстановка параметрів у рівняння для обох моделей")
    
    bd_col1, bd_col2 = st.columns(2)
    with bd_col1:
        with st.expander(f"Деталізація для Моделі 1 ({m1['label']})", expanded=False):
            bd_rows1 = [{
                "Член рівняння": "Вільний член (w₀ / Intercept)",
                "Значення (X)": "—",
                "Коефіцієнт (w)": f"{m1['intercept']:.6f}",
                "Внесок у ціну ($100k)": f"{m1['intercept']:.6f}"
            }]
            for name, c, v in zip(m1['feat_names'], m1['coefs'], trans1):
                contrib = c * v
                bd_rows1.append({
                    "Член рівняння": name,
                    "Значення (X)": f"{v:.4f}" if isinstance(v, float) else f"{v}",
                    "Коефіцієнт (w)": f"{c:.6f}" if abs(c) < 0.001 else f"{c:.4f}",
                    "Внесок у ціну ($100k)": f"{contrib:+.4f}"
                })
            st.dataframe(pd.DataFrame(bd_rows1), use_container_width=True, hide_index=True)
            st.markdown(f"**Сума внесків:** `{p1_100k:.4f}` ($100k) × $100,000 = **${p1_dollars:,.2f}**")
            
    with bd_col2:
        with st.expander(f"Деталізація для Моделі 2 ({m2['label']})", expanded=False):
            bd_rows2 = [{
                "Член рівняння": "Вільний член (w₀ / Intercept)",
                "Значення (X)": "—",
                "Коефіцієнт (w)": f"{m2['intercept']:.6f}",
                "Внесок у ціну ($100k)": f"{m2['intercept']:.6f}"
            }]
            for name, c, v in zip(m2['feat_names'], m2['coefs'], trans2):
                contrib = c * v
                bd_rows2.append({
                    "Член рівняння": name,
                    "Значення (X)": f"{v:.4f}" if isinstance(v, float) else f"{v}",
                    "Коефіцієнт (w)": f"{c:.6f}" if abs(c) < 0.001 else f"{c:.4f}",
                    "Внесок у ціну ($100k)": f"{contrib:+.4f}"
                })
            st.dataframe(pd.DataFrame(bd_rows2), use_container_width=True, hide_index=True)
            st.markdown(f"**Сума внесків:** `{p2_100k:.4f}` ($100k) × $100,000 = **${p2_dollars:,.2f}**")

# =========================================================
# Вкладка 3: Теоретичне пояснення
# =========================================================
with tab_theory:
    st.header("🧠 Теоретичне обґрунтування: Чому Test MSE кращий за Train MSE та що таке Overfitting")
    
    st.markdown("""
    ### 1. Чому помилка на тестових даних ($MSE_{\text{Test}}$) краще оцінює роботу моделі на нових даних?
    
    * **Мета машинного навчання — генералізація (узагальнення):** Головна цінність будь-якої моделі полягає не в тому, щоб відтворити минулі дані, а в тому, щоб **точно передбачати вартість нових об'єктів нерухомості**, яких вона ніколи раніше не бачила.
    * **Ізоляція тестової вибірки:** Тестова вибірка ($X_{\text{test}}, y_{\text{test}}$) є математичною імітацією майбутніх невідомих даних. Модель не використовує її під час пошуку коефіцієнтів. Тому саме $MSE_{\text{Test}}$ є **неупередженою (unbiased) оцінкою реальної якості моделі**.
    """)
    
    st.markdown("""
    ### 2. Чому низька помилка на Train сама по собі не гарантує якості й може свідчити про перенавчання (Overfitting)?
    
    * **Ілюзія досконалості:** Зі збільшенням складності моделі (наприклад, перехід від лінійної до кубічної регресії зі 164 ознаками) кількість ступенів свободи стає надлишковою. Модель отримує здатність підігнати криву під кожну випадкову точку, флуктуацію або шум навчального набору даних.
    * **Пастка Train MSE:** Через це $MSE_{\text{Train}}$ монотонно спадає (наприклад, для кубічної регресії він становить мінімальні `0.3572` проти `0.5179` у лінійної). Якщо оцінювати модель лише за навчальними даними, кубічна модель виглядає найкращою.
    * **Катастрофа на тесті:** Оскільки випадковий шум тренувальних даних не повторюється на нових будинках, кубічна модель видає колосальні похибки ($MSE_{\text{Test}} = 4.6013$ — зростання у **12.9 разів**!).
    """)
    
    st.divider()
    
    # Інформаційні блоки порівняння Bias-Variance Tradeoff
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.success("""
        ##### 🟢 Ознаки збалансованої моделі (наприклад, d = 2):
        * $MSE_{\text{Train}}$ зменшився (враховано нелінійні фактори).
        * $MSE_{\text{Test}}$ досяг абсолютного мінімуму (`0.4643`).
        * Розрив $\Delta MSE = MSE_{\text{Test}} - MSE_{\text{Train}}$ мінімальний (`+0.0436`, лише +10.4%).
        * **Висновок:** Модель засвоїла справжній закон ціноутворення, а не випадковий шум.
        """)
        
    with t_col2:
        st.error("""
        ##### 🔴 Ознаки перенавчання / Overfitting (наприклад, d = 3):
        * $MSE_{\text{Train}}$ екстремально низький (`0.3572`, найменший серед усіх).
        * $MSE_{\text{Test}}$ різко вибухає (`4.6013`, у ~13 разів більше!).
        * Розрив $\Delta MSE$ величезний (`+4.2442`, зростання на +1188%).
        * $R^2_{\text{Test}} < 0$ (модель прогнозує гірше, ніж просте середнє арифметичне).
        * **Висновок:** «Зазубрювання» навчальних точок замість узагальнення.
        """)

# =========================================================
# Вкладка 4: Набір даних California Housing
# =========================================================
with tab_data:
    st.header("📋 Перегляд та опис даних California Housing")
    
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Кількість записів (рядків)", f"{df.shape[0]:,}")
    dc2.metric("Кількість ознак (колонок)", f"{df.shape[1] - 1}")
    dc3.metric("Цільова змінна", "Price ($100k)")
    
    st.subheader("🔍 Перші 10 рядків таблиці")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📈 Описова статистика ознак")
    st.dataframe(df.describe().T, use_container_width=True)
    
    with st.expander("ℹ️ Детальний опис колонок датасету"):
        st.markdown("""
        * **MedInc**: Середній дохід мешканців у районі (у сотнях тисяч доларів США).
        * **HouseAge**: Середній вік будинків у районі (роки).
        * **AveRooms**: Середня кількість кімнат на будинок.
        * **AveBedrms**: Середня кількість спалень на будинок.
        * **Population**: Населення кварталу / району.
        * **AveOccup**: Середня кількість мешканців на один будинок.
        * **Latitude**: Географічна широта.
        * **Longitude**: Географічна довгота.
        * **Price ($100k)**: Середня вартість житла у районі (цільова змінна $y$).
        """)
