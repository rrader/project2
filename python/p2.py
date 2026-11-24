import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# Налаштування сторінки Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Дерево Прийняття Рішень — Прогноз Контрольної",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# СТИЛІЗАЦІЯ ІНТЕРФЕЙСУ
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 0.3rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(240, 136, 62, 0.3);
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffaa00;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .predict-pass {
        background: rgba(46, 160, 67, 0.15);
        border: 2px solid #2ea043;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #3fb950;
    }
    .predict-fail {
        background: rgba(248, 81, 73, 0.15);
        border: 2px solid #f85149;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #f85149;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ПОЧАТКОВИЙ НАБІР ДАНИХ
# ---------------------------------------------------------
DEFAULT_DATA = [
    {"Години відпочинку": 8.0, "Години навчання": 6.0, "Здав": 1},
    {"Години відпочинку": 4.0, "Години навчання": 1.5, "Здав": 0},
    {"Години відпочинку": 9.0, "Години навчання": 2.0, "Здав": 0},
    {"Години відпочинку": 7.0, "Години навчання": 5.0, "Здав": 1},
    {"Години відпочинку": 3.0, "Години навчання": 7.0, "Здав": 1},
    {"Години відпочинку": 10.0, "Години навчання": 1.0, "Здав": 0},
    {"Години відпочинку": 6.0, "Години навчання": 4.5, "Здав": 1},
    {"Години відпочинку": 5.0, "Години навчання": 3.0, "Здав": 0},
    {"Години відпочинку": 8.5, "Години навчання": 4.0, "Здав": 1},
    {"Години відпочинку": 2.0, "Години навчання": 2.0, "Здав": 0},
    {"Години відпочинку": 7.5, "Години навчання": 8.0, "Здав": 1},
    {"Години відпочинку": 11.0, "Години навчання": 0.5, "Здав": 0},
    {"Години відпочинку": 5.5, "Години навчання": 6.5, "Здав": 1},
    {"Години відпочинку": 6.0, "Години навчання": 2.5, "Здав": 0},
]

if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame(DEFAULT_DATA)

# ---------------------------------------------------------
# БОКОВА ПАНЕЛЬ: НАЛАШТУВАННЯ МОДЕЛІ ТА КЕРУВАННЯ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Гіперпараметри Дерева")

criterion = st.sidebar.selectbox(
    "Критерій розщеплення (Criterion)",
    options=["gini", "entropy"],
    index=0,
    help="Gini (неоднорідність Джині) вимірює чистоту вузла. Entropy (інформаційна ентропія) базується на зменшенні ентропії (Information Gain)."
)

depth_option = st.sidebar.slider(
    "Максимальна глибина дерева (max_depth)",
    min_value=1,
    max_value=6,
    value=3,
    help="Обмеження глибини допомагає уникнути перенавчання (overfitting)."
)

min_samples_split = st.sidebar.slider(
    "Мін. зразків для поділу вузла (min_samples_split)",
    min_value=2,
    max_value=10,
    value=2,
    step=1
)

min_samples_leaf = st.sidebar.slider(
    "Мін. зразків у листі (min_samples_leaf)",
    min_value=1,
    max_value=5,
    value=1,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.header("💾 Керування даними")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 Скинути дані", use_container_width=True):
        st.session_state.dataset = pd.DataFrame(DEFAULT_DATA)
        st.rerun()

with col_sb2:
    if st.button("🗑️ Очистити", use_container_width=True):
        st.session_state.dataset = pd.DataFrame(columns=["Години відпочинку", "Години навчання", "Здав"])
        st.rerun()

st.sidebar.markdown("### 🎲 Згенерувати випадкові записи")
n_random = st.sidebar.number_input("Кількість нових рядків", min_value=1, max_value=50, value=5, step=1)
if st.sidebar.button("➕ Додати згенеровані рядки", use_container_width=True):
    rng = np.random.default_rng()
    rest = np.round(rng.uniform(2.0, 11.0, size=int(n_random)), 1)
    study = np.round(rng.uniform(0.5, 9.0, size=int(n_random)), 1)
    # Реалістична логіка: імовірність здати більша при більшому навчанні та адекватному відпочинку
    prob = 1 / (1 + np.exp(-(study * 1.3 + (rest - 4.0) * 0.4 - 5.5)))
    passed = (rng.uniform(0, 1, size=int(n_random)) < prob).astype(int)
    
    new_rows = pd.DataFrame({
        "Години відпочинку": rest,
        "Години навчання": study,
        "Здав": passed
    })
    st.session_state.dataset = pd.concat([st.session_state.dataset, new_rows], ignore_index=True)
    st.rerun()

# ---------------------------------------------------------
# ОСНОВНИЙ ВМІСТ
# ---------------------------------------------------------
st.markdown('<div class="main-header-title">🌳 Дерево Прийняття Рішень: Прогноз Здачі Контрольної Роботи</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'Інтерактивний додаток для збору даних, створення <strong>pandas DataFrame</strong> та навчання моделі '
    'класифікації <strong>DecisionTreeClassifier</strong> на основі годин відпочинку та навчання.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# ВКЛАДКИ ДОДАТКА
# ---------------------------------------------------------
tab_data, tab_model, tab_predict, tab_theory = st.tabs([
    "📝 1. Введення та Таблиця Даних",
    "🌳 2. Навчання та Візуалізація Дерева",
    "🎯 3. Онлайн-Симулятор Прогнозу",
    "💡 4. Теорія та Пояснення"
])

# =========================================================
# TAB 1: ВВЕДЕННЯ ДАНИХ ТА PANDAS DATAFRAME
# =========================================================
with tab_data:
    st.markdown("### 📋 Форма швидкого додавання запису")
    st.write("Введіть показники учня та натисніть **«Додати запис до таблиці»**, або редагуйте рядки безпосередньо в таблиці нижче.")

    with st.form("add_student_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1])
        with c1:
            input_rest = st.number_input(
                "😴 Години відпочинку (на добу)",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5,
                help="Скільки годин учень спав та відпочивав напередодні"
            )
        with c2:
            input_study = st.number_input(
                "📚 Години навчання (на добу)",
                min_value=0.0,
                max_value=24.0,
                value=4.5,
                step=0.5,
                help="Скільки годин учень присвятив підготовці"
            )
        with c3:
            input_result = st.selectbox(
                "🎯 Результат контрольної",
                options=[1, 0],
                format_func=lambda x: "✅ Здав (1)" if x == 1 else "❌ Не здав (0)",
                help="1 — здав контрольну роботу, 0 — не здав"
            )
        with c4:
            st.write("&nbsp;")
            submitted = st.form_submit_button("➕ Додати запис", use_container_width=True)

        if submitted:
            new_entry = pd.DataFrame([{
                "Години відпочинку": float(input_rest),
                "Години навчання": float(input_study),
                "Здав": int(input_result)
            }])
            st.session_state.dataset = pd.concat([st.session_state.dataset, new_entry], ignore_index=True)
            st.success(f"Запис додано: Відпочинок {input_rest} год, Навчання {input_study} год -> Результат: {'Здав' if input_result==1 else 'Не здав'}")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Інтерактивна таблиця (pandas DataFrame)")
    st.info(
        "💡 **Підказка:** Ви можете клікати двічі по комірках таблиці нижче, щоб редагувати їх. "
        "Також можна додавати нові рядки кнопкою **«+»** у нижній частині таблиці або видаляти вибрані рядки!"
    )

    # Редактор даних Streamlit Data Editor
    edited_df = st.data_editor(
        st.session_state.dataset,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Години відпочинку": st.column_config.NumberColumn(
                "Години відпочинку (Rest)",
                min_value=0.0,
                max_value=24.0,
                step=0.1,
                format="%.1f год"
            ),
            "Години навчання": st.column_config.NumberColumn(
                "Години навчання (Study)",
                min_value=0.0,
                max_value=24.0,
                step=0.1,
                format="%.1f год"
            ),
            "Здав": st.column_config.SelectboxColumn(
                "Результат (1=Здав, 0=Не здав)",
                options=[0, 1],
                required=True
            ),
        },
        key="data_editor_widget"
    )

    # Оновлення стану DataFrame, якщо були зміни
    if not edited_df.equals(st.session_state.dataset):
        st.session_state.dataset = edited_df.copy()

    # Показ статистики DataFrame
    st.markdown("#### 📈 Зведена статистика сформованого DataFrame")
    total_count = len(st.session_state.dataset)
    if total_count > 0 and "Здав" in st.session_state.dataset.columns:
        passed_count = int((st.session_state.dataset["Здав"] == 1).sum())
        failed_count = int((st.session_state.dataset["Здав"] == 0).sum())
        avg_rest = st.session_state.dataset["Години відпочинку"].mean()
        avg_study = st.session_state.dataset["Години навчання"].mean()

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_count}</div><div class="metric-label">Всього учнів</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #3fb950;">{passed_count}</div><div class="metric-label">Здали (1)</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #f85149;">{failed_count}</div><div class="metric-label">Не здали (0)</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_rest:.1f}г</div><div class="metric-label">Сер. відпочинок</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_study:.1f}г</div><div class="metric-label">Сер. навчання</div></div>', unsafe_allow_html=True)
    else:
        st.warning("Таблиця порожня. Будь ласка, додайте хоча б кілька рядків.")

    # Експорт у CSV
    st.markdown("---")
    csv_bytes = st.session_state.dataset.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Завантажити поточний DataFrame як CSV",
        data=csv_bytes,
        file_name="study_exam_data.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 2: НАВЧАННЯ ТА ВІЗУАЛІЗАЦІЯ ДЕРЕВА
# =========================================================
with tab_model:
    st.markdown("### 🌳 Навчання класифікаційної моделі DecisionTreeClassifier")
    
    # Перевірка придатності даних для класифікації
    df_clean = st.session_state.dataset.dropna()
    total_valid = len(df_clean)

    if total_valid < 3:
        st.error("⚠️ Для навчання дерева рішень потрібно щонайменше 3 записи. Будь ласка, додайте записи на першій вкладці.")
    elif df_clean["Здав"].nunique() < 2:
        st.warning(
            "⚠️ У ваших даних зараз присутній лише один клас (або тільки 'Здав', або тільки 'Не здав'). "
            "Для повноцінної класифікації додайте приклади обох результатів!"
        )
    else:
        # Підготовка ознак X та цільової змінної y
        X = df_clean[["Години відпочинку", "Години навчання"]].values
        y = df_clean["Здав"].astype(int).values
        feature_names = ["Години відпочинку", "Години навчання"]
        class_names = ["Не здав (0)", "Здав (1)"]

        # Ініціалізація та навчання моделі
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=depth_option,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)

        st.success(f"✅ Модель успішно навчена на {total_valid} записах! Точність (Accuracy) на поточному наборі: **{acc * 100:.1f}%**")

        # -----------------------------------------------------
        # Графік 1: Візуалізація структури Дерева Рішень
        # -----------------------------------------------------
        st.markdown("#### 1. Графічна структура Дерева Рішень")
        st.write(
            "Кожен вузол показує умову розщеплення, критерій "
            f"({'Gini' if criterion=='gini' else 'Entropy'}), кількість зразків та розподіл класів. "
            "Зелений колір відповідає перевазі класу «Здав», помаранчевий — «Не здав»."
        )

        col_theme, _ = st.columns([2, 1])
        with col_theme:
            tree_theme = st.radio(
                "🎨 Фон та стиль схеми дерева:",
                options=["⚪ Світлий фон (рекомендовано, чіткі темні стрілки)", "🌙 Темний фон (яскраві золоті стрілки)"],
                index=0,
                horizontal=True
            )

        is_light = "Світлий" in tree_theme
        fig_tree, ax_tree = plt.subplots(figsize=(12, 6.5), dpi=150)
        
        if is_light:
            fig_tree.patch.set_facecolor('#ffffff')
            ax_tree.set_facecolor('#ffffff')
            arrow_color = '#1e293b'
            node_text_color = '#0f172a'
        else:
            fig_tree.patch.set_facecolor('#161b22')
            ax_tree.set_facecolor('#161b22')
            arrow_color = '#ffaa00'
            node_text_color = '#111111'

        annotations = plot_tree(
            clf,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            impurity=True,
            proportion=False,
            precision=2,
            ax=ax_tree,
            fontsize=9
        )

        # Явне налаштування кольорів стрілок та тексту вузлів для ідеальної видимості
        for ann in annotations:
            ann.set_color(node_text_color)
            if ann.arrow_patch is not None:
                ann.arrow_patch.set_color(arrow_color)
                ann.arrow_patch.set_edgecolor(arrow_color)
                ann.arrow_patch.set_linewidth(2.2)
        
        st.pyplot(fig_tree)
        plt.close(fig_tree)

        # -----------------------------------------------------
        # Графік 2: 2D Границя Прийняття Рішень (Decision Boundary)
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("#### 2. Двовимірна Границя Прийняття Рішень (Decision Boundary)")
        st.write(
            "Цей графік демонструє, як дерево прийняття рішень розбиває 2D-простір ознак на прямокутні зони. "
            "Зелена зона — модель прогнозує «Здав», червона зона — модель прогнозує «Не здав»."
        )

        fig_boundary, ax_b = plt.subplots(figsize=(10, 6), dpi=130)
        fig_boundary.patch.set_facecolor('#0e1117')
        ax_b.set_facecolor('#161b22')

        # Визначаємо межі координат
        x_min, x_max = max(0, X[:, 0].min() - 1), X[:, 0].max() + 1.5
        y_min, y_max = max(0, X[:, 1].min() - 1), X[:, 1].max() + 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

        # Передбачення на сітці
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Відображення контурних зон
        ax_b.contourf(xx, yy, Z, alpha=0.35, levels=[-0.5, 0.5, 1.5], colors=['#f85149', '#2ea043'])
        ax_b.contour(xx, yy, Z, colors='#ffaa00', linewidths=1.8, linestyles='--')

        # Відображення точок учнів
        mask_passed = (y == 1)
        mask_failed = (y == 0)

        ax_b.scatter(
            X[mask_passed, 0], X[mask_passed, 1],
            c='#3fb950', edgecolors='#ffffff', s=100, linewidths=1.5,
            label='Фактично Здав (1)', zorder=5
        )
        ax_b.scatter(
            X[mask_failed, 0], X[mask_failed, 1],
            c='#f85149', edgecolors='#ffffff', s=100, linewidths=1.5,
            marker='s', label='Фактично Не здав (0)', zorder=5
        )

        ax_b.set_xlabel("Години відпочинку (Rest Hours)", color='#f0f6fc', fontsize=11, labelpad=10)
        ax_b.set_ylabel("Години навчання (Study Hours)", color='#f0f6fc', fontsize=11, labelpad=10)
        ax_b.set_title(
            f"Границя рішень DecisionTree (max_depth={depth_option}, {criterion})",
            color='#ffaa00', fontsize=13, fontweight='bold', pad=12
        )
        ax_b.tick_params(colors='#8b949e')
        ax_b.grid(color='#30363d', linestyle=':', alpha=0.6)
        ax_b.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')

        st.pyplot(fig_boundary)
        plt.close(fig_boundary)

        # -----------------------------------------------------
        # Блок 3: Важливість ознак та Текстові правила
        # -----------------------------------------------------
        st.markdown("---")
        col_rules, col_imp = st.columns([1.2, 1])

        with col_rules:
            st.markdown("#### 📜 Згенеровані правила у вигляді тексту (if-then)")
            rules_str = export_text(clf, feature_names=feature_names)
            st.code(rules_str, language="text")

        with col_imp:
            st.markdown("#### ⚖️ Важливість ознак (Feature Importance)")
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({
                "Ознака": feature_names,
                "Важливість": importances
            }).sort_values(by="Важливість", ascending=True)

            fig_imp, ax_imp = plt.subplots(figsize=(6, 3.5), dpi=120)
            fig_imp.patch.set_facecolor('#0e1117')
            ax_imp.set_facecolor('#161b22')
            bars = ax_imp.barh(imp_df["Ознака"], imp_df["Важливість"], color='#ffaa00', height=0.45)
            ax_imp.set_xlim(0, 1.05)
            ax_imp.set_xlabel("Частка важливості", color='#f0f6fc')
            ax_imp.tick_params(colors='#f0f6fc')
            ax_imp.grid(axis='x', color='#30363d', linestyle=':', alpha=0.5)

            for bar in bars:
                w = bar.get_width()
                ax_imp.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%",
                            va='center', color='#ffaa00', fontweight='bold', fontsize=9)

            st.pyplot(fig_imp)
            plt.close(fig_imp)

        # -----------------------------------------------------
        # Блок 4: Матриця невідповідностей (Confusion Matrix)
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("#### 🎯 Матриця класифікації (Confusion Matrix)")
        cm = confusion_matrix(y, y_pred)
        col_cm1, col_cm2 = st.columns([1, 1.5])
        with col_cm1:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5), dpi=120)
            fig_cm.patch.set_facecolor('#0e1117')
            ax_cm.set_facecolor('#161b22')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Не здав", "Здав"])
            disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
            ax_cm.tick_params(colors='#f0f6fc')
            ax_cm.set_title("Confusion Matrix", color='#f0f6fc')
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        with col_cm2:
            st.write(
                f"- **Правильно передбачено (True Predictions):** {(y == y_pred).sum()} з {len(y)} зразків ({acc*100:.1f}%)\n"
                f"- **Істинно негативні (TN - Не здав):** {cm[0,0]}\n"
                f"- **Хибно позитивні (FP - помилково Здав):** {cm[0,1]}\n"
                f"- **Хибно негативні (FN - помилково Не здав):** {cm[1,0]}\n"
                f"- **Істинно позитивні (TP - Здав):** {cm[1,1]}"
            )

# =========================================================
# TAB 3: ОНЛАЙН-СИМУЛЯТОР ПРОГНОЗУ
# =========================================================
with tab_predict:
    st.markdown("### 🎯 Інтерактивний калькулятор прогнозу для нового учня")
    st.write("Вкажіть кількість годин відпочинку та навчання, і навчене дерево рішень миттєво зробить прогноз!")

    # Перевірка наявності навченої моделі
    df_clean = st.session_state.dataset.dropna()
    if len(df_clean) < 3 or df_clean["Здав"].nunique() < 2:
        st.warning("⚠️ Для прогнозування спершу додайте достатньо даних обох класів у вкладці «1. Введення даних».")
    else:
        # Навчання актуальної моделі
        X_all = df_clean[["Години відпочинку", "Години навчання"]].values
        y_all = df_clean["Здав"].astype(int).values
        sim_clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=depth_option,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        sim_clf.fit(X_all, y_all)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            sim_rest = st.slider("😴 Ваші години відпочинку:", min_value=0.0, max_value=16.0, value=7.5, step=0.5)
        with col_p2:
            sim_study = st.slider("📚 Ваші години навчання:", min_value=0.0, max_value=16.0, value=5.0, step=0.5)

        sample = np.array([[sim_rest, sim_study]])
        pred_class = sim_clf.predict(sample)[0]
        pred_prob = sim_clf.predict_proba(sample)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        if pred_class == 1:
            st.markdown(
                f"""
                <div class="predict-pass">
                    <h2>🎉 ПРОГНОЗ: КОНТРОЛЬНУ БУДЕ ЗДАНО! (Клас 1)</h2>
                    <p style="font-size: 1.2rem; margin: 8px 0;">
                        Впевненість моделі: <strong>{pred_prob[1]*100:.1f}%</strong>
                    </p>
                    <p style="color: #8b949e; font-size: 0.95rem;">
                        Поєднання {sim_study} годин підготовки та {sim_rest} годин відпочинку потрапляє в зону успішного складання!
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="predict-fail">
                    <h2>⚠️ ПРОГНОЗ: КОНТРОЛЬНУ НЕ БУДЕ ЗДАНО (Клас 0)</h2>
                    <p style="font-size: 1.2rem; margin: 8px 0;">
                        Ймовірність не здати: <strong>{pred_prob[0]*100:.1f}%</strong>
                    </p>
                    <p style="color: #8b949e; font-size: 0.95rem;">
                        Увага! При {sim_study} год навчання та {sim_rest} год відпочинку модель прогнозує ризик невдачі. Рекомендується збільшити час підготовки або збалансувати режим сну!
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# TAB 4: ТЕОРІЯ ТА ПОЯСНЕННЯ
# =========================================================
with tab_theory:
    st.markdown("### 💡 Як працює Дерево Прийняття Рішень (Decision Tree)?")
    st.markdown("""
    **Дерево прийняття рішень (Decision Tree)** — це один із найпопулярніших та найбільш інтерпретованих алгоритмів машинного навчання для задач класифікації та регресії.

    #### 1. Як будується дерево?
    1. **Кореневий вузол (Root Node):** Алгоритм бере всю вибірку учнів і шукає таку ознаку (наприклад, `Години навчання <= 3.5`) та таке порогове значення, яке найкраще розділяє учнів на тих, хто склав, і тих, хто не склав.
    2. **Критерій розділення:**
       - **Gini Impurity (Неоднорідність Джині):** Показує ймовірність помилкової класифікації навмання обраного елемента. Дорівнює $0$, коли всі елементи у вузлі належать до одного класу (чистий вузол).
       - **Entropy (Ентропія):** Міра невизначеності або хаосу в даних. Алгоритм намагається максимізувати *Information Gain* (зменшення ентропії).
    3. **Листяні вузли (Leaf Nodes):** Кінцеві вузли, де більше немає розгалужень. Вони видають фінальний клас: **1 (Здав)** або **0 (Не здав)**.

    #### 2. Переваги Decision Tree для нашої задачі:
    - 👁️ **Повна прозорість (White-box model):** Ми точно знаємо, за яким правилом учень потрапив у категорію "Здав" чи "Не здав".
    - ⚡ **Швидкість роботи:** Працює миттєво навіть безпосередньо у браузері через WebAssembly / Pyodide.
    - 📐 **Геометрична наочність:** Границя прийняття рішень складається з ортогональних прямих ліній, що розділяють площину `Години відпочинку` × `Години навчання`.

    #### 3. Чому важливо обмежувати глибину (`max_depth`)?
    Якщо дозволити дереву рости без обмежень, воно побудує окреме правило під кожен рядок вашої таблиці — це явище називається **перенавчанням (overfitting)**. Таке дерево матиме 100% точність на старих даних, але даватиме збої на нових учнях.
    """)
