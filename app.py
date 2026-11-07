"""Демо-приложение: отбор «хороших» вин логистической регрессией (ДЗ3 по ММАД).

Модель внутри приложения не обучается: она загружается из model.joblib,
порог по умолчанию берётся из config.json, тестовая выборка — из test.csv.
Запуск: streamlit run app.py

Кэширование (задание 4.5)
-------------------------
* Модель загружается через st.cache_resource. Это «ресурс»: один объект на весь
  сервер, который все сессии используют совместно, без копирования и без
  сериализации. Модель не меняется во время работы, поэтому её достаточно
  прочитать с диска один раз.
* Таблицы (test.csv, вероятности на тесте, точки ROC/PR-кривых) кэшируются через
  st.cache_data. Это «данные»: Streamlit хранит результат в сериализованном виде
  и каждому вызову отдаёт свою копию, поэтому случайное изменение DataFrame
  в одной части кода или в одной сессии не испортит кэш для остальных.
  Для объектов вроде модели такое копирование бессмысленно и дорого, а для
  небольших таблиц безопасно.
* Если вызывать joblib.load без декоратора, Streamlit будет заново читать и
  распаковывать модель с диска при каждом действии пользователя (каждое
  движение слайдера перезапускает скрипт сверху вниз). Приложение станет
  медленнее, а результат не изменится.
* Вероятности на тесте от порога не зависят, поэтому считаются один раз в
  test_proba(_model, X). Подчёркивание в _model говорит Streamlit не хешировать
  модель при поиске значения в кэше. При движении слайдера меняется только
  сравнение вероятностей с порогом.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
TARGET = 'target'

FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь, %',
    'is_red': 'Красное вино',
}

st.set_page_config(page_title='Подбор хороших вин', page_icon='🍷', layout='wide')


# ------------------------------------------------------------------ загрузка
@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    """Вероятности класса 1 на тесте: считаются один раз, от порога не зависят."""
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def curves(y_true, proba):
    """Точки ROC- и PR-кривых и площади под ними (тоже не зависят от порога)."""
    fpr, tpr, _ = metrics.roc_curve(y_true, proba)
    prec, rec, _ = metrics.precision_recall_curve(y_true, proba)
    return {
        'fpr': fpr, 'tpr': tpr, 'precision': prec, 'recall': rec,
        'roc_auc': metrics.roc_auc_score(y_true, proba),
        'ap': metrics.average_precision_score(y_true, proba),
    }


model = load_model()
config = load_config()
test_df = load_test()
FEATURES = list(getattr(model, 'feature_names_in_', test_df.columns.drop(TARGET)))
X_test = test_df[FEATURES]
y_test = test_df[TARGET].to_numpy()
p_test = test_proba(model, X_test)


def threshold_stats(y_true, proba, t):
    pred = proba >= t
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, fpr=fpr)


# ------------------------------------------------------------------ сайдбар
default_threshold = float(config.get('threshold', 0.5))
if 'threshold' not in st.session_state:
    st.session_state.threshold = default_threshold

with st.sidebar:
    st.header('Порог решения')
    st.slider('Вино считается хорошим, если вероятность ≥ порога',
              min_value=0.0, max_value=1.0, step=0.001, format='%.3f',
              key='threshold')

    def _reset_threshold():
        st.session_state.threshold = default_threshold

    st.button('Вернуть порог из config.json', on_click=_reset_threshold)
    st.caption(f'Порог по умолчанию: {default_threshold:.3f} — максимальный precision '
               'при recall ≥ 0.7 на валидации.')

threshold = st.session_state.threshold

st.title('🍷 Подбор хороших вин для подписки')
st.write('Модель по результатам лабораторного анализа оценивает вероятность того, что вино '
         'хорошее (оценка экспертов не ниже 7), и решает, включать ли его в подборку.')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

# ------------------------------------------------------------------ 4.1
with tab_one:
    st.subheader('Характеристики вина')
    values = {}
    numeric = [f for f in FEATURES if f != 'is_red']
    cols = st.columns(3)
    for i, feat in enumerate(numeric):
        col = X_test[feat]
        lo, hi, med = float(col.min()), float(col.max()), float(col.median())
        step = float(10 ** np.floor(np.log10(max((hi - lo) / 1000, 1e-6))))
        with cols[i % 3]:
            values[feat] = st.number_input(
                FEATURE_LABELS.get(feat, feat), min_value=lo, max_value=hi, value=med,
                step=step, format='%.4f' if step < 0.01 else '%.3f',
                help=f'Диапазон в тестовых данных: {lo:g} – {hi:g}, медиана {med:g}')
    if 'is_red' in FEATURES:
        color = st.radio('Тип вина', ['Белое', 'Красное'], horizontal=True,
                         index=int(X_test['is_red'].median()))
        values['is_red'] = 1 if color == 'Красное' else 0

    one = pd.DataFrame([values])[FEATURES]
    proba_one = float(model.predict_proba(one)[0, 1])
    good = proba_one >= threshold

    c1, c2 = st.columns(2)
    c1.metric('Вероятность, что вино хорошее', f'{proba_one:.1%}')
    c2.metric('Порог', f'{threshold:.3f}')
    if good:
        st.success('Решение модели: **включить в подборку** — вино хорошее.')
    else:
        st.warning('Решение модели: **не включать** — вино не дотягивает до порога.')

# ------------------------------------------------------------------ 4.2
with tab_metrics:
    s = threshold_stats(y_test, p_test, threshold)
    cv = curves(y_test, p_test)

    st.subheader(f'Качество на тестовой выборке при пороге {threshold:.3f}')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Precision', f'{s["precision"]:.3f}', help='Доля хороших среди отобранных')
    m2.metric('Recall', f'{s["recall"]:.3f}', help='Доля найденных хороших вин')
    m3.metric('F1', f'{s["f1"]:.3f}')
    m4.metric('Вин в подборке', f'{s["tp"] + s["fp"]} из {len(y_test)}')

    left, mid, right = st.columns(3)
    with left:
        cm = pd.DataFrame([[s['tn'], s['fp']], [s['fn'], s['tp']]],
                          index=['факт: не хорошее', 'факт: хорошее'],
                          columns=['модель: не хорошее', 'модель: хорошее'])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm.values, cmap='Blues')
        for (i, j), v in np.ndenumerate(cm.values):
            ax.text(j, i, v, ha='center', va='center', fontsize=14,
                    color='white' if v > cm.values.max() / 2 else 'black')
        ax.set_xticks([0, 1], ['не хорошее', 'хорошее'])
        ax.set_yticks([0, 1], ['не хорошее', 'хорошее'])
        ax.set_xlabel('Решение модели')
        ax.set_ylabel('Факт')
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        plt.close(fig)
    with mid:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(cv['fpr'], cv['tpr'], label=f'ROC-AUC = {cv["roc_auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.scatter(s['fpr'], s['recall'], color='red', s=60, zorder=5, label='текущий порог')
        ax.set(xlabel='FPR', ylabel='TPR (recall)', title='ROC-кривая')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(cv['recall'], cv['precision'], label=f'PR-AUC = {cv["ap"]:.3f}')
        ax.axhline(y_test.mean(), color='k', ls='--', lw=1, label='случайная модель')
        ax.scatter(s['recall'], s['precision'], color='red', s=60, zorder=5, label='текущий порог')
        ax.set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        '**Как порог меняет подборку.** '
        '**Повысить порог** — модель станет строже: в подборку попадёт меньше вин, среди них '
        'будет больше действительно хороших (precision растёт), но часть хороших вин '
        'окажется за бортом (recall падает), подборка станет беднее. '
        '**Понизить порог** — подборка станет шире и пропустит меньше хороших вин '
        '(recall растёт), но в неё попадёт больше посредственных, и клиенты чаще будут '
        'разочарованы (precision падает).')

# ------------------------------------------------------------------ 4.3
with tab_batch:
    st.subheader('Оценка списка вин из CSV')
    st.write(f'Загрузите CSV с признаками вин (столбец `{TARGET}` не нужен). '
             f'Решение принимается по порогу из сайдбара: **{threshold:.3f}**.')
    st.caption('Нужные столбцы: ' + ', '.join(f'`{f}`' for f in FEATURES))
    st.download_button('Скачать пример файла (10 вин из теста)',
                       X_test.head(10).to_csv(index=False).encode('utf-8'),
                       file_name='wine_example.csv', mime='text/csv')

    uploaded = st.file_uploader('CSV-файл', type=['csv'])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:  # битый или пустой файл
            st.error(f'Не удалось прочитать CSV: {exc}')
            batch = None

        if batch is not None:
            missing = [f for f in FEATURES if f not in batch.columns]
            if batch.empty:
                st.error('Файл пустой: в нём нет ни одной строки с вином.')
            elif missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(f'`{m}`' for m in missing))
            else:
                X_batch = batch[FEATURES].apply(pd.to_numeric, errors='coerce')
                bad_rows = X_batch.isna().any(axis=1)
                if bad_rows.all():
                    st.error('Ни в одной строке нет полного набора числовых признаков.')
                else:
                    if bad_rows.any():
                        st.warning(f'Строк с пустыми или нечисловыми значениями: {int(bad_rows.sum())}. '
                                   'Для них предсказание не делается.')
                    result = batch.copy()
                    result['вероятность_хорошее'] = np.nan
                    result.loc[~bad_rows, 'вероятность_хорошее'] = model.predict_proba(
                        X_batch[~bad_rows])[:, 1]
                    result['решение'] = np.where(
                        bad_rows, 'нет данных',
                        np.where(result['вероятность_хорошее'] >= threshold,
                                 'в подборку', 'не включать'))
                    n_good = int((result['решение'] == 'в подборку').sum())
                    st.write(f'Вин в файле: {len(result)}, в подборку: {n_good}.')
                    st.dataframe(result)
                    st.download_button('Скачать результат (CSV)',
                                       result.to_csv(index=False).encode('utf-8'),
                                       file_name='wine_predictions.csv', mime='text/csv')

# ------------------------------------------------------------------ 4.4
with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')
    st.write('Признаки перед обучением стандартизованы, поэтому коэффициенты сравнимы между '
             'собой: это изменение логарифма шансов «вино хорошее» при росте признака на одно '
             'стандартное отклонение.')
    lr = model[-1]
    coef = pd.Series(lr.coef_.ravel(), index=FEATURES).sort_values()
    labels = [FEATURE_LABELS.get(f, f) for f in coef.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, coef.values, color=['tab:red' if v < 0 else 'tab:green' for v in coef.values])
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Коэффициент (после масштабирования)')
    ax.set_title('Зелёные «голосуют» за хорошее вино, красные — против')
    st.pyplot(fig)
    plt.close(fig)

    pos = coef[coef > 0].sort_values(ascending=False).head(3)
    neg = coef[coef < 0].head(3)
    st.markdown(
        '**Сильнее всего за хорошее вино:** '
        + ', '.join(f'{FEATURE_LABELS.get(f, f).lower()} ({v:+.2f})' for f, v in pos.items())
        + '.  \n**Сильнее всего против:** '
        + ', '.join(f'{FEATURE_LABELS.get(f, f).lower()} ({v:+.2f})' for f, v in neg.items())
        + '.')
    st.caption('Признаки связаны между собой (например, плотность зависит от сахара и алкоголя), '
               'поэтому знак и размер отдельного коэффициента нельзя толковать как причинный '
               'эффект: модель распределяет «вклад» между коррелирующими признаками.')
