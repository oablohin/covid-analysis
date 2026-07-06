# -*- coding: utf-8 -*-
"""Generator for covid_complications.ipynb — late post-COVID complications analysis.
Restructured per supervisor's directive into two stages:
  Stage 1: complications vs disease severity (CT groups) + main risk drivers.
  Stage 2: risk groups — how factors influence each specific complication (OR analysis),
           multivariable logistic for frequent outcomes, focus on stroke/MI, ML where defensible.
Reads updated_data.xlsx, sheet '0бщ.755'. Self-contained notebook.
"""
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})

# ---------------------------------------------------------------- 0. Title
md("""# Анализ поздних (постковидных) осложнений

Цель работы — оценить поздние осложнения COVID-19 в зависимости от тяжести перенесённого
заболевания и выделить **группы риска** по основным клиническим факторам.

Анализ построен в два этапа (по постановке научного руководителя):

**Этап 1. Осложнения в зависимости от тяжести.**
Частота каждого осложнения по группам КТ (КТ0 / КТ1 / КТ2 / КТ3–4), проверка тренда
«тяжесть → риск» и ранжирование основных признаков, влияющих на общий риск поздних осложнений.

**Этап 2. Группы риска.**
Как факторы — пол, возраст, ИМТ, HOMA, сопутствующие болезни, вакцинация, АД, температура,
бактериальные осложнения (вторичная инфекция), ОРДС, дыхательная недостаточность, ГКС (гормоны) —
влияли на конкретные осложнения: пневмофиброз, повторные пневмонии, ХОБЛ/бронхиальную астму,
ЖКТ/НЖБП, СД/эндокринные заболевания, ССЗ, операции на венах и, в первую очередь, **ОНМК/ИМ**.

> **Источник данных.** `updated_data.xlsx`, лист `0бщ.755` — общая выборка 755 пациентов.
> Поздние осложнения закодированы в столбцах `DZ..EO`.

Дополнительно: индивидуальный SHAP-разбор типичных пациентов (этап 3), панкреонекроз (этап 4),
**сравнение осложнений по геновариантам — Ухань / Дельта / Омикрон (этап 5)** и **клинические схемы
риска для ОНМК/ИМ, повторных пневмоний и тромбозов (этап 6)**. Помимо восьми осложнений рассмотрены
ещё два поздних исхода — нарушения зрения/слуха и психоневрологические/обонятельные нарушения.

**Методика.** Для каждого осложнения частота и отношения шансов (OR) с 95% доверительными
интервалами оценены логистической регрессией; тренд по тяжести — логистической регрессией на
порядковую группу КТ. Для частых исходов дополнительно построены многофакторные модели и
предсказательные ML-модели с кросс-валидацией. Для редких исходов (< ~50 событий) приводятся
только однофакторные оценки с оговоркой о малой выборке (ML статистически некорректен).
""")

# ---------------------------------------------------------------- 0. Imports
md("## 0.1 Подготовка окружения")
code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from IPython.display import display

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
RANDOM_STATE = 42
print('Окружение готово')
""")

# ---------------------------------------------------------------- 1. Load
md("""## 0.2 Загрузка данных и формирование переменных

Загружаем лист `0бщ.755`. Осложнения приводим к бинарному виду (0 — нет, 1 — есть; значения 2+
означают несколько эпизодов и тоже считаются «есть»). Группа тяжести КТ формируется из `КТ(ИТОГ)`
объединением степеней 3 и 4 (КТ3–4) ввиду малочисленности крайней группы.""")
code("""
FILE_PATH = 'updated_data.xlsx'
SHEET = '0бщ.755'
df = pd.read_excel(FILE_PATH, sheet_name=SHEET, header=0)
N = len(df)
print(f'Загружено: {N} пациентов, {df.shape[1]} столбцов')

def col(i):
    \"\"\"Числовой столбец по позиционному индексу.\"\"\"
    return pd.to_numeric(df.iloc[:, i], errors='coerce')

def binarize(i):
    \"\"\"Бинаризация столбца: >0 -> 1.\"\"\"
    return (col(i).fillna(0) > 0).astype(int)

# --- осложнения-исходы (idx -> подпись) ---
COMP_IDX = {
    142: 'Пневмофиброз',
    140: 'Повторные пневмонии',
    143: 'ХОБЛ / бр. астма',
    144: 'ЖКТ / НЖБП',
    134: 'СД / эндокринные',
    130: 'ССЗ (новые)',
    132: 'Операции на венах',
    141: 'ОНМК / ИМ',
    135: 'Зрение / слух',
    136: 'Психика / ЦНС / обоняние',
}
OVERALL_IDX = 129  # общий флаг поздних осложнений

comp = pd.DataFrame({name: binarize(i) for i, name in COMP_IDX.items()})
comp['Любое осложнение'] = binarize(OVERALL_IDX)

# --- группа тяжести по КТ(ИТОГ) ---
kt = col(38)
kt_grp = np.where(kt >= 3, 3, kt)
KT_LABELS = {0: 'КТ0', 1: 'КТ1', 2: 'КТ2', 3: 'КТ3-4'}
kt_grp = pd.Series(kt_grp, index=df.index).map(KT_LABELS)
kt_order = ['КТ0', 'КТ1', 'КТ2', 'КТ3-4']
print('Группы тяжести:', kt_grp.value_counts().reindex(kt_order).to_dict())
""")

# ---------------------------------------------------------------- factors
md("""### Факторы риска

Непрерывные факторы используются «как есть» (в OR-оценках стандартизуются — OR на 1 стандартное
отклонение), бинарные — 0/1. Пол кодируется как «женский = 1». Бактериальные осложнения и
дыхательную недостаточность бинаризуем (наличие / отсутствие). Дополнительно вводится суммарное
число сопутствующих заболеваний.""")
code("""
# спецификация факторов: подпись -> (индекс, тип)
#   cont   — непрерывный (в OR стандартизуется)
#   bin    — бинарный >0
#   bin_f  — пол: женский=1 (исходно 1/2)
FACTOR_SPEC = [
    ('Возраст',            2,   'cont'),
    ('Пол (жен.)',         3,   'bin_f'),
    ('ИМТ',                31,  'cont'),
    ('HOMA',               114, 'cont'),
    ('АД сист.',           34,  'cont'),
    ('АД диаст.',          35,  'cont'),
    ('Температура',        20,  'cont'),
    ('Вакцинация',         10,  'bin'),
    ('Бактер. осложн.',    119, 'bin'),
    ('ОРДС',               109, 'bin'),
    ('Дых. недост. (ДН)',  103, 'bin'),
    ('ГКС (гормоны)',      92,  'bin'),
    ('Сопут.: ССС',        82,  'bin'),
    ('Сопут.: орг. дых.',  83,  'bin'),
    ('Сопут.: СД',         84,  'bin'),
    ('Сопут.: ЖКТ',        85,  'bin'),
]
COMORB_IDX = [82, 83, 84, 85, 86, 87, 88, 89]

def make_factor(idx, kind):
    if kind == 'cont':
        s = col(idx)
        return s.fillna(s.median())
    if kind == 'bin_f':
        return (col(idx) == 2).astype(int)
    return binarize(idx)  # 'bin'

factors = pd.DataFrame({name: make_factor(idx, kind) for name, idx, kind in FACTOR_SPEC})
factors['Число сопутств.'] = pd.concat([binarize(i) for i in COMORB_IDX], axis=1).sum(axis=1)

# какие факторы непрерывные (для стандартизации в OR)
CONT = {name for name, _, kind in FACTOR_SPEC if kind == 'cont'} | {'Число сопутств.'}
FACTOR_NAMES = list(factors.columns)
print(f'Факторов риска: {len(FACTOR_NAMES)}')
display(factors.describe().T[['mean', 'std', 'min', 'max']])
""")

# ---------------------------------------------------------------- helpers stats
md("### Вспомогательные функции (OR, тренд)")
code("""
def or_univariate(y, x, continuous):
    \"\"\"Однофакторное OR (95% ДИ, p) логистической регрессией.
    continuous=True -> стандартизация, OR на 1 SD.\"\"\"
    xv = (x - x.mean()) / x.std() if continuous else x.astype(float)
    X = sm.add_constant(pd.DataFrame({'x': xv}))
    try:
        r = sm.Logit(y, X).fit(disp=0)
        ci = np.exp(r.conf_int()).iloc[1]
        return np.exp(r.params.iloc[1]), ci.iloc[0], ci.iloc[1], r.pvalues.iloc[1]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

def trend_test(y, kt_numeric):
    \"\"\"Тренд «тяжесть -> риск»: OR на одну ступень КТ + p.\"\"\"
    X = sm.add_constant(pd.DataFrame({'kt': kt_numeric.astype(float)}))
    try:
        r = sm.Logit(y, X).fit(disp=0)
        return np.exp(r.params.iloc[1]), r.pvalues.iloc[1]
    except Exception:
        return np.nan, np.nan

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

print('OK')
""")

# ---------------------------------------------------------------- sample description
md("## 0.3 Описание выборки осложнений")
code("""
prev = pd.DataFrame({
    'Случаев': comp.sum().astype(int),
    'Частота, %': (100 * comp.mean()).round(1),
}).sort_values('Случаев', ascending=False)
display(prev)

fig, ax = plt.subplots(figsize=(8, 4.5))
order = prev.index.tolist()
sns.barplot(x=prev['Частота, %'], y=order, hue=order, palette='viridis', legend=False, ax=ax)
for i, v in enumerate(prev['Частота, %']):
    ax.text(v + 0.4, i, f'{v:.1f}%', va='center', fontsize=9)
ax.set_xlabel('Частота, %'); ax.set_ylabel('')
ax.set_title('Распространённость поздних осложнений (N = %d)' % N)
plt.tight_layout(); plt.show()
""")

# ============================================================ STAGE 1
md("""# ЭТАП 1. Осложнения в зависимости от тяжести (КТ)

Оцениваем, как частота каждого осложнения меняется по группам тяжести лёгочного поражения,
и проверяем статистическую значимость тренда.""")

md("## 1.1 Частота осложнений по группам КТ")
code("""
by_kt = comp.drop(columns=['Любое осложнение']).groupby(kt_grp).mean().reindex(kt_order) * 100
grp_sizes = kt_grp.value_counts().reindex(kt_order)

tbl = by_kt.T.round(1)
tbl.columns = [f'{c} (n={int(grp_sizes[c])})' for c in tbl.columns]
print('Частота осложнения, % по группам тяжести КТ:')
display(tbl)
""")
code("""
fig, ax = plt.subplots(figsize=(11, 5))
by_kt.T.plot(kind='bar', ax=ax, colormap='YlOrRd', width=0.8)
ax.set_ylabel('Частота, %'); ax.set_xlabel('')
ax.set_title('Частота поздних осложнений по тяжести КТ')
ax.legend(title='Группа КТ', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.show()
""")

md("## 1.2 Тест тренда «тяжесть → риск»")
code("""
kt_num = pd.Series(np.where(kt >= 3, 3, kt), index=df.index)
rows = []
for name in COMP_IDX.values():
    orr, p = trend_test(comp[name], kt_num)
    rows.append((name, orr, p, stars(p)))
trend = pd.DataFrame(rows, columns=['Осложнение', 'OR на ступень КТ', 'p', 'знач.'])
trend = trend.sort_values('OR на ступень КТ', ascending=False).reset_index(drop=True)
display(trend)

fig, ax = plt.subplots(figsize=(8, 4.5))
t = trend.sort_values('OR на ступень КТ')
colors = ['#b2182b' if p < 0.05 else '#bdbdbd' for p in t['p']]
ax.barh(t['Осложнение'], t['OR на ступень КТ'], color=colors)
ax.axvline(1, color='k', lw=0.8, ls='--')
for y_, (orr, st) in enumerate(zip(t['OR на ступень КТ'], t['знач.'])):
    ax.text(orr + 0.05, y_, f'{orr:.2f}{st}', va='center', fontsize=9)
ax.set_xlabel('OR на одну ступень тяжести КТ')
ax.set_title('Сила связи тяжести и риска осложнения')
plt.tight_layout(); plt.show()
""")
md("""Все осложнения демонстрируют статистически значимый рост частоты с увеличением тяжести КТ
(p < 0.001). Наиболее «зависимы от тяжести» пневмофиброз, ОНМК/ИМ, зрение/слух, ХОБЛ/астма и операции на венах.""")

md("""## 1.3 Основные признаки, влияющие на риск поздних осложнений

За исход берём общий флаг «любое позднее осложнение» (≈47% пациентов). Сначала однофакторные OR,
затем многофакторная логистическая модель (скорректированные OR) и, для устойчивости, важность
признаков по случайному лесу с кросс-валидацией.""")
code("""
y_any = comp['Любое осложнение']

# --- однофакторные OR ---
rows = []
for name in FACTOR_NAMES:
    orr, lo, hi, p = or_univariate(y_any, factors[name], name in CONT)
    rows.append((name, orr, lo, hi, p, stars(p)))
uni_any = pd.DataFrame(rows, columns=['Фактор', 'OR', 'ДИ low', 'ДИ high', 'p', 'знач.'])
uni_any['ед.'] = ['на 1 SD' if n in CONT else '0/1' for n in uni_any['Фактор']]
uni_any = uni_any.sort_values('p').reset_index(drop=True)
print('Однофакторные OR факторов риска для «любого позднего осложнения»:')
display(uni_any)
""")
code("""
# --- многофакторная логистическая модель (скорректированные OR) ---
Xm = factors.copy()
for c in CONT:
    Xm[c] = (Xm[c] - Xm[c].mean()) / Xm[c].std()
Xm = sm.add_constant(Xm)
mod_any = sm.Logit(y_any, Xm).fit(disp=0)
adj = pd.DataFrame({
    'OR (скорр.)': np.exp(mod_any.params),
    'ДИ low': np.exp(mod_any.conf_int()[0]),
    'ДИ high': np.exp(mod_any.conf_int()[1]),
    'p': mod_any.pvalues,
}).drop(index='const')
adj['знач.'] = adj['p'].map(stars)
adj = adj.sort_values('p')
print('Многофакторная модель: скорректированные OR (псевдо-R2 = %.3f)' % mod_any.prsquared)
display(adj)

# forest-plot скорректированных OR
fig, ax = plt.subplots(figsize=(8, 6))
a = adj.iloc[::-1]
ypos = np.arange(len(a))
ax.errorbar(a['OR (скорр.)'], ypos,
            xerr=[a['OR (скорр.)'] - a['ДИ low'], a['ДИ high'] - a['OR (скорр.)']],
            fmt='o', color='#2166ac', ecolor='#92c5de', capsize=3)
ax.axvline(1, color='k', lw=0.8, ls='--')
ax.set_yticks(ypos); ax.set_yticklabels(a.index)
ax.set_xscale('log'); ax.set_xlabel('OR (лог. шкала)')
ax.set_title('Скорректированные OR: «любое позднее осложнение»')
plt.tight_layout(); plt.show()
""")
code("""
# --- важность по случайному лесу + CV ROC-AUC (устойчивость ранжирования) ---
rf = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                            random_state=RANDOM_STATE, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
auc = cross_val_score(rf, factors.values, y_any, cv=cv, scoring='roc_auc')
print(f'RandomForest, CV ROC-AUC = {auc.mean():.3f} ± {auc.std():.3f}')

rf.fit(factors.values, y_any)
imp = pd.Series(rf.feature_importances_, index=FACTOR_NAMES).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
imp.plot(kind='barh', color='#1a9850', ax=ax)
ax.set_xlabel('Важность признака (Gini)')
ax.set_title('Важность факторов для риска любого осложнения (RandomForest)')
plt.tight_layout(); plt.show()
""")

# ============================================================ STAGE 2
md("""# ЭТАП 2. Группы риска: влияние факторов на конкретные осложнения

Для каждого осложнения оцениваем, как связаны с ним факторы риска. Базовый инструмент —
отношение шансов (OR): для непрерывных факторов на 1 SD, для бинарных — наличие/отсутствие.""")

md("""## 2.1 Матрица отношений шансов (фактор × осложнение)

Тепловая карта `log2(OR)`: красный — фактор повышает риск, синий — снижает. Звёздочкой отмечены
значимые связи (p < 0.05).""")
code("""
comp_names = list(COMP_IDX.values())
or_mat = pd.DataFrame(index=FACTOR_NAMES, columns=comp_names, dtype=float)
p_mat = pd.DataFrame(index=FACTOR_NAMES, columns=comp_names, dtype=float)
for cname in comp_names:
    for fname in FACTOR_NAMES:
        orr, lo, hi, p = or_univariate(comp[cname], factors[fname], fname in CONT)
        or_mat.loc[fname, cname] = orr
        p_mat.loc[fname, cname] = p

log2or = np.log2(or_mat.astype(float)).clip(-3, 3)
annot = or_mat.astype(float).round(2).astype(str)
annot = annot.where(p_mat >= 0.05, annot + '*')

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(log2or, annot=annot, fmt='', cmap='RdBu_r', center=0,
            linewidths=0.5, cbar_kws={'label': 'log2(OR)'}, ax=ax,
            annot_kws={'fontsize': 8})
ax.set_title('OR факторов риска по осложнениям (* p < 0.05)')
plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
plt.tight_layout(); plt.show()
""")
code("""
# наиболее сильные значимые связи (по всей матрице)
pairs = []
for cname in comp_names:
    for fname in FACTOR_NAMES:
        orr, p = or_mat.loc[fname, cname], p_mat.loc[fname, cname]
        if p < 0.05 and np.isfinite(orr):
            pairs.append((fname, cname, orr, p))
top = pd.DataFrame(pairs, columns=['Фактор', 'Осложнение', 'OR', 'p'])
top['|эффект|'] = np.abs(np.log2(top['OR']))
top = top.sort_values('|эффект|', ascending=False).drop(columns='|эффект|').head(20).reset_index(drop=True)
print('Топ-20 самых сильных значимых связей фактор → осложнение:')
display(top)
""")

md("""## 2.2 Многофакторная логистическая регрессия (частые исходы)

Для осложнений с достаточным числом случаев (повторные пневмонии, ЖКТ/НЖБП, ССЗ, зрение/слух) строим
многофакторные модели — скорректированные OR показывают независимый вклад каждого фактора.

> **Как читать скорректированные OR.** В многофакторной модели знак эффекта может отличаться от
> однофакторного: `OR < 1` означает, что *при прочих равных* фактор связан с **понижением** шансов.
> Такой разворот знака — следствие взаимной корреляции факторов (отрицательное конфаундинг) либо
> особенностей определения исхода, и не противоречит однофакторным оценкам, а уточняет их.""")
code("""
FREQUENT = ['Повторные пневмонии', 'ЖКТ / НЖБП', 'ССЗ (новые)', 'Зрение / слух']

def multivariable(cname):
    X = factors.copy()
    for c in CONT:
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = sm.add_constant(X)
    m = sm.Logit(comp[cname], X).fit(disp=0)
    out = pd.DataFrame({
        'OR (скорр.)': np.exp(m.params),
        'ДИ low': np.exp(m.conf_int()[0]),
        'ДИ high': np.exp(m.conf_int()[1]),
        'p': m.pvalues,
    }).drop(index='const')
    out['знач.'] = out['p'].map(stars)
    return out, m.prsquared

for cname in FREQUENT:
    res, pr2 = multivariable(cname)
    sig = res[res['p'] < 0.05].sort_values('p')
    print(f'\\n=== {cname}  (случаев={int(comp[cname].sum())}, псевдо-R2={pr2:.3f}) ===')
    print('Значимые независимые предикторы:')
    display(sig if len(sig) else res.sort_values('p').head(5))
""")
code("""
# forest-plot скорректированных OR для частых исходов
fig, axes = plt.subplots(1, len(FREQUENT), figsize=(15, 5.5), sharey=True)
for ax, cname in zip(axes, FREQUENT):
    res, _ = multivariable(cname)
    res = res.sort_values('OR (скорр.)')
    ypos = np.arange(len(res))
    sig = (res['p'] < 0.05).values
    ax.errorbar(res['OR (скорр.)'], ypos,
                xerr=[res['OR (скорр.)'] - res['ДИ low'], res['ДИ high'] - res['OR (скорр.)']],
                fmt='o', color='#999999', ecolor='#cccccc', capsize=2, zorder=1)
    ax.scatter(res['OR (скорр.)'].values[sig], ypos[sig], color='#b2182b', zorder=3, s=40)
    ax.axvline(1, color='k', lw=0.8, ls='--')
    ax.set_yticks(ypos); ax.set_yticklabels(res.index, fontsize=8)
    ax.set_xscale('log'); ax.set_xlabel('OR (лог.)')
    ax.set_title(cname, fontsize=10)
fig.suptitle('Скорректированные OR (красным — p < 0.05)', y=1.02)
plt.tight_layout(); plt.show()
""")

md("""## 2.3 Фокус: ОНМК / ИМ

Самое тяжёлое позднее осложнение (26 случаев, 3.4%). Ввиду редкости приводим профиль группы риска
(сравнение факторов у пациентов с осложнением и без) и однофакторные OR; многофакторные оценки
интерпретируем осторожно.""")
code("""
y_stroke = comp['ОНМК / ИМ']

# профиль группы риска: cases vs non-cases
prof = []
for name in FACTOR_NAMES:
    x = factors[name]
    if name in CONT:
        m1, m0 = x[y_stroke == 1].mean(), x[y_stroke == 0].mean()
        p = stats.mannwhitneyu(x[y_stroke == 1], x[y_stroke == 0]).pvalue
        prof.append((name, f'{m1:.1f}', f'{m0:.1f}', p, stars(p)))
    else:
        r1 = 100 * x[y_stroke == 1].mean(); r0 = 100 * x[y_stroke == 0].mean()
        tab = pd.crosstab(x, y_stroke)
        p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else np.nan
        prof.append((name, f'{r1:.0f}%', f'{r0:.0f}%', p, stars(p)))
prof = pd.DataFrame(prof, columns=['Фактор', 'ОНМК/ИМ (есть)', 'ОНМК/ИМ (нет)', 'p', 'знач.'])
prof = prof.sort_values('p').reset_index(drop=True)
print('Профиль группы риска ОНМК/ИМ (среднее для непрерывных, доля для бинарных):')
display(prof)
""")
code("""
# однофакторные OR для ОНМК/ИМ
rows = []
for name in FACTOR_NAMES:
    orr, lo, hi, p = or_univariate(y_stroke, factors[name], name in CONT)
    rows.append((name, orr, lo, hi, p, stars(p)))
uni_s = pd.DataFrame(rows, columns=['Фактор', 'OR', 'ДИ low', 'ДИ high', 'p', 'знач.'])
uni_s = uni_s[np.isfinite(uni_s['OR'])].sort_values('OR', ascending=False).reset_index(drop=True)
display(uni_s)

fig, ax = plt.subplots(figsize=(8, 6))
a = uni_s.sort_values('OR')
ypos = np.arange(len(a))
sig = (a['p'] < 0.05).values
ax.errorbar(a['OR'], ypos, xerr=[a['OR'] - a['ДИ low'], a['ДИ high'] - a['OR']],
            fmt='o', color='#999999', ecolor='#cccccc', capsize=2)
ax.scatter(a['OR'].values[sig], ypos[sig], color='#b2182b', zorder=3, s=40)
ax.axvline(1, color='k', lw=0.8, ls='--')
ax.set_yticks(ypos); ax.set_yticklabels(a['Фактор'], fontsize=8)
ax.set_xscale('log'); ax.set_xlabel('OR (лог. шкала)')
ax.set_title('Однофакторные OR факторов риска ОНМК/ИМ')
plt.tight_layout(); plt.show()
""")

md("""## 2.4 Редкие осложнения — однофакторные OR

Пневмофиброз, ХОБЛ/астма, СД/эндокринные, операции на венах — < ~50 событий. Приводим только
значимые однофакторные связи; многофакторные ML-модели для них статистически не обоснованы.""")
code("""
RARE = ['Пневмофиброз', 'ХОБЛ / бр. астма', 'СД / эндокринные', 'Операции на венах']
for cname in RARE:
    rows = []
    for name in FACTOR_NAMES:
        orr, lo, hi, p = or_univariate(comp[cname], factors[name], name in CONT)
        rows.append((name, orr, lo, hi, p))
    t = pd.DataFrame(rows, columns=['Фактор', 'OR', 'ДИ low', 'ДИ high', 'p'])
    t = t[(t['p'] < 0.05) & np.isfinite(t['OR'])].sort_values('OR', ascending=False)
    print(f'\\n=== {cname}  (случаев = {int(comp[cname].sum())}) — значимые факторы ===')
    display(t.reset_index(drop=True) if len(t) else 'значимых однофакторных связей не выявлено')
""")

md("""## 2.5 Предсказательные модели (только частые исходы)

Для исходов с достаточным числом случаев оцениваем предсказуемость по факторам риска: логистическая
регрессия и случайный лес, метрика — ROC-AUC по 5-блочной стратифицированной кросс-валидации
(честная оценка без переобучения; единичный train/test-сплит для редких событий неустойчив).""")
code("""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models = {
    'Логит. регрессия': make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)),
    'Случайный лес': RandomForestClassifier(n_estimators=400, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1),
}
rows = []
for cname in ['Любое осложнение'] + FREQUENT:
    y = comp[cname]
    row = {'Осложнение': cname, 'Случаев': int(y.sum())}
    for mname, mdl in models.items():
        s = cross_val_score(mdl, factors.values, y, cv=cv, scoring='roc_auc')
        row[mname] = f'{s.mean():.3f} ± {s.std():.3f}'
    rows.append(row)
display(pd.DataFrame(rows).set_index('Осложнение'))
""")

# ============================================================ 2.6 risk groups
md("""## 2.6 Поиск групп риска для каждого осложнения

Группы риска находим неглубоким деревом решений (глубина 2): оно разбивает пациентов на
подгруппы по сочетанию клинико-лабораторных признаков и выделяет подгруппу с максимальной частотой
осложнения. В предикторы добавлены **лабораторные показатели** (СРБ, ферритин, Д-димер, ЛДГ, ИЛ-6,
лейкоциты, СОЭ, глюкоза, креатинин, АЛТ, фибриноген), а также возраст, ИМТ, HOMA, SpO2, тяжесть КТ
и факторы течения (ОРДС, ДН, ГКС, бактериальные осложнения, вакцинация, пол).

Для каждой подгруппы приводится истинная частота осложнения, число пациентов и **lift** —
во сколько раз риск в подгруппе выше базового по всей выборке.

> **Оговорки.** (1) Частоты в подгруппах — **in-sample** (на тех же данных, без кросс-валидации),
> поэтому это *описание* выявленных групп риска, а не валидированный прогноз; реальная частота в новой
> выборке будет несколько ниже. (2) Здесь набор предикторов **шире**, чем в OR-анализе этапов 1.3–2.5:
> добавлены лабораторные показатели (ферритин, ЛДГ, Д-димер, ИЛ-6, СОЭ и др.), но не вынесены отдельно
> сопутствующие заболевания. Поэтому ведущие признаки тут (лабораторные) и в OR-матрице (течение и
> коморбидность) дополняют друг друга, а не противоречат.""")
code("""
from sklearn.tree import DecisionTreeClassifier, _tree

# --- панель лабораторных и клинических показателей (исходные/пиковые, без поздних анализов) ---
LAB_SPEC = [
    ('Возраст', 2), ('ИМТ', 31), ('SpO2 min', 33), ('КТ макс %', 39),
    ('Лейкоциты', 45), ('СОЭ', 50), ('СРБ', 74), ('Ферритин', 75), ('Д-димер', 76),
    ('ЛДГ', 73), ('АЛТ', 64), ('Креатинин', 68), ('Глюкоза макс', 66),
    ('HOMA', 114), ('ИЛ-6', 110), ('Фибриноген', 79),
]
labs = pd.DataFrame({n: col(i).fillna(col(i).median()) for n, i in LAB_SPEC})

# бинарные факторы течения для деревьев групп риска
BIN_SPEC = [('ОРДС', 109), ('Дых.недост.', 103), ('ГКС', 92),
            ('Бактер.осл.', 119), ('Вакцинация', 10), ('Пол жен.', 3)]
binf = {n: ((col(i) == 2).astype(int) if n == 'Пол жен.' else binarize(i)) for n, i in BIN_SPEC}
RG = pd.concat([labs, pd.DataFrame(binf)], axis=1)
BINARY = set(binf)

def _fmt(name, op, thr):
    if name in BINARY:
        return f"{name}: {'да' if op == '>' else 'нет'}"
    return f"{name} {'>' if op == '>' else '≤'} {thr:.0f}"

def _leaf_paths(tree, names):
    t = tree.tree_; paths = {}
    def rec(node, conds):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            nm = names[t.feature[node]]; thr = t.threshold[node]
            rec(t.children_left[node], conds + [(nm, '<=', thr)])
            rec(t.children_right[node], conds + [(nm, '>', thr)])
        else:
            paths[node] = conds
    rec(0, [])
    return paths

def risk_groups(y, depth=2, leaf=25):
    \"\"\"Подгруппы риска (правило, n, событий, истинная частота) по дереву решений.\"\"\"
    tr = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                random_state=RANDOM_STATE, class_weight='balanced').fit(RG, y)
    paths = _leaf_paths(tr, list(RG.columns))
    leafid = tr.apply(RG.values)
    rows = []
    for nd, conds in paths.items():
        m = leafid == nd
        rows.append((' и '.join(_fmt(*c) for c in conds), int(m.sum()),
                     int(y[m].sum()), float(y[m].mean())))
    return sorted(rows, key=lambda r: -r[3])

print('Панель показателей и функция групп риска готовы')
""")
code("""
# --- группа максимального риска для каждого осложнения ---
rows = []
for cname in COMP_IDX.values():
    y = comp[cname]; base = y.mean()
    rule, n, pos, rate = risk_groups(y)[0]
    rows.append({'Осложнение': cname, 'Группа риска (правило)': rule,
                 'N в группе': n, 'Событий': pos,
                 'Риск в группе, %': round(100 * rate, 1),
                 'Базовый риск, %': round(100 * base, 1),
                 'Lift': round(rate / base, 1)})
groups_tbl = pd.DataFrame(rows).set_index('Осложнение')
print('Подгруппа максимального риска по каждому осложнению:')
display(groups_tbl)
""")
code("""
# --- детализация: все подгруппы для ключевых осложнений ---
KEY = ['ОНМК / ИМ', 'ССЗ (новые)', 'ЖКТ / НЖБП', 'Повторные пневмонии']
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for ax, cname in zip(axes.ravel(), KEY):
    y = comp[cname]; base = y.mean()
    g = pd.DataFrame(risk_groups(y), columns=['Правило', 'N', 'Событий', 'Частота'])
    g = g.sort_values('Частота')
    colors = ['#b2182b' if r > base else '#9ecae1' for r in g['Частота']]
    ax.barh(range(len(g)), g['Частота'] * 100, color=colors)
    ax.axvline(base * 100, color='k', ls='--', lw=1, label=f'база {base*100:.1f}%')
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{r}\\n(n={n}, событий={e})" for r, n, e in
                        zip(g['Правило'], g['N'], g['Событий'])], fontsize=8)
    for i, (rate, n) in enumerate(zip(g['Частота'], g['N'])):
        ax.text(rate * 100 + 0.5, i, f'{rate*100:.0f}%', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Частота осложнения в подгруппе, %')
    ax.set_title(cname, fontsize=11); ax.legend(loc='lower right', fontsize=8)
plt.tight_layout(); plt.show()
""")
md("""Дерево решений выделяет клинически осмысленные группы риска: например, для **ОНМК/ИМ** ключевое
сочетание — высокий ферритин и инсулинорезистентность (HOMA), для **ЖКТ/НЖБП** — избыточная масса
тела и приём ГКС, для **повторных пневмоний** — инсулинорезистентность на фоне высокого ИЛ-6. Это
переводит статистические OR в практические критерии «кого относить к группе риска».""")

# ============================================================ STAGE 3 examples
md("""# ЭТАП 3. Клинико-лабораторный разбор конкретных примеров (SHAP)

Для каждого осложнения берём **репрезентативного пациента** с этим осложнением (ближайшего к центру
группы заболевших в пространстве лабораторных показателей) и объясняем предсказание модели методом
**SHAP** — так же, как в waterfall-разборе раздела 11.3 расширенного ноутбука.

Как читать диаграмму: внизу `E[f(x)]` — базовый (средний) риск осложнения по выборке, вверху
`f(x)` — итоговый риск для данного пациента. Каждая **стрелка** показывает вклад одного показателя:
**красная (вправо) повышает** риск, **синяя (влево) понижает**; слева подписано фактическое значение
показателя у пациента (напр. `СРБ = 142`). Сумма всех стрелок переводит базовый риск в итоговый.

> **Оговорка.** Для редких осложнений (ОНМК/ИМ, ХОБЛ/астма, СД, пневмофиброз — < ~50 событий)
> модель здесь обучается *в целях объяснения одного случая*, а не как валидированный прогноз
> (согласуется с принципом этапа 2.5). Waterfall показывает логику модели на конкретном пациенте,
> но величины вкладов для редких исходов следует трактовать качественно.""")
code("""
import shap

# робастный z-score лабораторных показателей — для выбора типичного пациента
iqr = (labs.quantile(0.75) - labs.quantile(0.25)).replace(0, 1.0)
zlab = (labs - labs.median()) / iqr

def representative_case(y):
    \"\"\"Индекс наиболее типичного заболевшего (минимум расстояния до центра группы).\"\"\"
    pos = np.where(y.values == 1)[0]
    centroid = zlab.iloc[pos].mean()
    dist = ((zlab.iloc[pos] - centroid) ** 2).sum(axis=1)
    return dist.idxmin()

def patient_shap(rf, explainer, ridx):
    \"\"\"SHAP-вклады по положительному классу для одного пациента (устойчиво к форме вывода).\"\"\"
    X_one = RG.loc[[ridx]]
    sv = explainer.shap_values(X_one, check_additivity=False)
    arr = np.array(sv)
    nf = RG.shape[1]
    if arr.ndim == 3:
        vals = arr[0, :, 1] if arr.shape[1] == nf else arr[1, 0, :]
    elif isinstance(sv, list):
        vals = sv[1][0]
    else:
        vals = arr[0]
    ev = explainer.expected_value
    base = float(ev[1]) if hasattr(ev, '__len__') else float(ev)
    return np.asarray(vals, dtype=float), base

print('Готово к построению SHAP-разборов')
""")
code("""
# waterfall-разбор репрезентативного пациента для каждого осложнения
comp_names = list(COMP_IDX.values())
for cname in comp_names:
    y = comp[cname]
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=-1).fit(RG, y)
    explainer = shap.TreeExplainer(rf)
    ridx = representative_case(y)
    vals, base = patient_shap(rf, explainer, ridx)

    age = labs.loc[ridx, 'Возраст']
    sex = 'жен' if (col(3) == 2).loc[ridx] else 'муж'
    kt_v = int(col(38).loc[ridx])

    explanation = shap.Explanation(values=vals, base_values=base,
                                   data=RG.loc[ridx].values, feature_names=list(RG.columns))
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f'{cname}: пациент №{ridx} ({sex}, {age:.0f} лет, КТ{kt_v})', fontsize=12)
    plt.tight_layout(); plt.show()
""")
md("""Waterfall-разбор показывает, какие именно показатели и в какую сторону «толкают» риск осложнения
у конкретного пациента, с их фактическими значениями. Видно, что у осложнений с
воспалительно-метаболическим механизмом (ОНМК/ИМ, ССЗ, ЖКТ/НЖБП) риск повышают высокие СРБ,
ферритин, ИЛ-6, глюкоза и HOMA, тогда как у фиброза и ХОБЛ/астмы — прежде всего тяжесть лёгочного
поражения (КТ, SpO2).""")

# ============================================================ STAGE 4 pancreonecrosis
md("""# ЭТАП 4. Прогнозирование панкреонекроза

Панкреонекроз — отдельно запрошенное осложнение. Оцениваем возможность его прогноза по имеющимся
данным.""")
code("""
PANCREO_IDX = 137
y_pn = binarize(PANCREO_IDX)
print(f'Панкреонекроз: {int(y_pn.sum())} случаев из {N} ({100*y_pn.mean():.1f}%)')

kt_num_pn = pd.Series(np.where(kt >= 3, 3, kt), index=df.index)
by_kt_pn = pd.DataFrame({
    'Всего': kt_num_pn.map(KT_LABELS).value_counts().reindex(kt_order),
    'Панкреонекроз': y_pn.groupby(kt_num_pn.map(KT_LABELS)).sum().reindex(kt_order).astype(int),
})
by_kt_pn['Частота, %'] = (100 * by_kt_pn['Панкреонекроз'] / by_kt_pn['Всего']).round(1)
print('\\nРаспределение по тяжести КТ:')
display(by_kt_pn)
""")
md("""**Вывод о возможности прогноза.** В выборке всего **8 случаев** панкреонекроза. По правилу
«не менее 10 событий на один предиктор» обучение и валидация многопризнаковой прогнозной модели
статистически некорректны — любая такая модель переобучится и не будет воспроизводимой. К тому же
в данных нет панкреас-специфичных маркеров (амилаза, липаза). Поэтому вместо «модели-прогноза»
приводим **описательный разбор**: однофакторные ассоциации и клинико-лабораторные профили всех 8
пациентов.""")
code("""
# однофакторные ассоциации (осторожно: широкие ДИ из-за 8 событий)
rows = []
for name in FACTOR_NAMES:
    orr, lo, hi, p = or_univariate(y_pn, factors[name], name in CONT)
    rows.append((name, orr, lo, hi, p))
pn_uni = pd.DataFrame(rows, columns=['Фактор', 'OR', 'ДИ low', 'ДИ high', 'p'])
pn_uni = pn_uni[np.isfinite(pn_uni['OR'])].sort_values('p').head(8).reset_index(drop=True)
print('Факторы, наиболее ассоциированные с панкреонекрозом (разведочно):')
display(pn_uni)
""")
code("""
# клинико-лабораторные профили всех пациентов с панкреонекрозом
pn_rows = np.where(y_pn.values == 1)[0]
profile = labs.loc[pn_rows].copy()
profile.insert(0, 'Пол', np.where((col(3) == 2).loc[pn_rows], 'жен', 'муж'))
profile.insert(1, 'КТ', col(38).loc[pn_rows].astype(int).values)
display(profile.T)

# сравнение средних: панкреонекроз vs остальные
cmp_tbl = pd.DataFrame({
    'Панкреонекроз (n=8)': labs.loc[pn_rows].mean().round(1),
    'Остальные': labs.loc[y_pn.values == 0].mean().round(1),
    'Медиана когорты': labs.median().round(1),
})
print('Средние значения показателей:')
display(cmp_tbl)
""")
md("""Панкреонекроз встречается преимущественно при тяжёлом течении, но и при лёгком (3 из 8 — КТ0):
группа малочисленна и неоднородна. Сильных воспроизводимых предикторов на таком числе наблюдений
выделить нельзя; для построения прогнозной модели потребуется накопление случаев и добавление
панкреас-специфичных лабораторных маркеров.""")

# ============================================================ STAGE 5 genovariant
md("""# ЭТАП 5. Осложнения в зависимости от геноварианта (Ухань / Дельта / Омикрон)

Сравниваем поздние осложнения между геновариантами вируса. Выборка собрана из отдельных листов
`УХАНЬ` (213), `ДЕЛЬТА` (272), `ОМИКРОН` (270) — это надёжнее, чем делить по году (год ≠ штамм).
Референс для сравнений — **Ухань**.

> **Важно — конфаундинг.** Геноварианты сильно различаются по тяжести (медиана КТ 1 / 1 / 0),
> возрасту (38 / 41 / 36) и охвату вакцинацией (0.5% / 30% / 89%). Поэтому **сырые** различия частот
> осложнений нельзя приписывать самому штамму — нужна поправка на тяжесть и возраст.""")
code("""
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency

STRAINS = ['УХАНЬ', 'ДЕЛЬТА', 'ОМИКРОН']
G = pd.concat([pd.read_excel(FILE_PATH, sheet_name=s, header=0).assign(strain=s)
               for s in STRAINS], ignore_index=True)
G['strain'] = pd.Categorical(G['strain'], categories=STRAINS, ordered=False)

def gcol(i):
    return pd.to_numeric(G.iloc[:, i], errors='coerce')
def gbin(i):
    return (gcol(i).fillna(0) > 0).astype(int)

profile = pd.DataFrame({
    'N': G['strain'].value_counts().reindex(STRAINS),
    'КТ(ИТОГ), медиана': gcol(38).groupby(G['strain']).median().reindex(STRAINS),
    'Возраст, медиана': gcol(2).groupby(G['strain']).median().reindex(STRAINS),
    'Вакцинация, %': (gbin(10).groupby(G['strain']).mean() * 100).reindex(STRAINS).round(1),
})
print('Профиль геновариантов:')
display(profile)
""")

md("## 5.1 Распространённость осложнений по геновариантам")
code("""
flags = pd.DataFrame({nm: gbin(i) for i, nm in COMP_IDX.items()})
flags['strain'] = G['strain'].values
prev_strain = (flags.groupby('strain').mean() * 100).reindex(STRAINS).round(1)
print('Частота осложнения, % по геновариантам:')
display(prev_strain.T)

fig, ax = plt.subplots(figsize=(12, 6))
prev_strain.T.plot(kind='bar', ax=ax, color=['#4575b4', '#fdae61', '#d73027'])
ax.set_ylabel('Частота, %'); ax.set_xlabel('')
ax.set_title('Частота поздних осложнений по геновариантам')
ax.legend(title='Геновариант')
plt.xticks(rotation=35, ha='right'); plt.tight_layout(); plt.show()
""")

md("""## 5.2 Различия частоты между штаммами (χ² + поправка на множественность)

Критерий χ² для каждого осложнения через три штамма; p корректируется методом Бенджамини–Хохберга
(FDR) на семейство тестов.""")
code("""
chi_targets = {OVERALL_IDX: 'Любое осложнение'}
chi_targets.update(COMP_IDX)
chi_rows = []
for i, nm in chi_targets.items():
    ct = pd.crosstab(G['strain'], gbin(i))
    chi2, p, dof, exp = chi2_contingency(ct)
    chi_rows.append([nm, round(chi2, 2), p, round(exp.min(), 1)])
chi = pd.DataFrame(chi_rows, columns=['Осложнение', 'хи²', 'p', 'min ожид.'])
chi['p (FDR)'] = multipletests(chi['p'], method='fdr_bh')[1]
chi['вывод'] = np.where(chi['p (FDR)'] < 0.05, 'значимо', 'поисково')
chi = chi.sort_values('p').reset_index(drop=True)
display(chi.round({'p': 4, 'p (FDR)': 4}))
""")
md("""Минимальные ожидаемые частоты во всех таблицах ≥ 5 (даже для ОНМК/ИМ — 7.3), поэтому χ²-приближение
корректно; ограничение — малое **число событий** в редких осложнениях (низкая мощность). После
FDR-поправки порог различий между штаммами выдерживает только **зрение/слух**; психоневрологические
осложнения (p≈0.02), пневмофиброз (p≈0.03) и ССЗ (p≈0.05) — поисковые.""")

md("""## 5.3 Сравнение с поправкой на тяжесть и возраст

Логистическая модель `осложнение ~ C(штамм) + КТ(ИТОГ) + возраст` (референс — Ухань). **Вакцинация
исключена из основной модели**: она почти коллинеарна штамму (0.5/30/89%), и её включение делает
эффект штамма неразделимым. Для зрения/слуха дополнительно показана модель чувствительности
с вакцинацией.""")
code("""
adf = pd.DataFrame({'strain': G['strain'].values, 'KT': gcol(38).values,
                    'age': gcol(2).values, 'vac': gbin(10).values})
adj_targets = {OVERALL_IDX: 'Любое осложнение', 140: 'Повторные пневмонии',
               144: 'ЖКТ / НЖБП', 130: 'ССЗ (новые)', 135: 'Зрение / слух'}
rows = []
for i, nm in adj_targets.items():
    adf['y'] = gbin(i).values
    m = smf.logit('y ~ C(strain) + KT + age', data=adf).fit(disp=0)
    for lvl in ['ДЕЛЬТА', 'ОМИКРОН']:
        k = 'C(strain)[T.' + lvl + ']'
        ci = np.exp(m.conf_int().loc[k])
        rows.append([nm, int(adf['y'].sum()), lvl, np.exp(m.params[k]),
                     ci.iloc[0], ci.iloc[1], m.pvalues[k]])
adj = pd.DataFrame(rows, columns=['Осложнение', 'событий', 'vs Ухань', 'OR', 'ДИ low', 'ДИ high', 'p'])
print('Скорректированные OR геноварианта (vs Ухань):')
display(adj.round(3))

fig, ax = plt.subplots(figsize=(8.5, 6.5))
ypos = 0; yt = []; yl = []
for nm in adj_targets.values():
    for lvl, c in [('ДЕЛЬТА', '#4575b4'), ('ОМИКРОН', '#d73027')]:
        r = adj[(adj['Осложнение'] == nm) & (adj['vs Ухань'] == lvl)].iloc[0]
        ax.errorbar(r['OR'], ypos, xerr=[[r['OR'] - r['ДИ low']], [r['ДИ high'] - r['OR']]],
                    fmt='o', color=c, capsize=3)
        yt.append(ypos); yl.append(nm + ' — ' + lvl); ypos += 1
    ypos += 0.6
ax.axvline(1, color='k', ls='--', lw=0.8)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8)
ax.set_xscale('log'); ax.set_xlabel('Скорр. OR vs Ухань (лог. шкала)')
ax.set_title('Эффект геноварианта с поправкой на тяжесть и возраст')
plt.tight_layout(); plt.show()

adf['y'] = gbin(135).values
ms = smf.logit('y ~ C(strain) + KT + age + vac', data=adf).fit(disp=0)
ko = 'C(strain)[T.ОМИКРОН]'
print('Чувствительность (зрение/слух, +вакцинация): Омикрон vs Ухань OR=%.2f (p=%.3f); '
      'вакцинация OR=%.2f (p=%.3f) -> эффект Омикрона устойчив.'
      % (np.exp(ms.params[ko]), ms.pvalues[ko], np.exp(ms.params['vac']), ms.pvalues['vac']))
""")
md("""После поправки на тяжесть и возраст **геновариант сам по себе не повышает** общий риск поздних
осложнений (Дельта и Омикрон vs Ухань — незначимо). Риск определяют тяжесть (КТ, OR≈2) и возраст
(≈1.07/год). Устойчивое исключение — **снижение нарушений зрения/слуха при Омикроне** (скорр. OR≈0.30):
этот сигнал значим и в χ² с FDR-поправкой (этап 5.2), и после поправки на тяжесть/возраст (этап 5.3 —
здесь p приведены без поправки на множественность). Оценка опирается на 24 случая в омикроновском плече
(в таблице 5.3 столбец «событий» = 147 — это объединённый по трём штаммам счёт) и неотделима от
вакцинации/календарного периода. Сигнал «больше новых ССЗ при Омикроне» (OR≈1.8, p≈0.03) — поисковый,
FDR-поправку не проходит.""")

md("""## 5.4 Предикторы осложнений внутри каждого штамма (поисковая карта)

Однофакторные OR общего риска отдельно в каждом штамме. Непрерывные факторы — OR **на 1 SD внутри
штамма** (разные абсолютные шаги, между штаммами в абсолюте не сравнимы), бинарные — наличие/отсутствие.""")
code("""
CONT_F = {'Возраст': 2, 'ИМТ': 31, 'HOMA': 114, '%КТ макс': 39, 'ДН (0-3)': 103}
BIN_F = {'ОРДС': 109, 'ГКС': 92, 'Вакцинация': 10, 'Бактер. осложн.': 119}

def strain_or(y, x, continuous):
    # возвращает (OR, аннотация); помечает неоценимые/неустойчивые ячейки
    if not continuous:
        nexp = int((x == 1).sum())
        if nexp < 5 or nexp > len(x) - 5:
            return np.nan, 'н/о'
        mincell = pd.crosstab(x, y).values.min()
    xv = (x - x.mean()) / x.std() if continuous else x.astype(float)
    try:
        r = sm.Logit(y, sm.add_constant(pd.DataFrame({'x': xv}))).fit(disp=0)
        orr = float(np.exp(r.params.iloc[1]))
    except Exception:
        return np.nan, 'н/о'
    if not continuous and mincell < 5:
        return orr, ('↑↑ неуст.' if orr > 1 else '↓↓ неуст.')
    return orr, ('%.2f' % orr)

rows_idx = list(CONT_F) + list(BIN_F)
or_strain = pd.DataFrame(index=rows_idx, columns=STRAINS, dtype=float)
ann_strain = pd.DataFrame(index=rows_idx, columns=STRAINS, dtype=object)
for s in STRAINS:
    sub = G[G['strain'] == s]
    y = (pd.to_numeric(sub.iloc[:, OVERALL_IDX], errors='coerce').fillna(0) > 0).astype(int)
    for nm, idx in CONT_F.items():
        x = pd.to_numeric(sub.iloc[:, idx], errors='coerce'); x = x.fillna(x.median())
        o, a = strain_or(y, x, True); or_strain.loc[nm, s] = o; ann_strain.loc[nm, s] = a
    for nm, idx in BIN_F.items():
        x = (pd.to_numeric(sub.iloc[:, idx], errors='coerce').fillna(0) > 0).astype(int)
        o, a = strain_or(y, x, False); or_strain.loc[nm, s] = o; ann_strain.loc[nm, s] = a

fig, ax = plt.subplots(figsize=(7.5, 7))
sns.heatmap(np.log2(or_strain.astype(float)).clip(-3, 3), annot=ann_strain.values, fmt='',
            cmap='RdBu_r', center=0, linewidths=0.5, cbar_kws={'label': 'log2(OR)'},
            annot_kws={'fontsize': 9}, ax=ax)
ax.set_title('Однофакторные OR общего риска внутри штамма (н/о — не оценимо)')
plt.yticks(rotation=0); plt.tight_layout(); plt.show()
""")
md("""Карта **поисковая** (без поправок и множественной коррекции). Основные предикторы (возраст, ИМТ,
HOMA, бактериальные осложнения) работают во всех штаммах. OR вакцинации в Ухане не оценим (вакцинирован
1 из 213 — разделение); OR для ОРДС внутри штаммов помечены как неустойчивые (ОРДС-позитивных всего
25 / 36 / 13; формальный критерий — минимальная клетка таблицы 2×2 < 5). Ослабление КТ/ДН/ГКС при
Омикроне частично отражает сужение диапазона тяжести
(медиана КТ = 0), а не обязательно иную биологию.""")

# ============================================================ STAGE 6 risk schemes
md("""# ЭТАП 6. Клинические схемы риска поздних осложнений

Компактные «схемы» (≤ 5–10 признаков) для трёх осложнений, каждая в двух читаемых формах:
**Форма A — дерево решений** (видны пороги-ветвления и риск в листьях) и **Форма B — балльная шкала**
(порог по каждому признаку и баллы по вкладу в логистическую модель).

> **Поисковый характер.** 3 цели × панель из ~8 признаков, отбор признаков по |β|, пороги по Юдену на
> той же выборке, банды по квантилям — это перебор на n=755, **завышающий видимую эффективность** всех
> трёх схем (особенно ОНМК/ИМ, где всего 26 событий). Поэтому рядом с in-sample-метрикой всегда
> приводится кросс-валидированный AUC. Схемы — **ориентировочные**, требуют внешней проверки.""")
code("""
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, roc_curve

def _f(i):
    s = col(i); return s.fillna(s.median())

PANEL = pd.DataFrame({
    'Возраст': _f(2),
    'Пол (жен.)': (col(3) == 2).astype(int),
    'ИМТ': _f(31),
    'HOMA': _f(114),
    'ТГ': _f(69),
    'Холестерин': col(70).clip(upper=15).fillna(col(70).median()),
    'Глюкоза макс': _f(66),
    '%КТ макс': _f(39),
    'Д-димер': _f(76),
    'СРБ': _f(74),
    'Ферритин': _f(75),
})
SCHEME_TARGETS = {
    'ОНМК / ИМ': (141, ['Возраст', 'Пол (жен.)', 'ИМТ', 'Глюкоза макс', '%КТ макс']),
    'Повторные пневмонии': (140, ['Возраст', 'Пол (жен.)', 'ИМТ', 'HOMA', 'ТГ', 'Холестерин', 'Глюкоза макс', '%КТ макс']),
    'Тромбозы / операции на венах': (132, ['Возраст', 'Пол (жен.)', 'ИМТ', 'Д-димер', '%КТ макс', 'СРБ', 'Глюкоза макс']),
}

def _youden(y, x, binary):
    if binary:
        return 0.5
    fpr, tpr, thr = roc_curve(y, x)
    return thr[np.argmax(tpr - fpr)]

def build_scorecard(y, Xp):
    cuts = {}; Xb = pd.DataFrame(index=Xp.index)
    for c in Xp.columns:
        binary = (c == 'Пол (жен.)'); cut = _youden(y, Xp[c], binary); cuts[c] = cut
        Xb[c] = Xp[c] if binary else (Xp[c] > cut).astype(int)
    r = sm.Logit(y, sm.add_constant(Xb)).fit(disp=0)
    betas = r.params.drop('const')
    kept = betas[betas.abs() >= 0.10]          # отбрасываем незначимые до расчёта делителя
    div = kept[kept > 0].min()
    pts = {c: max(0, int(round(kept[c] / div))) for c in kept.index}  # защитные признаки -> 0 баллов
    score = sum(Xb[c] * pts[c] for c in pts)
    return cuts, pts, score

def scheme_cv_auc(y, Xp):
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    al, at = [], []
    for tr, te in skf.split(Xp, y):
        ytr, yte = y.iloc[tr], y.iloc[te]
        Xtr = pd.DataFrame(); Xte = pd.DataFrame()
        for c in Xp.columns:
            binary = (c == 'Пол (жен.)'); cut = _youden(ytr, Xp[c].iloc[tr], binary)
            Xtr[c] = Xp[c].iloc[tr].values if binary else (Xp[c].iloc[tr].values > cut).astype(int)
            Xte[c] = Xp[c].iloc[te].values if binary else (Xp[c].iloc[te].values > cut).astype(int)
        lm = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr)
        al.append(roc_auc_score(yte, lm.predict_proba(Xte)[:, 1]))
        tm = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight='balanced',
                                    random_state=RANDOM_STATE).fit(Xp.iloc[tr], ytr)
        at.append(roc_auc_score(yte, tm.predict_proba(Xp.iloc[te])[:, 1]))
    return (np.mean(al), np.std(al)), (np.mean(at), np.std(at))

SCHEMES = {}

def render_scheme(nm):
    idx, feats = SCHEME_TARGETS[nm]
    y = (col(idx).fillna(0) > 0).astype(int)
    Xp = PANEL[feats]
    n_ev = int(y.sum())
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight='balanced',
                                  random_state=RANDOM_STATE).fit(Xp, y)
    cuts, pts, score = build_scorecard(y, Xp)
    (cvl, cvls), (cvt, cvts) = scheme_cv_auc(y, Xp)
    app = roc_auc_score(y, score)
    SCHEMES[nm] = {'n_ev': n_ev, 'app': app, 'cvl': cvl, 'cvls': cvls, 'cvt': cvt, 'cvts': cvts}

    def thr_label(c):
        return 'женский пол' if c == 'Пол (жен.)' else '> %.1f' % cuts[c]
    card = pd.DataFrame({'Порог': [thr_label(c) for c in pts], 'Баллы': [pts[c] for c in pts]},
                        index=list(pts))
    print('=== %s: событий=%d, EPV=%.1f ===' % (nm, n_ev, n_ev / len(feats)))
    print('AUC: баллы (in-sample)=%.3f | логит CV=%.3f±%.3f | дерево CV=%.3f±%.3f'
          % (app, cvl, cvls, cvt, cvts))
    display(card)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_axes([0.0, 0.0, 0.66, 1.0]); ax2 = fig.add_axes([0.74, 0.12, 0.24, 0.78])
    plot_tree(tree, feature_names=feats, filled=True, proportion=True,
              class_names=['нет', 'да'], rounded=True, fontsize=8, ax=ax1)
    ax1.set_title('Форма A — дерево решений: %s' % nm, fontsize=11)
    band = pd.qcut(score, 4, duplicates='drop')
    rb = pd.DataFrame({'band': band, 'y': y.values}).groupby('band', observed=True)['y'].agg(['mean', 'size'])
    rb['risk'] = rb['mean'] * 100
    ax2.bar(range(len(rb)), rb['risk'], color='#c0392b')
    ax2.set_xticks(range(len(rb)))
    ax2.set_xticklabels([str(b) for b in rb.index], rotation=30, ha='right', fontsize=7)
    for i, (rk, sz) in enumerate(zip(rb['risk'], rb['size'])):
        ax2.text(i, rk + 0.4, '%.0f%%\\nn=%d' % (rk, sz), ha='center', fontsize=7)
    ax2.set_ylabel('Риск, %'); ax2.set_title('Форма B — риск по сумме баллов', fontsize=10)
    plt.show()

print('Схемы готовы к построению; панель из %d признаков' % PANEL.shape[1])
""")

md("""## 6.1 ОНМК / ИМ

Всего **26 событий** (EPV ≈ 5) — схема **ориентировочная, гипотезо-порождающая**. Корневой признак
(возраст) воспроизводим; глубокие листья и AUC между фолдами нестабильны.""")
code("render_scheme('ОНМК / ИМ')")

md("""## 6.2 Повторные пневмонии

171 событие; ведущий профиль — **метаболический** (HOMA, возраст, ИМТ). Холестерин и %КТ исключены
из шкалы как незначимые (|β| < 0.10).""")
code("render_scheme('Повторные пневмонии')")

md("""## 6.3 Тромбозы / операции на венах

Позднее осложнение (`вены/опер`, idx 132, блок `DZ..EO`; 52 случая) — поздний венозный исход
(тромбозы / операции на венах). Это **прогностическая** схема: предикторы (Д-димер, СРБ, %КТ, ИМТ и др.)
измерены в исходном/остром периоде, **до** позднего исхода — утечки нет. Ведущий признак —
**Д-димер** (маркер тромбообразования).""")
code("render_scheme('Тромбозы / операции на венах')")

md("""## 6.4 Валидация схем — внутривыборочный против кросс-валидированного AUC

Кросс-валидация (5-блочная стратифицированная) с **переотбором порогов Юдена внутри каждого
обучающего фолда** (честная оценка). Балльная шкала с целочисленными весами — приближение
логистической модели, её AUC приводится отдельно.

> Примечание: для редких исходов (ОНМК/ИМ, тромбозы) часть квантильных границ суммы баллов совпадает,
> и банды риска схлопываются до трёх — это не пропущенная группа, а малое число различающихся сумм баллов.""")
code("""
auc_tbl = pd.DataFrame({
    'Событий': {nm: SCHEMES[nm]['n_ev'] for nm in SCHEMES},
    'AUC баллов (in-sample)': {nm: round(SCHEMES[nm]['app'], 3) for nm in SCHEMES},
    'Логит CV-AUC': {nm: '%.3f ± %.3f' % (SCHEMES[nm]['cvl'], SCHEMES[nm]['cvls']) for nm in SCHEMES},
    'Дерево CV-AUC': {nm: '%.3f ± %.3f' % (SCHEMES[nm]['cvt'], SCHEMES[nm]['cvts']) for nm in SCHEMES},
})
display(auc_tbl)
""")
md("""**Итог по схемам.** Тромбозы / операции на венах (позднее осложнение, 52 случая) — CV-AUC ≈0.82 (логит),
ведущий предиктор Д-димер. Повторные пневмонии — умеренная схема (CV-AUC ≈0.67), метаболический профиль. ОНМК/ИМ —
ориентировочная (CV-AUC ≈0.80±0.07 у логистической, ≈0.77±0.09 у дерева, 26 событий). Все схемы валидированы
**только внутри выборки** (n=755), без поправки на штамм/эру и без внешней валидации; «нулевой риск»
в нижних бандах — это верхняя граница ~0.6–0.8% (правило трёх), а не гарантированное отсутствие риска.""")

# ============================================================ conclusions
md("""# Выводы

**Этап 1 — тяжесть.** Частота всех поздних осложнений статистически значимо растёт с тяжестью
лёгочного поражения (тренд по КТ, p < 0.001 для каждого исхода). Сильнее всего от тяжести зависят
**пневмофиброз** (КТ0 0.7% → КТ3–4 ≈30%), **ОНМК/ИМ** (1% → 12.5%), **зрение/слух**, **ХОБЛ/астма**
и **операции на венах**. ЖКТ/НЖБП и повторные пневмонии — самые частые исходы (25% и 23%), также нарастающие
с тяжестью.

**Основные признаки риска (любое осложнение).** По **однофакторным** OR сильнее всего с риском
связаны маркеры тяжёлого течения (ОРДС, ДН, бактериальные осложнения) и метаболический профиль.
В **многофакторной** модели независимый вклад сохраняют **возраст, HOMA, ГКС и ИМТ** — тогда как
ОРДС и ДН после поправки на них уже незначимы (их эффект объясняется корреляцией с этими факторами).
Значимого эффекта вакцинации на риск поздних осложнений не выявлено (OR≈1, p≈0.8).

**Этап 2 — группы риска (по OR-матрице и моделям):**

- **ОНМК/ИМ** — наиболее значимые факторы: **ОРДС** (резкое повышение шансов), пожилой **возраст**,
  высокий **HOMA** и сопутствующие ССЗ; группа максимального риска — пожилые пациенты с тяжёлым
  течением и метаболическими нарушениями.
- **ССЗ (новые)** — независимые факторы: **возраст, ИМТ, HOMA**. Преморбидные ССС-заболевания
  связаны с «новыми ССЗ» **обратно** (8% против 18%, adjusted OR≈0.03) — по-видимому, в силу
  определения исхода (впервые возникшие ССЗ у пациентов без них в анамнезе), поэтому как фактор
  риска их трактовать нельзя.
- **ЖКТ / НЖБП** — ведущий независимый фактор **применение ГКС** (OR≈5.6), далее избыточная масса
  тела (ИМТ) и HOMA.
- **Повторные пневмонии** — по многофакторной модели значимы возраст, ГКС и бактериальные осложнения;
  ДН после поправки имеет **обратный** знак (OR≈0.37) и фактором риска не является.
- **Пневмофиброз / ХОБЛ-астма** — по однофакторным OR прежде всего тяжесть течения (ОРДС, ДН) и КТ.

**Группы риска (этап 2.6).** Деревья решений выделяют конкретные подгруппы с многократно повышенным
риском, например: ОНМК/ИМ — высокий ферритин + инсулинорезистентность (HOMA), риск ≈19% (×5–6 к
базовому); ЖКТ/НЖБП — избыточная масса тела + ГКС (≈53%); повторные пневмонии — HOMA + высокий ИЛ-6
(≈43%). Это переводит ассоциации в практические критерии отбора в группу риска.

**Клинико-лабораторный разбор (этап 3).** Для каждого осложнения разобран профиль типичного
пациента: воспалительно-метаболические осложнения (ОНМК/ИМ, ССЗ, ЖКТ/НЖБП) сопровождаются ростом
СРБ, ферритина, ИЛ-6, глюкозы и HOMA; фиброз и ХОБЛ/астма — в первую очередь тяжестью лёгочного
поражения (КТ, SpO2).

**Панкреонекроз (этап 4).** Всего 8 случаев — для воспроизводимой прогнозной модели данных
недостаточно (нужно ≥10 событий на предиктор и панкреас-специфичные маркеры). Приведён описательный
разбор; прогноз станет возможен по мере накопления случаев.

**Новые осложнения.** В анализ добавлены ещё два поздних исхода: **зрение/слух** (19.5%, частый) и
**психоневрологические нарушения / обоняние** (10.3%); оба проведены через этапы 1–3 и 5. Зрение/слух —
ведущий результат этапа 5 (единственное осложнение, выдерживающее FDR-поправку по штаммам).

**Геноварианты (этап 5).** Геновариант сам по себе не определяет частоту поздних осложнений: после
поправки на тяжесть (КТ) и возраст ни Дельта, ни Омикрон не дают независимого избытка риска по общему
флагу — риск определяют тяжесть (КТ OR≈2) и возраст (≈1.07/год). Единственное устойчивое штамм-
специфичное отличие — **снижение нарушений зрения/слуха при Омикроне** (8.9% против 23.9–26.5%,
выдерживает FDR-поправку; держится на 24 событиях и неотделимо от вакцинации/календарного периода).
Различия по психоневрологии, пневмофиброзу и ССЗ — поисковые, поправку не проходят. Сырые сравнения
конфаундированы (КТ 1/1/0, возраст 38/41/36, вакцинация 0.5/30/89%; вакцинация коллинеарна штамму).

**Клинические схемы (этап 6).** Схема повторных пневмоний (метаболический профиль HOMA/возраст/ИМТ,
CV-AUC≈0.67) — умеренная; схема ОНМК/ИМ (26 событий, CV-AUC≈0.80) — ориентировочная. Схема тромбозов /
операций на венах (idx 132, позднее осложнение, 52 случая) — прогностическая, ведущий предиктор Д-димер,
CV-AUC≈0.82. Все схемы валидированы только in-sample, пороги подобраны на той
же выборке, без поправки на штамм/эру — внешней валидации нет.

**Методическое замечание.** Следует различать два режима. *Прогностические* метрики (этап 2.5) даны
с кросс-валидацией и только для частых исходов. Деревья групп риска (2.6) и SHAP-разбор (3) применены
ко всем осложнениям, включая редкие, но в **описательно-объяснительном** режиме (in-sample, без
валидации) — это интерпретация выявленных закономерностей, а не готовый прогноз. Наборы предикторов
в OR-анализе (течение + коморбидность) и в деревьях/SHAP (с лабораторными показателями) различаются и
**дополняют** друг друга. Все выводы носят ассоциативный характер (наблюдательные данные) и требуют
клинической верификации.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open('covid_complications.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('wrote covid_complications.ipynb with', len(cells), 'cells')
