# Анализ данных COVID-19 и классификация степени поражения лёгких (КТ)

Ноутбук выполняет:
1. Разведочный анализ данных (EDA) по 4 листам: **общее**, **УХАНЬ**, **ДЕЛЬТА**, **ОМИКРОН**
2. Обучение и сравнение моделей классификации для предсказания **КТ(ИТОГ)** (степень поражения лёгких, 0–4):
   - Деревья решений (Decision Tree)
   - Логистическая регрессия (Logistic Regression)
   - Случайный лес (Random Forest)
   - Градиентный бустинг (Gradient Boosting)
3. Сравнение результатов между штаммами и моделями

**Признаки**: демографические (возраст, пол, группа крови), статус вакцинации, клинические показатели при поступлении (день заболевания, температура, ЧСС, АД, ИМТ), сопутствующие заболевания (органов дыхания, СД, ЖКТ, хр. гепатит, ХПН) и метаболизм (HOMA).

## О ноутбуке

Этот ноутбук решает две связанные задачи:

1. **Классификация** степени поражения лёгких при COVID-19 — целевая переменная **КТ(ИТОГ)** (5 классов от КТ-0 «норма» до КТ-4 «критическое поражение»).
2. **Регрессия и эмпирическая формула** — предсказание численного процента поражения лёгких **КТ(max)** (0–95%) одной компактной формулой, применимой у постели пациента.

Анализ выполняется отдельно для четырёх когорт: **общая выборка** и три штамма SARS-CoV-2 (**Уханьский**, **Дельта**, **Омикрон**) — это позволяет выявить штамм-специфичные клинические паттерны.

## Используемые библиотеки

| Библиотека | Назначение |
|---|---|
| **pandas, numpy** | Загрузка Excel, работа с табличными данными, числовые операции |
| **scipy.stats** (`skew`) | Расчёт асимметрии распределений — авто-детекция лог-нормальных признаков |
| **scikit-learn** | Все модели машинного обучения, метрики, валидация (см. ниже) |
| **shap** | SHAP-значения через TreeSHAP — индивидуальные и глобальные объяснения |
| **matplotlib, seaborn** | Визуализация: распределения, тепловые карты, confusion matrix, feature importance, scatter plots |

### Конкретные классы из `scikit-learn`

| Класс / функция | Где используется |
|---|---|
| `DecisionTreeClassifier`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier` | 4 классификатора КТ(ИТОГ) |
| `RidgeCV` | Линейная регрессия КТ(max) с L2-регуляризацией и CV-подбором α |
| `QuantileRegressor` | Квантильная регрессия для прогностических интервалов |
| `StandardScaler`, `Pipeline` | Стандартизация признаков (для LR и формулы) |
| `train_test_split`, `StratifiedKFold` | Разделение и кросс-валидация |
| `GridSearchCV` | Подбор гиперпараметров |
| `permutation_importance` | Permutation importance |
| `compute_sample_weight` | Веса классов для GB |
| `accuracy_score`, `f1_score`, `classification_report`, `confusion_matrix`, `r2_score`, `mean_absolute_error` | Метрики качества |

## Используемые подходы

### 1. Классификация КТ(ИТОГ) — 4 модели в сравнении

- **Decision Tree** — простая интерпретируемая модель; визуализируется деревом
- **Logistic Regression** — линейный baseline, работает на стандартизованных признаках через Pipeline
- **Random Forest** — ансамбль деревьев, устойчив к переобучению
- **Gradient Boosting** — последовательный бустинг

Для каждой модели:
- **Балансировка классов**: `class_weight='balanced'` (DT, LR, RF) и `sample_weight` (GB) — учитывают сильный дисбаланс (КТ-4 встречается в десятки раз реже КТ-1)
- **Подбор гиперпараметров**: `GridSearchCV` с **5-fold StratifiedKFold** по F1-weighted
- **Stratified train/test split** 80/20, `random_state=42` для воспроизводимости
- **Импутация пропусков**: медианой только по train (без data leakage)

### 2. Оценка важности признаков — 5 метрик в комбинации

Ни одна метрика важности не идеальна — пересечение нескольких даёт устойчивую интерпретацию.

| Метрика | Что измеряет |
|---|---|
| **Built-in feature importance** | Снижение Gini/MSE при разбиении узлов (RF, GB) |
| **Permutation importance** | Падение метрики на test при перетасовке признака; усреднение по LR + RF + GB (консенсус) |
| **Group permutation importance** | Перетасовка целых групп (метаболизм, цитокины, коагуляция, ...) — устраняет проблему мультиколлинеарности |
| **Drop-column importance** | Обучение модели без признака — уникальный (не воспроизводимый из остальных) вклад |
| **SHAP** | Аддитивное разложение через TreeSHAP — даёт индивидуальные и глобальные оценки |

### 3. Эмпирическая формула КТ(max) (раздел 12)

Регрессия числового процента поражения лёгких через линейные методы:

| Подраздел | Метод |
|---|---|
| **12.1** | Базовая Ridge-регрессия со стандартизованными признаками |
| **12.4** | Упрощённая формула — топ-5 признаков по permutation importance |
| **12.5** | Отдельные формулы для каждого штамма |
| **12.6** | Лог-преобразование скошенных признаков (асимметрия > 1.5) — адекватно лог-нормальной природе цитокинов и ферритина |
| **12.7** | Sample weights ∝ 1/частота(КТ-категории) — фокус на тяжёлых случаях |
| **12.8** | Квантильная регрессия (τ ∈ {0.1, 0.5, 0.9}) — прогностический интервал вместо точечного предсказания |
| **12.9** | Комбинированная формула: log + sample weights + параллельная квантильная τ=0.9 |

Все формулы выводятся в **двух видах**: стандартизованная (для сравнения значимости признаков) и в **исходных единицах** (для применения у постели пациента).


> **Источник данных (обновление).** Этот ноутбук — копия исходного анализа, пересчитанная на обновлённом наборе `updated_data.xlsx`: общая выборка увеличена до **755 пациентов**, а также добавлены столбцы поздних (постковидных) осложнений (`DZ..FL`). Схема клинических признаков идентична исходному файлу, поэтому методика анализа не изменилась — обновился только источник данных.

## 1. Импорт библиотек и загрузка данных

    Столбцы (признаки + целевая):
      Индекс   2 → возраст
      Индекс   3 → пол
      Индекс   7 → день заболевания
      Индекс  10 → вакцинация
      Индекс  20 → темпер
      Индекс  29 → группа крови
      Индекс  31 → ИМТ
      Индекс  32 → ЧСС (пост)
      Индекс  34 → АД(сис)
      Индекс  35 → АД(диас)
      Индекс  83 → Орг. Дых
      Индекс  84 → СД
      Индекс  85 → ЖКТ
      Индекс  86 → ХГ
      Индекс  87 → ХПН
      Индекс  97 → А/Т IgG
      Индекс 114 → HOMA
      Индекс  38 → КТ(ИТОГ)


        ОБЩЕЕ2: 755 строк, 18 столбцов
        УХАНЬ2: 213 строк, 18 столбцов
       ДЕЛЬТА2: 272 строк, 18 столбцов
      ОМИКРОН2: 270 строк, 18 столбцов


## 2. Предобработка данных

    Количество пропусков по столбцам:
    
    --- ОБЩЕЕ2 ---
      ИМТ: 1 (0.1%)
      ЧСС (пост): 1 (0.1%)
      АД(сис): 1 (0.1%)
      АД(диас): 1 (0.1%)
    
    --- УХАНЬ2 --- пропусков нет
    
    --- ДЕЛЬТА2 --- пропусков нет
    
    --- ОМИКРОН2 ---
      ИМТ: 1 (0.4%)
      ЧСС (пост): 1 (0.4%)
      АД(сис): 1 (0.4%)
      АД(диас): 1 (0.4%)
    


    
    ============================================================
      Описательная статистика: ОБЩЕЕ2
    ============================================================



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>возраст</th>
      <th>пол</th>
      <th>день заболевания</th>
      <th>вакцинация</th>
      <th>темпер</th>
      <th>группа крови</th>
      <th>ИМТ</th>
      <th>ЧСС (пост)</th>
      <th>АД(сис)</th>
      <th>АД(диас)</th>
      <th>Орг. Дых</th>
      <th>СД</th>
      <th>ЖКТ</th>
      <th>ХГ</th>
      <th>ХПН</th>
      <th>А/Т IgG</th>
      <th>HOMA</th>
      <th>КТ(ИТОГ)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>754.00</td>
      <td>754.00</td>
      <td>754.00</td>
      <td>754.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
      <td>755.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>37.87</td>
      <td>1.41</td>
      <td>4.08</td>
      <td>0.43</td>
      <td>37.99</td>
      <td>2.10</td>
      <td>27.11</td>
      <td>90.52</td>
      <td>139.31</td>
      <td>99.19</td>
      <td>0.05</td>
      <td>0.14</td>
      <td>0.53</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>164.24</td>
      <td>2.03</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.79</td>
      <td>0.49</td>
      <td>2.99</td>
      <td>0.50</td>
      <td>0.90</td>
      <td>0.93</td>
      <td>3.92</td>
      <td>40.60</td>
      <td>16.80</td>
      <td>14.41</td>
      <td>0.21</td>
      <td>0.34</td>
      <td>0.50</td>
      <td>0.19</td>
      <td>0.13</td>
      <td>415.94</td>
      <td>1.88</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>28.40</td>
      <td>1.00</td>
      <td>17.80</td>
      <td>42.00</td>
      <td>100.00</td>
      <td>60.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.33</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.00</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>37.40</td>
      <td>1.00</td>
      <td>24.40</td>
      <td>77.25</td>
      <td>128.00</td>
      <td>90.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.85</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>38.00</td>
      <td>2.00</td>
      <td>26.90</td>
      <td>88.00</td>
      <td>136.50</td>
      <td>100.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.35</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>46.00</td>
      <td>2.00</td>
      <td>6.00</td>
      <td>1.00</td>
      <td>38.50</td>
      <td>3.00</td>
      <td>29.69</td>
      <td>101.00</td>
      <td>150.00</td>
      <td>110.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>141.00</td>
      <td>2.51</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.00</td>
      <td>2.00</td>
      <td>18.00</td>
      <td>1.00</td>
      <td>40.30</td>
      <td>4.00</td>
      <td>40.20</td>
      <td>1102.00</td>
      <td>212.00</td>
      <td>149.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>5040.00</td>
      <td>16.21</td>
      <td>4.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    ============================================================
      Описательная статистика: УХАНЬ2
    ============================================================



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>возраст</th>
      <th>пол</th>
      <th>день заболевания</th>
      <th>вакцинация</th>
      <th>темпер</th>
      <th>группа крови</th>
      <th>ИМТ</th>
      <th>ЧСС (пост)</th>
      <th>АД(сис)</th>
      <th>АД(диас)</th>
      <th>Орг. Дых</th>
      <th>СД</th>
      <th>ЖКТ</th>
      <th>ХГ</th>
      <th>ХПН</th>
      <th>А/Т IgG</th>
      <th>HOMA</th>
      <th>КТ(ИТОГ)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>213.0</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
      <td>213.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.1</td>
      <td>1.18</td>
      <td>3.53</td>
      <td>0.00</td>
      <td>38.11</td>
      <td>2.11</td>
      <td>26.31</td>
      <td>86.74</td>
      <td>137.73</td>
      <td>96.98</td>
      <td>0.06</td>
      <td>0.11</td>
      <td>0.58</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>10.26</td>
      <td>1.74</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.8</td>
      <td>0.39</td>
      <td>2.49</td>
      <td>0.07</td>
      <td>0.87</td>
      <td>0.91</td>
      <td>3.88</td>
      <td>17.59</td>
      <td>17.20</td>
      <td>14.31</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>0.49</td>
      <td>0.14</td>
      <td>0.10</td>
      <td>53.90</td>
      <td>1.75</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>36.00</td>
      <td>1.00</td>
      <td>18.60</td>
      <td>42.00</td>
      <td>100.00</td>
      <td>60.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.33</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>37.40</td>
      <td>1.00</td>
      <td>23.60</td>
      <td>75.00</td>
      <td>125.00</td>
      <td>88.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.74</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>38.0</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>38.00</td>
      <td>2.00</td>
      <td>26.20</td>
      <td>85.00</td>
      <td>135.00</td>
      <td>98.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.13</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>45.0</td>
      <td>1.00</td>
      <td>6.00</td>
      <td>0.00</td>
      <td>38.80</td>
      <td>3.00</td>
      <td>28.60</td>
      <td>99.00</td>
      <td>145.00</td>
      <td>105.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.18</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>72.0</td>
      <td>2.00</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>40.10</td>
      <td>4.00</td>
      <td>38.50</td>
      <td>138.00</td>
      <td>212.00</td>
      <td>149.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>500.00</td>
      <td>13.90</td>
      <td>4.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    ============================================================
      Описательная статистика: ДЕЛЬТА2
    ============================================================



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>возраст</th>
      <th>пол</th>
      <th>день заболевания</th>
      <th>вакцинация</th>
      <th>темпер</th>
      <th>группа крови</th>
      <th>ИМТ</th>
      <th>ЧСС (пост)</th>
      <th>АД(сис)</th>
      <th>АД(диас)</th>
      <th>Орг. Дых</th>
      <th>СД</th>
      <th>ЖКТ</th>
      <th>ХГ</th>
      <th>ХПН</th>
      <th>А/Т IgG</th>
      <th>HOMA</th>
      <th>КТ(ИТОГ)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
      <td>272.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.35</td>
      <td>1.42</td>
      <td>4.87</td>
      <td>0.30</td>
      <td>38.07</td>
      <td>2.07</td>
      <td>27.73</td>
      <td>94.86</td>
      <td>142.64</td>
      <td>101.52</td>
      <td>0.05</td>
      <td>0.19</td>
      <td>0.65</td>
      <td>0.04</td>
      <td>0.01</td>
      <td>59.52</td>
      <td>2.09</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.36</td>
      <td>0.49</td>
      <td>3.37</td>
      <td>0.46</td>
      <td>0.85</td>
      <td>0.92</td>
      <td>3.97</td>
      <td>63.86</td>
      <td>16.97</td>
      <td>14.80</td>
      <td>0.21</td>
      <td>0.39</td>
      <td>0.48</td>
      <td>0.19</td>
      <td>0.12</td>
      <td>132.50</td>
      <td>1.71</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>36.00</td>
      <td>1.00</td>
      <td>17.80</td>
      <td>54.00</td>
      <td>105.00</td>
      <td>61.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.45</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.75</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>37.50</td>
      <td>1.00</td>
      <td>25.38</td>
      <td>79.00</td>
      <td>130.00</td>
      <td>90.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.96</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>41.00</td>
      <td>1.00</td>
      <td>4.00</td>
      <td>0.00</td>
      <td>38.00</td>
      <td>2.00</td>
      <td>27.80</td>
      <td>89.00</td>
      <td>140.00</td>
      <td>100.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.55</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.00</td>
      <td>2.00</td>
      <td>7.00</td>
      <td>1.00</td>
      <td>38.52</td>
      <td>3.00</td>
      <td>30.24</td>
      <td>105.00</td>
      <td>155.00</td>
      <td>112.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>12.58</td>
      <td>2.69</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>71.00</td>
      <td>2.00</td>
      <td>18.00</td>
      <td>1.00</td>
      <td>40.10</td>
      <td>4.00</td>
      <td>40.20</td>
      <td>1102.00</td>
      <td>200.00</td>
      <td>143.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>500.00</td>
      <td>15.30</td>
      <td>4.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    ============================================================
      Описательная статистика: ОМИКРОН2
    ============================================================



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>возраст</th>
      <th>пол</th>
      <th>день заболевания</th>
      <th>вакцинация</th>
      <th>темпер</th>
      <th>группа крови</th>
      <th>ИМТ</th>
      <th>ЧСС (пост)</th>
      <th>АД(сис)</th>
      <th>АД(диас)</th>
      <th>Орг. Дых</th>
      <th>СД</th>
      <th>ЖКТ</th>
      <th>ХГ</th>
      <th>ХПН</th>
      <th>А/Т IgG</th>
      <th>HOMA</th>
      <th>КТ(ИТОГ)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>269.00</td>
      <td>269.00</td>
      <td>269.00</td>
      <td>269.00</td>
      <td>270.00</td>
      <td>270.0</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
      <td>270.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.76</td>
      <td>1.58</td>
      <td>3.73</td>
      <td>0.89</td>
      <td>37.82</td>
      <td>2.11</td>
      <td>27.11</td>
      <td>89.12</td>
      <td>137.21</td>
      <td>98.60</td>
      <td>0.04</td>
      <td>0.1</td>
      <td>0.37</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>391.20</td>
      <td>2.20</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.60</td>
      <td>0.49</td>
      <td>2.79</td>
      <td>0.31</td>
      <td>0.95</td>
      <td>0.96</td>
      <td>3.81</td>
      <td>15.18</td>
      <td>15.82</td>
      <td>13.79</td>
      <td>0.19</td>
      <td>0.3</td>
      <td>0.48</td>
      <td>0.21</td>
      <td>0.17</td>
      <td>619.19</td>
      <td>2.13</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>28.40</td>
      <td>1.00</td>
      <td>19.40</td>
      <td>49.00</td>
      <td>110.00</td>
      <td>60.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>37.40</td>
      <td>1.00</td>
      <td>24.30</td>
      <td>79.00</td>
      <td>125.00</td>
      <td>90.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>34.20</td>
      <td>0.86</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>1.00</td>
      <td>37.80</td>
      <td>2.00</td>
      <td>26.60</td>
      <td>89.00</td>
      <td>135.00</td>
      <td>100.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>200.00</td>
      <td>1.30</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>46.00</td>
      <td>2.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>38.40</td>
      <td>3.00</td>
      <td>29.50</td>
      <td>100.00</td>
      <td>145.00</td>
      <td>110.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>500.00</td>
      <td>2.61</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.00</td>
      <td>2.00</td>
      <td>15.00</td>
      <td>1.00</td>
      <td>40.30</td>
      <td>4.00</td>
      <td>38.80</td>
      <td>127.00</td>
      <td>200.00</td>
      <td>126.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>5040.00</td>
      <td>16.21</td>
      <td>3.00</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Разведочный анализ данных (EDA)

### 3.1 Распределение целевой переменной КТ(ИТОГ)


    
![png](covid_analysis_updated_files/covid_analysis_updated_12_0.png)
    


### 3.2 Матрицы корреляций


    
![png](covid_analysis_updated_files/covid_analysis_updated_14_0.png)
    


### 3.3 Boxplot-ы признаков в разрезе КТ(ИТОГ)


    
![png](covid_analysis_updated_files/covid_analysis_updated_16_0.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_16_1.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_16_2.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_16_3.png)
    


### 3.4 Гистограммы распределений признаков


    
![png](covid_analysis_updated_files/covid_analysis_updated_18_0.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_18_1.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_18_2.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_18_3.png)
    


### 3.5 Сравнительная таблица средних по штаммам


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ОБЩЕЕ2</th>
      <th>УХАНЬ2</th>
      <th>ДЕЛЬТА2</th>
      <th>ОМИКРОН2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>возраст</th>
      <td>37.87</td>
      <td>36.10</td>
      <td>40.35</td>
      <td>36.76</td>
    </tr>
    <tr>
      <th>пол</th>
      <td>1.41</td>
      <td>1.18</td>
      <td>1.42</td>
      <td>1.58</td>
    </tr>
    <tr>
      <th>день заболевания</th>
      <td>4.08</td>
      <td>3.53</td>
      <td>4.87</td>
      <td>3.73</td>
    </tr>
    <tr>
      <th>вакцинация</th>
      <td>0.43</td>
      <td>0.00</td>
      <td>0.30</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>темпер</th>
      <td>37.99</td>
      <td>38.11</td>
      <td>38.07</td>
      <td>37.82</td>
    </tr>
    <tr>
      <th>группа крови</th>
      <td>2.10</td>
      <td>2.11</td>
      <td>2.07</td>
      <td>2.11</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>27.11</td>
      <td>26.31</td>
      <td>27.73</td>
      <td>27.11</td>
    </tr>
    <tr>
      <th>ЧСС (пост)</th>
      <td>90.52</td>
      <td>86.74</td>
      <td>94.86</td>
      <td>89.12</td>
    </tr>
    <tr>
      <th>АД(сис)</th>
      <td>139.31</td>
      <td>137.73</td>
      <td>142.64</td>
      <td>137.21</td>
    </tr>
    <tr>
      <th>АД(диас)</th>
      <td>99.19</td>
      <td>96.98</td>
      <td>101.52</td>
      <td>98.60</td>
    </tr>
    <tr>
      <th>Орг. Дых</th>
      <td>0.05</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>СД</th>
      <td>0.14</td>
      <td>0.11</td>
      <td>0.19</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>ЖКТ</th>
      <td>0.53</td>
      <td>0.58</td>
      <td>0.65</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>ХГ</th>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>ХПН</th>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>А/Т IgG</th>
      <td>164.24</td>
      <td>10.26</td>
      <td>59.52</td>
      <td>391.20</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>2.03</td>
      <td>1.74</td>
      <td>2.09</td>
      <td>2.20</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_analysis_updated_files/covid_analysis_updated_20_1.png)
    


## 4. Подготовка данных к моделированию

        ОБЩЕЕ2: train=604, test=151, классы в train: {0: np.int64(242), 1: np.int64(197), 2: np.int64(114), 3: np.int64(42), 4: np.int64(9)}
        УХАНЬ2: train=170, test=43, классы в train: {0: np.int64(67), 1: np.int64(54), 2: np.int64(32), 3: np.int64(14), 4: np.int64(3)}
       ДЕЛЬТА2: train=217, test=55, классы в train: {0: np.int64(60), 1: np.int64(83), 2: np.int64(46), 3: np.int64(22), 4: np.int64(6)}
      ОМИКРОН2: train=216, test=54, классы в train: {0: np.int64(115), 1: np.int64(59), 2: np.int64(35), 3: np.int64(7)}


## 5. Деревья решений — обучение и оценка

    
    ============================================================
      Дерево решений: ОБЩЕЕ2
    ============================================================


    
    Лучшие параметры: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 5}
    Лучший F1 (CV): 0.5083
    
    Тестовые метрики:
      Accuracy:        0.4503
      F1 (weighted):   0.4580
      F1 (macro):      0.3589
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.68      0.67      0.68        61
               1       0.48      0.27      0.34        49
               2       0.21      0.36      0.27        28
               3       0.30      0.27      0.29        11
               4       0.14      0.50      0.22         2
    
        accuracy                           0.45       151
       macro avg       0.36      0.41      0.36       151
    weighted avg       0.50      0.45      0.46       151
    
    
    ============================================================
      Дерево решений: УХАНЬ2
    ============================================================


    
    Лучшие параметры: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 20}
    Лучший F1 (CV): 0.4829
    
    Тестовые метрики:
      Accuracy:        0.3953
      F1 (weighted):   0.4074
      F1 (macro):      0.2717
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.69      0.53      0.60        17
               1       0.33      0.43      0.38        14
               2       0.14      0.12      0.13         8
               3       0.25      0.25      0.25         4
               4       0.00      0.00      0.00         0
    
        accuracy                           0.40        43
       macro avg       0.28      0.27      0.27        43
    weighted avg       0.43      0.40      0.41        43
    
    
    ============================================================
      Дерево решений: ДЕЛЬТА2
    ============================================================


    
    Лучшие параметры: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 10, 'min_samples_split': 5}
    Лучший F1 (CV): 0.4541
    
    Тестовые метрики:
      Accuracy:        0.3273
      F1 (weighted):   0.3468
      F1 (macro):      0.2429
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.43      0.60      0.50        15
               1       0.56      0.24      0.33        21
               2       0.44      0.33      0.38        12
               3       0.00      0.00      0.00         5
               4       0.00      0.00      0.00         2
    
        accuracy                           0.33        55
       macro avg       0.29      0.23      0.24        55
    weighted avg       0.43      0.33      0.35        55
    
    
    ============================================================
      Дерево решений: ОМИКРОН2
    ============================================================


    
    Лучшие параметры: {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 5, 'min_samples_split': 20}
    Лучший F1 (CV): 0.6805
    
    Тестовые метрики:
      Accuracy:        0.4815
      F1 (weighted):   0.5101
      F1 (macro):      0.3207
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.86      0.62      0.72        29
               1       0.27      0.27      0.27        15
               2       0.22      0.44      0.30         9
               3       0.00      0.00      0.00         1
    
        accuracy                           0.48        54
       macro avg       0.34      0.33      0.32        54
    weighted avg       0.57      0.48      0.51        54
    


### 5.1 Матрицы ошибок (Decision Tree)


    
![png](covid_analysis_updated_files/covid_analysis_updated_26_0.png)
    


### 5.2 Визуализация деревьев решений


    
![png](covid_analysis_updated_files/covid_analysis_updated_28_0.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_28_1.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_28_2.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_28_3.png)
    


### 5.3 Важность признаков (Decision Tree)


    
![png](covid_analysis_updated_files/covid_analysis_updated_30_0.png)
    


## 6. Логистическая регрессия

    
    ============================================================
      Логистическая регрессия: ОБЩЕЕ2
    ============================================================


    
    Лучшие параметры: C=0.5, penalty=l2
    Лучший F1 (CV): 0.5564
    
    Тестовые метрики:
      Accuracy:        0.5828
      F1 (weighted):   0.5927
      F1 (macro):      0.4854
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.79      0.72      0.75        61
               1       0.59      0.45      0.51        49
               2       0.45      0.46      0.46        28
               3       0.36      0.73      0.48        11
               4       0.14      0.50      0.22         2
    
        accuracy                           0.58       151
       macro avg       0.47      0.57      0.49       151
    weighted avg       0.62      0.58      0.59       151
    
    
    ============================================================
      Логистическая регрессия: УХАНЬ2
    ============================================================


    
    Лучшие параметры: C=0.1, penalty=l2
    Лучший F1 (CV): 0.5909
    
    Тестовые метрики:
      Accuracy:        0.3721
      F1 (weighted):   0.3765
      F1 (macro):      0.2454
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.65      0.65      0.65        17
               1       0.27      0.21      0.24        14
               2       0.11      0.12      0.12         8
               3       0.20      0.25      0.22         4
               4       0.00      0.00      0.00         0
    
        accuracy                           0.37        43
       macro avg       0.25      0.25      0.25        43
    weighted avg       0.38      0.37      0.38        43
    
    
    ============================================================
      Логистическая регрессия: ДЕЛЬТА2
    ============================================================


    
    Лучшие параметры: C=0.1, penalty=l2
    Лучший F1 (CV): 0.5144
    
    Тестовые метрики:
      Accuracy:        0.4364
      F1 (weighted):   0.4359
      F1 (macro):      0.3940
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.56      0.67      0.61        15
               1       0.54      0.33      0.41        21
               2       0.33      0.33      0.33        12
               3       0.29      0.40      0.33         5
               4       0.20      0.50      0.29         2
    
        accuracy                           0.44        55
       macro avg       0.38      0.45      0.39        55
    weighted avg       0.46      0.44      0.44        55
    
    
    ============================================================
      Логистическая регрессия: ОМИКРОН2
    ============================================================


    
    Лучшие параметры: C=0.5, penalty=l2
    Лучший F1 (CV): 0.6680
    
    Тестовые метрики:
      Accuracy:        0.6667
      F1 (weighted):   0.6758
      F1 (macro):      0.7101
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.84      0.72      0.78        29
               1       0.53      0.60      0.56        15
               2       0.45      0.56      0.50         9
               3       1.00      1.00      1.00         1
    
        accuracy                           0.67        54
       macro avg       0.71      0.72      0.71        54
    weighted avg       0.69      0.67      0.68        54
    


### 6.1 Коэффициенты логистической регрессии

Коэффициенты получены на **стандартизованных** признаках (StandardScaler). Величина коэффициента отражает силу влияния признака в единицах стандартного отклонения, а не в исходных единицах измерения. Знак показывает направление связи с классом.


    
![png](covid_analysis_updated_files/covid_analysis_updated_34_0.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_34_1.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_34_2.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_34_3.png)
    


### 6.2 Матрицы ошибок (Logistic Regression)


    
![png](covid_analysis_updated_files/covid_analysis_updated_36_0.png)
    


## 7. Случайный лес (Random Forest)

    
    ============================================================
      Random Forest: ОБЩЕЕ2
    ============================================================


    
    Лучшие параметры: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'n_estimators': 200}
    Лучший F1 (CV): 0.5931
    
    Тестовые метрики:
      Accuracy:        0.6026
      F1 (weighted):   0.6024
      F1 (macro):      0.4355
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.77      0.72      0.75        61
               1       0.57      0.59      0.58        49
               2       0.41      0.50      0.45        28
               3       0.44      0.36      0.40        11
               4       0.00      0.00      0.00         2
    
        accuracy                           0.60       151
       macro avg       0.44      0.44      0.44       151
    weighted avg       0.61      0.60      0.60       151
    
    
    ============================================================
      Random Forest: УХАНЬ2
    ============================================================


    
    Лучшие параметры: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 300}
    Лучший F1 (CV): 0.5918
    
    Тестовые метрики:
      Accuracy:        0.4419
      F1 (weighted):   0.4232
      F1 (macro):      0.3697
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.65      0.65      0.65        17
               1       0.35      0.43      0.39        14
               2       0.00      0.00      0.00         8
               3       0.40      0.50      0.44         4
    
        accuracy                           0.44        43
       macro avg       0.35      0.39      0.37        43
    weighted avg       0.41      0.44      0.42        43
    
    
    ============================================================
      Random Forest: ДЕЛЬТА2
    ============================================================


    
    Лучшие параметры: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 300}
    Лучший F1 (CV): 0.5197
    
    Тестовые метрики:
      Accuracy:        0.3818
      F1 (weighted):   0.3926
      F1 (macro):      0.3109
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.67      0.53      0.59        15
               1       0.41      0.33      0.37        21
               2       0.25      0.33      0.29        12
               3       0.25      0.40      0.31         5
               4       0.00      0.00      0.00         2
    
        accuracy                           0.38        55
       macro avg       0.32      0.32      0.31        55
    weighted avg       0.42      0.38      0.39        55
    
    
    ============================================================
      Random Forest: ОМИКРОН2
    ============================================================


    
    Лучшие параметры: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'n_estimators': 200}
    Лучший F1 (CV): 0.7259
    
    Тестовые метрики:
      Accuracy:        0.6481
      F1 (weighted):   0.6449
      F1 (macro):      0.4503
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.79      0.76      0.77        29
               1       0.47      0.60      0.53        15
               2       0.57      0.44      0.50         9
               3       0.00      0.00      0.00         1
    
        accuracy                           0.65        54
       macro avg       0.46      0.45      0.45        54
    weighted avg       0.65      0.65      0.64        54
    


### 7.1 Важность признаков (Random Forest)


    
![png](covid_analysis_updated_files/covid_analysis_updated_40_0.png)
    


### 7.2 Матрицы ошибок (Random Forest)


    
![png](covid_analysis_updated_files/covid_analysis_updated_42_0.png)
    


## 8. Градиентный бустинг (Gradient Boosting)

    
    ============================================================
      Gradient Boosting: ОБЩЕЕ2
    ============================================================


    
    Лучшие параметры: {'learning_rate': 0.02, 'max_depth': 5, 'min_samples_leaf': 10, 'n_estimators': 300, 'subsample': 0.8}
    Лучший F1 (CV): 0.4442
    
    Тестовые метрики:
      Accuracy:        0.5563
      F1 (weighted):   0.5601
      F1 (macro):      0.4849
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.74      0.70      0.72        61
               1       0.53      0.51      0.52        49
               2       0.34      0.43      0.38        28
               3       0.33      0.27      0.30        11
               4       0.50      0.50      0.50         2
    
        accuracy                           0.56       151
       macro avg       0.49      0.48      0.48       151
    weighted avg       0.57      0.56      0.56       151
    
    
    ============================================================
      Gradient Boosting: УХАНЬ2
    ============================================================


    
    Лучшие параметры: {'learning_rate': 0.02, 'max_depth': 5, 'min_samples_leaf': 5, 'n_estimators': 100, 'subsample': 1.0}
    Лучший F1 (CV): 0.3699
    
    Тестовые метрики:
      Accuracy:        0.4419
      F1 (weighted):   0.4343
      F1 (macro):      0.2937
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.58      0.65      0.61        17
               1       0.43      0.43      0.43        14
               2       0.17      0.12      0.14         8
               3       0.33      0.25      0.29         4
               4       0.00      0.00      0.00         0
    
        accuracy                           0.44        43
       macro avg       0.30      0.29      0.29        43
    weighted avg       0.43      0.44      0.43        43
    
    
    ============================================================
      Gradient Boosting: ДЕЛЬТА2
    ============================================================


    
    Лучшие параметры: {'learning_rate': 0.02, 'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 100, 'subsample': 1.0}
    Лучший F1 (CV): 0.5534
    
    Тестовые метрики:
      Accuracy:        0.3636
      F1 (weighted):   0.3683
      F1 (macro):      0.2926
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.67      0.67      0.67        15
               1       0.38      0.29      0.32        21
               2       0.25      0.25      0.25        12
               3       0.00      0.00      0.00         5
               4       0.14      0.50      0.22         2
    
        accuracy                           0.36        55
       macro avg       0.29      0.34      0.29        55
    weighted avg       0.38      0.36      0.37        55
    
    
    ============================================================
      Gradient Boosting: ОМИКРОН2
    ============================================================


    
    Лучшие параметры: {'learning_rate': 0.02, 'max_depth': 3, 'min_samples_leaf': 5, 'n_estimators': 100, 'subsample': 1.0}
    Лучший F1 (CV): 0.7101
    
    Тестовые метрики:
      Accuracy:        0.7222
      F1 (weighted):   0.7241
      F1 (macro):      0.5060
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.88      0.76      0.81        29
               1       0.68      0.87      0.76        15
               2       0.44      0.44      0.44         9
               3       0.00      0.00      0.00         1
    
        accuracy                           0.72        54
       macro avg       0.50      0.52      0.51        54
    weighted avg       0.74      0.72      0.72        54
    


### 8.1 Важность признаков (Gradient Boosting)


    
![png](covid_analysis_updated_files/covid_analysis_updated_46_0.png)
    


### 8.2 Матрицы ошибок (Gradient Boosting)


    
![png](covid_analysis_updated_files/covid_analysis_updated_48_0.png)
    


## 9. Сравнение всех моделей


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Accuracy (test)</th>
      <th colspan="4" halign="left">F1 test (macro)</th>
      <th colspan="4" halign="left">F1 test (weighted)</th>
    </tr>
    <tr>
      <th>Модель</th>
      <th>Decision Tree</th>
      <th>Gradient Boosting</th>
      <th>Logistic Regression</th>
      <th>Random Forest</th>
      <th>Decision Tree</th>
      <th>Gradient Boosting</th>
      <th>Logistic Regression</th>
      <th>Random Forest</th>
      <th>Decision Tree</th>
      <th>Gradient Boosting</th>
      <th>Logistic Regression</th>
      <th>Random Forest</th>
    </tr>
    <tr>
      <th>Штамм</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ДЕЛЬТА2</th>
      <td>0.3273</td>
      <td>0.3636</td>
      <td>0.4364</td>
      <td>0.3818</td>
      <td>0.2429</td>
      <td>0.2926</td>
      <td>0.3940</td>
      <td>0.3109</td>
      <td>0.3468</td>
      <td>0.3683</td>
      <td>0.4359</td>
      <td>0.3926</td>
    </tr>
    <tr>
      <th>ОБЩЕЕ2</th>
      <td>0.4503</td>
      <td>0.5563</td>
      <td>0.5828</td>
      <td>0.6026</td>
      <td>0.3589</td>
      <td>0.4849</td>
      <td>0.4854</td>
      <td>0.4355</td>
      <td>0.4580</td>
      <td>0.5601</td>
      <td>0.5927</td>
      <td>0.6024</td>
    </tr>
    <tr>
      <th>ОМИКРОН2</th>
      <td>0.4815</td>
      <td>0.7222</td>
      <td>0.6667</td>
      <td>0.6481</td>
      <td>0.3207</td>
      <td>0.5060</td>
      <td>0.7101</td>
      <td>0.4503</td>
      <td>0.5101</td>
      <td>0.7241</td>
      <td>0.6758</td>
      <td>0.6449</td>
    </tr>
    <tr>
      <th>УХАНЬ2</th>
      <td>0.3953</td>
      <td>0.4419</td>
      <td>0.3721</td>
      <td>0.4419</td>
      <td>0.2717</td>
      <td>0.2937</td>
      <td>0.2454</td>
      <td>0.3697</td>
      <td>0.4074</td>
      <td>0.4343</td>
      <td>0.3765</td>
      <td>0.4232</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_analysis_updated_files/covid_analysis_updated_51_0.png)
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_52_0.png)
    


## 10. Permutation Importance — надёжная оценка важности признаков

**Permutation importance** — модельно-независимый метод: оценивает важность признака как падение качества модели при случайной перетасовке его значений. В отличие от встроенного `feature_importances_` (который зависит от структуры конкретной модели), permutation importance напрямую измеряет вклад признака в **обобщающую способность** модели.

Считаем по тестовой выборке (n_repeats=20, scoring='f1_weighted') для трёх моделей: **Logistic Regression**, **Random Forest**, **Gradient Boosting**. Decision Tree исключён из итогового агрегирования как наименее устойчивая модель.

    Logistic Regression: готово


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
      warnings.warn(


    Random Forest: готово


    Gradient Boosting: готово


### 10.1 Permutation importance по моделям


    
![png](covid_analysis_updated_files/covid_analysis_updated_56_0.png)
    


### 10.2 Консенсус-важность (среднее по LR, RF, GB)

Усредняем permutation importance по трём моделям для каждого штамма — это финальная оценка значимости признаков.


    
![png](covid_analysis_updated_files/covid_analysis_updated_58_0.png)
    


    
    Топ-5 признаков по консенсус-важности (среднее по штаммам):
      день заболевания: 0.0510
      АД(сис): 0.0130
      ЧСС (пост): 0.0098
      ИМТ: 0.0094
      возраст: 0.0084


### 10.3 Group permutation importance

Перемешиваем **группы коррелированных признаков совместно**. Это решает проблему «ложно-нулевой важности» из-за коллинеарности: если HOMA, СД и ИМТ несут общий сигнал метаболического синдрома, мы измеряем важность всей группы, а не каждого признака отдельно.

**Группы (на основе медицинской логики и корреляций):**
- **Метаболическая**: HOMA, СД, ИМТ
- **Гемодинамическая**: ЧСС, АД(сис), АД(диас)
- **Демография**: возраст, пол, группа крови
- **Клиническая фаза**: день заболевания, темпер, вакцинация
- **Коморбидности**: Орг. Дых, ЖКТ, ХГ, ХПН

    Logistic Regression: готово


    Random Forest: готово


    Gradient Boosting: готово



    
![png](covid_analysis_updated_files/covid_analysis_updated_60_3.png)
    


    
    Топ групп по среднему по штаммам:
      Клиническая фаза: 0.0858
      Гемодинамическая: 0.0290
      Метаболическая: 0.0289
      Демография: 0.0125
      Гуморальный иммунитет: 0.0034
      Коморбидности: -0.0039


### 10.4 Drop-column importance

Самая «честная» оценка важности: для каждого признака **переобучаем** модель без него и измеряем падение F1 на тесте. Гиперпараметры берём из ранее подобранных моделей (без повторного GridSearch — иначе расчёт занял бы часы).

Drop-column importance даёт абсолютную оценку **уникального** вклада признака. В отличие от permutation, она не страдает от артефактов перемешивания, но всё ещё может занижать важность коллинеарных признаков (модель компенсирует через корреляты).

    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(
    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1135: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', and C=np.inf instead of penalty=None.
      warnings.warn(


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn(


    Logistic Regression: готово


    Random Forest: готово


    Gradient Boosting: готово



    
![png](covid_analysis_updated_files/covid_analysis_updated_62_60.png)
    


    
    Топ-5 признаков по drop-column importance (среднее по штаммам):
      день заболевания: 0.0332
      ИМТ: 0.0089
      АД(сис): 0.0082
      АД(диас): 0.0012
      HOMA: 0.0004


### 10.5 Сводная таблица: четыре метрики важности

Сравним все метрики важности рядом, чтобы выявить устойчивые признаки и эффекты коллинеарности.


    
![png](covid_analysis_updated_files/covid_analysis_updated_64_0.png)
    


    
    Исходные значения (без нормализации):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Built-in (RF+GB)</th>
      <th>Permutation</th>
      <th>Drop-column</th>
      <th>LR |coef|</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>день заболевания</th>
      <td>0.0833</td>
      <td>0.0510</td>
      <td>0.0332</td>
      <td>0.3195</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>0.1194</td>
      <td>0.0094</td>
      <td>0.0089</td>
      <td>0.2820</td>
    </tr>
    <tr>
      <th>АД(сис)</th>
      <td>0.0759</td>
      <td>0.0130</td>
      <td>0.0082</td>
      <td>0.2529</td>
    </tr>
    <tr>
      <th>АД(диас)</th>
      <td>0.0627</td>
      <td>0.0069</td>
      <td>0.0012</td>
      <td>0.2123</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>0.1696</td>
      <td>0.0080</td>
      <td>0.0004</td>
      <td>0.3735</td>
    </tr>
    <tr>
      <th>СД</th>
      <td>0.0095</td>
      <td>-0.0075</td>
      <td>-0.0004</td>
      <td>0.2289</td>
    </tr>
    <tr>
      <th>ХГ</th>
      <td>0.0024</td>
      <td>0.0009</td>
      <td>-0.0005</td>
      <td>0.1772</td>
    </tr>
    <tr>
      <th>ЧСС (пост)</th>
      <td>0.1926</td>
      <td>0.0098</td>
      <td>-0.0010</td>
      <td>0.5727</td>
    </tr>
    <tr>
      <th>вакцинация</th>
      <td>0.0060</td>
      <td>0.0061</td>
      <td>-0.0015</td>
      <td>0.2533</td>
    </tr>
    <tr>
      <th>Орг. Дых</th>
      <td>0.0191</td>
      <td>-0.0019</td>
      <td>-0.0017</td>
      <td>0.1641</td>
    </tr>
    <tr>
      <th>группа крови</th>
      <td>0.0308</td>
      <td>0.0067</td>
      <td>-0.0025</td>
      <td>0.2924</td>
    </tr>
    <tr>
      <th>возраст</th>
      <td>0.0788</td>
      <td>0.0084</td>
      <td>-0.0030</td>
      <td>0.3240</td>
    </tr>
    <tr>
      <th>темпер</th>
      <td>0.0870</td>
      <td>0.0062</td>
      <td>-0.0032</td>
      <td>0.4208</td>
    </tr>
    <tr>
      <th>ХПН</th>
      <td>0.0001</td>
      <td>0.0004</td>
      <td>-0.0042</td>
      <td>0.1232</td>
    </tr>
    <tr>
      <th>А/Т IgG</th>
      <td>0.0405</td>
      <td>0.0028</td>
      <td>-0.0043</td>
      <td>0.2963</td>
    </tr>
    <tr>
      <th>пол</th>
      <td>0.0086</td>
      <td>-0.0031</td>
      <td>-0.0135</td>
      <td>0.1634</td>
    </tr>
    <tr>
      <th>ЖКТ</th>
      <td>0.0138</td>
      <td>-0.0036</td>
      <td>-0.0193</td>
      <td>0.2982</td>
    </tr>
  </tbody>
</table>
</div>


## 11. SHAP — индивидуальные и глобальные объяснения

**SHAP** (SHapley Additive exPlanations) основан на теории кооперативных игр: каждый признак — игрок, его SHAP-значение — справедливый предельный вклад в предсказание. Это единственный метод, удовлетворяющий аксиомам эффективности, симметрии, аддитивности и линейности одновременно.

**В отличие от permutation importance**, SHAP позволяет:
- объяснить **отдельное предсказание** для конкретного пациента (waterfall plot)
- показать **направление** влияния признака (повышает/понижает риск)
- использовать TreeSHAP — точный и быстрый алгоритм для tree-based моделей

Считаем SHAP для **Random Forest** через TreeSHAP. Для multiclass GradientBoostingClassifier текущая версия SHAP не поддерживает TreeSHAP, поэтому ограничиваемся одной моделью.

    ОБЩЕЕ2: shape=(151, 17, 5)
    УХАНЬ2: shape=(43, 17, 5)
    ДЕЛЬТА2: shape=(55, 17, 5)


    ОМИКРОН2: shape=(54, 17, 4)


### 11.1 Глобальная SHAP-важность по штаммам

Сравнение глобальной SHAP-важности (среднее модуля SHAP-значений) с уже рассчитанными метриками. Видно направление и распределение вклада признаков.


    
![png](covid_analysis_updated_files/covid_analysis_updated_68_0.png)
    


    
    Топ-5 признаков по SHAP-важности (среднее по штаммам):
      HOMA: 0.0463
      ЧСС (пост): 0.0423
      день заболевания: 0.0317
      ИМТ: 0.0306
      темпер: 0.0272


### 11.2 Summary plots — направление влияния признаков

В отличие от bar-чарта, **beeswarm summary plot** показывает каждого пациента (точку) и направление влияния. Цвет — значение признака (красный = высокое, синий = низкое); ось X — SHAP-значение (вклад в предсказание).

Покажем для **ОБЩЕЕ2** (наибольшая выборка) для класса **КТ-3** (тяжёлое поражение) — какие признаки повышают/понижают риск тяжёлого исхода.

    SHAP summary для класса КТ-3 (Random Forest, ОБЩЕЕ2)



    
![png](covid_analysis_updated_files/covid_analysis_updated_70_1.png)
    


### 11.3 Индивидуальные объяснения (waterfall plots)

Самое практически ценное применение SHAP — объяснение конкретного предсказания. **Waterfall plot** показывает, как каждый признак двигает предсказание от базового ожидания (среднего по выборке) к итоговому значению для данного пациента.

Покажем 5 примеров из ОБЩЕЕ2: пациент с КТ-0 (норма), КТ-1 (минимальное), КТ-2 (умеренное), КТ-3 (тяжёлое), КТ-4 (критическое) — на модели Random Forest. Класс КТ-4 представлен в выборке очень редко (1–4 пациента в test), поэтому если в test-выборке нет такого пациента, ищем его в train (SHAP пересчитываем по запросу).

    
    === Пациент 2 (test) с истинным классом КТ-0 ===



    
![png](covid_analysis_updated_files/covid_analysis_updated_72_1.png)
    


    
    === Пациент 0 (test) с истинным классом КТ-1 ===



    
![png](covid_analysis_updated_files/covid_analysis_updated_72_3.png)
    


    
    === Пациент 4 (test) с истинным классом КТ-2 ===



    
![png](covid_analysis_updated_files/covid_analysis_updated_72_5.png)
    


    
    === Пациент 11 (test) с истинным классом КТ-3 ===



    
![png](covid_analysis_updated_files/covid_analysis_updated_72_7.png)
    


    
    === Пациент 6 (test) с истинным классом КТ-4 ===



    
![png](covid_analysis_updated_files/covid_analysis_updated_72_9.png)
    


### 11.4 Сравнение SHAP с другими метриками важности


    
![png](covid_analysis_updated_files/covid_analysis_updated_74_0.png)
    


    
    Исходные значения (отсортировано по SHAP):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Built-in (RF+GB)</th>
      <th>Permutation</th>
      <th>Drop-column</th>
      <th>LR |coef|</th>
      <th>SHAP (RF)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HOMA</th>
      <td>0.1696</td>
      <td>0.0080</td>
      <td>0.0004</td>
      <td>0.3735</td>
      <td>0.0463</td>
    </tr>
    <tr>
      <th>ЧСС (пост)</th>
      <td>0.1926</td>
      <td>0.0098</td>
      <td>-0.0010</td>
      <td>0.5727</td>
      <td>0.0423</td>
    </tr>
    <tr>
      <th>день заболевания</th>
      <td>0.0833</td>
      <td>0.0510</td>
      <td>0.0332</td>
      <td>0.3195</td>
      <td>0.0317</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>0.1194</td>
      <td>0.0094</td>
      <td>0.0089</td>
      <td>0.2820</td>
      <td>0.0306</td>
    </tr>
    <tr>
      <th>темпер</th>
      <td>0.0870</td>
      <td>0.0062</td>
      <td>-0.0032</td>
      <td>0.4208</td>
      <td>0.0272</td>
    </tr>
    <tr>
      <th>АД(сис)</th>
      <td>0.0759</td>
      <td>0.0130</td>
      <td>0.0082</td>
      <td>0.2529</td>
      <td>0.0213</td>
    </tr>
    <tr>
      <th>возраст</th>
      <td>0.0788</td>
      <td>0.0084</td>
      <td>-0.0030</td>
      <td>0.3240</td>
      <td>0.0196</td>
    </tr>
    <tr>
      <th>АД(диас)</th>
      <td>0.0627</td>
      <td>0.0069</td>
      <td>0.0012</td>
      <td>0.2123</td>
      <td>0.0188</td>
    </tr>
    <tr>
      <th>А/Т IgG</th>
      <td>0.0405</td>
      <td>0.0028</td>
      <td>-0.0043</td>
      <td>0.2963</td>
      <td>0.0114</td>
    </tr>
    <tr>
      <th>ЖКТ</th>
      <td>0.0138</td>
      <td>-0.0036</td>
      <td>-0.0193</td>
      <td>0.2982</td>
      <td>0.0107</td>
    </tr>
    <tr>
      <th>группа крови</th>
      <td>0.0308</td>
      <td>0.0067</td>
      <td>-0.0025</td>
      <td>0.2924</td>
      <td>0.0082</td>
    </tr>
    <tr>
      <th>СД</th>
      <td>0.0095</td>
      <td>-0.0075</td>
      <td>-0.0004</td>
      <td>0.2289</td>
      <td>0.0052</td>
    </tr>
    <tr>
      <th>Орг. Дых</th>
      <td>0.0191</td>
      <td>-0.0019</td>
      <td>-0.0017</td>
      <td>0.1641</td>
      <td>0.0050</td>
    </tr>
    <tr>
      <th>вакцинация</th>
      <td>0.0060</td>
      <td>0.0061</td>
      <td>-0.0015</td>
      <td>0.2533</td>
      <td>0.0034</td>
    </tr>
    <tr>
      <th>пол</th>
      <td>0.0086</td>
      <td>-0.0031</td>
      <td>-0.0135</td>
      <td>0.1634</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <th>ХГ</th>
      <td>0.0024</td>
      <td>0.0009</td>
      <td>-0.0005</td>
      <td>0.1772</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>ХПН</th>
      <td>0.0001</td>
      <td>0.0004</td>
      <td>-0.0042</td>
      <td>0.1232</td>
      <td>0.0001</td>
    </tr>
  </tbody>
</table>
</div>


## 12. Эмпирическая формула для КТ(max)

В отличие от классификации (КТ(ИТОГ), 5 классов), здесь используем **числовую переменную КТ(max)** (столбец AN, 0–95% поражения лёгких) — это позволит вывести **одну компактную линейную формулу** для расчёта степени поражения, которую может применить врач у постели пациента.

**План:**
1. Обучаем Ridge-регрессию на стандартизованных признаках (даёт устойчивые коэффициенты при коллинеарности)
2. Выводим формулу в двух видах: стандартизованная (для сравнения значимости признаков) и в исходных единицах (для практического применения)
3. Валидация: R², MAE, scatter plot предсказание vs истинное значение
4. Конвертация числового предсказания в категории КТ(ИТОГ) по медицинским порогам — сравнение с реальной классификацией
5. Упрощённая формула из топ-5 признаков (для запоминания)

        ОБЩЕЕ2: train=604, test=151, КТ(max) range [0, 95], mean=16.7


        УХАНЬ2: train=170, test=43, КТ(max) range [0, 95], mean=16.8


       ДЕЛЬТА2: train=217, test=55, КТ(max) range [0, 95], mean=21.9


      ОМИКРОН2: train=216, test=54, КТ(max) range [0, 70], mean=11.4


### 12.1 Обучение и вывод формулы

Для **ОБЩЕЕ2** обучаем Ridge-регрессию (с кросс-валидацией для выбора α). Признаки стандартизуются — это даёт **сопоставимые коэффициенты** (вес признака = его влияние при изменении на 1 стандартное отклонение).

    Выбранная alpha: 100.0
    Intercept (стандартизованная форма): 17.02
    
    === Формула (стандартизованные признаки) ===
    КТ(max) = 17.02
             + 4.742 · z(темпер)
             + 4.207 · z(HOMA)
             + 3.074 · z(день заболевания)
             - 2.707 · z(вакцинация)
             + 2.484 · z(АД(сис))
             + 1.601 · z(возраст)
             + 1.549 · z(ИМТ)
             + 1.244 · z(ЖКТ)
             + 1.101 · z(ХГ)
             + 1.048 · z(пол)
             + 0.997 · z(АД(диас))
             + 0.754 · z(Орг. Дых)
             - 0.614 · z(А/Т IgG)
             + 0.572 · z(ЧСС (пост))
             + 0.351 · z(ХПН)
             + 0.283 · z(СД)
             - 0.125 · z(группа крови)
    где z(x) = (x - mean) / std
    
    === Формула (исходные единицы) ===
    КТ(max) = -232.27
             - 5.4810 · вакцинация
             + 5.4248 · ХГ
             + 5.0892 · темпер
             + 3.7864 · Орг. Дых
             + 2.7538 · ХПН
             + 2.5047 · ЖКТ
             + 2.1695 · HOMA
             + 2.1347 · пол
             + 1.0136 · день заболевания
             + 0.8145 · СД
             + 0.4000 · ИМТ
             + 0.1499 · АД(сис)
             - 0.1349 · группа крови
             + 0.1261 · возраст
             + 0.0714 · АД(диас)
             + 0.0129 · ЧСС (пост)
             - 0.0015 · А/Т IgG


### 12.2 Валидация формулы

Точность в числовой форме: **R²** (доля объяснённой дисперсии), **MAE** (средняя ошибка в %).

    R² (test): 0.431
    MAE (test): 9.71% поражения лёгких



    
![png](covid_analysis_updated_files/covid_analysis_updated_80_1.png)
    


### 12.3 Конвертация в категории КТ(ИТОГ)

Применяем стандартные медицинские пороги к предсказанному КТ(max) и сравниваем с реальной КТ(ИТОГ):
- 0% → КТ-0 (норма)
- 1–25% → КТ-1 (минимальное)
- 26–50% → КТ-2 (умеренное)
- 51–75% → КТ-3 (тяжёлое)
- 76–100% → КТ-4 (критическое)

    Точность по категориям после конвертации: 0.606
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.67      0.36      0.47        11
               1       0.64      0.82      0.72        17
               2       0.40      1.00      0.57         2
               3       0.00      0.00      0.00         2
               4       0.00      0.00      0.00         1
    
        accuracy                           0.61        33
       macro avg       0.34      0.44      0.35        33
    weighted avg       0.57      0.61      0.56        33
    



    
![png](covid_analysis_updated_files/covid_analysis_updated_82_1.png)
    


    
    Для сравнения — лучшая модель классификации:
      лучшая classification (RF/GB/LR/DT) на ОБЩЕЕ2: Accuracy = 0.603


### 12.4 Упрощённая формула — топ-5 признаков

Для **практического применения** часто нужна формула из 3–5 признаков, которые врач сможет запомнить и применить «у постели». Берём топ-5 по консенсус permutation importance и переобучаем.

    Топ-5 признаков: ['день заболевания', 'АД(сис)', 'ЧСС (пост)', 'ИМТ', 'возраст']
    
    === Упрощённая формула (5 признаков, исходные единицы) ===
    КТ(max) ≈ -65.46
               + 1.317 · день заболевания
               + 0.992 · ИМТ
               + 0.284 · АД(сис)
               + 0.197 · возраст
               + 0.030 · ЧСС (пост)
    
    R²:  0.311  (полная: 0.431)
    MAE: 10.96%  (полная: 9.71%)
    
    === Сравнение полной vs упрощённой формулы ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Полная (16 признаков)</th>
      <th>Упрощённая (5 признаков)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R²</th>
      <td>0.431</td>
      <td>0.311</td>
    </tr>
    <tr>
      <th>MAE (%)</th>
      <td>9.706</td>
      <td>10.961</td>
    </tr>
    <tr>
      <th>Число признаков</th>
      <td>17.000</td>
      <td>5.000</td>
    </tr>
  </tbody>
</table>
</div>


### 12.6 Логарифмирование скошенных признаков

Многие лабораторные и метаболические показатели имеют **лог-нормальное распределение** (хвост в сторону больших значений): цитокины, ферритин, д-димер, ЛДГ, СРБ, HOMA. Линейная Ridge-регрессия на исходных значениях недооценивает крайние значения — а именно они отличают тяжёлые случаи КТ-3/КТ-4.

**Идея:** заменить `x` на `log(1 + x)` для признаков с асимметрией > 1.5. Формула остаётся линейной по log-признакам, что клинически интерпретируется как «риск растёт мультипликативно при росте маркера».

    Признаков с log-преобразованием (3): ['ЧСС (пост)', 'А/Т IgG', 'HOMA']
    
    Ridge на log-признаках:
      alpha: 100.0
      R²:  0.455  (без log: 0.431, Δ=+0.024)
      MAE: 9.70%  (без log: 9.71%, Δ=-0.01)
      Точность КТ-категорий: 0.576  (без log: 0.606)
      Macro-F1 КТ-категорий: 0.305
    
    === Формула 12.6 (log + стандартизованные признаки) ===
    КТ(max) = 17.02
             + 4.914 · z(log(1 + HOMA))
             + 4.040 · z(темпер)
             + 2.869 · z(день заболевания)
             + 2.670 · z(log(1 + ЧСС (пост)))
             + 2.266 · z(АД(сис))
             - 2.022 · z(вакцинация)
             + 1.574 · z(возраст)
             - 1.532 · z(log(1 + А/Т IgG))
             + 1.020 · z(пол)
             + 0.938 · z(ИМТ)
             + 0.926 · z(ХГ)
             + 0.912 · z(ЖКТ)
             + 0.837 · z(Орг. Дых)
             + 0.292 · z(АД(диас))
             + 0.216 · z(ХПН)
             - 0.130 · z(группа крови)
             + 0.072 · z(СД)
    где z(·) = (· - mean) / std (по train)
    
    === Формула 12.6 (исходные единицы) ===
    КТ(max) = -250.86
             + 12.2492 · log(1 + ЧСС (пост))
             + 10.4893 · log(1 + HOMA)
             + 4.5620 · ХГ
             + 4.3361 · темпер
             + 4.1999 · Орг. Дых
             - 4.0933 · вакцинация
             + 2.0772 · пол
             + 1.8353 · ЖКТ
             + 1.6933 · ХПН
             + 0.9458 · день заболевания
             - 0.5663 · log(1 + А/Т IgG)
             + 0.2424 · ИМТ
             + 0.2060 · СД
             - 0.1410 · группа крови
             + 0.1367 · АД(сис)
             + 0.1240 · возраст
             + 0.0209 · АД(диас)


### 12.7 Sample weights — фокус на тяжёлых случаях

Стандартная Ridge минимизирует MSE равномерно по всем пациентам. При перекосе классов (КТ-1/КТ-2 — большинство, КТ-3/КТ-4 — единицы) это приводит к **усадке предсказаний к среднему** ≈18% — крайние значения недостижимы, формула почти никогда не предсказывает >50%.

**Решение:** взвесить наблюдения по обратной частоте КТ-категории. Тяжёлые пациенты получают больший вес — формула учится предсказывать высокие значения, ценой небольшой потери точности на массовых КТ-1.

    Распределение категорий в train:
      КТ-0: 243 пациентов
      КТ-1: 199 пациентов
      КТ-2: 111 пациентов
      КТ-3: 40 пациентов
      КТ-4: 11 пациентов


    
    Ridge с sample_weight:
      R²:  0.170  (без весов: 0.431, Δ=-0.261)
      MAE: 12.78%  (без весов: 9.71%, Δ=+3.07)
      Точность КТ-категорий: 0.364  (без весов: 0.606)
      Macro-F1 КТ-категорий: 0.271
    
    Classification report (с sample_weight):
                  precision    recall  f1-score   support
    
               0       1.00      0.27      0.43        11
               1       0.47      0.41      0.44        17
               2       0.09      0.50      0.15         2
               3       0.25      0.50      0.33         2
               4       0.00      0.00      0.00         1
    
        accuracy                           0.36        33
       macro avg       0.36      0.34      0.27        33
    weighted avg       0.59      0.36      0.40        33
    


### 12.8 Квантильная регрессия (τ=0.9) — верхняя граница риска

Линейная регрессия предсказывает **условное среднее** КТ(max). Для клиники полезнее знать **потенциал тяжести** — какая степень поражения возможна с вероятностью 90%? Это даёт **квантильная регрессия** с τ=0.9: предсказание накрывает 90% реальных значений сверху.

Возвращаем три формулы: τ=0.5 (медиана, аналог Ridge), τ=0.9 (верхняя граница), τ=0.1 (нижняя граница) — это формирует **прогностический интервал** вместо точечного числа.

    Квантильная регрессия:
      τ=0.1: R²=-0.685, MAE=15.47%, доля под кривой=1.00
      τ=0.5: R²=0.394, MAE=9.79%, доля под кривой=0.59
      τ=0.9: R²=-0.338, MAE=17.37%, доля под кривой=0.85
    
    Категоризация τ=0.9 vs реальная КТ(ИТОГ):
      Accuracy: 0.152
      Macro-F1: 0.074



    
![png](covid_analysis_updated_files/covid_analysis_updated_90_1.png)
    


    
    === Формула верхней границы τ=0.9 (исходные единицы), топ-7 ===
    КТ(max, τ=0.9) ≈ -282.01
                 + 7.7501 · Орг. Дых
                 - 6.4011 · вакцинация
                 + 6.0757 · темпер
                 + 4.5986 · ХГ
                 + 4.2515 · HOMA
                 + 3.3241 · СД
                 + 0.9466 · ЖКТ


### 12.9 Комбинированная формула: log + sample_weight + квантиль

Объединяем три приёма: log-преобразование скошенных лабораторных показателей, sample weights для редких категорий, и параллельно квантильную регрессию для верхней границы. Это итоговая клинически применимая модель.

    === Комбинированная Ridge (log + sample_weight) ===
      alpha: 100.0
      R²: 0.194, MAE: 12.33%
      Acc категорий: 0.485, Macro-F1: 0.328
    
    === Формула 12.9 (log + веса, стандартизованные признаки) ===
    КТ(max) = 24.37
             + 8.096 · z(log(1 + HOMA))
             + 6.976 · z(log(1 + ЧСС (пост)))
             + 6.145 · z(темпер)
             - 4.197 · z(вакцинация)
             + 4.073 · z(возраст)
             - 2.813 · z(группа крови)
             - 2.703 · z(log(1 + А/Т IgG))
             + 2.569 · z(ЖКТ)
             + 1.842 · z(Орг. Дых)
             + 1.629 · z(день заболевания)
             + 1.570 · z(АД(сис))
             + 1.060 · z(пол)
             - 0.541 · z(СД)
             - 0.476 · z(ХГ)
             - 0.416 · z(ХПН)
             + 0.107 · z(АД(диас))
             - 0.006 · z(ИМТ)
    где z(·) = (· - mean) / std (по train)
    
    === Формула 12.9 (исходные единицы) ===
    КТ(max) = -409.49
             + 32.0020 · log(1 + ЧСС (пост))
             + 17.2811 · log(1 + HOMA)
             + 9.2470 · Орг. Дых
             - 8.4969 · вакцинация
             + 6.5952 · темпер
             + 5.1700 · ЖКТ
             - 3.2593 · ХПН
             - 3.0405 · группа крови
             - 2.3471 · ХГ
             + 2.1580 · пол
             - 1.5554 · СД
             - 0.9992 · log(1 + А/Т IgG)
             + 0.5370 · день заболевания
             + 0.3209 · возраст
             + 0.0947 · АД(сис)
             + 0.0076 · АД(диас)
             - 0.0016 · ИМТ
    
    === Упрощённая формула 12.9 — топ-5 признаков ===
    (топ-5 по |стандартизованному коэф.|: ['HOMA', 'ЧСС (пост)', 'темпер', 'вакцинация', 'возраст'])
    КТ(max) ≈ -473.18
               + 41.210 · log(1 + ЧСС (пост))
               + 22.132 · log(1 + HOMA)
               - 14.016 · вакцинация
               + 7.406 · темпер
               + 0.380 · возраст
    
    R²: 0.133, MAE: 12.72%, Acc: 0.424, Macro-F1: 0.231
    
    === Comparison: все варианты ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Базовая Ridge</th>
      <th>12.6 +log</th>
      <th>12.7 +веса</th>
      <th>12.8 квантиль τ=0.9</th>
      <th>12.9 log+веса</th>
      <th>12.9 log+веса+τ=0.9</th>
      <th>12.9 упрощённая (5 призн.)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R²</th>
      <td>0.431</td>
      <td>0.455</td>
      <td>0.170</td>
      <td>-0.338</td>
      <td>0.194</td>
      <td>-1.697</td>
      <td>0.133</td>
    </tr>
    <tr>
      <th>MAE (%)</th>
      <td>9.706</td>
      <td>9.697</td>
      <td>12.779</td>
      <td>17.367</td>
      <td>12.330</td>
      <td>26.428</td>
      <td>12.722</td>
    </tr>
    <tr>
      <th>Точность категорий</th>
      <td>0.606</td>
      <td>0.576</td>
      <td>0.364</td>
      <td>0.152</td>
      <td>0.485</td>
      <td>0.152</td>
      <td>0.424</td>
    </tr>
    <tr>
      <th>Macro-F1</th>
      <td>0.352</td>
      <td>0.305</td>
      <td>0.271</td>
      <td>0.074</td>
      <td>0.328</td>
      <td>0.311</td>
      <td>0.231</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_analysis_updated_files/covid_analysis_updated_92_2.png)
    


    
    Лучшая формула 12.9 по Macro-F1: 12.9 log + веса
      Accuracy: 0.485, Macro-F1: 0.328



    
![png](covid_analysis_updated_files/covid_analysis_updated_92_4.png)
    


    
    Classification report (12.9 log + веса):
                  precision    recall  f1-score   support
    
               0       0.80      0.36      0.50        11
               1       0.62      0.59      0.61        17
               2       0.12      0.50      0.20         2
               3       0.25      0.50      0.33         2
               4       0.00      0.00      0.00         1
    
        accuracy                           0.48        33
       macro avg       0.36      0.39      0.33        33
    weighted avg       0.61      0.48      0.51        33
    


## 13. Выводы

    ======================================================================
      ИТОГОВЫЕ ВЫВОДЫ
    ======================================================================
    
    1. Лучшая модель по Accuracy для каждого штамма:
           ОБЩЕЕ2: Random Forest (Accuracy=0.6026, F1w=0.6024)
           УХАНЬ2: Random Forest (Accuracy=0.4419, F1w=0.4232)
          ДЕЛЬТА2: Logistic Regression (Accuracy=0.4364, F1w=0.4359)
         ОМИКРОН2: Gradient Boosting (Accuracy=0.7222, F1w=0.7241)
    
    2. Средние метрики по всем штаммам:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy (test)</th>
      <th>F1 test (weighted)</th>
      <th>F1 test (macro)</th>
    </tr>
    <tr>
      <th>Модель</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.5210</td>
      <td>0.5217</td>
      <td>0.3943</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.5145</td>
      <td>0.5202</td>
      <td>0.4587</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.5186</td>
      <td>0.5158</td>
      <td>0.3916</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.4136</td>
      <td>0.4306</td>
      <td>0.2986</td>
    </tr>
  </tbody>
</table>
</div>


    
    3. Топ-5 признаков по консенсус permutation importance (LR + RF + GB):
       день заболевания: 0.0510
       АД(сис): 0.0130
       ЧСС (пост): 0.0098
       ИМТ: 0.0094
       возраст: 0.0084
    
    4. Топ-3 признаков по консенсус-важности для каждого штамма:
           ОБЩЕЕ2: день заболевания (0.083), темпер (0.036), HOMA (0.014)
           УХАНЬ2: возраст (0.007), АД(диас) (0.002), Орг. Дых (0.002)
          ДЕЛЬТА2: день заболевания (0.043), возраст (0.024), АД(сис) (0.015)
         ОМИКРОН2: день заболевания (0.080), ЧСС (пост) (0.079), HOMA (0.049)

