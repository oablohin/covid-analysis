# Анализ поздних (постковидных) осложнений

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


## 0.1 Подготовка окружения

    Окружение готово


## 0.2 Загрузка данных и формирование переменных

Загружаем лист `0бщ.755`. Осложнения приводим к бинарному виду (0 — нет, 1 — есть; значения 2+
означают несколько эпизодов и тоже считаются «есть»). Группа тяжести КТ формируется из `КТ(ИТОГ)`
объединением степеней 3 и 4 (КТ3–4) ввиду малочисленности крайней группы.

    Загружено: 755 пациентов, 168 столбцов
    Группы тяжести: {'КТ0': 303, 'КТ1': 246, 'КТ2': 142, 'КТ3-4': 64}


### Факторы риска

Непрерывные факторы используются «как есть» (в OR-оценках стандартизуются — OR на 1 стандартное
отклонение), бинарные — 0/1. Пол кодируется как «женский = 1». Бактериальные осложнения и
дыхательную недостаточность бинаризуем (наличие / отсутствие). Дополнительно вводится суммарное
число сопутствующих заболеваний.

    Факторов риска: 17



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
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Возраст</th>
      <td>37.866</td>
      <td>12.789</td>
      <td>18.000</td>
      <td>80.000</td>
    </tr>
    <tr>
      <th>Пол (жен.)</th>
      <td>0.408</td>
      <td>0.492</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>27.106</td>
      <td>3.922</td>
      <td>17.800</td>
      <td>40.200</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>2.031</td>
      <td>1.884</td>
      <td>0.330</td>
      <td>16.210</td>
    </tr>
    <tr>
      <th>АД сист.</th>
      <td>139.311</td>
      <td>16.787</td>
      <td>100.000</td>
      <td>212.000</td>
    </tr>
    <tr>
      <th>АД диаст.</th>
      <td>99.196</td>
      <td>14.400</td>
      <td>60.000</td>
      <td>149.000</td>
    </tr>
    <tr>
      <th>Температура</th>
      <td>37.992</td>
      <td>0.901</td>
      <td>28.400</td>
      <td>40.300</td>
    </tr>
    <tr>
      <th>Вакцинация</th>
      <td>0.428</td>
      <td>0.495</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Бактер. осложн.</th>
      <td>0.297</td>
      <td>0.457</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>ОРДС</th>
      <td>0.098</td>
      <td>0.298</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Дых. недост. (ДН)</th>
      <td>0.257</td>
      <td>0.437</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>ГКС (гормоны)</th>
      <td>0.405</td>
      <td>0.491</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Сопут.: ССС</th>
      <td>0.368</td>
      <td>0.483</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Сопут.: орг. дых.</th>
      <td>0.046</td>
      <td>0.210</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Сопут.: СД</th>
      <td>0.135</td>
      <td>0.342</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Сопут.: ЖКТ</th>
      <td>0.530</td>
      <td>0.499</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Число сопутств.</th>
      <td>1.400</td>
      <td>1.240</td>
      <td>0.000</td>
      <td>5.000</td>
    </tr>
  </tbody>
</table>
</div>


### Вспомогательные функции (OR, тренд)

    OK


## 0.3 Описание выборки осложнений


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
      <th>Случаев</th>
      <th>Частота, %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Любое осложнение</th>
      <td>357</td>
      <td>47.300</td>
    </tr>
    <tr>
      <th>ЖКТ / НЖБП</th>
      <td>191</td>
      <td>25.300</td>
    </tr>
    <tr>
      <th>Повторные пневмонии</th>
      <td>171</td>
      <td>22.600</td>
    </tr>
    <tr>
      <th>Зрение / слух</th>
      <td>147</td>
      <td>19.500</td>
    </tr>
    <tr>
      <th>ССЗ (новые)</th>
      <td>111</td>
      <td>14.700</td>
    </tr>
    <tr>
      <th>Психика / ЦНС / обоняние</th>
      <td>78</td>
      <td>10.300</td>
    </tr>
    <tr>
      <th>Операции на венах</th>
      <td>52</td>
      <td>6.900</td>
    </tr>
    <tr>
      <th>СД / эндокринные</th>
      <td>38</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>Пневмофиброз</th>
      <td>35</td>
      <td>4.600</td>
    </tr>
    <tr>
      <th>ОНМК / ИМ</th>
      <td>26</td>
      <td>3.400</td>
    </tr>
    <tr>
      <th>ХОБЛ / бр. астма</th>
      <td>24</td>
      <td>3.200</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_10_1.png)
    


# ЭТАП 1. Осложнения в зависимости от тяжести (КТ)

Оцениваем, как частота каждого осложнения меняется по группам тяжести лёгочного поражения,
и проверяем статистическую значимость тренда.

## 1.1 Частота осложнений по группам КТ

    Частота осложнения, % по группам тяжести КТ:



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
      <th>КТ0 (n=303)</th>
      <th>КТ1 (n=246)</th>
      <th>КТ2 (n=142)</th>
      <th>КТ3-4 (n=64)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Пневмофиброз</th>
      <td>0.700</td>
      <td>2.000</td>
      <td>6.300</td>
      <td>29.700</td>
    </tr>
    <tr>
      <th>Повторные пневмонии</th>
      <td>15.200</td>
      <td>24.000</td>
      <td>27.500</td>
      <td>42.200</td>
    </tr>
    <tr>
      <th>ХОБЛ / бр. астма</th>
      <td>1.300</td>
      <td>2.400</td>
      <td>4.200</td>
      <td>12.500</td>
    </tr>
    <tr>
      <th>ЖКТ / НЖБП</th>
      <td>11.200</td>
      <td>26.800</td>
      <td>41.500</td>
      <td>50.000</td>
    </tr>
    <tr>
      <th>СД / эндокринные</th>
      <td>3.000</td>
      <td>2.400</td>
      <td>11.300</td>
      <td>10.900</td>
    </tr>
    <tr>
      <th>ССЗ (новые)</th>
      <td>9.200</td>
      <td>14.200</td>
      <td>22.500</td>
      <td>25.000</td>
    </tr>
    <tr>
      <th>Операции на венах</th>
      <td>2.600</td>
      <td>6.100</td>
      <td>10.600</td>
      <td>21.900</td>
    </tr>
    <tr>
      <th>ОНМК / ИМ</th>
      <td>1.000</td>
      <td>1.600</td>
      <td>7.700</td>
      <td>12.500</td>
    </tr>
    <tr>
      <th>Зрение / слух</th>
      <td>7.300</td>
      <td>17.500</td>
      <td>31.000</td>
      <td>59.400</td>
    </tr>
    <tr>
      <th>Психика / ЦНС / обоняние</th>
      <td>5.300</td>
      <td>11.800</td>
      <td>12.700</td>
      <td>23.400</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_14_0.png)
    


## 1.2 Тест тренда «тяжесть → риск»


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
      <th>Осложнение</th>
      <th>OR на ступень КТ</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Пневмофиброз</td>
      <td>4.360</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ОНМК / ИМ</td>
      <td>2.644</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Зрение / слух</td>
      <td>2.545</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ХОБЛ / бр. астма</td>
      <td>2.198</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Операции на венах</td>
      <td>2.123</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ЖКТ / НЖБП</td>
      <td>2.064</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>6</th>
      <td>СД / эндокринные</td>
      <td>1.841</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Психика / ЦНС / обоняние</td>
      <td>1.656</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ССЗ (новые)</td>
      <td>1.549</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Повторные пневмонии</td>
      <td>1.533</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_16_1.png)
    


Все осложнения демонстрируют статистически значимый рост частоты с увеличением тяжести КТ
(p < 0.001). Наиболее «зависимы от тяжести» пневмофиброз, ОНМК/ИМ, зрение/слух, ХОБЛ/астма и операции на венах.

## 1.3 Основные признаки, влияющие на риск поздних осложнений

За исход берём общий флаг «любое позднее осложнение» (≈47% пациентов). Сначала однофакторные OR,
затем многофакторная логистическая модель (скорректированные OR) и, для устойчивости, важность
признаков по случайному лесу с кросс-валидацией.

    Однофакторные OR факторов риска для «любого позднего осложнения»:



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
      <th>Фактор</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
      <th>знач.</th>
      <th>ед.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Возраст</td>
      <td>3.042</td>
      <td>2.511</td>
      <td>3.685</td>
      <td>0.000</td>
      <td>***</td>
      <td>на 1 SD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ИМТ</td>
      <td>2.956</td>
      <td>2.429</td>
      <td>3.597</td>
      <td>0.000</td>
      <td>***</td>
      <td>на 1 SD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HOMA</td>
      <td>5.560</td>
      <td>4.030</td>
      <td>7.671</td>
      <td>0.000</td>
      <td>***</td>
      <td>на 1 SD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ГКС (гормоны)</td>
      <td>5.108</td>
      <td>3.725</td>
      <td>7.004</td>
      <td>0.000</td>
      <td>***</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Число сопутств.</td>
      <td>2.349</td>
      <td>1.978</td>
      <td>2.789</td>
      <td>0.000</td>
      <td>***</td>
      <td>на 1 SD</td>
    </tr>
    <tr>
      <th>5</th>
      <td>АД сист.</td>
      <td>2.339</td>
      <td>1.961</td>
      <td>2.791</td>
      <td>0.000</td>
      <td>***</td>
      <td>на 1 SD</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Бактер. осложн.</td>
      <td>4.787</td>
      <td>3.392</td>
      <td>6.756</td>
      <td>0.000</td>
      <td>***</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Дых. недост. (ДН)</td>
      <td>5.419</td>
      <td>3.734</td>
      <td>7.865</td>
      <td>0.000</td>
      <td>***</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>АД диаст.</td>
      <td>2.103</td>
      <td>1.775</td>
      <td>2.492</td>
      <td>0.000</td>
      <td>***</td>
      <td>на 1 SD</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Сопут.: ЖКТ</td>
      <td>3.272</td>
      <td>2.424</td>
      <td>4.416</td>
      <td>0.000</td>
      <td>***</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Сопут.: ССС</td>
      <td>3.354</td>
      <td>2.460</td>
      <td>4.574</td>
      <td>0.000</td>
      <td>***</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Сопут.: СД</td>
      <td>5.636</td>
      <td>3.374</td>
      <td>9.412</td>
      <td>0.000</td>
      <td>***</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ОРДС</td>
      <td>11.057</td>
      <td>5.227</td>
      <td>23.390</td>
      <td>0.000</td>
      <td>***</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Температура</td>
      <td>1.633</td>
      <td>1.385</td>
      <td>1.926</td>
      <td>0.000</td>
      <td>***</td>
      <td>на 1 SD</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Сопут.: орг. дых.</td>
      <td>2.536</td>
      <td>1.224</td>
      <td>5.254</td>
      <td>0.012</td>
      <td>*</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Пол (жен.)</td>
      <td>1.150</td>
      <td>0.860</td>
      <td>1.538</td>
      <td>0.345</td>
      <td></td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Вакцинация</td>
      <td>0.963</td>
      <td>0.722</td>
      <td>1.286</td>
      <td>0.799</td>
      <td></td>
      <td>0/1</td>
    </tr>
  </tbody>
</table>
</div>


    Многофакторная модель: скорректированные OR (псевдо-R2 = 0.298)



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
      <th>OR (скорр.)</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Возраст</th>
      <td>2.158</td>
      <td>1.687</td>
      <td>2.759</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>2.552</td>
      <td>1.777</td>
      <td>3.663</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>ГКС (гормоны)</th>
      <td>2.431</td>
      <td>1.508</td>
      <td>3.919</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>1.344</td>
      <td>1.024</td>
      <td>1.765</td>
      <td>0.033</td>
      <td>*</td>
    </tr>
    <tr>
      <th>Сопут.: ССС</th>
      <td>0.550</td>
      <td>0.308</td>
      <td>0.980</td>
      <td>0.042</td>
      <td>*</td>
    </tr>
    <tr>
      <th>ОРДС</th>
      <td>1.794</td>
      <td>0.712</td>
      <td>4.516</td>
      <td>0.215</td>
      <td></td>
    </tr>
    <tr>
      <th>Бактер. осложн.</th>
      <td>1.310</td>
      <td>0.806</td>
      <td>2.128</td>
      <td>0.276</td>
      <td></td>
    </tr>
    <tr>
      <th>Пол (жен.)</th>
      <td>1.147</td>
      <td>0.767</td>
      <td>1.715</td>
      <td>0.504</td>
      <td></td>
    </tr>
    <tr>
      <th>АД диаст.</th>
      <td>0.899</td>
      <td>0.638</td>
      <td>1.267</td>
      <td>0.544</td>
      <td></td>
    </tr>
    <tr>
      <th>АД сист.</th>
      <td>1.112</td>
      <td>0.773</td>
      <td>1.599</td>
      <td>0.567</td>
      <td></td>
    </tr>
    <tr>
      <th>Дых. недост. (ДН)</th>
      <td>0.847</td>
      <td>0.478</td>
      <td>1.502</td>
      <td>0.569</td>
      <td></td>
    </tr>
    <tr>
      <th>Сопут.: СД</th>
      <td>1.209</td>
      <td>0.572</td>
      <td>2.552</td>
      <td>0.619</td>
      <td></td>
    </tr>
    <tr>
      <th>Сопут.: орг. дых.</th>
      <td>1.241</td>
      <td>0.461</td>
      <td>3.342</td>
      <td>0.670</td>
      <td></td>
    </tr>
    <tr>
      <th>Температура</th>
      <td>0.956</td>
      <td>0.765</td>
      <td>1.193</td>
      <td>0.688</td>
      <td></td>
    </tr>
    <tr>
      <th>Число сопутств.</th>
      <td>1.064</td>
      <td>0.675</td>
      <td>1.676</td>
      <td>0.789</td>
      <td></td>
    </tr>
    <tr>
      <th>Вакцинация</th>
      <td>1.051</td>
      <td>0.680</td>
      <td>1.624</td>
      <td>0.822</td>
      <td></td>
    </tr>
    <tr>
      <th>Сопут.: ЖКТ</th>
      <td>0.998</td>
      <td>0.566</td>
      <td>1.759</td>
      <td>0.993</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_20_2.png)
    


    RandomForest, CV ROC-AUC = 0.816 ± 0.049



    
![png](covid_complications_files/covid_complications_21_1.png)
    


# ЭТАП 2. Группы риска: влияние факторов на конкретные осложнения

Для каждого осложнения оцениваем, как связаны с ним факторы риска. Базовый инструмент —
отношение шансов (OR): для непрерывных факторов на 1 SD, для бинарных — наличие/отсутствие.

## 2.1 Матрица отношений шансов (фактор × осложнение)

Тепловая карта `log2(OR)`: красный — фактор повышает риск, синий — снижает. Звёздочкой отмечены
значимые связи (p < 0.05).


    
![png](covid_complications_files/covid_complications_24_0.png)
    


    Топ-20 самых сильных значимых связей фактор → осложнение:



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
      <th>Фактор</th>
      <th>Осложнение</th>
      <th>OR</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ОРДС</td>
      <td>Пневмофиброз</td>
      <td>14.358</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Дых. недост. (ДН)</td>
      <td>Пневмофиброз</td>
      <td>13.349</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ГКС (гормоны)</td>
      <td>Пневмофиброз</td>
      <td>9.652</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ОРДС</td>
      <td>ОНМК / ИМ</td>
      <td>9.221</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Бактер. осложн.</td>
      <td>Пневмофиброз</td>
      <td>8.960</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ГКС (гормоны)</td>
      <td>Зрение / слух</td>
      <td>8.232</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ГКС (гормоны)</td>
      <td>ЖКТ / НЖБП</td>
      <td>6.819</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ОРДС</td>
      <td>Зрение / слух</td>
      <td>6.311</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Сопут.: ССС</td>
      <td>Пневмофиброз</td>
      <td>6.306</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Дых. недост. (ДН)</td>
      <td>ОНМК / ИМ</td>
      <td>5.891</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Бактер. осложн.</td>
      <td>ОНМК / ИМ</td>
      <td>5.712</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Дых. недост. (ДН)</td>
      <td>Зрение / слух</td>
      <td>5.587</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Сопут.: ССС</td>
      <td>ХОБЛ / бр. астма</td>
      <td>5.435</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Сопут.: ССС</td>
      <td>СД / эндокринные</td>
      <td>5.230</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Сопут.: ССС</td>
      <td>Зрение / слух</td>
      <td>4.943</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ОРДС</td>
      <td>Операции на венах</td>
      <td>4.943</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Сопут.: ССС</td>
      <td>ОНМК / ИМ</td>
      <td>4.926</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Сопут.: СД</td>
      <td>Зрение / слух</td>
      <td>4.726</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Сопут.: орг. дых.</td>
      <td>ХОБЛ / бр. астма</td>
      <td>4.516</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Сопут.: СД</td>
      <td>ОНМК / ИМ</td>
      <td>4.327</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>


## 2.2 Многофакторная логистическая регрессия (частые исходы)

Для осложнений с достаточным числом случаев (повторные пневмонии, ЖКТ/НЖБП, ССЗ, зрение/слух) строим
многофакторные модели — скорректированные OR показывают независимый вклад каждого фактора.

> **Как читать скорректированные OR.** В многофакторной модели знак эффекта может отличаться от
> однофакторного: `OR < 1` означает, что *при прочих равных* фактор связан с **понижением** шансов.
> Такой разворот знака — следствие взаимной корреляции факторов (отрицательное конфаундинг) либо
> особенностей определения исхода, и не противоречит однофакторным оценкам, а уточняет их.

    
    === Повторные пневмонии  (случаев=171, псевдо-R2=0.097) ===
    Значимые независимые предикторы:



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
      <th>OR (скорр.)</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Дых. недост. (ДН)</th>
      <td>0.366</td>
      <td>0.199</td>
      <td>0.673</td>
      <td>0.001</td>
      <td>**</td>
    </tr>
    <tr>
      <th>Возраст</th>
      <td>1.349</td>
      <td>1.068</td>
      <td>1.705</td>
      <td>0.012</td>
      <td>*</td>
    </tr>
    <tr>
      <th>ГКС (гормоны)</th>
      <td>1.850</td>
      <td>1.115</td>
      <td>3.068</td>
      <td>0.017</td>
      <td>*</td>
    </tr>
    <tr>
      <th>Бактер. осложн.</th>
      <td>1.777</td>
      <td>1.090</td>
      <td>2.897</td>
      <td>0.021</td>
      <td>*</td>
    </tr>
  </tbody>
</table>
</div>


    
    === ЖКТ / НЖБП  (случаев=191, псевдо-R2=0.250) ===
    Значимые независимые предикторы:



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
      <th>OR (скорр.)</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ГКС (гормоны)</th>
      <td>5.636</td>
      <td>3.280</td>
      <td>9.684</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>1.664</td>
      <td>1.265</td>
      <td>2.190</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>АД сист.</th>
      <td>1.638</td>
      <td>1.128</td>
      <td>2.377</td>
      <td>0.009</td>
      <td>**</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>1.296</td>
      <td>1.030</td>
      <td>1.630</td>
      <td>0.027</td>
      <td>*</td>
    </tr>
    <tr>
      <th>Дых. недост. (ДН)</th>
      <td>0.554</td>
      <td>0.311</td>
      <td>0.988</td>
      <td>0.045</td>
      <td>*</td>
    </tr>
  </tbody>
</table>
</div>


    
    === ССЗ (новые)  (случаев=111, псевдо-R2=0.272) ===
    Значимые независимые предикторы:



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
      <th>OR (скорр.)</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Сопут.: ССС</th>
      <td>0.031</td>
      <td>0.012</td>
      <td>0.078</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>Возраст</th>
      <td>2.546</td>
      <td>1.858</td>
      <td>3.488</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>1.822</td>
      <td>1.305</td>
      <td>2.542</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>1.487</td>
      <td>1.159</td>
      <td>1.907</td>
      <td>0.002</td>
      <td>**</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Зрение / слух  (случаев=147, псевдо-R2=0.243) ===
    Значимые независимые предикторы:



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
      <th>OR (скорр.)</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ГКС (гормоны)</th>
      <td>3.066</td>
      <td>1.718</td>
      <td>5.472</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>Возраст</th>
      <td>1.652</td>
      <td>1.249</td>
      <td>2.185</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>Вакцинация</th>
      <td>0.460</td>
      <td>0.265</td>
      <td>0.797</td>
      <td>0.006</td>
      <td>**</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_28_0.png)
    


## 2.3 Фокус: ОНМК / ИМ

Самое тяжёлое позднее осложнение (26 случаев, 3.4%). Ввиду редкости приводим профиль группы риска
(сравнение факторов у пациентов с осложнением и без) и однофакторные OR; многофакторные оценки
интерпретируем осторожно.

    Профиль группы риска ОНМК/ИМ (среднее для непрерывных, доля для бинарных):



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
      <th>Фактор</th>
      <th>ОНМК/ИМ (есть)</th>
      <th>ОНМК/ИМ (нет)</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HOMA</td>
      <td>4.9</td>
      <td>1.9</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Возраст</td>
      <td>50.8</td>
      <td>37.4</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ОРДС</td>
      <td>46%</td>
      <td>9%</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Число сопутств.</td>
      <td>2.6</td>
      <td>1.4</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>4</th>
      <td>АД сист.</td>
      <td>156.9</td>
      <td>138.7</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Дых. недост. (ДН)</td>
      <td>65%</td>
      <td>24%</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ИМТ</td>
      <td>31.3</td>
      <td>27.0</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Бактер. осложн.</td>
      <td>69%</td>
      <td>28%</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>8</th>
      <td>АД диаст.</td>
      <td>111.7</td>
      <td>98.8</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Сопут.: ССС</td>
      <td>73%</td>
      <td>36%</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Температура</td>
      <td>38.6</td>
      <td>38.0</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Сопут.: СД</td>
      <td>38%</td>
      <td>13%</td>
      <td>0.001</td>
      <td>**</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Сопут.: ЖКТ</td>
      <td>81%</td>
      <td>52%</td>
      <td>0.004</td>
      <td>**</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Сопут.: орг. дых.</td>
      <td>15%</td>
      <td>4%</td>
      <td>0.028</td>
      <td>*</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ГКС (гормоны)</td>
      <td>62%</td>
      <td>40%</td>
      <td>0.040</td>
      <td>*</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Вакцинация</td>
      <td>38%</td>
      <td>43%</td>
      <td>0.692</td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Пол (жен.)</td>
      <td>42%</td>
      <td>41%</td>
      <td>1.000</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Фактор</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
      <th>знач.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ОРДС</td>
      <td>9.221</td>
      <td>4.087</td>
      <td>20.807</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Дых. недост. (ДН)</td>
      <td>5.891</td>
      <td>2.580</td>
      <td>13.449</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Бактер. осложн.</td>
      <td>5.712</td>
      <td>2.446</td>
      <td>13.342</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Сопут.: ССС</td>
      <td>4.926</td>
      <td>2.044</td>
      <td>11.872</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Сопут.: СД</td>
      <td>4.327</td>
      <td>1.906</td>
      <td>9.823</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Сопут.: орг. дых.</td>
      <td>4.094</td>
      <td>1.330</td>
      <td>12.603</td>
      <td>0.014</td>
      <td>*</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Сопут.: ЖКТ</td>
      <td>3.879</td>
      <td>1.447</td>
      <td>10.397</td>
      <td>0.007</td>
      <td>**</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Возраст</td>
      <td>2.819</td>
      <td>1.872</td>
      <td>4.247</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>8</th>
      <td>АД диаст.</td>
      <td>2.793</td>
      <td>1.770</td>
      <td>4.405</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ИМТ</td>
      <td>2.696</td>
      <td>1.858</td>
      <td>3.911</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>10</th>
      <td>АД сист.</td>
      <td>2.452</td>
      <td>1.729</td>
      <td>3.478</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Число сопутств.</td>
      <td>2.450</td>
      <td>1.689</td>
      <td>3.554</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ГКС (гормоны)</td>
      <td>2.422</td>
      <td>1.084</td>
      <td>5.411</td>
      <td>0.031</td>
      <td>*</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Температура</td>
      <td>2.216</td>
      <td>1.458</td>
      <td>3.369</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HOMA</td>
      <td>2.014</td>
      <td>1.588</td>
      <td>2.555</td>
      <td>0.000</td>
      <td>***</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Пол (жен.)</td>
      <td>1.067</td>
      <td>0.483</td>
      <td>2.355</td>
      <td>0.873</td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Вакцинация</td>
      <td>0.831</td>
      <td>0.372</td>
      <td>1.855</td>
      <td>0.651</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_31_1.png)
    


## 2.4 Редкие осложнения — однофакторные OR

Пневмофиброз, ХОБЛ/астма, СД/эндокринные, операции на венах — < ~50 событий. Приводим только
значимые однофакторные связи; многофакторные ML-модели для них статистически не обоснованы.

    
    === Пневмофиброз  (случаев = 35) — значимые факторы ===



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
      <th>Фактор</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ОРДС</td>
      <td>14.358</td>
      <td>6.991</td>
      <td>29.487</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Дых. недост. (ДН)</td>
      <td>13.349</td>
      <td>5.727</td>
      <td>31.114</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ГКС (гормоны)</td>
      <td>9.652</td>
      <td>3.701</td>
      <td>25.173</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Бактер. осложн.</td>
      <td>8.960</td>
      <td>4.003</td>
      <td>20.057</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Сопут.: ССС</td>
      <td>6.306</td>
      <td>2.823</td>
      <td>14.087</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Сопут.: орг. дых.</td>
      <td>3.833</td>
      <td>1.389</td>
      <td>10.576</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Сопут.: СД</td>
      <td>3.168</td>
      <td>1.501</td>
      <td>6.685</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Сопут.: ЖКТ</td>
      <td>3.140</td>
      <td>1.407</td>
      <td>7.004</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Температура</td>
      <td>2.467</td>
      <td>1.701</td>
      <td>3.577</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>АД диаст.</td>
      <td>2.223</td>
      <td>1.516</td>
      <td>3.260</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>АД сист.</td>
      <td>2.146</td>
      <td>1.581</td>
      <td>2.914</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Число сопутств.</td>
      <td>2.124</td>
      <td>1.544</td>
      <td>2.923</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ИМТ</td>
      <td>2.053</td>
      <td>1.491</td>
      <td>2.828</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Возраст</td>
      <td>1.973</td>
      <td>1.405</td>
      <td>2.771</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HOMA</td>
      <td>1.663</td>
      <td>1.343</td>
      <td>2.058</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Вакцинация</td>
      <td>0.381</td>
      <td>0.171</td>
      <td>0.850</td>
      <td>0.018</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Пол (жен.)</td>
      <td>0.348</td>
      <td>0.150</td>
      <td>0.807</td>
      <td>0.014</td>
    </tr>
  </tbody>
</table>
</div>


    
    === ХОБЛ / бр. астма  (случаев = 24) — значимые факторы ===



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
      <th>Фактор</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Сопут.: ССС</td>
      <td>5.435</td>
      <td>2.131</td>
      <td>13.861</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Сопут.: орг. дых.</td>
      <td>4.516</td>
      <td>1.456</td>
      <td>14.011</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Дых. недост. (ДН)</td>
      <td>4.286</td>
      <td>1.871</td>
      <td>9.816</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Сопут.: СД</td>
      <td>4.116</td>
      <td>1.751</td>
      <td>9.674</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Бактер. осложн.</td>
      <td>3.473</td>
      <td>1.519</td>
      <td>7.943</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ОРДС</td>
      <td>3.250</td>
      <td>1.248</td>
      <td>8.463</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Сопут.: ЖКТ</td>
      <td>2.741</td>
      <td>1.076</td>
      <td>6.983</td>
      <td>0.035</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ГКС (гормоны)</td>
      <td>2.520</td>
      <td>1.088</td>
      <td>5.835</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Число сопутств.</td>
      <td>2.173</td>
      <td>1.489</td>
      <td>3.172</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Возраст</td>
      <td>2.161</td>
      <td>1.440</td>
      <td>3.243</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ИМТ</td>
      <td>1.959</td>
      <td>1.345</td>
      <td>2.853</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>АД диаст.</td>
      <td>1.636</td>
      <td>1.062</td>
      <td>2.522</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>12</th>
      <td>АД сист.</td>
      <td>1.622</td>
      <td>1.131</td>
      <td>2.326</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HOMA</td>
      <td>1.493</td>
      <td>1.169</td>
      <td>1.907</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>


    
    === СД / эндокринные  (случаев = 38) — значимые факторы ===



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
      <th>Фактор</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Сопут.: ССС</td>
      <td>5.230</td>
      <td>2.500</td>
      <td>10.943</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Дых. недост. (ДН)</td>
      <td>3.884</td>
      <td>2.004</td>
      <td>7.530</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Сопут.: ЖКТ</td>
      <td>3.517</td>
      <td>1.591</td>
      <td>7.776</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ОРДС</td>
      <td>3.113</td>
      <td>1.413</td>
      <td>6.860</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ГКС (гормоны)</td>
      <td>2.644</td>
      <td>1.345</td>
      <td>5.199</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ИМТ</td>
      <td>2.149</td>
      <td>1.575</td>
      <td>2.931</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Бактер. осложн.</td>
      <td>1.994</td>
      <td>1.031</td>
      <td>3.857</td>
      <td>0.040</td>
    </tr>
    <tr>
      <th>7</th>
      <td>АД диаст.</td>
      <td>1.856</td>
      <td>1.299</td>
      <td>2.650</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>8</th>
      <td>АД сист.</td>
      <td>1.851</td>
      <td>1.381</td>
      <td>2.480</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HOMA</td>
      <td>1.705</td>
      <td>1.382</td>
      <td>2.104</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Возраст</td>
      <td>1.594</td>
      <td>1.156</td>
      <td>2.197</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Число сопутств.</td>
      <td>1.560</td>
      <td>1.152</td>
      <td>2.112</td>
      <td>0.004</td>
    </tr>
  </tbody>
</table>
</div>


    
    === Операции на венах  (случаев = 52) — значимые факторы ===



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
      <th>Фактор</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ОРДС</td>
      <td>4.943</td>
      <td>2.587</td>
      <td>9.442</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Бактер. осложн.</td>
      <td>3.901</td>
      <td>2.188</td>
      <td>6.954</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Дых. недост. (ДН)</td>
      <td>3.774</td>
      <td>2.129</td>
      <td>6.689</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Сопут.: орг. дых.</td>
      <td>3.750</td>
      <td>1.553</td>
      <td>9.054</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Сопут.: ЖКТ</td>
      <td>3.572</td>
      <td>1.806</td>
      <td>7.062</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Сопут.: СД</td>
      <td>3.189</td>
      <td>1.697</td>
      <td>5.991</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ГКС (гормоны)</td>
      <td>2.993</td>
      <td>1.657</td>
      <td>5.406</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Сопут.: ССС</td>
      <td>2.972</td>
      <td>1.665</td>
      <td>5.308</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>АД диаст.</td>
      <td>2.279</td>
      <td>1.645</td>
      <td>3.158</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Число сопутств.</td>
      <td>2.166</td>
      <td>1.652</td>
      <td>2.840</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ИМТ</td>
      <td>2.019</td>
      <td>1.537</td>
      <td>2.652</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>HOMA</td>
      <td>1.991</td>
      <td>1.616</td>
      <td>2.451</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>АД сист.</td>
      <td>1.938</td>
      <td>1.495</td>
      <td>2.514</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Температура</td>
      <td>1.761</td>
      <td>1.303</td>
      <td>2.379</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Возраст</td>
      <td>1.759</td>
      <td>1.327</td>
      <td>2.331</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>


## 2.5 Предсказательные модели (только частые исходы)

Для исходов с достаточным числом случаев оцениваем предсказуемость по факторам риска: логистическая
регрессия и случайный лес, метрика — ROC-AUC по 5-блочной стратифицированной кросс-валидации
(честная оценка без переобучения; единичный train/test-сплит для редких событий неустойчив).


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
      <th>Случаев</th>
      <th>Логит. регрессия</th>
      <th>Случайный лес</th>
    </tr>
    <tr>
      <th>Осложнение</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Любое осложнение</th>
      <td>357</td>
      <td>0.843 ± 0.043</td>
      <td>0.816 ± 0.049</td>
    </tr>
    <tr>
      <th>Повторные пневмонии</th>
      <td>171</td>
      <td>0.684 ± 0.052</td>
      <td>0.660 ± 0.048</td>
    </tr>
    <tr>
      <th>ЖКТ / НЖБП</th>
      <td>191</td>
      <td>0.813 ± 0.032</td>
      <td>0.812 ± 0.032</td>
    </tr>
    <tr>
      <th>ССЗ (новые)</th>
      <td>111</td>
      <td>0.817 ± 0.033</td>
      <td>0.845 ± 0.033</td>
    </tr>
    <tr>
      <th>Зрение / слух</th>
      <td>147</td>
      <td>0.810 ± 0.011</td>
      <td>0.788 ± 0.036</td>
    </tr>
  </tbody>
</table>
</div>


## 2.6 Поиск групп риска для каждого осложнения

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
> коморбидность) дополняют друг друга, а не противоречат.

    Панель показателей и функция групп риска готовы


    Подгруппа максимального риска по каждому осложнению:



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
      <th>Группа риска (правило)</th>
      <th>N в группе</th>
      <th>Событий</th>
      <th>Риск в группе, %</th>
      <th>Базовый риск, %</th>
      <th>Lift</th>
    </tr>
    <tr>
      <th>Осложнение</th>
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
      <th>Пневмофиброз</th>
      <td>ЛДГ &gt; 439 и Д-димер &gt; 0</td>
      <td>101</td>
      <td>27</td>
      <td>26.700</td>
      <td>4.600</td>
      <td>5.800</td>
    </tr>
    <tr>
      <th>Повторные пневмонии</th>
      <td>HOMA &gt; 1 и ИЛ-6 &gt; 5</td>
      <td>174</td>
      <td>75</td>
      <td>43.100</td>
      <td>22.600</td>
      <td>1.900</td>
    </tr>
    <tr>
      <th>ХОБЛ / бр. астма</th>
      <td>СОЭ &gt; 16 и HOMA &gt; 1</td>
      <td>271</td>
      <td>23</td>
      <td>8.500</td>
      <td>3.200</td>
      <td>2.700</td>
    </tr>
    <tr>
      <th>ЖКТ / НЖБП</th>
      <td>ИМТ &gt; 27 и ГКС: да</td>
      <td>227</td>
      <td>120</td>
      <td>52.900</td>
      <td>25.300</td>
      <td>2.100</td>
    </tr>
    <tr>
      <th>СД / эндокринные</th>
      <td>АЛТ &gt; 41 и HOMA &gt; 3</td>
      <td>98</td>
      <td>20</td>
      <td>20.400</td>
      <td>5.000</td>
      <td>4.100</td>
    </tr>
    <tr>
      <th>ССЗ (новые)</th>
      <td>Возраст &gt; 30 и HOMA &gt; 3</td>
      <td>121</td>
      <td>43</td>
      <td>35.500</td>
      <td>14.700</td>
      <td>2.400</td>
    </tr>
    <tr>
      <th>Операции на венах</th>
      <td>Д-димер &gt; 0 и HOMA &gt; 1</td>
      <td>166</td>
      <td>42</td>
      <td>25.300</td>
      <td>6.900</td>
      <td>3.700</td>
    </tr>
    <tr>
      <th>ОНМК / ИМ</th>
      <td>Ферритин &gt; 418 и HOMA &gt; 2</td>
      <td>93</td>
      <td>18</td>
      <td>19.400</td>
      <td>3.400</td>
      <td>5.600</td>
    </tr>
    <tr>
      <th>Зрение / слух</th>
      <td>ГКС: да и Возраст &gt; 42</td>
      <td>176</td>
      <td>84</td>
      <td>47.700</td>
      <td>19.500</td>
      <td>2.500</td>
    </tr>
    <tr>
      <th>Психика / ЦНС / обоняние</th>
      <td>Возраст &gt; 26 и СОЭ &gt; 24</td>
      <td>236</td>
      <td>46</td>
      <td>19.500</td>
      <td>10.300</td>
      <td>1.900</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_39_0.png)
    


Дерево решений выделяет клинически осмысленные группы риска: например, для **ОНМК/ИМ** ключевое
сочетание — высокий ферритин и инсулинорезистентность (HOMA), для **ЖКТ/НЖБП** — избыточная масса
тела и приём ГКС, для **повторных пневмоний** — инсулинорезистентность на фоне высокого ИЛ-6. Это
переводит статистические OR в практические критерии «кого относить к группе риска».

# ЭТАП 3. Клинико-лабораторный разбор конкретных примеров (SHAP)

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
> но величины вкладов для редких исходов следует трактовать качественно.

    Готово к построению SHAP-разборов



    
![png](covid_complications_files/covid_complications_43_0.png)
    



    
![png](covid_complications_files/covid_complications_43_1.png)
    



    
![png](covid_complications_files/covid_complications_43_2.png)
    



    
![png](covid_complications_files/covid_complications_43_3.png)
    



    
![png](covid_complications_files/covid_complications_43_4.png)
    



    
![png](covid_complications_files/covid_complications_43_5.png)
    



    
![png](covid_complications_files/covid_complications_43_6.png)
    



    
![png](covid_complications_files/covid_complications_43_7.png)
    



    
![png](covid_complications_files/covid_complications_43_8.png)
    



    
![png](covid_complications_files/covid_complications_43_9.png)
    


Waterfall-разбор показывает, какие именно показатели и в какую сторону «толкают» риск осложнения
у конкретного пациента, с их фактическими значениями. Видно, что у осложнений с
воспалительно-метаболическим механизмом (ОНМК/ИМ, ССЗ, ЖКТ/НЖБП) риск повышают высокие СРБ,
ферритин, ИЛ-6, глюкоза и HOMA, тогда как у фиброза и ХОБЛ/астмы — прежде всего тяжесть лёгочного
поражения (КТ, SpO2).

# ЭТАП 4. Прогнозирование панкреонекроза

Панкреонекроз — отдельно запрошенное осложнение. Оцениваем возможность его прогноза по имеющимся
данным.

    Панкреонекроз: 8 случаев из 755 (1.1%)
    
    Распределение по тяжести КТ:



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
      <th>Всего</th>
      <th>Панкреонекроз</th>
      <th>Частота, %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>КТ0</th>
      <td>303</td>
      <td>3</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>КТ1</th>
      <td>246</td>
      <td>1</td>
      <td>0.400</td>
    </tr>
    <tr>
      <th>КТ2</th>
      <td>142</td>
      <td>0</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>КТ3-4</th>
      <td>64</td>
      <td>4</td>
      <td>6.200</td>
    </tr>
  </tbody>
</table>
</div>


**Вывод о возможности прогноза.** В выборке всего **8 случаев** панкреонекроза. По правилу
«не менее 10 событий на один предиктор» обучение и валидация многопризнаковой прогнозной модели
статистически некорректны — любая такая модель переобучится и не будет воспроизводимой. К тому же
в данных нет панкреас-специфичных маркеров (амилаза, липаза). Поэтому вместо «модели-прогноза»
приводим **описательный разбор**: однофакторные ассоциации и клинико-лабораторные профили всех 8
пациентов.

    Факторы, наиболее ассоциированные с панкреонекрозом (разведочно):


    /home/user/code/covid_an/.venv/lib/python3.13/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "



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
      <th>Фактор</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>АД сист.</td>
      <td>2.695</td>
      <td>1.534</td>
      <td>4.735</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ОРДС</td>
      <td>9.671</td>
      <td>2.367</td>
      <td>39.518</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>АД диаст.</td>
      <td>2.872</td>
      <td>1.341</td>
      <td>6.151</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Температура</td>
      <td>2.495</td>
      <td>1.189</td>
      <td>5.235</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Сопут.: ССС</td>
      <td>12.295</td>
      <td>1.505</td>
      <td>100.464</td>
      <td>0.019</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HOMA</td>
      <td>1.480</td>
      <td>1.015</td>
      <td>2.157</td>
      <td>0.041</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ГКС (гормоны)</td>
      <td>4.470</td>
      <td>0.896</td>
      <td>22.295</td>
      <td>0.068</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ИМТ</td>
      <td>1.736</td>
      <td>0.919</td>
      <td>3.277</td>
      <td>0.089</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>187</th>
      <th>230</th>
      <th>245</th>
      <th>272</th>
      <th>300</th>
      <th>482</th>
      <th>514</th>
      <th>539</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Пол</th>
      <td>муж</td>
      <td>муж</td>
      <td>муж</td>
      <td>муж</td>
      <td>муж</td>
      <td>муж</td>
      <td>жен</td>
      <td>муж</td>
    </tr>
    <tr>
      <th>КТ</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Возраст</th>
      <td>54</td>
      <td>31</td>
      <td>26</td>
      <td>51</td>
      <td>50</td>
      <td>49</td>
      <td>53</td>
      <td>43</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>30.100</td>
      <td>27.500</td>
      <td>25.600</td>
      <td>33.800</td>
      <td>31.200</td>
      <td>28.900</td>
      <td>30.700</td>
      <td>27.920</td>
    </tr>
    <tr>
      <th>SpO2 min</th>
      <td>88.000</td>
      <td>90.000</td>
      <td>96.000</td>
      <td>87.000</td>
      <td>78.000</td>
      <td>95.000</td>
      <td>95.000</td>
      <td>94.000</td>
    </tr>
    <tr>
      <th>КТ макс %</th>
      <td>70</td>
      <td>60</td>
      <td>0</td>
      <td>70</td>
      <td>65</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Лейкоциты</th>
      <td>3.640</td>
      <td>4.660</td>
      <td>4.860</td>
      <td>2.880</td>
      <td>9.710</td>
      <td>4.790</td>
      <td>3.780</td>
      <td>5.390</td>
    </tr>
    <tr>
      <th>СОЭ</th>
      <td>24</td>
      <td>41</td>
      <td>4</td>
      <td>25</td>
      <td>47</td>
      <td>8</td>
      <td>43</td>
      <td>9</td>
    </tr>
    <tr>
      <th>СРБ</th>
      <td>89.720</td>
      <td>85.600</td>
      <td>3.490</td>
      <td>80.700</td>
      <td>106.800</td>
      <td>5.300</td>
      <td>25.200</td>
      <td>13.200</td>
    </tr>
    <tr>
      <th>Ферритин</th>
      <td>614.000</td>
      <td>525.000</td>
      <td>54.600</td>
      <td>253.400</td>
      <td>529.000</td>
      <td>38.600</td>
      <td>162.000</td>
      <td>213.500</td>
    </tr>
    <tr>
      <th>Д-димер</th>
      <td>1.000</td>
      <td>0.710</td>
      <td>0.250</td>
      <td>1.230</td>
      <td>0.690</td>
      <td>0.250</td>
      <td>0.250</td>
      <td>0.600</td>
    </tr>
    <tr>
      <th>ЛДГ</th>
      <td>618.000</td>
      <td>701.000</td>
      <td>316.000</td>
      <td>567.000</td>
      <td>554.000</td>
      <td>228.000</td>
      <td>389.500</td>
      <td>314.500</td>
    </tr>
    <tr>
      <th>АЛТ</th>
      <td>318.000</td>
      <td>211.000</td>
      <td>62.600</td>
      <td>388.000</td>
      <td>53.800</td>
      <td>50.300</td>
      <td>74.200</td>
      <td>71.800</td>
    </tr>
    <tr>
      <th>Креатинин</th>
      <td>107.000</td>
      <td>93.400</td>
      <td>95.100</td>
      <td>126.000</td>
      <td>101.000</td>
      <td>69.500</td>
      <td>99.700</td>
      <td>72.900</td>
    </tr>
    <tr>
      <th>Глюкоза макс</th>
      <td>6.720</td>
      <td>6.680</td>
      <td>4.640</td>
      <td>18.150</td>
      <td>7.070</td>
      <td>5.430</td>
      <td>8.250</td>
      <td>5.490</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>6.240</td>
      <td>2.740</td>
      <td>0.610</td>
      <td>5.380</td>
      <td>4.100</td>
      <td>1.550</td>
      <td>4.260</td>
      <td>2.880</td>
    </tr>
    <tr>
      <th>ИЛ-6</th>
      <td>69.700</td>
      <td>64.300</td>
      <td>0.320</td>
      <td>69.300</td>
      <td>46.700</td>
      <td>0.000</td>
      <td>1.130</td>
      <td>0.370</td>
    </tr>
    <tr>
      <th>Фибриноген</th>
      <td>5.480</td>
      <td>6.450</td>
      <td>2.870</td>
      <td>4.340</td>
      <td>6.430</td>
      <td>1.700</td>
      <td>2.800</td>
      <td>2.420</td>
    </tr>
  </tbody>
</table>
</div>


    Средние значения показателей:



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
      <th>Панкреонекроз (n=8)</th>
      <th>Остальные</th>
      <th>Медиана когорты</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Возраст</th>
      <td>44.600</td>
      <td>37.800</td>
      <td>39.000</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>29.500</td>
      <td>27.100</td>
      <td>26.900</td>
    </tr>
    <tr>
      <th>SpO2 min</th>
      <td>90.400</td>
      <td>94.100</td>
      <td>95.000</td>
    </tr>
    <tr>
      <th>КТ макс %</th>
      <td>35.600</td>
      <td>16.500</td>
      <td>10.000</td>
    </tr>
    <tr>
      <th>Лейкоциты</th>
      <td>5.000</td>
      <td>5.000</td>
      <td>4.600</td>
    </tr>
    <tr>
      <th>СОЭ</th>
      <td>25.100</td>
      <td>18.300</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>СРБ</th>
      <td>51.300</td>
      <td>30.300</td>
      <td>14.400</td>
    </tr>
    <tr>
      <th>Ферритин</th>
      <td>298.800</td>
      <td>208.000</td>
      <td>125.200</td>
    </tr>
    <tr>
      <th>Д-димер</th>
      <td>0.600</td>
      <td>0.400</td>
      <td>0.200</td>
    </tr>
    <tr>
      <th>ЛДГ</th>
      <td>461.000</td>
      <td>336.900</td>
      <td>299.000</td>
    </tr>
    <tr>
      <th>АЛТ</th>
      <td>153.700</td>
      <td>71.100</td>
      <td>48.900</td>
    </tr>
    <tr>
      <th>Креатинин</th>
      <td>95.600</td>
      <td>91.600</td>
      <td>90.200</td>
    </tr>
    <tr>
      <th>Глюкоза макс</th>
      <td>7.800</td>
      <td>6.000</td>
      <td>5.500</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>3.500</td>
      <td>2.000</td>
      <td>1.400</td>
    </tr>
    <tr>
      <th>ИЛ-6</th>
      <td>31.500</td>
      <td>7.800</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Фибриноген</th>
      <td>4.100</td>
      <td>3.500</td>
      <td>3.200</td>
    </tr>
  </tbody>
</table>
</div>


Панкреонекроз встречается преимущественно при тяжёлом течении, но и при лёгком (3 из 8 — КТ0):
группа малочисленна и неоднородна. Сильных воспроизводимых предикторов на таком числе наблюдений
выделить нельзя; для построения прогнозной модели потребуется накопление случаев и добавление
панкреас-специфичных лабораторных маркеров.

# ЭТАП 5. Осложнения в зависимости от геноварианта (Ухань / Дельта / Омикрон)

Сравниваем поздние осложнения между геновариантами вируса. Выборка собрана из отдельных листов
`УХАНЬ` (213), `ДЕЛЬТА` (272), `ОМИКРОН` (270) — это надёжнее, чем делить по году (год ≠ штамм).
Референс для сравнений — **Ухань**.

> **Важно — конфаундинг.** Геноварианты сильно различаются по тяжести (медиана КТ 1 / 1 / 0),
> возрасту (38 / 41 / 36) и охвату вакцинацией (0.5% / 30% / 89%). Поэтому **сырые** различия частот
> осложнений нельзя приписывать самому штамму — нужна поправка на тяжесть и возраст.

    Профиль геновариантов:



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
      <th>N</th>
      <th>КТ(ИТОГ), медиана</th>
      <th>Возраст, медиана</th>
      <th>Вакцинация, %</th>
    </tr>
    <tr>
      <th>strain</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>УХАНЬ</th>
      <td>213</td>
      <td>1.000</td>
      <td>38.000</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>ДЕЛЬТА</th>
      <td>272</td>
      <td>1.000</td>
      <td>41.000</td>
      <td>29.800</td>
    </tr>
    <tr>
      <th>ОМИКРОН</th>
      <td>270</td>
      <td>0.000</td>
      <td>36.000</td>
      <td>89.300</td>
    </tr>
  </tbody>
</table>
</div>


## 5.1 Распространённость осложнений по геновариантам

    Частота осложнения, % по геновариантам:



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
      <th>strain</th>
      <th>УХАНЬ</th>
      <th>ДЕЛЬТА</th>
      <th>ОМИКРОН</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Пневмофиброз</th>
      <td>3.300</td>
      <td>7.400</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>Повторные пневмонии</th>
      <td>22.500</td>
      <td>22.800</td>
      <td>22.600</td>
    </tr>
    <tr>
      <th>ХОБЛ / бр. астма</th>
      <td>3.800</td>
      <td>2.900</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>ЖКТ / НЖБП</th>
      <td>26.300</td>
      <td>28.700</td>
      <td>21.100</td>
    </tr>
    <tr>
      <th>СД / эндокринные</th>
      <td>6.100</td>
      <td>5.900</td>
      <td>3.300</td>
    </tr>
    <tr>
      <th>ССЗ (новые)</th>
      <td>12.700</td>
      <td>12.100</td>
      <td>18.900</td>
    </tr>
    <tr>
      <th>Операции на венах</th>
      <td>6.100</td>
      <td>7.400</td>
      <td>7.000</td>
    </tr>
    <tr>
      <th>ОНМК / ИМ</th>
      <td>2.300</td>
      <td>3.300</td>
      <td>4.400</td>
    </tr>
    <tr>
      <th>Зрение / слух</th>
      <td>23.900</td>
      <td>26.500</td>
      <td>8.900</td>
    </tr>
    <tr>
      <th>Психика / ЦНС / обоняние</th>
      <td>12.200</td>
      <td>12.900</td>
      <td>6.300</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_54_2.png)
    


## 5.2 Различия частоты между штаммами (χ² + поправка на множественность)

Критерий χ² для каждого осложнения через три штамма; p корректируется методом Бенджамини–Хохберга
(FDR) на семейство тестов.


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
      <th>Осложнение</th>
      <th>хи²</th>
      <th>p</th>
      <th>min ожид.</th>
      <th>p (FDR)</th>
      <th>вывод</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Зрение / слух</td>
      <td>30.500</td>
      <td>0.000</td>
      <td>41.500</td>
      <td>0.000</td>
      <td>значимо</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Психика / ЦНС / обоняние</td>
      <td>7.440</td>
      <td>0.024</td>
      <td>22.000</td>
      <td>0.104</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Пневмофиброз</td>
      <td>7.130</td>
      <td>0.028</td>
      <td>9.900</td>
      <td>0.104</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ССЗ (новые)</td>
      <td>5.900</td>
      <td>0.052</td>
      <td>31.300</td>
      <td>0.144</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ЖКТ / НЖБП</td>
      <td>4.260</td>
      <td>0.119</td>
      <td>53.900</td>
      <td>0.262</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>5</th>
      <td>СД / эндокринные</td>
      <td>2.550</td>
      <td>0.279</td>
      <td>10.700</td>
      <td>0.512</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Любое осложнение</td>
      <td>2.110</td>
      <td>0.347</td>
      <td>100.700</td>
      <td>0.546</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ОНМК / ИМ</td>
      <td>1.600</td>
      <td>0.450</td>
      <td>7.300</td>
      <td>0.619</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ХОБЛ / бр. астма</td>
      <td>0.320</td>
      <td>0.852</td>
      <td>6.800</td>
      <td>0.944</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Операции на венах</td>
      <td>0.310</td>
      <td>0.858</td>
      <td>14.700</td>
      <td>0.944</td>
      <td>поисково</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Повторные пневмонии</td>
      <td>0.010</td>
      <td>0.997</td>
      <td>48.200</td>
      <td>0.997</td>
      <td>поисково</td>
    </tr>
  </tbody>
</table>
</div>


Минимальные ожидаемые частоты во всех таблицах ≥ 5 (даже для ОНМК/ИМ — 7.3), поэтому χ²-приближение
корректно; ограничение — малое **число событий** в редких осложнениях (низкая мощность). После
FDR-поправки порог различий между штаммами выдерживает только **зрение/слух**; психоневрологические
осложнения (p≈0.02), пневмофиброз (p≈0.03) и ССЗ (p≈0.05) — поисковые.

## 5.3 Сравнение с поправкой на тяжесть и возраст

Логистическая модель `осложнение ~ C(штамм) + КТ(ИТОГ) + возраст` (референс — Ухань). **Вакцинация
исключена из основной модели**: она почти коллинеарна штамму (0.5/30/89%), и её включение делает
эффект штамма неразделимым. Для зрения/слуха дополнительно показана модель чувствительности
с вакцинацией.

    Скорректированные OR геноварианта (vs Ухань):



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
      <th>Осложнение</th>
      <th>событий</th>
      <th>vs Ухань</th>
      <th>OR</th>
      <th>ДИ low</th>
      <th>ДИ high</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Любое осложнение</td>
      <td>357</td>
      <td>ДЕЛЬТА</td>
      <td>0.868</td>
      <td>0.567</td>
      <td>1.329</td>
      <td>0.516</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Любое осложнение</td>
      <td>357</td>
      <td>ОМИКРОН</td>
      <td>1.316</td>
      <td>0.859</td>
      <td>2.018</td>
      <td>0.207</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Повторные пневмонии</td>
      <td>171</td>
      <td>ДЕЛЬТА</td>
      <td>0.827</td>
      <td>0.529</td>
      <td>1.292</td>
      <td>0.404</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Повторные пневмонии</td>
      <td>171</td>
      <td>ОМИКРОН</td>
      <td>1.077</td>
      <td>0.686</td>
      <td>1.690</td>
      <td>0.748</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ЖКТ / НЖБП</td>
      <td>191</td>
      <td>ДЕЛЬТА</td>
      <td>0.870</td>
      <td>0.564</td>
      <td>1.342</td>
      <td>0.528</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ЖКТ / НЖБП</td>
      <td>191</td>
      <td>ОМИКРОН</td>
      <td>0.842</td>
      <td>0.534</td>
      <td>1.327</td>
      <td>0.459</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ССЗ (новые)</td>
      <td>111</td>
      <td>ДЕЛЬТА</td>
      <td>0.731</td>
      <td>0.415</td>
      <td>1.289</td>
      <td>0.279</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ССЗ (новые)</td>
      <td>111</td>
      <td>ОМИКРОН</td>
      <td>1.838</td>
      <td>1.073</td>
      <td>3.150</td>
      <td>0.027</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Зрение / слух</td>
      <td>147</td>
      <td>ДЕЛЬТА</td>
      <td>0.803</td>
      <td>0.504</td>
      <td>1.279</td>
      <td>0.355</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Зрение / слух</td>
      <td>147</td>
      <td>ОМИКРОН</td>
      <td>0.300</td>
      <td>0.168</td>
      <td>0.536</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_59_2.png)
    


    Чувствительность (зрение/слух, +вакцинация): Омикрон vs Ухань OR=0.29 (p=0.002); вакцинация OR=1.05 (p=0.866) -> эффект Омикрона устойчив.


После поправки на тяжесть и возраст **геновариант сам по себе не повышает** общий риск поздних
осложнений (Дельта и Омикрон vs Ухань — незначимо). Риск определяют тяжесть (КТ, OR≈2) и возраст
(≈1.07/год). Устойчивое исключение — **снижение нарушений зрения/слуха при Омикроне** (скорр. OR≈0.30):
этот сигнал значим и в χ² с FDR-поправкой (этап 5.2), и после поправки на тяжесть/возраст (этап 5.3 —
здесь p приведены без поправки на множественность). Оценка опирается на 24 случая в омикроновском плече
(в таблице 5.3 столбец «событий» = 147 — это объединённый по трём штаммам счёт) и неотделима от
вакцинации/календарного периода. Сигнал «больше новых ССЗ при Омикроне» (OR≈1.8, p≈0.03) — поисковый,
FDR-поправку не проходит.

## 5.4 Предикторы осложнений внутри каждого штамма (поисковая карта)

Однофакторные OR общего риска отдельно в каждом штамме. Непрерывные факторы — OR **на 1 SD внутри
штамма** (разные абсолютные шаги, между штаммами в абсолюте не сравнимы), бинарные — наличие/отсутствие.


    
![png](covid_complications_files/covid_complications_62_0.png)
    


Карта **поисковая** (без поправок и множественной коррекции). Основные предикторы (возраст, ИМТ,
HOMA, бактериальные осложнения) работают во всех штаммах. OR вакцинации в Ухане не оценим (вакцинирован
1 из 213 — разделение); OR для ОРДС внутри штаммов помечены как неустойчивые (ОРДС-позитивных всего
25 / 36 / 13; формальный критерий — минимальная клетка таблицы 2×2 < 5). Ослабление КТ/ДН/ГКС при
Омикроне частично отражает сужение диапазона тяжести
(медиана КТ = 0), а не обязательно иную биологию.

# ЭТАП 6. Клинические схемы риска поздних осложнений

Компактные «схемы» (≤ 5–10 признаков) для трёх осложнений, каждая в двух читаемых формах:
**Форма A — дерево решений** (видны пороги-ветвления и риск в листьях) и **Форма B — балльная шкала**
(порог по каждому признаку и баллы по вкладу в логистическую модель).

> **Поисковый характер.** 3 цели × панель из ~8 признаков, отбор признаков по |β|, пороги по Юдену на
> той же выборке, банды по квантилям — это перебор на n=755, **завышающий видимую эффективность** всех
> трёх схем (особенно ОНМК/ИМ, где всего 26 событий). Поэтому рядом с in-sample-метрикой всегда
> приводится кросс-валидированный AUC. Схемы — **ориентировочные**, требуют внешней проверки.

    Схемы готовы к построению; панель из 11 признаков


## 6.1 ОНМК / ИМ

Всего **26 событий** (EPV ≈ 5) — схема **ориентировочная, гипотезо-порождающая**. Корневой признак
(возраст) воспроизводим; глубокие листья и AUC между фолдами нестабильны.

    === ОНМК / ИМ: событий=26, EPV=5.2 ===
    AUC: баллы (in-sample)=0.850 | логит CV=0.796±0.065 | дерево CV=0.770±0.090



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
      <th>Порог</th>
      <th>Баллы</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Возраст</th>
      <td>&gt; 43.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>&gt; 28.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Глюкоза макс</th>
      <td>&gt; 5.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>%КТ макс</th>
      <td>&gt; 27.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_67_2.png)
    


## 6.2 Повторные пневмонии

171 событие; ведущий профиль — **метаболический** (HOMA, возраст, ИМТ). Холестерин и %КТ исключены
из шкалы как незначимые (|β| < 0.10).

    === Повторные пневмонии: событий=171, EPV=21.4 ===
    AUC: баллы (in-sample)=0.700 | логит CV=0.665±0.024 | дерево CV=0.608±0.041



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
      <th>Порог</th>
      <th>Баллы</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Возраст</th>
      <td>&gt; 39.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Пол (жен.)</th>
      <td>женский пол</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>&gt; 27.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>HOMA</th>
      <td>&gt; 1.3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>ТГ</th>
      <td>&gt; 0.9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Глюкоза макс</th>
      <td>&gt; 6.4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_69_2.png)
    


## 6.3 Тромбозы / операции на венах

Позднее осложнение (`вены/опер`, idx 132, блок `DZ..EO`; 52 случая) — поздний венозный исход
(тромбозы / операции на венах). Это **прогностическая** схема: предикторы (Д-димер, СРБ, %КТ, ИМТ и др.)
измерены в исходном/остром периоде, **до** позднего исхода — утечки нет. Ведущий признак —
**Д-димер** (маркер тромбообразования).

    === Тромбозы / операции на венах: событий=52, EPV=7.4 ===
    AUC: баллы (in-sample)=0.853 | логит CV=0.815±0.053 | дерево CV=0.771±0.070



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
      <th>Порог</th>
      <th>Баллы</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Возраст</th>
      <td>&gt; 34.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Пол (жен.)</th>
      <td>женский пол</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ИМТ</th>
      <td>&gt; 27.2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Д-димер</th>
      <td>&gt; 0.4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>%КТ макс</th>
      <td>&gt; 17.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>СРБ</th>
      <td>&gt; 19.8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Глюкоза макс</th>
      <td>&gt; 5.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](covid_complications_files/covid_complications_71_2.png)
    


## 6.4 Валидация схем — внутривыборочный против кросс-валидированного AUC

Кросс-валидация (5-блочная стратифицированная) с **переотбором порогов Юдена внутри каждого
обучающего фолда** (честная оценка). Балльная шкала с целочисленными весами — приближение
логистической модели, её AUC приводится отдельно.

> Примечание: для редких исходов (ОНМК/ИМ, тромбозы) часть квантильных границ суммы баллов совпадает,
> и банды риска схлопываются до трёх — это не пропущенная группа, а малое число различающихся сумм баллов.


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
      <th>Событий</th>
      <th>AUC баллов (in-sample)</th>
      <th>Логит CV-AUC</th>
      <th>Дерево CV-AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ОНМК / ИМ</th>
      <td>26</td>
      <td>0.850</td>
      <td>0.796 ± 0.065</td>
      <td>0.770 ± 0.090</td>
    </tr>
    <tr>
      <th>Повторные пневмонии</th>
      <td>171</td>
      <td>0.700</td>
      <td>0.665 ± 0.024</td>
      <td>0.608 ± 0.041</td>
    </tr>
    <tr>
      <th>Тромбозы / операции на венах</th>
      <td>52</td>
      <td>0.853</td>
      <td>0.815 ± 0.053</td>
      <td>0.771 ± 0.070</td>
    </tr>
  </tbody>
</table>
</div>


**Итог по схемам.** Тромбозы / операции на венах (позднее осложнение, 52 случая) — CV-AUC ≈0.82 (логит),
ведущий предиктор Д-димер. Повторные пневмонии — умеренная схема (CV-AUC ≈0.67), метаболический профиль. ОНМК/ИМ —
ориентировочная (CV-AUC ≈0.80±0.07 у логистической, ≈0.77±0.09 у дерева, 26 событий). Все схемы валидированы
**только внутри выборки** (n=755), без поправки на штамм/эру и без внешней валидации; «нулевой риск»
в нижних бандах — это верхняя граница ~0.6–0.8% (правило трёх), а не гарантированное отсутствие риска.

# Выводы

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

