import json, copy

OLD_HEAD = ("FILE_PATH = 'initial_data.xlsx'\n"
            "SHEET_NAMES = ['ОБЩЕЕ2', 'УХАНЬ2', 'ДЕЛЬТА2', 'ОМИКРОН2']")

NEW_HEAD = ("FILE_PATH = 'updated_data.xlsx'\n"
            "# Данные обновлены: файл updated_data.xlsx (общая выборка — 755 пациентов,\n"
            "# добавлены столбцы поздних осложнений DZ..FL). Логические имена когорт\n"
            "# сохранены прежними, поэтому весь последующий код не меняется.\n"
            "SHEET_MAP = {'ОБЩЕЕ2': '0бщ.755', 'УХАНЬ2': 'УХАНЬ', 'ДЕЛЬТА2': 'ДЕЛЬТА', 'ОМИКРОН2': 'ОМИКРОН'}\n"
            "SHEET_NAMES = ['ОБЩЕЕ2', 'УХАНЬ2', 'ДЕЛЬТА2', 'ОМИКРОН2']\n\n"
            "# Прозрачная подмена имён листов: любой вызов pd.read_excel(..., sheet_name='ОБЩЕЕ2')\n"
            "# в последующих ячейках автоматически читает соответствующий лист обновлённого файла.\n"
            "_orig_read_excel = pd.read_excel\n"
            "def _read_excel_mapped(*a, **kw):\n"
            "    sn = kw.get('sheet_name')\n"
            "    if isinstance(sn, str):\n"
            "        kw['sheet_name'] = SHEET_MAP.get(sn, sn)\n"
            "    elif isinstance(sn, (list, tuple)):\n"
            "        kw['sheet_name'] = [SHEET_MAP.get(s, s) if isinstance(s, str) else s for s in sn]\n"
            "    return _orig_read_excel(*a, **kw)\n"
            "pd.read_excel = _read_excel_mapped")

OLD_READ = "df_raw = pd.read_excel(FILE_PATH, sheet_name=sheet)"
NEW_READ = "df_raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_MAP[sheet])"

NOTE = ("\n\n> **Источник данных (обновление).** Этот ноутбук — копия исходного анализа, "
        "пересчитанная на обновлённом наборе `updated_data.xlsx`: общая выборка увеличена "
        "до **755 пациентов**, а также добавлены столбцы поздних (постковидных) осложнений "
        "(`DZ..FL`). Схема клинических признаков идентична исходному файлу, поэтому методика "
        "анализа не изменилась — обновился только источник данных.")

def patch(src_path, dst_path):
    nb = json.load(open(src_path, encoding='utf-8'))
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and OLD_HEAD in src:
            src = src.replace(OLD_HEAD, NEW_HEAD).replace(OLD_READ, NEW_READ)
            cell['source'] = src.splitlines(keepends=True)
        # add data-source note to the "О ноутбуке" markdown cell
        if cell['cell_type'] == 'markdown' and src.strip().startswith('## О ноутбуке'):
            cell['source'] = (src + NOTE).splitlines(keepends=True)
        # clear stale outputs
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    json.dump(nb, open(dst_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote', dst_path)

patch('covid_analysis.ipynb', 'covid_analysis_updated.ipynb')
patch('covid_analysis_extended.ipynb', 'covid_analysis_extended_updated.ipynb')
