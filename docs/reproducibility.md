# Запуск и проверка воспроизводимости

## Проверенный запуск

Опубликованная [закреплённая версия 1 Kaggle Notebook](https://www.kaggle.com/code/sevaperetiatko/sleep-edf-sbmteam?scriptVersionId=366571096)
скачана с Kaggle вместе с исходными выводами ячеек. Копия находится в
[final/notebooks/06_final.ipynb](../final/notebooks/06_final.ipynb),
[PDF](../final/pdf/06_final.pdf) сформирован из этой копии.

| Проверка | Результат |
|---|---:|
| Финальная конфигурация | `157_pretrained` |
| Полных тестовых ID | 66 222 |
| Несовпадающих меток | 0 |
| Статус | `EXACT_MATCH` |
| Время inference pipeline | 805.92 с |
| Время всех ячеек с подготовкой среды | 925.9 с |
| Профиль | Kaggle T4 × 2; вычисления на `cuda:0` |
| Internet | Off |

Отчёт и контрольные суммы — в [final/evidence](../final/evidence/).
Производственный код был зафиксирован commit `f0c9db1873517bef41cd6c28d7ce30354f80a3ad`
в code/weights Input version 2. Обновление документации репозитория не меняет этот Input.

## Inputs

| Назначение | Владелец / Dataset | Версия |
|---|---|---:|
| Сигналы и метаданные | [levshaazz/sleep-edf-full-psg-course-data](https://www.kaggle.com/datasets/levshaazz/sleep-edf-full-psg-course-data/versions/1) | 1 |
| Код и девять checkpoints | [sevaperetiatko/sleep-edf-final-offline-inputs](https://www.kaggle.com/datasets/sevaperetiatko/sleep-edf-final-offline-inputs/versions/2) | 2 |
| Linux Python и готовые зависимости | [sevaperetiatko/sleep-edf-final-runtime-linux](https://www.kaggle.com/datasets/sevaperetiatko/sleep-edf-final-runtime-linux/versions/1) | 1 |

В исходной поставке 152 записи: train — 302 653 эпохи, 58 людей; val — 43 206 эпох,
8 людей; test — 66 222 эпохи. Test labels не используются и в поставке не читаются.
Файлы сверяются с MANIFEST; joins и CSV строятся по ID, а не по порядку строк.

## Kaggle

1. Импортируйте `final/notebooks/06_final.ipynb`.
2. Подключите три Input нужных версий из таблицы.
3. Выберите T4 × 2 и Internet Off. `OUTPUT_SECOND_SUBMISSION=False` — финальный `157`.
4. Выполните Run All. Сохраните версию с outputs и выгрузите итоговый ZIP.

Ноутбук восстанавливает готовые Python 3.12.12 и 3.14.2 из локального runtime Input,
проверяет SHA и версии distributions. Установки пакетов и загрузки весов из сети нет.
Признаки и нейросетевые прогнозы вычисляются заново из raw test.

Выходы: `submission.csv`, `157_probabilities.npz`, component probabilities,
`run_report.json`, manifests и компактный ZIP. Флаг дополнительного экспорта
может создать `submission_2.csv` для компонента `113`; отдельный полный экспорт `113`
не заявляется результатом опубликованного запуска с флагом False.

## Локальная подготовка

Нужны `uv` и два окружения, поскольку 106-feature extractor был зафиксирован отдельно:

```sh
uv sync --project final --locked --python 3.12.12
uv sync --project final/environments/features --locked --python 3.14.2
```

На Linux, из корня репозитория:

```sh
uv run --project final --locked --no-dev python final/run.py \
  --data-root /path/to/full_psg_v1 \
  --output-root final/outputs/local \
  --feature-python final/environments/features/.venv/bin/python \
  --device cuda:0 --threads 4 --feature-workers 2
```

CUDA-профиль намеренно ограничен T4. На Windows путь к feature Python заканчивается
на `.venv/Scripts/python.exe`; CPU доступен для разработки, но полный CPU-run
не проверен на двухчасовой лимит. `--smoke-epochs 4` — только структурная проверка.

Runtime собирается заранее на Linux через `final/offline/build_runtime.py` и два
`uv.lock`. Code/weights ZIP создаёт `python final/package.py bundle`.
Подготовка может использовать сеть; финальный Kaggle-run — полностью offline.
Локальные ZIP, данные и виртуальные окружения не хранятся в Git.

## Проверки

```sh
uv run --project final --locked --group dev ruff check final
uv run --project final --locked --group dev python -m pytest final/tests -q
```

Тесты проверяют состояния моделей, временные границы, схему submission,
локальное восстановление среды и опубликованный notebook. Они не заменяют
полный запуск, подтверждённый сохранёнными Kaggle outputs.

## Валидация и ограничения

Обучаемые компоненты использовали subject-grouped folds внутри train.
Нормализация обучаемых признаков оценивалась по train соответствующего fold,
итоговый refit — по всему train; provided val не участвует в таком fit.

Validation многократно использовалась для выбора архитектур и смесей. Дополнительные
12 людей из train служили диагностике и не являются независимым holdout для моделей,
дообучавшихся на train. Author pretraining также может пересекаться с людьми курса:
для AnySleep по опубликованным split найдено пересечение 6 из 8 validation людей;
полное исключение пересечений всех внешних checkpoints не заявляется.

Число эпох не равно числу независимых наблюдений. Эти результаты описывают
учебный benchmark, а не независимую клиническую валидацию. Битовая повторяемость
нового обучения не заявляется; проверенное требование — совпадение финальных
целочисленных меток из фиксированных состояний и среды.
