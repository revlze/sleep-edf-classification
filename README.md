# Sleep-EDF 2026

Классификация пяти стадий сна по EEG/EOG. Основная метрика — macro-F1;
валидация внутри train группируется по subject_id.

## Запуск

Python 3.14, зависимости закреплены в uv.lock.

```sh
uv sync --locked
uv run python -m src.run configs/001_linear.yaml --smoke
```

Для полноценного запуска уберите `--smoke`. Пути задаются через
`--data-root` и `--output-root`; по умолчанию это `data/raw` и `artefacts`.
Если данные лежат непосредственно в `data/`, добавьте `--data-root data`.
Локальный запуск — `notebooks/02_train.ipynb`: выбирает LR или бустинги.
По умолчанию выбран только `015_tree`, `STAGE="fixed"`: один fit LightGBM,
855 готовых признаков, 420 итераций, без поиска и CV. Кэш должен существовать.

## Структура

- `src/` — данные, признаки, модели, обучение и оценка.
- `configs/` — LR `001–007_linear`, LightGBM/CatBoost `009–014_tree`.
- `notebooks/01_eda.ipynb` — исследование данных.
- `notebooks/02_train.ipynb` — обучение; `03_eval.ipynb` — сравнение; `04_submit.ipynb` — submission.
- `notebooks/00_kaggle.ipynb` — запуск на Kaggle и выгрузка результатов одним ZIP.
- `experiments.csv` — индекс завершённых экспериментов.
- `data/raw/`, `artefacts/`, `submissions/`, `docs/` — локальные файлы вне Git.

Поставка `full_psg_v1`: в корне `labels.csv`, `sample_submission.csv`,
`epochs/{train,val,test}/*.npz`. Классы: W, N1, N2, N3, REM.
Для LR `MANIFEST.json` не требуется: кэш учитывает метки, параметры, размеры
и время изменения NPZ. Проверки по исходным контрольным суммам не выполняются.
Признаки и завершённые trials переиспользуются; прерванный fold обучается заново.
Готовые кэши читаются параллельно; создание одного NPZ сериализуется отдельно.
Разные эксперименты и инференс завершённой модели могут работать одновременно.
Запись одного CSV и экспорт результатов защищены отдельными блокировками.

## Проверки

```sh
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest
```
