# Финальный inference113 /099

Единственный notebook: [notebooks/06_final.ipynb](notebooks/06_final.ipynb).
Исходное состояние моделей уже зафиксировано в weights/; авторские определения —
src/sleep_edf_final/vendor/. Основная модель113 использует099-подобную базовую смесь
и heads102/106; её веса, контекст и floating-point порядок сохранены.

## Kaggle

1. Подключите `levshaazz/sleep-edf-full-psg-course-data`, **version1**.
2. Подключите ZIP `sleep-edf-final-inputs.zip`, подготовленный `python final/package.py bundle`.
   В нём final-код и48.5MiB необходимых весов, без данных, старых probabilities и кэшей.
3. Откройте06_final.ipynb, выберите T4x2, Internet On для установки pinned dependencies.
4. `OUTPUT_SECOND_SUBMISSION=False` создаёт только `submission.csv` (113).
   `True` дополнительно создаёт `submission_2.csv` (099). Остальные значения уже настроены.
5. Run All. Сначала106 признаков из raw test в Python3.14, затем frozen backbones и
   heads в Python3.12. Один GPU-процесс использует cuda:0; GPU-память не объединяется.
6. Для полного запуска требуется `run_report.json: status=EXACT_MATCH`, epochs66222
   и label_mismatches0 у каждого включённого решения. Скачайте результаты одним ZIP.

Загрузка teacher-only документов в Kaggle не требуется. В Input нет токенов или
персональных данных; private GitHub для inference не используется.

## Локальный CLI (Linux)

```sh
uv sync --project final --locked --no-dev --python 3.12.12
uv sync --project final/environments/features --locked --no-dev --python 3.14.2
uv run --project final --locked --no-dev python final/run.py \
  --data-root archive/data/raw --output-root final/outputs/run \
  --feature-python final/environments/features/.venv/bin/python \
  --device cuda:0 --threads 4 --feature-workers 2 --output-second-submission
```

На Windows feature-python указывает на `.venv/Scripts/python.exe`. Профиль CPU
предусмотрен для разработки; совпадение всех выбранных labels в нём отдельно не заявляется.
`--smoke-epochs 4` обрабатывает только начало одной записи и не является оценкой качества.

Повторный запуск может переиспользовать только кэш этого же кода/данных/профиля в output-root.
Исходные research predictions не используются. При несовпадении contracts старый результат
сохраняется и требуется другой output-root. ZIP результатов исключает временные кэши.

## Файлы и статус

- configs/113_pretrained.yaml и099_pretrained.yaml — только frozen inference, training disabled.
- weights/manifest.json — точные SHA-256 восьми файлов весов и provenance.
- submissions/ — выбранные исторические CSV для проверки; runtime пишет собственные новые CSV.
- experiments.csv — две исторические validation-строки; время смеси не равно времени всего pipeline.
- release_metadata.json — версии публикаций/Inputs и submission IDs; неизвестные поля явно null.
- pdf/ — место для PDF после выполненного Kaggle notebook. Сейчас PDF ещё нет.

Приватный пакет для преподавателя собирается отдельно и не публикуется в Git.
До полного запуска06 не готовы executed outputs/PDF, Kaggle version и submission IDs.
