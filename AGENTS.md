# Работа над финальным Sleep-EDF

Пиши по-русски. Исследования завершены: финалы157_pretrained и113_pretrained.
Не меняй checkpoints, preprocessing, контекст, batches или ансамбль без явного запроса.
Точное совпадение всех test id,label важнее совпадения округлённой метрики.

- final/ — автономный inference без imports из archive/, два frozen YAML и необходимые веса.
- archive/ — сохранённая исследовательская часть; подробные исторические инструкции в archive/AGENTS.md.
- experiments.csv — вся история; final/experiments.csv — только выбранные два исследования.
- archive/docs/ — игнорируемые черновики. Публичные объяснения — аккуратные final/docs/*.md.
- moodle_package/ и team_info.csv — приватные, игнорируемые материалы. Персональные данные не публиковать.

Используй uv и два закреплённых окружения: final/ (Python3.12, PyTorch2.6) и
final/environments/features/ (Python3.14, исходные NumPy/SciPy/antropy).
Код извлечения106 признаков не переносить в другое окружение без проверки чисел.
Классы W,N1,N2,N3,REM; сигналы100Hz,30s,3 канала вµV; никакого чтения test truth.
Старые research submissions в final/submissions/ используются только для сравнения,
не для генерации или коррекции выходных меток. При расхождении сохраняй диагностику и останавливай финал.

Основной notebook — final/notebooks/06_final.ipynb. По умолчанию только157;
OUTPUT_SECOND_SUBMISSION=True добавляет113 как submission_2.csv.
Одна публикация notebook на два решения согласована пользователем с преподавателем.
Полный Kaggle-run06 ещё не выполнен; не объявлять exact reproduction или PDF готовыми.

До правок проверяй Git diff, сохраняй результаты. Не запускай долгое обучение.
Веса только выбранных компонентов; SHA и vendor licenses обязательны.
Проверки: uv run --project final ruff check final; uv run --project final python -m pytest final/tests.

Финальный runtime обязан работать с Internet Off от начала до конца.
final/offline/build_runtime.py использует uv/сеть только при предварительной Linux-сборке.
final/offline_runtime.py в Kaggle только проверяет/распаковывает архивы и создаёт venv
через stdlib без pip. Не возвращать pip/uv sync/download в notebook06.
Новые Inputs: sleep-edf-final-157-113-inputs.zip (code kindv3) и отдельный
sleep-edf-final-runtime-linux.zip. Старый code Input v1 не считать offline-готовым.
Runtime-бинарники, build caches и teacher-only файлы вне Git. Численные модели не меняются.

Финал157 = .5U-Sleep156 + .5ансамбль113; второй CSV —113. Девять файлов весов.
U-Sleep использует нашу схему019: per-segment clip→RobustScaler, core128/halo64,
среднее двух EEG/EOG пар. Runtime ZIP прежний, lock-файлы не менять.
