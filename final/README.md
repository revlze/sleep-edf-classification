# Финальный inference113 /099

Единственный notebook: [notebooks/06_final.ipynb](notebooks/06_final.ipynb).
Исходное состояние моделей уже зафиксировано в weights/; авторские определения —
src/sleep_edf_final/vendor/. Основная модель113 использует099-подобную базовую смесь
и heads102/106; её веса, контекст и floating-point порядок сохранены.

## Kaggle: Internet Off

1. Подключите `levshaazz/sleep-edf-full-psg-course-data`, **version1**.
2. Подключите новый `sleep-edf-final-offline-inputs.zip` — код и веса. Прежний
   `sleep-edf-final-inputs.zip` v1 с сетевой установкой для финала не используется.
3. Подключите `sleep-edf-final-runtime-linux.zip` — готовые Python3.12.12/3.14.2
   и пакеты Linux x86_64, собранные строго по двум uv.lock. Он существенно больше
   model Input, потому что включает PyTorch и CUDA-библиотеки. Внутри сохранены лицензии.
   Внутренние архивы имеют расширение.bin: это защищает их от автоматической
   распаковки Kaggle при загрузке; notebook извлечёт их сам после проверки SHA.
4. Откройте06_final.ipynb, выберите T4x2 и **Internet Off до запуска**.
5. `OUTPUT_SECOND_SUBMISSION=False`: только113. `True`:113 и099 без повторения backbones.
6. Run All. Notebook локально распакует оба окружения, создаст venv без pip,
   восстановит site-packages и проверит Python/версии всех distributions и SHA архивов.
   `uv sync`, `pip install`, скачивание Python и GitHub во время исполнения не вызываются.
7. Затем106 признаков заново из raw test, frozen backbones, heads и CSV.
   Требуются status=EXACT_MATCH, epochs66222, label_mismatches0 для включённых решений
   и полный notebook runtime≤7200s. Последняя ячейка экспортирует компактный ZIP.

Runtime Input можно держать приватным на этапе тестирования. Перед дедлайном обеспечьте
преподавателю доступ именно к его фиксированной версии; public notebook не открывает
автоматически приватные Inputs. В нём нет личной формы или teacher-only отчётности.

## Подготовка offline Input (только сборщик использует сеть)

На Linux x86_64, например в WSL Ubuntu, из корня репозитория:

```sh
python3 final/offline/build_runtime.py --final-root final \
  --work-dir /tmp/sleep-edf-offline-runtime-v1 \
  --output final/bundles/sleep-edf-final-runtime-linux.zip
python3 final/package.py bundle
```

Сборщик загружает закреплённый uv с проверкой SHA, устанавливает managed Python и
выполняет `uv sync --locked --no-dev`. В Input сохраняются Python и site-packages
отдельно: на Kaggle venv создаётся заново с правильными локальными путями. Windows
.venv не переносится. Lock-файлы/модели/численные настройки не изменяются.

Инвентарь пакетов, SHA каждого tar и исходных lock-файлов — offline_manifest.json.
Runtime ZIP не коммитится в Git; исходник сборщика и локального загрузчика коммитится.
Предвычисленных признаков или готовых probabilities в runtime нет.

Готовый ZIP проверен холодным восстановлением внутри Linux `unshare --net` и
реальным CPU inference четырёх эпох с обоими CSV. SHA ZIP и границы проверки —
offline/verification.json. Полный T4-run и его длительность пока предстоит проверить.

Повторный запуск может переиспользовать только кэш той же версии в output-root.
Неполностью распакованное окружение сохраняется; при прерывании выберите новый
RUNTIME_CACHE. Исследовательские prediction caches не используются.

## Файлы и статус

- configs/113_pretrained.yaml и099_pretrained.yaml — только frozen inference, training disabled.
- weights/manifest.json — точные SHA-256 восьми файлов весов и provenance.
- submissions/ — выбранные исторические CSV для проверки; runtime пишет собственные новые CSV.
- experiments.csv — две исторические validation-строки; время смеси не равно времени всего pipeline.
- release_metadata.json — версии публикаций/Inputs и submission IDs; неизвестные поля явно null.
- pdf/ — место для PDF после выполненного Kaggle notebook. Сейчас PDF ещё нет.

Приватный пакет для преподавателя собирается отдельно и не публикуется в Git.
До полного запуска06 не готовы executed outputs/PDF, Kaggle version и submission IDs.
