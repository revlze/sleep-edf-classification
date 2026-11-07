# Финальное решение Sleep-EDF

Основное решение —157: равное среднее U-Sleep156 и ансамбля113.
Дополнительное решение —113. Выбор выполнен по provided validation macro-F1;
для дополнительного анализа использовалась выборка12 людей из train.

## Запуск на Kaggle

1. Импортируйте notebooks/06_final.ipynb.
2. Подключите исходный Sleep-EDF Dataset version1, новый
   sleep-edf-final-157-113-inputs.zip и прежний sleep-edf-final-runtime-linux.zip.
3. Выберите T4×2 и Internet Off. GitHub и токен при запуске не требуются.
4. OUTPUT_SECOND_SUBMISSION=False создаёт submission.csv для157.
   True дополнительно создаёт submission_2.csv для113.
5. Выполните Run All. Ноутбук восстановит фиксированное окружение, рассчитает
   признаки и прогнозы из raw test, сохранит CSV и ZIP результатов.
6. Для обоих решений ожидаются66222 ID и0 несовпадающих меток, status=EXACT_MATCH.
   Полное время, включая подготовку окружения, должно быть не более7200 секунд.

Официальные веса и собственные checkpoints находятся в weights/; точные SHA-256
и источники — weights/manifest.json. Нового обучения и скачивания весов нет.
Окружение Linux уже подготовлено: новые зависимости не требуются, runtime Input прежний.
Код/веса упаковываются командой python final/package.py bundle из корня проекта.

## Состав

- configs/157_pretrained.yaml и113_pretrained.yaml — фиксированные настройки.
- src/ — автономный inference, без imports из archive/.
- submissions/ — выбранные CSV только для проверки после inference.
- experiments.csv — validation-метрики двух выбранных решений.
- release_metadata.json — точные версии notebook/Inputs и ID отправок; ожидают заполнения после Kaggle-run.
- docs/ — устройство решения, история исследований, источники и ограничения.

Исходный notebook ещё нужно выполнить на Kaggle и сохранить его версию с outputs.
Выполненный notebook и его PDF добавляются в отдельный приватный пакет Moodle.
