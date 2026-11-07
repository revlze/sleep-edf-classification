# Финальный inference

`157 = 0.50 × U-Sleep156 + 0.50 × ансамбль113`.
Все модели frozen. Код автономен: imports из `archive/` не используются.

- [Выполненный notebook](notebooks/06_final.ipynb) и [PDF](pdf/06_final.pdf).
- [Точный Kaggle-run](https://www.kaggle.com/code/sevaperetiatko/sleep-edf-sbmteam?scriptVersionId=366571096).
- [Запуск и версии Inputs](../docs/reproducibility.md).
- [Архитектура](../docs/solution.md), [источники](../docs/sources.md).

## Kaggle

Подключите исходный Sleep-EDF Dataset v1, code/weights Input v2 и Linux runtime Input v1.
Выберите T4 × 2, Internet Off и выполните Run All.
`OUTPUT_SECOND_SUBMISSION=False` создаёт `submission.csv` для `157`.
True дополнительно экспортирует `113` как `submission_2.csv`.

## Файлы

| Путь | Назначение |
|---|---|
| `src/` | Подготовка сигналов, backbones, heads, строгая проверка CSV |
| `configs/` | Зафиксированные `157` и его компонент `113` |
| `weights/manifest.json` | SHA-256 девяти нужных файлов весов |
| `environments/features/` | Исходная среда 106-feature extractor, Python 3.14 |
| `offline/` и `offline_runtime.py` | Сборка и локальное восстановление Linux runtime |
| `evidence/` | Подтверждение опубликованного Kaggle-run |
| `submissions/` | Исторические CSV только для сравнения после inference |
| `tests/` | Короткие проверки контрактов и границ |

Зафиксированное воспроизведение: `EXACT_MATCH`, 66 222 ID, 0 различий меток,
925.9 секунды для всех ячеек. Reference CSV не подменяет результат вычислений.
При несовпадении код сохраняет диагностику и завершает запуск с ошибкой.

Для нового Input: `python final/package.py bundle` из корня репозитория.
Для точного воспроизведения опубликованной отправки используйте версии Inputs из инструкции.
