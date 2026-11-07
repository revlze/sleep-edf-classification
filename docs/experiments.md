# Исследования и эксперименты

[Полный журнал](../experiments.csv) содержит 158 записей: 149 со статусом `complete`,
8 `archived` и 1 `interrupted`. ID общие для всех семейств; ниже приведены только
завершённые эксперименты и их provided validation macro-F1.

## Основные контрольные точки

| ID | Что проверяли | Val macro-F1 |
|---|---|---:|
| [001_linear](../archive/configs/001_linear.yaml) | 106 признаков, Logistic Regression | 0.714794 |
| [005_linear](../archive/configs/005_linear.yaml) | LR с соседними эпохами ±2 | 0.759776 |
| [007_linear](../archive/configs/007_linear.yaml) | LR с multi-resolution | 0.764349 |
| [011_tree](../archive/configs/011_tree.yaml) | LightGBM с контекстом | 0.763927 |
| [015_tree](../archive/configs/015_tree.yaml) | LightGBM с multi-resolution | 0.763601 |
| [019_pretrained](../archive/configs/019_pretrained.yaml) | Среднее трёх AnySleep | 0.838570 |
| [041_pretrained](../archive/configs/041_pretrained.yaml) | Исходная TinySleepNet | 0.815991 |
| [053_pretrained](../archive/configs/053_pretrained.yaml) | AnySleep + Tiny, 50/50 | 0.850075 |
| [082_pretrained](../archive/configs/082_pretrained.yaml) | Tiny classifier FT | 0.820470 |
| [085_pretrained](../archive/configs/085_pretrained.yaml) | Tiny full FT | 0.819201 |
| [099_pretrained](../archive/configs/099_pretrained.yaml) | Базовая смесь AnySleep/Tiny/Tiny | 0.852765 |
| [102_pretrained](../archive/configs/102_pretrained.yaml) | Physiology-aware gate | 0.852832 |
| [106_pretrained](../archive/configs/106_pretrained.yaml) | Residual TCN | 0.848499 |
| [113_pretrained](../archive/configs/113_pretrained.yaml) | Базовая смесь + Gate + TCN | 0.853972 |
| [118_pretrained](../archive/configs/118_pretrained.yaml) | Исходная Tiny с PhysioEx voting | 0.813133 |
| [120_pretrained](../archive/configs/120_pretrained.yaml) | Tiny full FT с PhysioEx voting | 0.821011 |
| [149_pretrained](../archive/configs/149_pretrained.yaml) | AnySleep: весь непрерывный сегмент | 0.822239 |
| [156_pretrained](../archive/configs/156_pretrained.yaml) | U-Sleep с нашей блочной схемой | 0.926024 |
| [157_pretrained](../archive/configs/157_pretrained.yaml) | U-Sleep + 113, 50/50; финальная отправка | 0.923672 |
| [159_pretrained](../archive/configs/159_pretrained.yaml) | U-Sleep + 113, 90/10 | 0.927125 |
| [160_pretrained](../archive/configs/160_pretrained.yaml) | Четыре базовых эксперта | 0.902364 |
| [163_pretrained](../archive/configs/163_pretrained.yaml) | Четыре эксперта + старые поправки | 0.899825 |
| [164_pretrained](../archive/configs/164_pretrained.yaml) | Четыре эксперта + половинные поправки | 0.902776 |

## Как развивалось решение

### 1. EDA и табличные контрольные модели

Начали с проверки временной сетки, разрывов, распределения стадий и разделения
людей между train/val. На 106 физиологических EEG/EOG-признаках обучили Logistic
Regression, сравнив преобразования, веса классов и регуляризацию по grouped folds.
Основное улучшение линейной модели дал контекст соседних эпох ±2. Multi-resolution
добавил небольшой прирост. LightGBM оказался конкурентоспособным табличным baseline;
добавление всех multi-resolution признаков не улучшило его результат.
CatBoost оказался слишком дорогим для полученного качества.

### 2. Pretrained-модели и комбинации экспертов

Проверили AnySleep, U-Sleep, SleepFM и другие frozen источники, различные каналы,
fusion и способы временного декодирования. Среднее трёх AnySleep стало основной
контрольной моделью. Исходная TinySleepNet была слабее отдельно, но полезна в смеси.
Classifier/full fine-tuning Tiny добавили два отличающихся эксперта; смесь `099`
зафиксирована с весами `0.50 / 0.25 / 0.25`.

### 3. Корректоры над готовыми представлениями

Сравнили gates, residual temporal heads и фиксированные итоговые смеси. Для CV heads
использовали nested cross-fitting исходных обучаемых экспертов: outer-held прогнозы
не участвовали в обучении source-моделей для outer-train. Конфигурация `113`
объединила базовую смесь, physiology-aware gate `102` и TCN `106`.

### 4. Проверки контекста и стратегии inference

Проверили официальный PhysioEx voting TinySleepNet с усреднением logits, полную
Tiny fine-tuning с покрытием записей и несколько downstream направлений.
Увеличение блоков AnySleep вплоть до full-segment не улучшило `019` в этой серии.
Эти результаты относятся к конкретным рецептам и не доказывают бесполезность
самих семейств методов. Для дальнейшего сравнения сохранялся `113`.

### 5. U-Sleep и интеграция четвёртого эксперта

Официальный `Pretrained_Model.ckpt` U-Sleep проверили с нашей схемой preprocessing
и core128/halo64. Затем сравнили фиксированные смеси с `113` и перенос старых
Gate/TCN-поправок в смесь четырёх экспертов. Простые смеси оказались сильнее
проверенных переносов поправок. Код `165` готовит переобучение Gate4/TCN4, но
завершённого результата `165` в журнале нет, поэтому он не включён в таблицу.

Для финальной отправки команда выбрала `157`. Это не максимальный validation
score таблицы: `159` и чистая `156` выше по validation. Поэтому отдельно
показываем выбранную конфигурацию, validation и public leaderboard, не подменяя
одну оценку другой.

## Как читать журнал

- `macro_f1` и `f1_*` относятся к validation указанного протокола, а не leaderboard.
- Smoke-проверки не считаются полноценными экспериментами.
- `archived` и `interrupted` не приравниваются к завершённым моделям.
- Для смесей `infer_seconds` может означать только смешивание готовых probabilities.
  Полное время финального raw-test pipeline приведено в отчёте Kaggle-run.
- `artefacts_path` — историческое место локального запуска. Большие артефакты,
  вероятности, кэши и исходные данные не распространяются вместе с журналом.

Исследовательские реализации сохранены в [archive](../archive/README.md).
