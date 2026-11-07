# Устройство финального решения

## Сигналы и временная сетка

Одна эпоха — 30 секунд, 3 000 отсчётов при 100 Гц: EEG Fpz-Cz,
EEG Pz-Oz и horizontal EOG, единицы — микровольты.
Классы везде упорядочены как `W, N1, N2, N3, REM`.
Запись разбивается на непрерывные сегменты по `epoch_index`.
Контекст и фильтрация не переносятся через пропущенные эпохи.

## Базовые модели

| Компонент | Сигнал и preprocessing | Inference |
|---|---|---|
| `019`: AnySleep run1/2/3 | Три канала; µV → V, resample до 128 Гц, clipping ±20 IQR, затем RobustScaler | Core 128, halo 64, encoder grid 16; равное среднее трёх checkpoint |
| `156`: U-Sleep | Тот же preprocessing отдельно по непрерывным сегментам; пары Fpz-Cz/EOG и Pz-Oz/EOG | Core 128, halo 64, grid 16; среднее softmax двух пар |
| `082`: TinySleepNet classifier FT | Fpz-Cz, 100 Гц, Butterworth 0.3–40 Гц, order 5 | Начальная последовательность и rolling L=20; последняя позиция следующих окон |
| `085`: TinySleepNet full FT | Тот же канал и preprocessing | Тот же rolling inference, другое дообученное состояние |

U-Sleep загружается из официального `Pretrained_Model.ckpt`, с авторской архитектурой
и строгой проверкой состояния. Дополнительного high-pass, Hann-взвешивания и смешивания
с full-record проходом в финальной схеме нет. Все backbones frozen.

## Gate, temporal correction и смесь

Сначала строится базовая смесь:

```text
B = 0.50 P019 + 0.25 P082 + 0.25 P085
P113 = 0.50 B + 0.25 P102 + 0.25 P106
P157 = 0.50 P156 + 0.50 P113
```

`102` — MLP gate: 30 confidence/disagreement признаков трёх экспертов и
106 физиологических признаков, слои `136 → 64 → 64 → 3`.
Выход задаёт динамические веса экспертов; начальная смесь — `0.50 / 0.25 / 0.25`.

`106` — residual TCN с контекстом 20 эпох: 30 confidence признаков,
128 contextual embeddings оригинальной TinySleepNet и 6 pre-classifier features
AnySleep run1. Используются исходные frozen представления, а не embeddings `085`.
TCN корректирует логарифмы базовой смеси; padding маскируется.

Mean/std обоих heads взяты из сохранённого full-train refit. В inference они не
переоцениваются. Физиологические признаки рассчитываются из raw test тем же
106-feature extractor, который использовался в исследовании.

## Численные соглашения

- Backbones — float32, без TF32. Heads — исходный CUDA float16 autocast.
- Внутри `113` три экспертных распределения сначала нормализуются в float32.
  Базовая смесь вычисляется в float64, без промежуточного float32 округления.
- `113` нормализуется и сохраняется в float32. Смешивание `156 + 113`
  выполняется в float64 с нормализацией строк и возвратом к float32.
- Сохранены исследовательские batches: Tiny CNN 64 / LSTM 16;
  embeddings CNN/LSTM 256, AnySleep embeddings до 4 блоков; heads 4096 / 512.
- Seed 42, deterministic algorithms, CuDNN benchmark и TF32 выключены,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Это зафиксированная конфигурация для Tesla T4. Изменение batch size, preprocessing,
порядка операций или среды может изменить пограничные предсказания.

## Зафиксированные состояния

В [manifest весов](../final/weights/manifest.json) перечислены все девять файлов,
их SHA-256 и происхождение: три AnySleep, официальный U-Sleep, оригинальная
TinySleepNet, два Tiny refit и два downstream head. Сторонние определения
архитектур находятся рядом с notices в `final/src/sleep_edf_final/vendor/`.
