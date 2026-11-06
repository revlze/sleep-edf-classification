# Sleep-EDF 2026

Классификация стадий W, N1, N2, N3 и REM по двум EEG и одному EOG. Решение SBMTeam
объединяет AnySleep, дообученные TinySleepNet и небольшие heads над их прогнозами.
Исследования завершены; выбраны основной ансамбль **113** и резервный **099**.

| Решение | Provided validation macro-F1 | Public leaderboard¹ |
|---|---:|---:|
| 113: B + physiology gate + residual TCN | 0.853972 | 0.85207 |
| 099: 50% AnySleep + 25% Tiny082 + 25% Tiny085 | 0.852765 | 0.84951 |

¹ По сохранённым отправкам/скриншоту команды. Это отдельная оценка, не validation и
не private leaderboard. Многократно использованный val не является независимым holdout;
авторское pretraining AnySleep пересекается с частью validation-людей.

## Финальный запуск

[final/notebooks/06_final.ipynb](final/notebooks/06_final.ipynb) — единый Kaggle notebook.
По умолчанию создаёт `submission.csv` для113. `OUTPUT_SECOND_SUBMISSION=True`
дополнительно создаёт `submission_2.csv` для099. Все прогнозы вычисляются из raw test;
обучения и чтения test labels нет. Интернет нужен только для установки зависимостей.

Подключаются исходный full_psg_v1 **version1** и подготовленный Input с кодом/весами.
Инструкции и упаковка Input — в [final/README.md](final/README.md).
Код не зависит от приватного GitHub или Kaggle Secrets.

**Статус воспроизведения:** локальный структурный smoke пройден. Полный запуск06 на T4
и проверка точного совпадения всех66222 пар id,label с выбранными отправками ещё ожидаются.
Пустой notebook без outputs и отсутствие PDF не выдаются за завершённую сдачу.

## Устройство решения и исследования

- [Решение и численные соглашения](final/docs/solution.md)
- [История экспериментов и отрицательные результаты](final/docs/research.md)
- [Воспроизводимость и ограничения](final/docs/reproducibility.md)
- [Внешний код и веса](final/docs/third_party.md)

```text
final/             Frozen inference, notebook06, два конфига, веса, тесты и документация
archive/           Исторические src, configs, notebooks и окружения исследований
experiments.csv    Полная история149 записей; final/experiments.csv содержит выбранные2
```

Все численные validation-метрики журнала сохранены; изменены только пути после переноса.
Локальные данные, кэши, архивы прогонов и старые рабочие заметки остаются вне Git.
Материалы преподавателю и персональные формы готовятся отдельно от публичного решения.
