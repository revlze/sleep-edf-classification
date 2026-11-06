# История исследования

Полный индекс — ../../experiments.csv:149 записей с зафиксированными статусами.
Глобальные ID не переиспользовались; отменённые и прерванные варианты не выдаются за
завершённые. Ниже выдержка именно provided validation, без подмены public leaderboard
или дополнительной диагностической выборкой из train.

| ID | Проверка | Val macro-F1 |
|---|---|---:|
| 001_linear | none; quantile; weights=none | 0.714794 |
| 002_linear | none; quantile; weights=sqrt | 0.723669 |
| 005_linear | neighbors; quantile; weights=sqrt | 0.759776 |
| 007_linear | multires; quantile; weights=sqrt | 0.764349 |
| 011_tree | neighbors; native_nan; weights=sqrt | 0.763927 |
| 015_tree | multires; native_nan; weights=sqrt | 0.763601 |
| 019_pretrained | AnySleep run1/2/3; fixed equal probability mean | 0.838570 |
| 041_pretrained | PhysioEx TinySleepNet fold0; EEG Fpz-Cz; 20-epoch rolling context | 0.815991 |
| 053_pretrained | Fixed probability blend: 50% AnySleep 019 + 50% TinySleepNet Fpz 041 | 0.850075 |
| 082_pretrained | Tiny041 classifier: lr=0.001, class_weight=none, focal_gamma=0 | 0.820470 |
| 085_pretrained | Tiny041 full: lr=1e-05, class_weight=sqrt, focal_gamma=0 | 0.819201 |
| 099_pretrained | Candidate B: 50% AnySleep019 + 25% Tiny082 classifier-only + 25% Tiny085 sqrt-weight full FT | 0.852765 |
| 102_pretrained | G3 physiology-aware residual gate with existing106 features | 0.852832 |
| 106_pretrained | Residual TCN context20 on frozen embeddings | 0.848499 |
| 113_pretrained | Fixed mixture: 0.5*B + 0.25*BestGate + 0.25*BestTemporal | 0.853972 |
| 117_pretrained | Full TinySleepNet fine-tuning with record coverage and subject-balanced sleep/random windows | 0.828861 |
| 118_pretrained | PhysioEx L20 logit voting on frozen 041_pretrained | 0.813133 |
| 119_pretrained | PhysioEx L20 logit voting on frozen 082_pretrained | 0.819485 |
| 120_pretrained | PhysioEx L20 logit voting on frozen 085_pretrained | 0.821011 |
| 122_pretrained | Fixed voting mixture: 0.5*019_pretrained + 0.25*119_pretrained + 0.25*120_pretrained | 0.853469 |
| 147_pretrained | Frozen AnySleep run1/2/3: core256/halo128 | 0.833853 |
| 148_pretrained | Frozen AnySleep run1/2/3: core512/halo256 | 0.829860 |
| 149_pretrained | Frozen AnySleep run1/2/3: whole continuous segment | 0.822239 |

## Что проверяли

1. EDA и физиологические признаки.106 признаков дают полезную линейную контрольную
   точку. Умеренные sqrt weights и QuantileTransformer оказались полезнее полной
   балансировки и проверенного Log+Robust варианта. Наибольший прирост дал контекст±2.
2. LightGBM улучшал табличный baseline; CatBoost оказался слишком дорогим. Добавление
   multi-resolution признаков к фиксированному бустингу не дало нужного выигрыша.
3. Frozen staging: AnySleep, U-Sleep, SleepFM и новые источники. Проверяли fusion,
   частоту выходов, контекст, Viterbi, residual corrections и fine-tuning.
4. Tiny082/085 дополнили AnySleep019. Зафиксировали B099 с весами50/25/25.
5. Gate100–104 и temporal106–110 выбирались по train OOF. Для корректной head CV
   применяли nested source cross-fitting.113 объединяет B, выбранный gate102 иTCN106.
6. Более дорогое TinyFT117, PhysioEx voting118–123, physiology/contrastive/calibration/
   fusion124–146 и рост AnySleep context147–155 не заменили выбранный финал.
   Это результаты конкретных проверенных recipes, а не доказательство бесполезности
   семейств методов вообще.153–155 — диагностика со старыми heads, не переобученная113.

## Как делали выбор

Основной инструмент — provided validation macro-F1 с F1 классов, confusion и людьми;
обучаемые настройки/CV/early stopping выполнялись внутри train по subject groups.
Выбор итоговой архитектуры многократно использовал val. Резервный099 проще, чем113.
Дополнительные12 людей из train использовались для наглядной диагностики и не считаются
независимой оценкой моделей, дообученных на train. Test labels не читались.

Время в журнале некоторых ансамблей — только смешивание готовых probabilities; его
нельзя трактовать как время полного запуска от raw test. Его зафиксирует финальный06.
Сохранённые конфигурации и источник каждого этапа находятся в archive/; подробные
локальные ZIP/кэши и черновые аудиты не публикуются в Git.
