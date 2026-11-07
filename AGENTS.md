# Работа над Sleep-EDF 2026

## Срочная проверка U-Sleep RBD/PD

Ветка feat/usleep-rbd-pd создана от master; финальный пакет остаётся в своей ветке.
156_pretrained — официальный Pretrained_Model.ckpt из U-Sleep-for-RBD-PD;
157–159 — фиксированные смеси156/113:50/50,75/25,90/10. Никакого обучения.
CLI: python -m src.usleep_rbd_run;05_pretrained по умолчанию all, полный val/test.
По явному запросу пользователя156–159 заменены ДО новой оценки: используем нашу
прежнюю схему019, не рецепт из чужого notebook. Старые результаты удалены по запросу.
Preprocessing — preprocess_segment(anysleep) по каждому непрерывному сегменту;
core128/halo64, encoder grid16, только центральные прогнозы, среднее EEG/EOG-пар.
Без high-pass, Hann, full-context смеси и внешнего целевого macro-F1.
В Input только официальный source/checkpoint и сохранённые113 val/test probabilities.
Сначала standalone val, затем фиксированные смеси; CUDA OOM сохраняет явную ошибку.
Input ZIP прежний: там нет обработанных сигналов или inference probabilities156.
Исходные113 и финальный пакет не менять. Новые метрики появляются только после Kaggle-run.

## Цель и стиль работы

Развиваем воспроизводимый исследовательский ML-проект для Kaggle-классификации
стадий сна. Приоритеты: корректная валидация, понятный код, сравнимые эксперименты.
Основная среда обучения — Kaggle; локально готовим код и выполняем небольшие проверки.

Пиши по-русски, профессионально и прямо. Не выдумывай метрики, выполненные запуски
и возможности кода. Если сведений не хватает, укажи конкретную неопределённость
и способ проверки. Отличай план, каркас, smoke-run и полноценный эксперимент.

README держи кратким: назначение проекта, запуск, структура и проверки.
Подробности реализации, замеры и рабочие заметки пиши в игнорируемую docs/.
Смена индикации или fingerprint не делает прошлые результаты недействительными;
сохраняй артефакты и метрики, не предлагай их удалять ради нового кода.

Перед правками проверяй фактические файлы и Git diff. Сохраняй пользовательские
изменения. Следуй KISS и DRY: небольшие функции, pathlib, явные пути и type hints
для основных интерфейсов. Не добавляй универсальный framework или зависимости
без использующей их задачи. Каркас уже создан по согласованию с пользователем;
новые модули и абстракции добавляй по мере появления функциональности.

## Структура и текущий этап

- `src/data.py` — записи, метки и метаданные; соединение по ID.
- `src/preprocessing.py` — преобразования сигналов.
- `src/features.py` — спектральные, статистические и контекстные признаки.
- `src/datasets.py` — выборка эпох, непрерывные сегменты, окна и маски.
- `src/models/` — отдельные семейства и адаптеры pretrained-моделей.
- `src/training/` — отдельные процедуры для табличных и нейросетевых моделей.
- `src/evaluation.py` — метрики и протоколы оценки.
- `src/inference.py` — предсказания и submission.
- `src/experiment.py` — конфигурация, организация запуска и учёт результатов.
- `src/run.py` — место для будущей CLI-точки входа.
- `configs/` — YAML экспериментов.
- `notebooks/` — тонкие Jupyter `.ipynb`: `01_eda`, `02_train`, `03_eval`, `04_submit`.
- `data/raw/` — исходная поставка вне Git.
- `artefacts/` — результаты и необходимые кэши вне Git; имя сохраняем.
- `submissions/` — CSV предсказаний вне Git.
- `docs/` — локальные рабочие заметки вне Git; существенные выводы сообщай в чате.
- `tests/` — локальные содержательные проверки; в Git только `.gitkeep`.
- `experiments.csv` — компактный независимый индекс экспериментов в Git.

Реализованы загрузчики и валидация в data.py, анализ в eda.py и 01_eda.ipynb.
Полный проход сигналов EDA ограничен train/val; для test — метаданные и заголовки.
Реализован LR pipeline: 106 признаков, контекст, групповые folds, fixed/search C,
refit, CLI и инференс. Конфигурации 001–007 занимают ID.
Линейная серия завершена; 005 и 007 — сохранённые контрольные варианты.
Для бустингов завершены 009–011_tree (LightGBM); 012_tree частично выполнен.
CatBoost закрыт, 012–014 отменены и запрещены к запуску; результаты сохраняются.
015_tree завершён: фиксированный LightGBM, 855 признаков, sqrt-веса,
параметры 011 tuned, 420 итераций, один fit на всём train, затем val.
Нет CV, screening, search, выбора родителя и early stopping; OOF отсутствует.
Табличная серия закрыта. Сохраняем 005/007_linear и 011_tree как контрольные
модели; 015_tree — завершённая проверка multi-resolution без выигрыша macro-F1.
Для zero-shot серии завершены 016_pretrained (AnySleep run1), 017_pretrained
(U-Sleep run1, голосование двух EEG/EOG-пар), 018_pretrained (SleepFMStager,
три BAS-канала) и 019_pretrained (равное среднее AnySleep run1/2/3).
CLI исходных 016–019 — python -m src.pretrained.
Серия 020–043 завершена на Kaggle: полный val → test, результаты импортированы,
вероятности и submissions проверены, в experiments.csv добавлены 24 строки.
Оригинальные ZIP — artefacts/archives/pretrained; аудит импорта —
artefacts/imports/pretrained_series; сравнение — docs/pretrained_series_results.md.
К 019 добавлен test, прежние val/config/metrics сохранены. 022 дал macro-F1
0.83862 против 0.83857 у 019; существенный выигрыш не подтверждён.
044–045 завершены на Kaggle и импортированы: OOF, refit checkpoints, val/test,
submissions и две строки experiments.csv. Веса — artefacts/trained-checkpoints;
аудит — artefacts/imports/pretrained_correction. При импорте исправлена только
сериализация NumPy-ключей истории в 12 fold last.pt; все тензоры сохранены,
исходные ZIP неизменны. Подробности — docs/correction_import_notes.md.
046–048 завершены на Kaggle и импортированы: best checkpoints по train subject
holdout, val/test probabilities, submissions и три строки experiments.csv.
Веса — artefacts/trained-checkpoints/{full,smoke}/<ID>/fit; аудит —
artefacts/imports/pretrained_finetune. В 6 last.pt нормализованы только NumPy-ключи
истории, численное состояние сохранено; оригинальные ZIP не изменены.
Технические подробности — docs/finetune_import_notes.md. Анализ качества не выполнялся.
Из сетки смесей 019 + 041 пользователь оставил только 053_pretrained: 50/50.
Его конфигурация — configs/053_pretrained.yaml; submission — submissions/053_pretrained.csv
и копия в submissions/ensemble_019_041. Остальные новые смеси 049–052 и 054–057
архивированы: YAML перенесены в artefacts/<ID>/<ID>.yaml, численные результаты
сохранены, в experiments.csv статус archived. Лишние CSV-копии удалены по запросу.
Номера архивированных экспериментов не переиспользуются.
Назначение конфигураций:
020–028 — AnySleep Early/Late Fusion и фиксированные ансамбли;
029–036 — HF4/HF8 soft/hard и ансамбли, 034/036 запускаются отдельным этапом;
037–040 — train-only дискриминативный Viterbi; 041–043 — TinySleepNet по двум EEG;
044–045 — residual CNN на вероятностях, grouped CV и full-train refit;
046–048 — classifier/partial/full fine-tuning AnySleep run1 с train subject holdout.
CLI новой серии — python -m src.pretrained_series; тот же 05_pretrained.ipynb.
По умолчанию группа cheap (A–D), SMOKE_ONLY=True, ALLOW_TRAINING=False.
E/F требуют явного --allow-training. Модели проходят val → test, сохраняют
вероятности и submission; test labels не читаются. MAP Viterbi labels и marginals
сохраняются раздельно; метрики и submission используют MAP labels.
019 val переиспользуется из проверенного Input; недостающий test дополняется.
Вероятности backbone кэшируются независимо; записи сохраняются для resume.
Компактный ZIP включает результаты/вероятности/код, исключает сигналы и веса.
Обученные best/last checkpoints выгружаются отдельно. Одно устройство на процесс;
две T4 не объединяются в 32 ГБ. Подробности — docs/pretrained_series_runbook.md.
Smoke не попадает в журнал.
NPZ предсказаний должны читаться с allow_pickle=False: subject_id сохраняем
как Unicode, без object-массивов. Исходные Kaggle ZIP сохранены в
artefacts/archives/pretrained; проверка импорта и исправление сериализации —
в artefacts/imports/pretrained. Вероятности и метки при исправлении не менялись.
008 подтянут из master и оставлен нетронутым; в новой серии не используется.
Серия 058–096 завершена пользователем на Kaggle и импортирована: 39 полных прогонов,
val/test probabilities и submissions проверены, добавлены 39 complete-строк в журнал.
097 прерван во время OOF fold0, сохранён как interrupted без выдуманных метрик;
098 выполнен только как smoke, в журнал не добавлен. Веса 096/097 пользователь
решил не сохранять. ZIP — artefacts/archives/pretrained; аудит импорта —
artefacts/imports/pretrained_expansion. Анализ качества не выполнялся.
058–063 — шесть оригинальных frozen staging-моделей; 064–081 — фиксированные смеси;
082–086 — TinySleepNet classifier/full fine-tuning; 087–091 — frozen embeddings;
092–095 — N1/N2-N3 specialists; 096–097 — SleepGPT encoder + обучаемая staging head;
098 — выбор двух новых frozen источников по train-level оценке и смесь с 053.
Пользователь выбрал только B из временных ансамблей A–M. Он сохранён как
099_pretrained: 50%019 + 25%082 + 25%085, по готовым val/test probabilities,
без нового обучения. Это offline_saved_probability_ensemble, отдельный от CLI
новой серии. Временная папка A–M удалена.
CLI — python -m src.expansion_run; тот же 05_pretrained.ipynb. По умолчанию frozen,
SMOKE_ONLY=True, ALLOW_TRAINING=False. Любые обучаемые варианты требуют --allow-training.
Три общих GroupKFold по subject_id внутри train; refit после выбора только по train.
Training smoke использует реальные окна около начала сна по train labels:
старый prefix399 состоял целиком из Wake. Полный train/val/test не меняется,
сигналы выбранного smoke выравниваются по ID, его метрики не являются оценкой качества.
Промежуточный early stopping использует фиксированные блоки по всей длине held-out
записей; итоговый OOF покрывает fold полностью. Tiny BN running stats фиксированы.
Кэш embeddings/preprocessing отдельный от OUTPUT_ROOT. Старые probabilities и
результаты сохраняются. D3/D4/E читают только готовые 106 признаков, включая test;
автоматического переизвлечения нет. Проверяются definition, source, labels и IDs.
Официальные веса новых моделей закреплены в src/expansion_assets.json и проверяются
по SHA-256; нет fallback на случайные параметры. SeqSleepNet-phan опубликован с
метаданными MASS, хотя исходное задание называло Sleep-EDF. SleepFM-2 не получает
фиктивного ID: указанный авторами публичный репозиторий вернул 404 при проверке.
SleepGPT использует опубликованный large encoder, EDF channels 3/5/7, оригинальный
STFT и новую обучаемую Linear(1536,5); partial размораживает blocks10/11 + norm.
Компактный ZIP исключает сигналы, embeddings и веса; best/last и classifier.joblib
экспортируются отдельно. Подробности — docs/expansion_runbook.md.
Серия 100–117 выполнена пользователем на Kaggle и импортирована:18 полных прогонов
и18 smoke. В experiments.csv добавлены18 complete-строк с validation-метриками;
прежние93 строки сохранены. Вероятности, ID, submissions, OOF, наличие best/last
и целостность архивов проверены без загрузки моделей и анализа качества.
Результаты — artefacts/<ID>, smoke — artefacts/smoke/<ID>, веса —
artefacts/trained-checkpoints/{full,smoke}, включая shared nested экспертов.
Готовые embeddings, folds и nested OOF — artefacts/cache/refinement/{full,smoke}.
Оригинальные триZIP — artefacts/archives/pretrained; аудит и копии журнала —
artefacts/imports/pretrained_refinement/20261105T181752Z. Checkpoints не изменялись.
CLI — python -m src.refinement_run; launcher — тот же 05_pretrained.ipynb.
100–104 — residual softmax gates с prior019/082/085=0.5/0.25/0.25 и CE+KL;
105 — официальный wav2sleep-eog с одним horizontalEOG в слоте EOG-L, без EOG-R.
Авторская архитектура поддерживает subset сигналов; горизонтальное дифференциальное
отведение отличается от авторского EOG-L, это явное ограничение рецепта.
106–110 — residual TCN/BiGRU на frozen Tiny128+AnySleep6, контекст20/100;
110 добавляет center-driven loss соседей. Это downstream модели, не воспроизведение
архитектур SleepLT/CSPT-SleepNet.111–116 — шесть фиксированных итоговых смесей.
BestGate/BestTemporal выбираются только по train OOF.117 — последний дорогой full
TinyFT с покрытием20-эпоховых tiles, sleep/random supplements и равным суммарным
весом людей в CE. Для сегментов короче20 полные обучающие окна не создаются.
Порядок групп: prepare -> gates -> wav2sleep -> temporal -> blends -> wav_blend ->
tiny -> tiny_blends. Самостоятельно полную серию не запускать.
Для корректной downstream CV простого повторного разбиения готового082/085 OOF
недостаточно: возникает утечка через обучение экспертов на других folds.
prepare один раз строит inner GroupKFold2 внутри каждого из3 outer folds:
12 внутренних fits082/085 с заранее фиксированными5 эпохами, без outer held labels.
Outer held вероятности исходных082/085 переиспользуются; итоговые heads обучаются
на исходных train OOF, а val/test используют исходные full-train refit вероятности.
Уменьшенное число людей в inner source models относительно final experts —
ограничение nested stacking; исходный author-pretraining overlap остаётся отдельным.
Smoke структурный, не независимая оценка качества. Отдельные smoke/full кэши.
Старые train019, OOF082/085, val/test и106 признаки упакованы без нового инференса
в artefacts/cachepacks/sleep-edf-refinement-inputs.zip. Старые результаты сохранены.
Embeddings прежние ZIP не содержат; при наличии отдельного кэша проверяем ID,
данные, checkpoint assets и слой. Иначе извлекаем один раз с увеличенными GPU batches.
Vectorized GPU window gathers для heads; memory-mapped pool и cached metadata
для Tiny исключают повторное распаковывание NPZ и hash/header чтения в каждом batch.
Новые Tiny CNN/LSTM batches256; AnySleep до4 одинаковых по длине исходных блоков;
117 оценивает GPU-память одним backward на16 окнах без обновления параметров,
выбирает физический batch до128, сохраняет размер и уменьшает его при CUDA OOM.
Один процесс использует одну GPU. После prepare группы gates/temporal по умолчанию
делятся на две очереди cuda:0/cuda:1 (PARALLEL_DEVICES=[] отключает параллелизм),
CPU-потоки делятся между процессами. Для каждого процесса отдельный failure-report
и один display_id; Stop завершает оба дерева uv-процессов. Остальные группы последовательные.
Общий индикатор показывает номер конфигурации,
этап/число этапов, fold/refit, эпоху/batch, время конфигурации и этапа; refresh7секунд.
Результаты, best/last веса и reusable embeddings/nested/folds экспортируются тремя
ZIP; сигналы/pools не входят в переносимый кэш. Resume требует соответствующих
RESUME_ROOT, RESUME_TRAINED_ROOT и RESUME_CACHE_ROOT и той же версии кода.
Серия118–123 выполнена пользователем наKaggle и импортирована:6 полных прогонов
и6 smoke. В experiments.csv добавлены6 complete-строк с полной validation;
прежние111 строк сохранены. ПровереныCRC/SHA, ID/CSV, logits/counts/fallback,
воспроизведение старого rolling и фиксированные смеси без загрузки моделей.
Результаты — artefacts/<ID>, smoke — artefacts/smoke/<ID>, submissions — стандартно.
CNN-кэш — artefacts/cache/voting:80 full и4 smoke записи, два уникальных backbone.
ZIP — artefacts/archives/pretrained; аудит и общий source snapshot —
artefacts/imports/pretrained_voting/20261105T210959Z. Анализ качества не выполнялся.
CLI python -m src.voting_run, тот же05_pretrained.ipynb.
118/119/120 — PhysioEx logit voting исходных041/082/085, L20.121 — 50%019+50%118;
122 — 50%019+25%119+25%120;123 — 50%019+25%118+12.5%119+12.5%120.
Обучения, CV и поиска весов нет. Старые rolling adapter/training inference не изменены.
Авторский voting закреплён PhysioEx66b24320:offset0..19, только полные окна,
logits всех позиций усредняются по фактическому числу голосов, затем softmax.
Forward/accumulation float32 без autocast; соответствие авторскому циклу проверено.
Сегменты короче20 автор пропускает. Для них явно сохраняется native-short rolling
fallback с official vote_count0; все ID остаются. В full val6 таких эпох, test11.
Основные метрики/CSV охватывают все ID; отдельно сохраняются covered-only метрики,
counts/fallback mask и mean_logits. Не выдавать fallback за официальный voting.
Voting и paired rolling получаются из одних CNN features и тех же LSTM окон.
Для full сверяется воспроизведение старых041/082/085 probabilities; smoke отличается
граничным preprocessing из-за prefix-селекции и используется только для структуры.
Сравнения на val — со041/082/085/053/099/113, по классам/людям, исправленным и новым
ошибкам, истинным переходам N1/N2 и N2/N3. Test labels не читаются.
Frozen refit checkpoints082/085 строго восстанавливаются из best.pt с проверкой
SHA, contract, refit epoch и trainable manifest. Original041/082 CNN реально
идентичны,085 отличается. Кэш2048 CNN features учитывает параметры И buffers,
данные, IDs и preprocessing; signals один раз на запись, CNN256 и LSTM512 наGPU.
Предыдущие contextual128/AnySleep6 embeddings и106-фичи для voting не применимы.
Новый Input bundle — artefacts/cachepacks/sleep-edf-voting-inputs.zip (26.5MiB):
официальные Tiny assets, full-refit best082/085 и готовые val/test references.
Внешние скачивания весов не нужны. Порядок:prepare smoke -> prepare full -> all.
prepare строит только два frozen CNN backbone по val/test; модели идут двумя GPU
очередями, после barrier121–123 читают готовые probabilities. PARALLEL_DEVICES=[]
включает последовательный режим. CPU-потоки делятся между очередями; Stop прежний.
ZIP результатов накопительный, с одним общим source snapshot. Отдельный CNN ZIP
сливает writable/read-only voting caches, исключает signals; может быть около1.8GB
до сжатия полного val/test. Новые веса не создаются и не экспортируются.
Resume:RESUME_ROOT для результатов; voting CNN_INPUTS обнаруживается вInput и
читается без копирования, RESUME_CACHE_ROOT необязателен. Следующий ID —147.
Smoke не записывается в experiments.csv; при импорте voting полные метрики
сохранялись только для всех43206 эпох val, отдельно от covered-only оценки.
Серия124–146 выполнена на Kaggle и импортирована:23 full+23 smoke. CLI `python -m src.physiology_run`,
launcher тот же05_pretrained.ipynb, та же веткаfeat/pretrained-refinement.
124/125 — native YASA0.7 EEG+EOG без EMG/демографии, classifierSHA закреплён;
126–128 — frozen Tiny contextual128 + AnySleep pre-classifier6, MLP CE/SupCon/hard negatives;
129–131 — компактная обучаемая адаптация CalibSleep (не zero-shot/точная реплика):
три channel-specific CNN, GPU STFT, cross-attention/gating, soft train-fold transition prior.
132/133_tree — event80 / event80+106, ровно2 LightGBM trials на3 общих subject folds;
134/135 — bounded class-wise residual113 без/с event80;136–145 — фиксированные75/25 смеси;
146 — .5*113+.25*BestContrastive+.25*BestCalib по train OOF. F1/F2/F3 переиспользуют
существующие IDs136–145 черезfamily_selection.json, новых дубликатов смесей нет.
В experiments.csv добавлены23 complete-строки по полной validation; прежние117 строк сохранены, всего140. Smoke в журнал не добавлен.
Старые predictions/OOF/embeddings134/106 и6 nested Tiny experts упакованы без нового
инференса в artefacts/cachepacks/sleep-edf-physiology-inputs.zip (~409MiB).
Новый frozen embedding reader проверяет ID/данные/слой/assets/SHA и запрещает extraction fallback.
Для fusion CV повторного разбиения глобального102/106 OOF недостаточно: подготовка строит
четыре непересекающиеся subject роли A(experts)/B(upstream heads)/C(correction)/D(held)
в каждом из3 outer folds. Переиспользуются6 сохранённых inner0 Tiny экспертов, новые
backbones НЕ обучаются; обучаются6 небольших102/106 heads только наB. C/D не входят
в A/B.082 использует готовые128 features после проверки всех frozen параметров и buffers;
085 использует прежние held probabilities наB/C, новый inference только наD.
Это честная role-split CV с меньшими upstream/calibration выборками, не точный recursive
nested refit исходной113; ограничение явно сохраняется. Final correction fit — на исходных
aligned train OOF; val/test — исходные full-train refit probabilities. Smoke только структурный.
Event detection выполняется один раз на непрерывный сегмент без hypnogram: peak counts
и точное перекрытие длительности с30s эпохами;5s powers/EOG dynamics, всего80 признаков.
YASA-сегмент из1 эпохи: один повтор terminal sample для strict-window API, time_norm=0;
ограничение и сообщения о старом pickle classifier сохраняются в metadata, ID не теряются.
Группы: contrastive -> yasa -> features -> events -> blends -> signals -> calib ->
fusion_prepare -> fusion -> blends -> final; all запускает эту цепочку smoke, затемfull.
По умолчанию features, SMOKE_ONLY=True, ALLOW_TRAINING=False. NN группы имеют2GPU очереди,
CPU threads делятся; feature workers также делят CPU. Batch SupCon640 / dual-stream512 /
residual4096, AMP, GPU STFT, mmap signals, OOM автоматом уменьшает batch и повторяет эпоху.
Один display на процесс,7s refresh, config/stage/fold/epoch/batch и реальное время.
Экспорт3 накопительных ZIP: results (код/метрики/predictions/submissions), cache (events/
roleOOF без signals/pools), trained (best/last/LightGBM). Resume требует всех3 соответствующих
корней и той же версии кода. Изменён только isolated pretrained env: YASA0.7.0, pandas2.2.3
(совместимость SciPy1.15); корневое окружение и завершённые recipes не менялись.

Импорт124–146: результаты artefacts/<ID>, smoke artefacts/smoke/<ID>,23 CSV в submissions/.
Веса — artefacts/trained-checkpoints/{full,smoke}/<ID>, включая6 role_stack heads в каждомрежиме;
переносимый кэш — artefacts/cache/physiology/{full,smoke,folds}. Проверены CRC/SHA архивов,
val/test IDs и CSV, арифметика метрик, OOF/folds, смеси/селекторы, полнота event/YASA/role
кэша и наличие best/last/LightGBM с доступнымиSHA. Модели не загружались, обучение не запускалось.
Во всех23 full были только missing-experiments.csv ошибки уже после сохранения численных
результатов. Журнал и общие CSV восстановлены; оригинальные errorJSON сохранены отдельно
в аудите. Отложенные этим сбоем парные сравнения восстановлены из готовых вероятностей
штатной функцией, без нового inference и без интерпретации качества моделей.
Исходные3 ZIP неизменными перемещены в artefacts/archives/pretrained. Аудит и source snapshot:
artefacts/imports/pretrained_physiology/20261106T173329Z. Из снимка вZIP фильтром*.lock был
исключён environments/pretrained/uv.lock: он восстановлен из заявленного9fbaf0c с точнымSHA.
Численные данные, параметры и checkpoint-файлы при импорте не изменялись.
В fusion_prepare обнаружено многократное чтение z['probabilities'] внутри цикла по строкам;
этот известный bottleneck пока не исправлялся при импорте. Полную завершённую серию не повторять.

Серия147–155 выполнена пользователем на Kaggle и импортирована:9 full+9 smoke для проверки
длины AnySleep context. CLI `python -m src.context_run`, тот же05_pretrained.ipynb.
147=core256/halo128,148=core512/halo256,149=целый непрерывный сегмент;150–152 — B
с соответствующим новым019;153–155 — только диагностическая подстановка в113
с сохранёнными outputs102/106. Это НЕ корректно переобученная113; диагностические
варианты исключены из выбора. A1 переиспользует019/099/113, нового ID не получает.
Официальные run1/2/3, preprocessing, float32, encoder grid16 и границы сегментов
сохранены. Только batch одинаковых длин уменьшается при OOM; при batch1 OOM
вариант явно skipped_oom, зависимые смеси skipped_dependency, без скрытого
перехода к меньшему контексту/precision или падения остальных вариантов.
Один GPU-процесс, preprocessing один раз на сегмент для всех режимов и checkpoints;
MAX_BATCH4, все доступные CPU threads. Full-record всегда batch1.
Порядок одного launcher-run: smoke -> все full val -> зафиксировать выбор по val
macro-F1 простого B (baseline099 тоже кандидат) -> test ВСЕХ успешных вариантов.
TEST_POLICY=all по умолчанию; test labels не читаются. Результаты и9 submissions
экспортируются одним накопительным ZIP без signals/embeddings/весов, с uv.lock.
Перед test строки журнала validation_only; после успешного test complete. Smoke
в журнал не попадает. Локально проверены только синтетические alignment/OOM и
короткий двухэпоховый smoke с настоящими checkpoints, не полный эксперимент.
Input — artefacts/cachepacks/sleep-edf-anysleep-context-inputs.zip (~50MiB):
три официальных AnySleep веса/source и сохранённые val/test019/082/085/099/102/106/113.
Старые feature/embedding/CNN кэши и FT weights для этой серии не нужны.
Resume: исходный Input bundle, тот же CODE_REF и RESUME_ROOT с ZIP результатов;
record probability caches включены в результат, сигналы пересчитываются только
для незавершённых записей. Старые результаты/конфигурации не меняются. Следующий ID156.

Импорт147–155:9 complete-строк по полной validation добавлены в experiments.csv,
прежние140 строк сохранены; всего149. Все9 full охватывают43206 val и66222 test эпох,
все9 smoke —160 val/test. Проверены CRC/SHA всех778 файлов ZIP, ID/метаданные,
confusion/F1, CSV и фиксированные смеси,126 record probability caches, длины окон
и границы сегментов.149 действительно выполнил целые непрерывные сегменты batch1
на val/test без temporal fallback; ошибок и OOM-пропусков нет. Модели при импорте
не загружались, inference/обучение и анализ качества не выполнялись.
Результаты — artefacts/<ID>, smoke — artefacts/smoke/<ID>,9 CSV — submissions/<ID>.csv.
Оригинальный ZIP — artefacts/archives/pretrained; аудит, source snapshot и общие
timing/selection отчёты — artefacts/imports/pretrained_context/20261106T191847Z.

Исследования завершены по решению пользователя: основной финал113_pretrained,
второй099_pretrained (B). Новые эксперименты и изменения модели не планировать
без нового запроса. Финальные публичные Kaggle notebooks и пакет для сдачи ещё
не подготовлены; текущий05 — исследовательский launcher последней серии.
Следующий этап согласуется отдельно: два выполненных публичных notebook с точным
воспроизведением всех test id,label из сохранённых состояний, без обучения,
с закреплёнными Inputs/кодом/зависимостями и SHA-256 весов/CSV. Валидационные
метрики experiments.csv не подменять числами дополнительной оценки или Kaggle.

Исправлена инициализация experiments.csv в новом OUTPUT_ROOT: smoke не пишет журнал,
а прежний full падал после сохранения metrics/probabilities/submission на отсутствии CSV.
upsert_result теперь поддерживает новый журнал под общей блокировкой.05 перед запуском,
после него и перед экспортом вызывает src.result_recovery: восстановление индекса/копий CSV
по завершённым metrics+provenance, удаление только исправленных missing-journal ошибок.
Численные исходники и guarded implementation124–146 не менялись, checkpoints совместимы;
не создавать новый OUTPUT_ROOT и не пересчитывать завершённые модели ради этого исправления.

Другие семейства моделей остаются каркасом. Проверенные команды и ограничения
смотри в README. Полную серию выполняем на Kaggle; smoke не является экспериментом
качества и не записывается в experiments.csv.

## Окружение

Используй `uv` для управления зависимостями и запуска Python. Объявляй зависимости
в `pyproject.toml`, фиксируй их в `uv.lock`; оба файла должны версионироваться.
Python 3.14 выбран в `.python-version`, требование проекта — `>=3.14`. Не меняй
версию без конкретного обоснования совместимостью библиотек или средой Kaggle.
Pretrained имеет отдельные environments/pretrained/pyproject.toml и uv.lock:
Python 3.12, PyTorch 2.6 (CUDA 12.4 на Linux, CPU на Windows), Braindecode на
фиксированном Git revision. Это изолированное окружение для официальных моделей;
корневое окружение LR/бустингов не меняется. Общие модули, импортируемые pretrained,
должны оставаться совместимыми с Python 3.12. Ruff targets для новых файлов заданы явно.
В Kaggle явно проверяй Python, GPU и CUDA перед выбором сборки PyTorch.

Подключены NumPy, Polars, SciPy, pandas, Matplotlib, Seaborn, tqdm и ранее добавленный marimo;
в dev — ipykernel, Ruff, pytest, nbformat и nbclient.
Для LR добавлены OmegaConf, scikit-learn, antropy, joblib и threadpoolctl. Бустинг,
PyTorch, pretrained-библиотеки и другие инструменты добавляются вместе с задачами,
которые их используют. MNE не нужен только ради чтения готовых NPZ.
MLflow необязателен и не является основой текущего pipeline.

Для shared/exclusive блокировок на Windows используется portalocker[win32];
на macOS/Linux остаётся стандартный fcntl. Эта зависимость условная по платформе.

Для деревьев добавлены LightGBM и CatBoost. По умолчанию training.threads=auto
использует все доступные логические CPU кроме одного; -1 использует все.
Число потоков можно передать через --threads. Не меняй параметры завершённых LR.
Деревья читают только готовый кэш и сохранённые full-data folds: --cache-root и
--fold-root могут указывать на read-only Input. При несовместимости не запускай
извлечение автоматически. Инструкции и замеры — в docs/boosting_runbook.md.
02_train по умолчанию запускает только 015_tree, STAGE="fixed".
Обучение запускает пользователь. Не возобновляй старую автоматическую серию.

Не смешивай установку зависимостей через `uv` с неучтёнными установками в notebook.
`uv sync` не переключает уже запущенное ядро Kaggle; окружение процесса обучения
и ядро notebook нужно различать. Пути `/kaggle/input` проверяй по фактическому
подключению dataset; результаты пиши в доступный для записи каталог.

`notebooks/00_kaggle.ipynb` — launcher для приватного GitHub через Kaggle Secrets
(`GITHUB_TOKEN`, read-only Contents). Не записывай секреты в URL, файлы и вывод.
Он запускает uv CLI в отдельном Python 3.14.2, сохраняет результаты и копию кода
в /kaggle/working; окружение и загрузки uv держит в /tmp. Ветки/коммиты сначала
должны быть отправлены на GitHub; автоматически не коммить и не пушь.
Вывод дочернего процесса notebooks передавай через src.notebook.run_command.
Показывай один обновляемый индикатор на этап (все записи / всё обучение),
с интервалом 7 секунд, без индикаторов на каждый файл, пакет, C или fold.
В Jupyter обновляй один display_id; не печатай carriage-return поток в ячейку.
Не выдумывай процент сходимости и точное ETA.
Для выгрузки результатов используй src.export: один ZIP со всем OUTPUT_ROOT,
включая кэши, журнал и снимок кода. Launcher показывает одну ссылку для скачивания.

## Конфигурации и журнал

Каждый эксперимент получает глобальный последовательный ID с `001`, общий для
всех семейств. Формат — `NNN_однослово`: `001_linear`, `002_tree`, `003_pretrained`.
Слово обозначает семейство, а не комбинацию техник. Номера не переиспользуются.
Новый seed или изменение фиксированных параметров означает новый ID.
Исключение — заранее объявленный hyperparameter search: его trials являются частью
одного эксперимента и не получают отдельные глобальные ID. Сохраняй диапазон,
порядок trials, folds, все результаты и выбранные параметры. Фиксированный baseline
и подобранную версию сравнивай отдельно на одних folds; в experiments.csv — итоговый
выбранный вариант. Для точного повторения
той же конфигурации не создавай новый ID и не перезаписывай результат молча.

Пути согласованы: `configs/<ID>.yaml`, `artefacts/<ID>/`, `submissions/<ID>.csv`.
В начале YAML кратко опиши проверяемую гипотезу. Включай ID, seed, пути и версию
данных, preprocessing, модель, обучение и оценку. Читай YAML через OmegaConf после
подключения этой функциональности; сохраняй разрешённую конфигурацию тоже в YAML.
Не занимай ID фиктивными конфигурациями и не создавай сложное наследование заранее.

`experiments.csv` хранит одну итоговую строку на конфигурацию: ID, путь к YAML,
модель, короткую характеристику, seed, протокол оценки, macro-F1, F1 пяти классов,
время обучения и инференса в секундах, статус и путь к артефактам. Не записывай
несуществующие запуски и нулевые метрики вместо отсутствующих. Пока запусков нет,
в CSV только заголовок. Обновление журнала реализуй в одном общем месте.

Сохраняй только полезные результаты: необходимые веса, конфигурацию, метрики,
время, confusion matrix и вероятности с ID для анализа и ансамбля. Фиксируй версию
данных, commit кода и окружение. Для pretrained фиксируй источник, revision/checksum
checkpoint, лицензию, порядок классов и точное преобразование входа.
MLflow можно добавить позже поверх общего результата запуска, без обязательного
сервера и без зависимости CSV от MLflow.

## Данные и корректность

Источник поставки — подключённый пользователем dataset full_psg_v1;
владельца Kaggle и slug не подставляй без проверки фактического подключения.

Локальные пути: `data/raw/epochs/{train,val,test}/`, рядом `labels.csv`,
`sample_submission.csv`, `team_info.csv`, `data_description.md`, `MANIFEST.json`.
Сверяй фактическую поставку с оригинальным описанием и SHA-256 из manifest;
не подставляй описание другой редакции.
`team_info.csv` — форма команды, не таблица испытуемых. `subject_id` есть в метках.

Ожидаемая редакция — `full_psg_v1`. Один NPZ соответствует записи; ключи:
`x`, `ids`, `epoch_index`, `fs`, `channel_names`. Эпоха длится 30 секунд при 100 Гц;
`x`: float32, `(N, 3000, 3)`. По постановке единицы — микровольты. Порядок каналов:
`EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`. EOG не является третьим EEG-каналом.
Преобразование единиц и осей делай явно в соответствии с моделью; перевод в вольты
не является универсальным требованием для всех моделей.

Классы: `0=W`, `1=N1`, `2=N2`, `3=N3`, `4=REM`. Movement/unknown не добавляем.
Соединяй сигналы с метками по `id`. Проверяй уникальность ID, соответствие записей
и эпох, форму, частоту, каналы, конечность сигналов, допустимость меток и баланс
классов. Проверку нескольких файлов не называй проверкой всего набора.

Контекстные окна, фильтрация, ресемплинг и сглаживание учитывают границы записей
и разрывы `epoch_index`. Соседние строки не обязательно соседние интервалы времени.
Не восстанавливай отсутствующие сигналы выдуманными значениями без отдельного
обоснования. Не смешивай поставку с кэшами других редакций. Ключ кэша учитывает
версию данных и preprocessing, а для embeddings — checkpoint и fold encoder.
Не загружай весь набор сигналов в RAM без оценки памяти.

По описанию поставки тестовые сигналы не используются для обучения, псевдоразметки,
адаптации или оценки общих статистик. Допустим фиксированный preprocessing каждой
записи отдельно и её контекст. Не сопоставляй тест с исходными гипнограммами
для восстановления скрытых меток. Соблюдай оригинальные правила соревнования.

## Валидация и предсказания

Сохраняй выданное разделение. Собственные holdout/CV внутри train группируй по
`subject_id`, учитывая несколько ночей человека. Случайное разделение эпох
недопустимо. Проверяй отсутствие пересечений испытуемых по доступным метаданным;
отсутствие совпадающих ID эпох само по себе этого не доказывает.

Обучаемые scaler, отбор признаков и другие обучаемые преобразования выполняют
`fit` только на тренировочной части fold. Фиксированная нормализация отдельной
записи при инференсе pretrained-модели допустима как явно заданная часть рецепта,
без использования меток; её нельзя путать с обучением общего scaler на val/test.

Основная метрика — macro-F1 по всем оцениваемым эпохам с фиксированными пятью
классами. Дополняй метриками классов и confusion matrix. Средний F1 по людям,
внутренняя CV, выданная val и Kaggle — разные оценки. Многократно использованная
для подбора val не является независимой финальной проверкой. Фиксируй seed,
версию разбиения и ограничения воспроизводимости.

Модели возвращают вероятности в порядке W, N1, N2, N3, REM с привязкой к ID.
Epoch encoders должны поддерживать извлечение embeddings при их реализации.
Для ансамбля используй согласованные validation/OOF-предсказания; обучающие
предсказания не подходят для подбора весов. Границы контекста действуют и здесь.

Итоговый CSV: `id,label`, целочисленные классы 0–4, точные ID и порядок строк
`sample_submission.csv`. Связывай его с конфигурацией, данными и checkpoint.

## Проверки и завершение

Доступные проверки окружения и оформления:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
```

При появлении функциональности добавляй локальные проверки критичных свойств:
привязка меток, группы испытуемых, границы контекста, формы и единицы, метрики,
загрузка checkpoint и сборка submission. Не пиши проверки, повторяющие реализацию.
Локальные pytest-проверки покрывают данные, признаки, folds, поиск C, checkpoint
и submission; команда запуска — uv run python -m src.run configs/001_linear.yaml.
Для вычислительных изменений сначала выполняй небольшой содержательный smoke-run.
Полноценное обучение запускай, когда оно необходимо для поставленной задачи.

В Git должны попадать код, notebooks без outputs, конфигурации, README,
`experiments.csv`, `.python-version` и `uv.lock`. Исключай данные, окружения, веса,
локальные результаты, кэши, заметки и содержимое `tests/`, кроме `.gitkeep`.
Поддерживай `.gitignore`. Не коммить и не публикуй без запроса пользователя.
В итоговом ответе сообщай изменения, реально выполненные проверки и ограничения.

## Интеграция U-Sleep160–165

160–164: Base4 и фиксированные поправки старых102/106 с проекцией на симплекс.
165: новые Gate4/TCN4, только головы, прежние рецепты102/106 и prior1/3,1/3,1/6,1/6.
CLI python -m src.integration_run; notebook05 сначала quick, отдельный ZIP, затем train.
Новый Input sleep-edf-usleep-integration-inputs.zip содержит frozen embeddings,106 features,
nested OOF,folds, saved probabilities и официальный U-Sleep checkpoint. Старые модели
не обучать и не пересчитывать; единственный новый frozen inference —156 по train.
Исходные113/156–159 сохранять. Новые результаты/веса/кэш — OUTPUT_ROOT/phase1 и phase2.
Resume: два отдельных ZIP; численные контракты проверяются. Полный Kaggle-run ещё не выполнен.
