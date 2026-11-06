# Внешние реализации и checkpoints

- AnySleep: https://github.com/dslaborg/AnySleep , revision899bfff284f3eb3e7448bc90990f2230757470a2.
  Три опубликованных run1/2/3 checkpoint, одинаковая архитектура и разные состояния.
  Авторская standalone реализация сохранена без изменения вычислений; MIT notice
  находится в src/sleep_edf_final/vendor/AnySleep-LICENSE.txt.
- PhysioEx TinySleepNet: https://github.com/guidogagl/physioex ,
  revision66b24320aa39bc635e304b0891eff41131ca54a5. АрхитектураFeatureExtractor/Classifier/
  TinySleepNet извлекается из закреплённых AST, без training/demo imports.
  MIT notice — vendor/PhysioEx-LICENSE.txt.
- Исходный Tiny checkpoint: https://huggingface.co/4rooms/physioex/tree/fc362d1976a203f9bf47cd5072845615d573aae2/tinysleepnet-supratak .
  Config указывает sleepedf fold0, raw pipeline, один100Hz EEG, L20,5классов.
- Собственные full-train refit082/085 и downstream heads102/106 сохранены вместе
  с исходными контрактами и seed42. Их обучение описано в archive configs/code.
-106 физиологических признаков — собственная адаптация семейств из работ Van Der Donckt;
  это не побитовая репликация сторонней системы. Используются NumPy/SciPy/antropy.

SHA-256, размеры и источники каждого необходимого файла весов — weights/manifest.json.
Нейросетевые веса не скачиваются с плавающей ветки во время inference.
Файлы чужих лицензий сохранены рядом с vendored исходниками.

## Offline-окружения

Отдельный runtime Input содержит CPython3.12.12/3.14.2 и distributions из двух
uv.lock, включая PyTorch/CUDA/NVIDIA, NumPy/SciPy и LLVM/numba. Это сборник
зависимостей с разными лицензиями, не единый новый MIT-пакет. Их LICENSE/notices
сохраняются внутри Python и site-packages; версии и SHA архивов перечислены в
offline_manifest.json. Код загрузчика не скачивает и не переустанавливает пакеты.
