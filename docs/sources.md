# Источники и условия использования

## Архитектуры и веса

| Компонент | Источник | Зафиксированное состояние |
|---|---|---|
| AnySleep | [dslaborg/AnySleep](https://github.com/dslaborg/AnySleep) | Revision `899bfff284f3eb3e7448bc90990f2230757470a2`, run1/2/3 |
| U-Sleep | [jesperstroem/U-Sleep-for-RBD-PD](https://github.com/jesperstroem/U-Sleep-for-RBD-PD) | Revision `19e72b632b1bd87650f7d560f42aa32d47bd9a9b`, `Pretrained_Model.ckpt` |
| TinySleepNet / PhysioEx | [guidogagl/physioex](https://github.com/guidogagl/physioex) | Revision `66b24320aa39bc635e304b0891eff41131ca54a5` |
| Исходный Tiny checkpoint | [4rooms/physioex](https://huggingface.co/4rooms/physioex/tree/fc362d1976a203f9bf47cd5072845615d573aae2/tinysleepnet-supratak) | Revision `fc362d1976a203f9bf47cd5072845615d573aae2`, sleepedf fold0 |
| Собственное дообучение | `archive/src/training` и `archive/configs` | Full-train refit `082`, `085`, `102`, `106`, seed 42 |

Полные SHA-256, размеры и происхождение — в [weights/manifest.json](../final/weights/manifest.json).
Загрузка строго проверяет имена и формы параметров; случайные веса не служат fallback.

## Notices

- Авторский AnySleep и PhysioEx source сопровождаются их MIT notices.
- Для U-Sleep-for-RBD-PD авторы указывают **non-commercial research only**.
  Это условие не заменяется MIT-лицензией других компонентов.
- Копии notices находятся в [vendor](../final/src/sleep_edf_final/vendor/).
  Единую новую лицензию на все сторонние исходники и веса репозиторий не назначает.

106 физиологических признаков — собственная реализация спектральных,
статистических и контекстных характеристик, вдохновлённая семействами признаков
из исследований автоматического staging. Это не побитовая репликация иной системы.

## Данные и зависимости

Данные получаются отдельно из [официальной поставки курса](https://www.kaggle.com/datasets/levshaazz/sleep-edf-full-psg-course-data/versions/1).
Исходные записи и метки не включены в Git.

Версии inference-зависимостей зафиксированы в двух `uv.lock`. Linux runtime содержит
CPython, PyTorch/CUDA/NVIDIA, NumPy/SciPy, scikit-learn, antropy/numba и их зависимости.
Это набор пакетов с разными лицензиями; исходные LICENSE/notices сохранены в runtime.
Его версии и SHA перечислены в [инвентаре](../final/evidence/offline_runtime_manifest.json).
