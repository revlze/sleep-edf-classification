# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1", "matplotlib==3.11.2"]
# ///
"""Рисунки документации из raw train, открытых меток и сохранённых оценок.

uv run --script docs/figures/generate.py --data-root data
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("W", "N1", "N2", "N3", "REM")
CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
COLORS = ("#D9A83F", "#40AAA9", "#5B8BD3", "#8C6DBE", "#D27199")
INK, MUTED, GRID = "#193047", "#52667B", "#DCE4EC"
DARK, DARK_TEXT = "#112335", "#C1D1DF"
VERSIONS = {"numpy": "2.5.4", "scipy": "1.18.1", "matplotlib": "3.11.2"}
EXAMPLE_INDICES = (1892, 750, 1252, 844, 1307)
COVER_INDEX = 1279
DPI = 140
CHECKPOINTS = (
    ("001_linear", "Logistic Regression · 106 признаков"),
    ("005_linear", "Logistic Regression · контекст ±2"),
    ("011_tree", "LightGBM · контекст"),
    ("019_pretrained", "AnySleep · три состояния"),
    ("053_pretrained", "AnySleep + TinySleepNet · 50/50"),
    ("113_pretrained", "Базовая смесь + Gate + TCN"),
    ("156_pretrained", "U-Sleep"),
    ("157_pretrained", "U-Sleep + ансамбль · 50/50"),
    ("159_pretrained", "U-Sleep + ансамбль · 90/10"),
)


def number(value, digits=0):
    if digits:
        return f"{value:.{digits}f}".replace(".", ",")
    return f"{int(value):,}".replace(",", " ")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Record:
    ids: np.ndarray
    indices: np.ndarray
    labels: np.ndarray
    signals: np.ndarray | None

    def position(self, index):
        found = np.flatnonzero(self.indices == index)
        require(len(found) == 1, f"Нет единственной эпохи {index}")
        return int(found[0])

    def epoch(self, index):
        require(self.signals is not None, "Сигналы записи не загружены")
        sample = self.signals[self.position(index)]
        require(np.isfinite(sample).all(), f"NaN/Inf в эпохе {index}")
        return sample


def read_labels(path):
    counts = {split: Counter() for split in ("train", "val")}
    subjects = {split: set() for split in counts}
    selected, seen = {}, set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "label", "split", "record_id", "epoch_index", "subject_id"}
        require(required <= set(reader.fieldnames or []), "Неверная схема labels.csv")
        for row in reader:
            split, label, identity = row["split"], int(row["label"]), row["id"]
            require(split in counts and label in range(5), "Неверный split или label")
            require(identity not in seen, f"Повтор ID: {identity}")
            seen.add(identity)
            counts[split][label] += 1
            subjects[split].add(row["subject_id"])
            if row["record_id"] in ("SC4011", "SC4092"):
                require(split == "train", "Примеры должны принадлежать train")
                selected[identity] = (label, int(row["epoch_index"]), row["record_id"])
    return counts, subjects, selected


def read_record(path, label_rows, *, signals):
    with np.load(path, allow_pickle=False) as source:
        require(
            set(source.files) == {"x", "ids", "epoch_index", "fs", "channel_names"},
            f"Неверные ключи NPZ: {path.name}",
        )
        require(int(source["fs"]) == 100, "Ожидается 100 Гц")
        require(tuple(source["channel_names"].tolist()) == CHANNELS, "Порядок каналов")
        identities = source["ids"].astype(str)
        indices = source["epoch_index"].copy()
        values = source["x"].copy() if signals else None
    require(len(identities) == len(set(identities)), "Повтор ID в NPZ")
    require(indices.shape == identities.shape, "Несогласованные длины NPZ")
    require(np.all(np.diff(indices) > 0), "Индексы должны строго возрастать")
    labels = []
    for identity, index in zip(identities, indices, strict=True):
        require(identity in label_rows, f"Нет открытой метки для {identity}")
        label, label_index, record_id = label_rows[identity]
        require(label_index == int(index) and record_id == path.stem, "Join по ID")
        labels.append(label)
    if signals:
        require(values.dtype == np.float32, "Ожидается raw float32")
        require(values.shape == (len(indices), 3000, 3), "Неверная форма сигнала")
    return Record(identities, indices, np.array(labels), values)


def read_evidence(path, counts, subjects):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    require(tuple(metrics["stage_order"]) == STAGES, "Порядок стадий в оценке")
    require(metrics["evaluation_set"] == "provided validation", "Другая оценка")
    matrix = np.array(metrics["confusion_matrix"], dtype=np.int64)
    support = matrix.sum(axis=1)
    require(matrix.shape == (5, 5) and np.all(matrix >= 0), "Матрица ошибок")
    require(support.tolist() == [counts["val"][k] for k in range(5)], "Объём val")
    require(int(matrix.sum()) == metrics["epoch_count"], "Число эпох в оценке")
    require(len(subjects["val"]) == metrics["subject_count"], "Число людей в val")
    f1 = 2 * matrix.diagonal() / (support + matrix.sum(axis=0))
    require(np.allclose(f1, metrics["f1"], rtol=0, atol=1e-12), "F1 и матрица")
    require(
        math.isclose(float(f1.mean()), metrics["macro_f1"], abs_tol=1e-12), "macro-F1"
    )
    return metrics


def read_experiments(path, metrics):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["experiment_id"]: row for row in csv.DictReader(stream)}
    selected = []
    for identity, label in CHECKPOINTS:
        row = rows[identity]
        require(row["status"] == "complete", f"Незавершённый эксперимент: {identity}")
        require("provided_val" in row["evaluation_set"], f"Другая оценка: {identity}")
        selected.append(
            {
                "id": identity,
                "label": label,
                "macro_f1": float(row["macro_f1"]),
                "evaluation_set": row["evaluation_set"],
            }
        )
    require(
        math.isclose(selected[-2]["macro_f1"], metrics["macro_f1"], abs_tol=1e-12),
        "Оценка 157 в журнале и evidence расходится",
    )
    return selected


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def heading(fig, title, subtitle):
    fig.text(0.045, 0.945, title, fontsize=23, weight="bold", va="top")
    fig.text(0.045, 0.885, subtitle, fontsize=13, color=MUTED, va="top")


def save(fig, output, name):
    fig.savefig(
        output / name,
        dpi=DPI,
        facecolor=fig.get_facecolor(),
        metadata={"Software": "Matplotlib 3.11.2"},
    )
    plt.close(fig)


def cover(record, output):
    fig = plt.figure(figsize=(14.4, 6), facecolor=DARK)
    fig.text(
        0.045,
        0.93,
        "Определение стадии сна",
        color="#F2F6FA",
        fontsize=28,
        weight="bold",
        va="top",
    )
    fig.text(
        0.045,
        0.84,
        "30 секунд EEG/EOG → одна из пяти стадий",
        color=DARK_TEXT,
        fontsize=17,
        va="top",
    )
    panel_color = "#193247"
    for x, width in ((0.045, 0.425), (0.60, 0.355)):
        fig.add_artist(
            FancyBboxPatch(
                (x, 0.22),
                width,
                0.51,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=fig.transFigure,
                facecolor=panel_color,
                edgecolor="#2E4A60",
                linewidth=1,
                zorder=-1,
            )
        )
    fig.text(
        0.045 + 0.425 / 2,
        0.675,
        "Сигналы за 30 секунд",
        fontsize=17,
        color="#F2F6FA",
        weight="bold",
        ha="center",
        va="center",
    )
    fig.text(
        0.60 + 0.355 / 2,
        0.675,
        "Одна из пяти стадий",
        fontsize=17,
        color="#F2F6FA",
        weight="bold",
        ha="center",
        va="center",
    )
    left = fig.add_axes((0.125, 0.32, 0.32, 0.29), facecolor=panel_color)
    sample = record.epoch(COVER_INDEX)
    shifted = sample + np.array([320, 160, 0])
    for channel, color in enumerate(("#80CDD2", "#A1BCE8", "#D5AAD0")):
        left.plot(
            np.arange(3000) / 100,
            shifted[:, channel],
            color=color,
            linewidth=0.8,
        )
    left.set(xlim=(0, 30), ylim=(shifted.min() - 25, shifted.max() + 30))
    left.set_yticks([320, 160, 0], ["EEG\nFpz-Cz", "EEG\nPz-Oz", "EOG"])
    left.set_xticks([0, 15, 30])
    left.set_xlabel("Время внутри эпохи, с", fontsize=12)
    left.spines["left"].set_visible(False)
    left.tick_params(axis="y", length=0, labelsize=11, pad=10)
    left.tick_params(colors=DARK_TEXT)
    left.xaxis.label.set_color(DARK_TEXT)
    left.spines["bottom"].set_color("#496479")
    left.plot([28.8, 28.8], [55, 105], color=DARK_TEXT, linewidth=1.5)
    left.text(27.8, 80, "50 мкВ", ha="right", va="center", color=DARK_TEXT, fontsize=10)
    fig.add_artist(
        FancyArrowPatch(
            (0.495, 0.45),
            (0.575, 0.45),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2,
            color="#9BB4C9",
        )
    )
    fig.text(
        0.535,
        0.53,
        "Задача\nмодели",
        ha="center",
        va="center",
        fontsize=13,
        color=DARK_TEXT,
    )
    descriptions = ("Бодрствование", "Сон N1", "Сон N2", "Сон N3", "REM-сон")
    for index, (stage, description, color) in enumerate(
        zip(STAGES, descriptions, COLORS, strict=True)
    ):
        y = 0.60 - index * 0.075
        if stage == "N2":
            fig.add_artist(
                FancyBboxPatch(
                    (0.618, y - 0.031),
                    0.32,
                    0.062,
                    boxstyle="round,pad=0.004,rounding_size=0.008",
                    transform=fig.transFigure,
                    facecolor="#263F5B",
                    edgecolor=color,
                    linewidth=1.4,
                    zorder=-1,
                )
            )
            fig.text(
                0.925,
                y,
                "этот пример",
                fontsize=11,
                color="#B9D4FF",
                ha="right",
                va="center",
            )
        fig.text(0.635, y, stage, fontsize=20, weight="bold", color=color, va="center")
        fig.text(0.71, y, description, fontsize=15, color="#EDF3F8", va="center")
    fig.text(
        0.045,
        0.125,
        "train · SC4011 · эпоха 1279 · N2 — метка примера из labels.csv",
        fontsize=12,
        color=DARK_TEXT,
    )
    fig.text(
        0.045,
        0.065,
        "EEG — два канала ЭЭГ; EOG — канал ЭОГ. Сигналы сдвинуты по вертикали, амплитуды сохранены.",
        fontsize=11,
        color=DARK_TEXT,
    )
    save(fig, output, "cover.png")


def atlas(record, output):
    samples = np.stack([record.epoch(index) for index in EXAMPLE_INDICES])
    limits = np.ceil(np.max(np.abs(samples), axis=(0, 1)) / 25) * 25
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(14.4, 8.5),
        gridspec_kw={
            "left": 0.09,
            "right": 0.975,
            "top": 0.77,
            "bottom": 0.145,
            "hspace": 0.38,
            "wspace": 0.20,
        },
    )
    heading(
        fig,
        "Как выглядят пять стадий сна",
        "Train · SC4011 · 100 Гц · каждая колонка — одна размеченная 30-секундная эпоха",
    )
    for column, (sample, index) in enumerate(
        zip(samples, EXAMPLE_INDICES, strict=True)
    ):
        for channel, ax in enumerate(axes[:, column]):
            ax.plot(
                np.arange(3000) / 100,
                sample[:, channel],
                color=COLORS[column],
                linewidth=0.7,
            )
            ax.set(xlim=(0, 30), ylim=(-limits[channel], limits[channel]))
            ax.set_xticks([0, 15, 30])
            ax.set_yticks([-limits[channel], 0, limits[channel]])
            ax.tick_params(labelleft=column == 0, labelbottom=channel == 2)
            ax.axhline(0, color=GRID, linewidth=0.6, zorder=-1)
            ax.grid(axis="x", alpha=0.35)
            if column == 0:
                ax.set_ylabel(f"{CHANNELS[channel]}\nмкВ", labelpad=9)
            if channel == 2:
                ax.set_xlabel("Время, с")
            if channel == 0:
                ax.set_title(
                    f"{STAGES[column]}\nэпоха {index:05d}",
                    color=COLORS[column],
                    fontsize=15,
                    pad=13,
                )
    fig.text(
        0.045,
        0.055,
        "Без дополнительной фильтрации. Масштаб каждого канала одинаков для всех пяти стадий.",
        fontsize=12,
        color=MUTED,
    )
    save(fig, output, "stage_examples.png")


def errors(metrics, output):
    matrix = np.array(metrics["confusion_matrix"])
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list(
        "validation", ["#F1F5FA", "#9DBCE0", "#205C95"]
    )
    fig = plt.figure(figsize=(14.4, 6.9))
    heading(
        fig,
        "Где ошибается финальный ансамбль",
        "Валидация · 43 206 эпох · 8 людей · финальная модель 157",
    )
    ax = fig.add_axes((0.075, 0.24, 0.405, 0.52))
    ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(5), STAGES)
    ax.set_yticks(range(5), STAGES)
    ax.set(xlabel="Предсказанная стадия", ylabel="Истинная стадия")
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0, pad=8)
    for row in range(5):
        for column in range(5):
            ax.text(
                column,
                row,
                f"{number(fractions[row, column] * 100, 1)}%\n{number(matrix[row, column])}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if fractions[row, column] > 0.6 else INK,
            )
    ax.set_title("Доля внутри истинной стадии", fontsize=15, pad=16)
    bar = fig.add_axes((0.625, 0.24, 0.32, 0.52))
    bar.barh(range(5), np.ones(5), color="#F0F3F7", height=0.57, zorder=0)
    bar.barh(range(5), metrics["f1"], color=COLORS, height=0.57)
    bar.set_yticks(range(5), STAGES)
    bar.invert_yaxis()
    bar.set(xlim=(0, 1.13), xlabel="F1")
    bar.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,50", "0,75", "1"])
    bar.spines["left"].set_visible(False)
    bar.tick_params(axis="y", length=0, pad=10)
    bar.set_title("F1 каждой стадии", fontsize=15, pad=16)
    for i, value in enumerate(metrics["f1"]):
        bar.text(
            1.025,
            i,
            number(value, 3),
            va="center",
            fontsize=13,
            weight="bold" if i == 1 else "normal",
        )
    fig.text(
        0.045,
        0.11,
        "Цвет матрицы — доля строки; в ячейке — процент и число эпох. N1 имеет наименьший F1.",
        fontsize=12,
        color=MUTED,
    )
    save(fig, output, "validation_errors.png")


def gaps(record):
    return [
        {
            "start_epoch": int(a + 1),
            "end_epoch_exclusive": int(b),
            "missing_epochs": int(b - a - 1),
        }
        for a, b in zip(record.indices[:-1], record.indices[1:], strict=True)
        if b - a > 1
    ]


def challenges(counts, subjects, record, output):
    missing = gaps(record)
    largest = max(missing, key=lambda gap: gap["missing_epochs"])
    require(largest["missing_epochs"] == 802, "Изменился диагностический разрыв")
    fig = plt.figure(figsize=(14.4, 9.2))
    heading(
        fig,
        "Что усложняет работу с данными",
        "Доли стадий и разрывы исходной 30-секундной сетки",
    )
    ax = fig.add_axes((0.085, 0.51, 0.87, 0.29))
    for i, split in enumerate(("train", "val")):
        total = sum(counts[split].values())
        percentages = np.array([counts[split][k] / total * 100 for k in range(5)])
        positions = np.arange(5) + (i - 0.5) * 0.32
        ax.bar(
            positions,
            percentages,
            width=0.29,
            color=COLORS if i == 0 else "white",
            edgecolor=COLORS,
            hatch=None if i == 0 else "///",
            linewidth=1.5,
            zorder=3,
        )
        for x, value in zip(positions, percentages, strict=True):
            ax.text(x, value + 1.8, f"{number(value, 1)}%", ha="center", fontsize=12)
    ax.set_xticks(range(5), STAGES)
    ax.set(ylim=(0, 82), ylabel="Доля эпох, %")
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(axis="y", zorder=0, alpha=0.4)
    legend = [
        Patch(
            facecolor="#718BA5",
            label=f"Train · {number(sum(counts['train'].values()))} эпох · {len(subjects['train'])} людей",
        ),
        Patch(
            facecolor="white",
            edgecolor="#718BA5",
            hatch="///",
            label=f"Val · {number(sum(counts['val'].values()))} эпох · {len(subjects['val'])} людей",
        ),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=12)
    fig.text(
        0.085,
        0.83,
        "А. Бодрствование занимает большую часть эпох",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.085,
        0.405,
        "Б. SC4092: сохранённые эпохи не образуют непрерывную запись",
        fontsize=16,
        weight="bold",
    )
    timeline = fig.add_axes((0.085, 0.15, 0.87, 0.20))
    boundaries = np.r_[
        0, np.flatnonzero(np.diff(record.indices) != 1) + 1, len(record.indices)
    ]
    spans = [
        (record.indices[a] / 120, (b - a) / 120)
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    timeline.broken_barh(spans, (0.25, 0.33), facecolors="#43859F")
    for gap in missing:
        timeline.add_patch(
            Rectangle(
                (gap["start_epoch"] / 120, 0.25),
                gap["missing_epochs"] / 120,
                0.33,
                facecolor="white",
                edgecolor="#BAC8D4",
                hatch="////",
                linewidth=0.5,
            )
        )
    middle = (largest["start_epoch"] + largest["end_epoch_exclusive"]) / 240
    timeline.text(
        middle,
        0.76,
        "802 эпохи · 6 ч 41 мин\nсодержание неизвестно",
        ha="center",
        va="center",
        fontsize=13,
    )
    timeline.text(0, 0.77, "SC4092 · train", fontsize=13, va="center")
    timeline.set(xlim=(0, 24.2), ylim=(0, 1), xlabel="Время от начала исходной PSG, ч")
    timeline.set_xticks([0, 4, 8, 12, 16, 20, 24])
    timeline.set_yticks([])
    timeline.spines["left"].set_visible(False)
    timeline.legend(
        handles=[
            Patch(facecolor="#43859F", label="Сохранённые эпохи"),
            Patch(
                facecolor="white",
                edgecolor="#BAC8D4",
                hatch="////",
                label="Пропущенные эпохи",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0, -0.44),
        ncol=2,
        frameon=False,
        fontsize=12,
        borderaxespad=0,
    )
    fig.text(
        0.045,
        0.035,
        f"{len(missing)} разрывов → {len(missing) + 1} непрерывных сегментов. Контекст через разрыв не переносится.",
        fontsize=12,
        color=MUTED,
    )
    save(fig, output, "data_challenges.png")


def spectra(record, output):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.4, 6.6),
        gridspec_kw={
            "left": 0.08,
            "right": 0.97,
            "top": 0.69,
            "bottom": 0.26,
            "wspace": 0.32,
        },
    )
    heading(
        fig,
        "Те же примеры — в частотном представлении",
        "Train · SC4011 · те же пять 30-секундных эпох, что и в атласе",
    )
    for stage, index in enumerate(EXAMPLE_INDICES):
        frequency, psd = welch(
            record.epoch(index),
            fs=100,
            window="hann",
            nperseg=400,
            noverlap=200,
            detrend="constant",
            scaling="density",
            axis=0,
        )
        for channel, ax in enumerate(axes):
            ax.semilogy(
                frequency[frequency >= 0.5],
                psd[frequency >= 0.5, channel],
                color=COLORS[stage],
                linewidth=1.4,
            )
    for channel, ax in enumerate(axes):
        ax.set(xlim=(0.5, 40), xlabel="Частота, Гц", ylabel="PSD, мкВ²/Гц")
        ax.set_xticks([0.5, 10, 20, 30, 40], ["0,5", "10", "20", "30", "40"])
        ax.set_title(CHANNELS[channel], pad=14)
        ax.grid(alpha=0.4)
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=3, label=stage)
            for stage, color in zip(STAGES, COLORS, strict=True)
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.82),
        ncol=5,
        frameon=False,
    )
    fig.text(
        0.045,
        0.13,
        "Welch: окно Hann 4 с, перекрытие 50%; перед PSD вычитается среднее каждого окна.",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.065,
        "Это отдельные примеры, а не типичные спектры стадий или усреднение по людям.",
        fontsize=12,
        color=MUTED,
    )
    save(fig, output, "example_spectra.png")


def experiments(rows, output):
    fig = plt.figure(figsize=(14.4, 8.7))
    heading(
        fig,
        "Как менялось качество решения",
        "Девять контрольных точек · macro-F1 на выданной валидации · выше лучше",
    )
    ax = fig.add_axes((0.40, 0.19, 0.55, 0.59))
    values = [row["macro_f1"] for row in rows]
    colors = ["#176C80" if row["id"] == "157_pretrained" else "#96B5D9" for row in rows]
    ax.barh(range(len(rows)), values, color=colors, height=0.63)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows], fontsize=13)
    ax.tick_params(axis="y", length=0, pad=13)
    for label, row in zip(ax.get_yticklabels(), rows, strict=True):
        if row["id"] == "157_pretrained":
            label.set_weight("bold")
            label.set_color("#176C80")
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.06), xlabel="Валидационный macro-F1")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0,2", "0,4", "0,6", "0,8", "1"])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.45)
    ax.set_axisbelow(True)
    for i, row in enumerate(rows):
        ax.text(
            row["macro_f1"] + 0.012,
            i,
            number(row["macro_f1"], 6),
            va="center",
            fontsize=13,
            weight="bold" if row["id"] == "157_pretrained" else "normal",
        )
    fig.legend(
        handles=[Patch(facecolor="#176C80", label="Выбранная финальная отправка")],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.835),
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.045,
        0.09,
        "У чистой U-Sleep и смеси 90/10 validation выше, чем у выбранной смеси 50/50.",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.04,
        "Validation использовалась многократно. Public leaderboard — отдельная оценка.",
        fontsize=12,
        color=MUTED,
    )
    save(fig, output, "experiment_milestones.png")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(
    data_root, output, record, missing_record, counts, subjects, metrics, rows
):
    files = {
        "dataset:labels.csv": data_root / "labels.csv",
        "dataset:epochs/train/SC4011.npz": data_root / "epochs/train/SC4011.npz",
        "dataset:epochs/train/SC4092.npz": data_root / "epochs/train/SC4092.npz",
        "repo:final/evidence/validation_metrics.json": ROOT
        / "final/evidence/validation_metrics.json",
        "repo:experiments.csv": ROOT / "experiments.csv",
        "repo:docs/figures/generate.py": Path(__file__).resolve(),
    }
    samples = [
        {
            "stage": stage,
            "id": str(record.ids[record.position(index)]),
            "epoch_index": index,
        }
        for stage, index in zip(STAGES, EXAMPLE_INDICES, strict=True)
    ]
    metadata = {
        "schema_version": 1,
        "description": "Рисунки из открытого train, labels.csv и сохранённых результатов; модели не запускались.",
        "runtime": {
            "python": sys.version.split()[0],
            "packages": VERSIONS,
            "font": "DejaVu Sans",
        },
        "sources": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
        "signal": {
            "fs_hz": 100,
            "epoch_seconds": 30,
            "samples_per_epoch": 3000,
            "units": "µV",
            "channels": CHANNELS,
            "extra_filtering": False,
        },
        "stage_order": STAGES,
        "palette": dict(zip(STAGES, COLORS, strict=True)),
        "examples": samples,
        "figures": {
            "cover.png": {
                "record": "SC4011",
                "split": "train",
                "signal_id": "full_SC4011_01279",
                "signal_seconds": [0, 30],
                "display_channel_offsets_uv": [320, 160, 0],
                "diagram": "Схема задачи: 30 секунд трёх сигналов → одна из пяти стадий",
                "highlighted_stage": "N2",
                "label_source": "dataset:labels.csv",
                "model_execution": False,
            },
            "stage_examples.png": {
                "examples": [sample["id"] for sample in samples],
                "duration_seconds": 30,
                "scale": "Единая для каждого канала между стадиями",
            },
            "validation_errors.png": {
                "experiment_id": "157_pretrained",
                "evaluation_set": metrics["evaluation_set"],
                "epoch_count": metrics["epoch_count"],
                "subject_count": metrics["subject_count"],
                "matrix_rows": "Истинные стадии",
                "matrix_columns": "Предсказанные стадии",
                "matrix_color": "Нормализация по строке",
                "f1": metrics["f1"],
            },
            "data_challenges.png": {
                "class_counts": {
                    split: {stage: counts[split][k] for k, stage in enumerate(STAGES)}
                    for split in ("train", "val")
                },
                "subject_counts": {
                    split: len(subjects[split]) for split in ("train", "val")
                },
                "gap_record": "SC4092",
                "split": "train",
                "gaps": gaps(missing_record),
                "missing_interval_contents": "Неизвестно",
            },
            "example_spectra.png": {
                "examples": [sample["id"] for sample in samples],
                "method": "Welch",
                "window": "hann",
                "nperseg": 400,
                "noverlap": 200,
                "detrend": "constant",
                "scaling": "density",
                "units": "µV²/Hz",
                "display_frequency_hz": [0.5, 40],
                "aggregation_across_epochs": False,
            },
            "experiment_milestones.png": {
                "metric": "provided validation macro-F1",
                "higher_is_better": True,
                "axis_starts_at": 0,
                "selected": "157_pretrained",
                "experiments": rows,
            },
        },
    }
    for name, entry in metadata["figures"].items():
        entry["png_sha256"] = sha256(output / name)
    (output / "sources.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Каталог исходных NPZ и labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Каталог шести PNG и sources.json",
    )
    args = parser.parse_args()
    data_root, output = (
        args.data_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    for package, expected in VERSIONS.items():
        require(
            version(package) == expected,
            f"Ожидается {package}=={expected}; используйте uv run --script",
        )
    required = [
        data_root / "labels.csv",
        data_root / "epochs/train/SC4011.npz",
        data_root / "epochs/train/SC4092.npz",
        ROOT / "final/evidence/validation_metrics.json",
        ROOT / "experiments.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Для рисунков нужен исходный файл: {path}")
    counts, subjects, label_rows = read_labels(data_root / "labels.csv")
    record = read_record(required[1], label_rows, signals=True)
    missing_record = read_record(required[2], label_rows, signals=False)
    for stage, index in enumerate(EXAMPLE_INDICES):
        require(
            record.labels[record.position(index)] == stage,
            f"Изменилась стадия эпохи {index}",
        )
    require(
        record.labels[record.position(COVER_INDEX)] == 2, "Обложка должна показывать N2"
    )
    metrics = read_evidence(required[3], counts, subjects)
    rows = read_experiments(required[4], metrics)
    output.mkdir(parents=True, exist_ok=True)
    style()
    cover(record, output)
    atlas(record, output)
    errors(metrics, output)
    challenges(counts, subjects, missing_record, output)
    spectra(record, output)
    experiments(rows, output)
    provenance(
        data_root, output, record, missing_record, counts, subjects, metrics, rows
    )
    print(f"Созданы шесть рисунков и sources.json: {output}")


if __name__ == "__main__":
    main()
