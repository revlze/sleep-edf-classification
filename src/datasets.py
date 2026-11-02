"""Контекст: только непрерывные окна исходной временной сетки одной записи."""

import numpy as np


def window_indices(
    indices: np.ndarray, left: int, right: int
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-left, right + 1)
    rows = np.arange(len(indices))[:, None] + offsets
    inside = (rows >= 0) & (rows < len(indices))
    clipped = np.clip(rows, 0, len(indices) - 1)
    valid = inside.all(axis=1) & (indices[clipped] == indices[:, None] + offsets).all(
        axis=1
    )
    return clipped, valid


def add_context(
    base: np.ndarray,
    indices: np.ndarray,
    names: list[str],
    mode: str,
    long_windows: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if mode == "none":
        return base, names
    blocks = [base]
    columns = [f"t0__{name}" for name in names]
    flags = []
    flag_names = []
    for offset in (-2, -1, 1, 2):
        rows, valid = window_indices(indices, max(-offset, 0), max(offset, 0))
        source = rows[:, 0 if offset < 0 else -1]
        block = base[source].copy()
        block[~valid] = np.nan
        blocks.append(block)
        columns += [f"t{offset:+d}__{name}" for name in names]
        flags.append((~valid).astype(np.float32))
        flag_names.append(f"missing_t{offset:+d}")
    if mode == "multires":
        if long_windows is None:
            raise ValueError("Missing 60/90 s features")
        for key in ("left60", "right60", "center90"):
            columns += [f"{key}__{name}" for name in names]
            left, right = {"left60": (1, 0), "right60": (0, 1), "center90": (1, 1)}[key]
            _, valid = window_indices(indices, left, right)
            block = long_windows[key].copy()
            block[~valid] = np.nan
            blocks.append(block)
            flags.append((~valid).astype(np.float32))
            flag_names.append(f"missing_{key}")
    elif mode != "neighbors":
        raise ValueError(mode)
    return np.column_stack([*blocks, *flags]).astype(np.float32), columns + flag_names
