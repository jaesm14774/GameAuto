"""陣型函式 — elftw 遊戲特定的陣型計算"""

from typing import Dict, List, Tuple


def formation_front_row(units: List[Dict]) -> List[Tuple[Dict, Tuple[int, int]]]:
    """
    預設陣型：所有角色排在第一排 (Row 0)。

    邏輯：
    - 已在 Row 0 的角色留在原位（不移動）
    - 其他行的角色搬到 Row 0 的空位
    - 超過 5 個溢出到 Row 1
    """
    # 分離前排和後排角色
    front_units = [u for u in units if u["row"] == 0]
    back_units = [u for u in units if u["row"] != 0]

    # 前排角色佔用的欄位
    occupied_cols = {u["col"] for u in front_units}
    # Row 0 剩餘可用的欄位（依序分配）
    available_cols = [c for c in range(5) if c not in occupied_cols]

    result = []

    # 前排角色：維持原位
    for u in front_units:
        result.append((u, (0, u["col"])))

    # 後排角色：填入 Row 0 空位，溢出到 Row 1
    for i, u in enumerate(back_units):
        if i < len(available_cols):
            result.append((u, (0, available_cols[i])))
        else:
            overflow_col = i - len(available_cols)
            result.append((u, (1, overflow_col % 5)))

    return result


def formation_angel_column(units: List[Dict]) -> List[Tuple[Dict, Tuple[int, int]]]:
    """
    聖光天使陣型：聖光天使在 Row 0 中央 (col 2)，
    其餘角色排在同一行 (col 2) 的 Row 1, 2, 3。
    超過 4 個角色時，溢出到前排兩側。

    邏輯：
    - 已在正確位置的角色不移動
    - 其餘角色填入空的目標格
    """
    angel = None
    others = []
    for unit in units:
        name = unit.get("name", "")
        if "聖光" in name or "天使" in name or "angel" in name.lower() or "victory" in name.lower():
            angel = unit
        else:
            others.append(unit)

    center_col = 2

    # 目標格位（依優先序）
    # 聖光天使佔 (0, 2)，其餘排在後方同列，溢出到兩側
    target_slots = [
        (1, center_col), (2, center_col), (3, center_col),
        (0, 1), (0, 3), (0, 0), (0, 4),
    ]

    result = []

    # 聖光天使 → (0, 2)
    if angel:
        result.append((angel, (0, center_col)))

    # 找出已在目標格的角色（不需移動）
    assigned_slots = set()
    unassigned = []
    for unit in others:
        pos = (unit["row"], unit["col"])
        if pos in target_slots and pos not in assigned_slots:
            result.append((unit, pos))
            assigned_slots.add(pos)
        else:
            unassigned.append(unit)

    # 剩餘角色填入空的目標格
    remaining_slots = [s for s in target_slots if s not in assigned_slots]
    for i, unit in enumerate(unassigned):
        if i < len(remaining_slots):
            result.append((unit, remaining_slots[i]))
        else:
            result.append((unit, (3, i % 5)))

    return result
