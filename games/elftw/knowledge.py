"""角色知識庫 — 自動掃描並快取角色資訊，提供智慧站位與購買決策"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ======================================================================
# 角色職責 → 偏好站位對照表
# ======================================================================
# row 0 = 前排 (靠敵方), row 3 = 後排 (靠商店)
ROLE_POSITION: Dict[str, Dict[str, Any]] = {
    "tank":       {"row": 0, "col_pref": "center"},   # 前排中央，吸收傷害
    "warrior":    {"row": 0, "col_pref": "any"},       # 前排，泛用近戰
    "melee_dps":  {"row": 0, "col_pref": "side"},      # 前排兩側，側翼輸出
    "assassin":   {"row": 3, "col_pref": "corner"},    # 後排角落，跳後排
    "mage":       {"row": 3, "col_pref": "center"},    # 後排中央，安全輸出
    "ranged_dps": {"row": 2, "col_pref": "center"},    # 中後排，遠程輸出
    "support":    {"row": 1, "col_pref": "center"},    # 中排中央，靠近隊友
    "healer":     {"row": 1, "col_pref": "center"},    # 中排中央，治療範圍
    "summoner":   {"row": 2, "col_pref": "any"},       # 中後排，召喚物擋前
    "unknown":    {"row": 0, "col_pref": "any"},       # 未知角色預設前排
}

# 各偏好方向的欄位優先序
COL_PRIORITY: Dict[str, List[int]] = {
    "center": [2, 1, 3, 0, 4],
    "side":   [0, 4, 1, 3, 2],
    "corner": [0, 4, 1, 3, 2],
    "any":    [2, 1, 3, 0, 4],
}


# ======================================================================
# CharacterInfo — 單一角色資訊
# ======================================================================
class CharacterInfo:
    """儲存單一角色的完整資訊（由 LLM 掃描取得）"""

    def __init__(self, data: Dict[str, Any]):
        self.name: str = data.get("name", "")
        self.role: str = data.get("role", "unknown")
        self.classes: List[str] = data.get("classes", [])
        self.abilities: List[str] = data.get("abilities", [])
        self.ability_desc: str = data.get("ability_desc", "")
        self.position_pref: str = data.get("position_pref", "front")
        self.priority: int = data.get("priority", 50)
        self.notes: str = data.get("notes", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "classes": self.classes,
            "abilities": self.abilities,
            "ability_desc": self.ability_desc,
            "position_pref": self.position_pref,
            "priority": self.priority,
            "notes": self.notes,
        }

    @property
    def preferred_row(self) -> int:
        return ROLE_POSITION.get(self.role, ROLE_POSITION["unknown"])["row"]

    @property
    def col_preference(self) -> List[int]:
        pref = ROLE_POSITION.get(self.role, ROLE_POSITION["unknown"])["col_pref"]
        return COL_PRIORITY.get(pref, COL_PRIORITY["any"])

    def __repr__(self) -> str:
        return f"<{self.name} role={self.role} pos={self.position_pref} pri={self.priority}>"


# ======================================================================
# CharacterKnowledgeBase — 知識庫管理器
# ======================================================================
class CharacterKnowledgeBase:
    """
    角色知識庫：載入/儲存角色資訊，提供智慧站位與購買決策。

    資料來源：
    - 自動掃描（discover 模式，LLM 逐一讀取英雄詳情）
    - 手動編輯 JSON 檔案
    """

    def __init__(self, path: str = "characters_elftw.json"):
        self.path = Path(path)
        self._chars: Dict[str, CharacterInfo] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for cd in data.get("characters", []):
                    info = CharacterInfo(cd)
                    if info.name:
                        self._chars[info.name] = info
                log.info("知識庫載入: %d 個角色 (%s)", len(self._chars), self.path)
            except Exception as e:
                log.warning("知識庫載入失敗: %s", e)

    def save(self):
        data = {"characters": [c.to_dict() for c in self._chars.values()]}
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("知識庫已儲存: %d 個角色 → %s", len(self._chars), self.path)

    def add(self, char_data: Dict[str, Any]):
        info = CharacterInfo(char_data)
        if not info.name:
            log.warning("跳過無名角色")
            return
        self._chars[info.name] = info
        log.info("新增角色: %s (role=%s, pos=%s, pri=%d)",
                 info.name, info.role, info.position_pref, info.priority)

    def get(self, name: str) -> Optional[CharacterInfo]:
        """精確匹配 → 子字串匹配"""
        if name in self._chars:
            return self._chars[name]
        for cname, info in self._chars.items():
            if cname in name or name in cname:
                return info
        return None

    def get_all(self) -> List[CharacterInfo]:
        return list(self._chars.values())

    @property
    def is_empty(self) -> bool:
        return len(self._chars) == 0

    def __len__(self) -> int:
        return len(self._chars)

    # ------------------------------------------------------------------
    # 智慧站位
    # ------------------------------------------------------------------
    def compute_formation(
        self, board_units: List[Dict],
    ) -> List[Tuple[Dict, Tuple[int, int]]]:
        """
        根據角色職責計算最佳陣型。

        Args:
            board_units: [{"name": "角色名", "row": 0, "col": 2}, ...]

        Returns:
            [(unit_info, (target_row, target_col)), ...]

        邏輯：
        1. 查知識庫取得每個角色的 role → 偏好 row + col 優先序
        2. 按 row 排序（tank 優先搶前排）
        3. 貪心分配格位，避免衝突
        4. 溢出時往相鄰 row 安排
        """
        if not board_units:
            return []

        # 建立待分配清單
        todo: List[Tuple[Dict, int, List[int]]] = []
        for unit in board_units:
            char = self.get(unit.get("name", ""))
            if char:
                todo.append((unit, char.preferred_row, char.col_preference))
            else:
                todo.append((unit, 0, COL_PRIORITY["any"]))

        # tank/warrior (row 0) 優先分配，確保前排有人擋
        todo.sort(key=lambda x: x[1])

        occupied: set = set()
        result: List[Tuple[Dict, Tuple[int, int]]] = []

        for unit, target_row, col_prefs in todo:
            placed = False
            # 先試偏好 row，再試相鄰 row
            for row_off in [0, 1, -1, 2, -2, 3]:
                actual_row = target_row + row_off
                if actual_row < 0 or actual_row > 3:
                    continue
                for col in col_prefs:
                    if (actual_row, col) not in occupied:
                        occupied.add((actual_row, col))
                        result.append((unit, (actual_row, col)))
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                # 最後手段：塞任何空格
                for r in range(4):
                    for c in range(5):
                        if (r, c) not in occupied:
                            occupied.add((r, c))
                            result.append((unit, (r, c)))
                            placed = True
                            break
                    if placed:
                        break

        return result

    # ------------------------------------------------------------------
    # 智慧購買
    # ------------------------------------------------------------------
    def rank_shop_cards(
        self, card_names: List[str],
    ) -> List[Tuple[str, int, bool]]:
        """
        排序商店卡片的購買優先序。

        Returns:
            [(name, priority, should_buy), ...]  priority 越小越想買
        """
        result = []
        for name in card_names:
            char = self.get(name)
            if char:
                result.append((name, char.priority, char.priority <= 70))
            else:
                # 未知角色：中等優先度，仍然購買
                result.append((name, 50, True))
        result.sort(key=lambda x: x[1])
        return result
