# GameAuto — Vision Agent 遊戲自動化框架

LLM 視覺代理驅動的遊戲自動化框架。Bot 直接「看」螢幕截圖，理解畫面並做出決策，無需手動截取模板或硬編碼規則。

## 架構

```
截圖 → 畫面變化偵測 → 狀態機偵測 → 規則引擎 / LLM 決策 → 執行動作 → 觀察結果 → 更新記憶 → 循環
```

系統採三層決策：

1. **規則引擎 (Strategy)** — 已知狀態用固定座標快速操作，不呼叫 LLM
2. **LLM 判讀** — 讀取金幣、人口等數值，做數據驅動決策
3. **LLM 兜底** — 未知狀態時截圖丟給 LLM 分析

### 核心模組 (`core/`)

| 模組 | 職責 |
|------|------|
| `agent.py` | 通用決策引擎基底類：整合狀態機 + 規則引擎 + LLM 兜底 |
| `vision.py` | Vision Client + Grid Overlay，LLM 視覺辨識 |
| `state_machine.py` | YAML 驅動狀態機，從配置讀取狀態與轉場規則 |
| `strategy.py` | Action 資料結構 + RuleEngine 規則引擎 + 策略抽象基底類 |
| `capture.py` | 螢幕擷取（MSS / DXCam / PyAutoGUI 自動選擇） |
| `controller.py` | ctypes 直接操控滑鼠，原生支援多螢幕負座標 |
| `screen_detector.py` | 像素差異偵測畫面變化，避免重複呼叫 LLM |
| `memory.py` | Agent Memory，儲存動作歷史與學習 pattern |

### 遊戲模組 (`games/`)

遊戲透過 `games/__init__.py` 的 `GAME_REGISTRY` 註冊，每個遊戲提供：
- **Strategy** — 繼承 `GameStrategy`，實作 `on_state()` 產生動作
- **Agent** — 繼承 `GameAgent`，覆寫 `resolve_action()` 處理遊戲特定動作類型

目前支援的遊戲：

| 遊戲 | 模組 | 說明 |
|------|------|------|
| 指尖棋兵 (ElfTW) | `games/elftw/` | 自走棋自動化：盲買循環 + 智慧購買 + 陣型調整 + 聖物選擇 |

#### ElfTW 模組結構

| 檔案 | 職責 |
|------|------|
| `strategy.py` | 準備階段盲買循環（人口等級分層）、狀態處理 |
| `agent.py` | LLM 金幣判讀、智慧購買、陣型調整、聖物選擇、角色掃描 |
| `knowledge.py` | 角色知識庫：職責站位對照、智慧購買排序、陣型計算 |
| `formations.py` | 陣型函式：前排陣型、聖光天使陣型 |

### 主程式 (`main.py`)

| 功能 | 說明 |
|------|------|
| `GameAutoBot` | 串接截圖、Agent、操作控制的自動化引擎 |
| `run()` | 主迴圈：截圖 → Agent 決策 → 執行動作 |
| `discover()` | 角色掃描模式：掃描英雄圖鑑建立知識庫 |
| `calibrate()` | 校準模式：即時顯示滑鼠相對遊戲窗口座標 |

## 準備階段策略

核心迴圈邏輯：每 **2 個盲買循環** 呼叫一次 LLM 判讀金幣，金幣 < 8 直接點準備。

### 盲買循環（依人口等級）

| 人口上限 | 循環順序 |
|----------|----------|
| < 4 | 人口 → 刷新 → 買 → 刷新 → 買 → 人口 → 刷新 → 買 |
| 4~5 | 刷新 → 買 → 人口 → 刷新 → 買 → 刷新 → 買 |
| >= 6 | (30%人口?) 刷新 → 買 → (30%人口?) 刷新 → 買 → 刷新 → 買 |

### 決策流程

```
盲買循環 1 → 盲買循環 2 → LLM 讀金幣
                              ├── gold < 8 → 點準備
                              └── gold >= 8 → 重置，回到盲買循環 1
                                               （安全網：25 秒超時或 8 tick 上限強制點準備）
```

## 安裝

```bash
pip install -r requirements.txt

copy .env.example .env
# 編輯 .env，填入 API Key
```

### API Key 取得方式

| 方案 | 取得連結 | 說明 |
|------|----------|------|
| Google Gemini（推薦） | https://aistudio.google.com/apikey | GCP Console 也可建立，支援 Gemini 2.5 Flash/Pro |
| OpenRouter（免費模型） | https://openrouter.ai/keys | 有免費額度，適合測試 |

## 啟動

```bash
# 確保雷電模擬器已開啟遊戲
python main.py -c configs/elftw.yaml

# 除錯模式
python main.py -c configs/elftw.yaml -v

# 校準模式：顯示滑鼠相對遊戲窗口座標（填入 config 的 positions）
python main.py --calibrate

# 角色掃描模式：掃描英雄圖鑑建立知識庫
python main.py --discover
```

停止：`Ctrl+C`

## 配置

`configs/elftw.yaml` 包含完整的遊戲配置：

```yaml
game:
  name: "elftw"
  window_title: "指尖棋兵"
  alt_titles: ["LDPlayer", "雷電模擬器"]

capture:
  mode: "fullscreen"    # fullscreen / window
  monitor: 2            # 螢幕索引

vision:
  grid_cols: 20
  grid_rows: 10
  min_interval: 2.0     # LLM 呼叫最小間隔（秒）
  thinking_level: "low"  # Gemini thinking level: low / medium / high

controller:
  click_delay: [0.02, 0.06]
  move_duration: [0.01, 0.04]

# 狀態定義 + 轉場規則
states:
  main_menu:
    detect: "看到對戰按鈕和合作按鈕"
  preparation:
    detect: "棋盤+底部商店卡牌+準備按鈕"
  battle:
    detect: "棋盤上單位戰鬥中，無可操作按鈕"
  # ... 其他狀態

transitions:
  - {from: main_menu, to: matchmaking, condition: "點擊對戰按鈕後"}
  - {from: preparation, to: battle, condition: "倒計時結束或點擊準備按鈕"}
  # ...

strategy:
  composition: "mage"
  max_prep_rounds: 2          # 盲買輪數（之後呼叫 LLM）
  knowledge_base: "characters_elftw.json"

  relics:                     # 聖物優先購買清單
    - {name: "精氣球", priority: 1}
    - {name: "鐵樹枝幹", priority: 2}
    # ...

  positions:                  # 固定座標（用 --calibrate 校準）
    card_slots: [{x: 1080, y: 1028}, ...]
    refresh_btn: {x: 1464, y: 1210}
    population_btn: {x: 1122, y: 1233}
    ready_btn: {x: 1615, y: 969}
    battle_btn: {x: 1120, y: 1084}
    board_grid: [...]         # 4x5 棋盤格座標
```

## 支援的視覺模型

`.env` 中設定 `LLM_MODEL`：

**Google Gemini API（推薦）**

| 模型 | 特點 |
|------|------|
| `gemini-2.5-flash` | 快速推理、1M context、高性價比（預設） |
| `gemini-2.5-pro` | 最強推理能力，較慢 |
| `gemini-2.0-flash` | 輕量快速 |

**OpenRouter 免費模型**

| 模型 | 特點 |
|------|------|
| `google/gemma-3-27b-it:free` | 支援視覺，免費 |
| `google/gemini-2.0-flash-exp:free` | 1M context，強視覺 |

## 專案結構

```
GameAuto/
├── main.py                    # 主程式：GameAutoBot + CLI
├── core/                      # 通用核心模組
│   ├── agent.py               # 通用決策引擎基底類
│   ├── vision.py              # Vision Client + Grid Overlay
│   ├── state_machine.py       # YAML 驅動狀態機
│   ├── strategy.py            # Action + RuleEngine + GameStrategy 抽象基底
│   ├── capture.py             # 螢幕擷取（MSS / DXCam / PyAutoGUI）
│   ├── controller.py          # 滑鼠操作（ctypes，多螢幕）
│   ├── screen_detector.py     # 畫面變化偵測
│   └── memory.py              # Agent Memory
├── games/                     # 遊戲專用模組
│   ├── __init__.py            # GAME_REGISTRY 遊戲註冊表
│   └── elftw/                 # 指尖棋兵
│       ├── agent.py           # ElfTWAgent：LLM 決策 + 角色掃描
│       ├── strategy.py        # ElfTWStrategy：盲買循環 + 狀態處理
│       ├── knowledge.py       # 角色知識庫 + 智慧站位 + 智慧購買
│       └── formations.py      # 陣型計算函式
├── configs/
│   └── elftw.yaml             # 遊戲配置（狀態 + 轉場 + 策略 + 座標）
├── .env                       # API Key（不進版控）
├── .env.example               # 環境變數範例
└── requirements.txt
```

## 擴充新遊戲

1. 在 `games/` 下建立新目錄（如 `games/mygame/`）
2. 實作 `MyGameStrategy(GameStrategy)` 和 `MyGameAgent(GameAgent)`
3. 在 `games/__init__.py` 的 `GAME_REGISTRY` 註冊
4. 建立 `configs/mygame.yaml` 配置檔
5. 啟動：`python main.py -c configs/mygame.yaml`
