# リポジトリ再編成 実装計画（v3 — レビュー反映版）

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** rag-gemini → rag-local リネーム + chat.py の機能分割（回答支援AI / 事務改定評価AI）+ apps/ 分離 + アーカイブ移動

**Architecture:** chat.py（1,122行）を共通UI部品（ui/shared.py）+ 回答支援UI（apps/answer-support/ui/chat.py）+ 事務改定評価UI（apps/revision-eval/ui/eval_ui.py）に3分割。共有コア（src/, config.py）は rag-local/ ルートに残し、apps/ からは sys.path で参照。将来的には `pyproject.toml` + `pip install -e .` による editable install への移行を推奨（N6）。

**Tech Stack:** Python, Streamlit, Git, Bash

**Design Doc:** `docs/plans/2026-02-17-rag-reorganization-design.md`

**Review:** 4観点レビュー実施済み（v2 → v3 で全指摘反映）

---

## chat.py 関数配置マップ（全28関数）

| 関数名 | 行 | 配置先 | 理由 |
|--------|-----|--------|------|
| `load_revision_correct_ids()` | 60-84 | **eval_ui** | 正解IDは評価専用 |
| `extract_bot_name_from_category()` | 87-99 | **eval_ui** | シナリオID構築（評価用） |
| `extract_bot_name_from_area()` | 102-115 | **eval_ui** | エリア→ボット変換（評価用） |
| `build_scenario_id()` | 118-139 | **eval_ui** | 通常検索のID構築だが正解判定にしか使わない |
| `build_scenario_id_from_area()` | 142-161 | **eval_ui** | 評価専用 |
| `check_if_correct()` | 164-183 | **eval_ui** | 正解判定（評価専用） |
| `get_available_business_areas()` | 187-197 | **chat** | 業務分野選択（通常検索専用） |
| `initialize_session_state()` | 200-246 | **shared（共通部）+ 両方（モード固有部）** | 共通部分（chat_history, processing_query, config初期化）を `initialize_common_session_state()` として shared.py に抽出。各UIはこれを呼んだ後にモード固有の状態を追加（W1） |
| `format_message()` | 248-261 | **shared** | 両モード共通 |
| `_create_badge()` | 264-268 | **shared** | 両モード共通 |
| `_create_category_badge()` | 270-281 | **shared** | 両モード共通 |
| `_create_correct_badge()` | 284-288 | **shared** | 両モード共通 |
| `_create_scenario_id_badge()` | 291-295 | **shared** | 両モード共通 |
| `_create_relevance_badge()` | 298-312 | **shared** | 両モード共通 |
| `_is_valid_llm_judgment()` | 315-320 | **shared** | 両モード共通 |
| `_create_llm_analysis_section()` | 323-333 | **shared** | 両モード共通 |
| `format_response_card()` | 335-357 | **shared** | 両モード共通 |
| `_needs_processor_reinit()` | 359-371 | **chat** | Processor は通常検索専用 |
| `_load_reference_data_for_business()` | 374-437 | **chat** | 業務分野参照データ（通常検索専用） |
| `execute_dual_provider_search()` | 440-480 | **eval_ui** | 両プロバイダー検索（評価専用） |
| `_search_with_provider()` | 483-589 | **eval_ui** | プロバイダー別検索（評価専用） |
| `_initialize_processor()` | 592-606 | **chat** | Processor 初期化（通常検索専用） |
| `process_query()` | 609-694 | **両方に分割** | 617-643: eval_ui / 644-694: chat |
| `run_llm_analysis()` | 697-741 | **chat** | LLM判断支援（通常検索の多段階モード用） |
| `save_chat_history()` | 743-775 | **両方に配置** | Processor依存を避けるため各UI側に配置（W2）。shared.pyには置かない |
| `_render_provider_results()` | 778-809 | **eval_ui** | プロバイダー別結果表示（評価専用） |
| `_render_vector_weight_slider()` | 812-829 | **shared（純粋関数化）** | 値を返すのみ。config更新は呼び出し側で行う（W3） |
| `run_streamlit_ui()` | 832-1123 | **両方に分割** | 945-1036: chat / 861-943: eval_ui |

### 分割後のファイルサイズ見込み

| ファイル | 行数 | 内容 |
|---------|------|------|
| `ui/shared.py` | ~200行 | UI部品10関数 + `initialize_common_session_state()` + CSS + imports（Processor非依存） |
| `apps/answer-support/ui/chat.py` | ~480行 | 回答支援UI 9関数（save_chat_history含む）+ run_streamlit_ui |
| `apps/revision-eval/ui/eval_ui.py` | ~530行 | 評価UI 11関数（save_chat_history含む）+ run_streamlit_ui |
| **合計** | ~1,210行 | 旧 chat.py 1,122行 + import/setup + 共通初期化抽出のオーバーヘッド |

---

### Task 1: ブランチ作成 + ディレクトリリネーム

**Files:**
- 操作: `rag-gemini/` → `rag-local/`

**Step 1: ブランチ作成**

```bash
git checkout -b refactor/rag-reorganization
```

**Step 2: rag-gemini → rag-local リネーム**

```bash
git mv rag-gemini rag-local
```

**Step 3: コミット**

```bash
git add rag-local
git commit -m "refactor: rag-gemini → rag-local リネーム"
```

---

### Task 2: main.py + evaluate_revisions.py 移動 + import 修正

**Files:**
- Move: `rag-local/main.py` → `rag-local/apps/answer-support/main.py`
- Move: `rag-local/scripts/evaluate_revisions.py` → `rag-local/apps/revision-eval/evaluate_revisions.py`
- Modify: 移動した2ファイルの import パス

> **Note**: chat.py は Task 3-5 で分割するため、ここでは移動しない。

**Step 1: ディレクトリ作成 + ファイル移動**

```bash
mkdir -p rag-local/apps/answer-support/ui
mkdir -p rag-local/apps/revision-eval/ui
git mv rag-local/main.py rag-local/apps/answer-support/main.py
git mv rag-local/scripts/evaluate_revisions.py rag-local/apps/revision-eval/evaluate_revisions.py
```

**Step 2: `apps/answer-support/main.py` の import 修正**

変更前 (行1-13):
```python
import sys
import os
import subprocess
import argparse
from dotenv import load_dotenv
from config import SearchConfig
...
load_dotenv()
```

変更後:
```python
import sys
import os
import subprocess
import argparse
from pathlib import Path

# apps/answer-support/ → rag-local/ ルートへのパス解決
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from config import SearchConfig
...
load_dotenv(PROJECT_ROOT / ".env")
```

変更前 (行127):
```python
config = SearchConfig(base_dir=os.path.dirname(os.path.abspath(__file__)))
```

変更後:
```python
config = SearchConfig(base_dir=str(PROJECT_ROOT))
```

変更前 (行149):
```python
process = subprocess.Popen([python_executable, "-m", "streamlit", "run", "ui/chat.py"])
```

変更後（**暫定**: 旧パスを維持。Task 4 で新 chat.py 作成後に最終パスに更新する — C2）:
```python
# Task 4 完了まで旧 chat.py を参照（中間状態での動作保証）
process = subprocess.Popen([python_executable, "-m", "streamlit", "run",
    str(PROJECT_ROOT / "ui" / "chat.py")])
```

> **注意**: この時点では `apps/answer-support/ui/chat.py` はまだ存在しないため、旧パスを維持する。Task 4 で新ファイル作成後にパスを `Path(__file__).parent / "ui" / "chat.py"` に更新する。

**Step 3: `apps/revision-eval/evaluate_revisions.py` の import 修正**

変更前 (行27-28):
```python
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

変更後:
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

**Step 4: スモークテスト**

```bash
cd rag-local
python apps/answer-support/main.py --help
python apps/revision-eval/evaluate_revisions.py --help
```

Expected: 両方ともヘルプが表示される（import エラーなし）

**Step 5: コミット**

```bash
git add rag-local/apps/answer-support/main.py rag-local/apps/revision-eval/evaluate_revisions.py
git commit -m "refactor: main.py + evaluate_revisions.py を apps/ に移動

- main.py → apps/answer-support/main.py（PROJECT_ROOT + sys.path 追加）
- evaluate_revisions.py → apps/revision-eval/evaluate_revisions.py（parent深度修正）"
```

---

### Task 3: 共通UI部品の抽出 → ui/shared.py

**Files:**
- Create: `rag-local/ui/shared.py`（chat.py から10関数 + 共通セッション初期化を抽出）
- Source: `rag-local/ui/chat.py`（元ファイル、Task 4-5 で残りを分割）

> **設計方針（レビュー反映）:**
> - shared.py は「純粋なUI部品 + 共通セッション初期化」に限定。Processor への依存を持たない（W2）
> - `save_chat_history()` は Processor に依存するため、shared.py ではなく各UI側に配置（W2）
> - `render_vector_weight_slider()` は値を返すだけの純粋関数に変更。config更新は呼び出し側で行う（W3）
> - `initialize_common_session_state()` で SearchConfig 初期化の共通部分を一元化し、重複による乖離バグを防止（W1）
> - このモジュールは `apps/answer-support/ui/chat.py` と `apps/revision-eval/ui/eval_ui.py` から共通利用される

**Step 1: `ui/shared.py` を作成**

以下の関数を chat.py から抽出して `ui/shared.py` に配置:

```python
"""共通UI部品 — 回答支援AI・事務改定評価AI 共用

このモジュールは以下から import される:
- apps/answer-support/ui/chat.py（回答支援UI）
- apps/revision-eval/ui/eval_ui.py（事務改定評価UI）

責務: UIレンダリング部品 + 共通CSS + 共通セッション状態初期化
Processor 等の重い依存は持たない（各UI側で管理する）。
"""
import sys
import os
from pathlib import Path
import html
from typing import Dict, List, Optional, Tuple

import streamlit as st

# rag-local ルートへのパス解決
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SearchConfig, load_settings
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 共通CSS
COMMON_STYLES = """
<style>
div.stButton > button:hover { background-color: #007bff; color: white; }
div.stButton > button:focus { background-color: #007bff; color: white; }
[data-baseweb="button"]:has(#save_chat_history_button) { background-color: #28a745 !important; color: white !important; }
[data-baseweb="button"]:has(#save_chat_history_button):hover { background-color: #1e7e34 !important; }
</style>
"""

def apply_common_styles():
    st.markdown(COMMON_STYLES, unsafe_allow_html=True)

# === 共通セッション状態初期化（W1: SearchConfig 初期化の重複回避）===
def initialize_common_session_state():
    """chat_history, processing_query, config の共通初期化。
    各UIはこの関数を呼んだ後、モード固有の状態を追加する。"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    if "config" not in st.session_state:
        # ... chat.py:220-244 の SearchConfig 初期化ロジック（25行）をそのまま移植 ...
        # 環境変数チェック + SearchConfig 生成
        pass

# === chat.py:248-261 ===
def format_message(message: str, is_user: bool = False) -> str:
    # ... そのまま移植 ...

# === chat.py:264-268 ===
def _create_badge(text: str, bg_color: str, text_color: str, bold: bool = True) -> str:
    # ... そのまま移植 ...

# === chat.py:270-281 ===
def _create_category_badge(category: str) -> str:
    # ... そのまま移植 ...

# === chat.py:284-288 ===
def _create_correct_badge(is_correct: bool) -> str:
    # ... そのまま移植 ...

# === chat.py:291-295 ===
def _create_scenario_id_badge(scenario_id: str) -> str:
    # ... そのまま移植 ...

# === chat.py:298-312 ===
def _create_relevance_badge(relevance_judgment: Optional[str]) -> str:
    # ... そのまま移植 ...

# === chat.py:315-320 ===
def _is_valid_llm_judgment(relevance_judgment: Optional[str]) -> bool:
    # ... そのまま移植 ...

# === chat.py:323-333 ===
def _create_llm_analysis_section(relevance_judgment: Optional[str], judgment_reason: Optional[str]) -> str:
    # ... そのまま移植 ...

# === chat.py:335-357（型ヒント追加 — N5）===
def format_response_card(
    number: int,
    similarity: float,
    query: str,
    answer: str,
    category: Optional[str] = None,
    relevance_judgment: Optional[str] = None,
    judgment_reason: Optional[str] = None,
    scenario_id: Optional[str] = None,
    is_correct: bool = False,
) -> str:
    # ... そのまま移植 ...

# === chat.py:812-829（純粋関数化 — W3）===
def render_vector_weight_slider(default_value: float, key: Optional[str] = None) -> float:
    """ベクトル重みスライダーを描画し、選択値を返す。
    config の更新は呼び出し側で行うこと（副作用なし）。"""
    vector_weight = st.slider(
        "検索バランス（ベクトル↔キーワード）",
        min_value=0.0, max_value=1.0,
        value=default_value, step=0.1,
        key=key,
        help="1.0 = ベクトル検索のみ、0.0 = キーワード検索のみ"
    )
    return vector_weight
    # 注: 旧実装では st.session_state.config を直接更新していたが、
    # shared.py の設計方針（副作用なし）に従い、値を返すのみに変更。
    # 呼び出し側で以下を実行:
    #   weight = render_vector_weight_slider(default)
    #   st.session_state.config.vector_weight = weight
    #   st.session_state.config.keyword_weight = 1.0 - weight

# 注: save_chat_history() は Processor に依存するため shared.py には含めない（W2）。
# 各UI（chat.py, eval_ui.py）にそれぞれ配置する。
```

> **重要**: 各関数の実装は chat.py からコピー。ロジック変更は `render_vector_weight_slider`（純粋関数化）と `initialize_common_session_state`（共通部分抽出）のみ。

> **実装注意（最終確認で発見）**: `initialize_common_session_state()` は `_ui_settings`（`load_settings("ui")` の結果）を参照する。shared.py のモジュールスコープに必ず以下を含めること（省略すると NameError）:
> ```python
> _ui_settings = load_settings("ui")
> ```

**Step 2: コミット**

```bash
git add rag-local/ui/shared.py
git commit -m "refactor: chat.py から共通UI部品を ui/shared.py に抽出

10関数 + initialize_common_session_state を抽出。
render_vector_weight_slider を純粋関数化（値を返すのみ、config更新は呼び出し側）。
save_chat_history は Processor 依存のため各UI側に配置（shared.py には含めない）。"
```

---

### Task 4: 回答支援UI → apps/answer-support/ui/chat.py + main.py パス最終修正

**Files:**
- Create: `rag-local/apps/answer-support/ui/chat.py`（回答支援モードのみ）
- Create: `rag-local/apps/answer-support/ui/__init__.py`
- Modify: `rag-local/apps/answer-support/main.py`（subprocess パスを最終パスに更新 — C2）

**Step 1: 新 chat.py を作成**

chat.py の回答支援モード部分のみで構成:

```python
"""回答支援AI — Streamlit UI"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
import re
import copy
import html
from typing import Dict, List

from config import SearchConfig, load_settings
from src.core.processor import Processor
from src.utils.logger import setup_logger
from src.utils.dynamic_db_manager import DynamicDBManager

# 共通UI部品
from ui.shared import (
    format_message, format_response_card, render_vector_weight_slider,
    apply_common_styles, initialize_common_session_state,
)

logger = setup_logger(__name__)
_ui_settings = load_settings("ui")
INPUT_FILE = PROJECT_ROOT / "data" / "input" / "multi_stage_input.xlsx"
DEFAULT_BUSINESS_AREAS = [...]  # chat.py:56 から

# === chat.py:187-197（C1: base_dir を PROJECT_ROOT に修正）===
@st.cache_data(ttl=60)
def get_available_business_areas() -> list:
    try:
        config = SearchConfig(base_dir=str(PROJECT_ROOT))  # C1: "." → PROJECT_ROOT
        # ... 残りはそのまま移植 ...
    except Exception:
        return DEFAULT_BUSINESS_AREAS

# === chat.py:200-246（W1: 共通部分は initialize_common_session_state() に委譲）===
def initialize_session_state():
    initialize_common_session_state()  # chat_history, processing_query, config の共通初期化
    if "business_area" not in st.session_state:
        st.session_state.business_area = "預金"
    # 注: dual_provider_mode, selected_revision, correct_ids 等の評価用状態は不要

# === chat.py:359-371 ===
def _needs_processor_reinit() -> bool:
    # ... そのまま移植 ...

# === chat.py:374-437 ===
def _load_reference_data_for_business(config, business_area):
    # ... そのまま移植 ...

# === chat.py:592-606 ===
def _initialize_processor():
    # ... そのまま移植 ...

# === chat.py:644-694（回答支援パスのみ）===
def process_query(query):
    st.session_state.processing_query = True
    try:
        query_number = len(st.session_state.chat_history) // 2 + 1
        # ... chat.py:644-694 の回答支援パスをそのまま移植 ...
        # dual_provider_mode 分岐は不要
        if _needs_processor_reinit():
            _initialize_processor()
        processor = st.session_state.processor
        results = processor.searcher.search(str(query_number), query, "")
        # ... 結果処理 ...
    except Exception as e:
        # ... エラー処理 ...
    finally:
        st.session_state.processing_query = False

# === chat.py:697-741 ===
def run_llm_analysis():
    # ... そのまま移植 ...

# === chat.py:743-775（W2: shared.py ではなくここに配置。Processor 依存あり）===
def save_chat_history():
    # ... そのまま移植 ...

# === run_streamlit_ui（回答支援モードのみ）===
def run_streamlit_ui():
    st.set_page_config(page_title="類似回答検索ボット", layout="wide", initial_sidebar_state="expanded")
    apply_common_styles()
    initialize_session_state()

    with st.sidebar:
        st.title("設定")
        # === chat.py:945-1036（回答支援サイドバー）===
        # 検索パラメータ、業務分野選択

        # W3: render_vector_weight_slider は値を返すのみ。config更新は呼び出し側で行う
        weight = render_vector_weight_slider(
            st.session_state.config.vector_weight, key="vector_weight_slider"
        )
        st.session_state.config.vector_weight = weight
        st.session_state.config.keyword_weight = 1.0 - weight

        st.markdown("---")
        if st.button("💾 チャット履歴を保存", use_container_width=True, key="save_chat_history_button"):
            save_chat_history()

    st.title(f"類似回答検索ボット【{st.session_state.business_area}】")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["type"] == "user":
                st.markdown(format_message(msg["text"], True), unsafe_allow_html=True)
            else:
                # === chat.py:1075-1093（回答支援結果表示）===
                if isinstance(msg["text"], list):
                    for idx, response in enumerate(msg["text"], 1):
                        card_html = format_response_card(
                            idx, response["Similarity"],
                            response["Search_Result_Q"], response["Search_Result_A"],
                            category=response.get("Search_Category"),
                            relevance_judgment=response.get("Relevance_Judgment"),
                            judgment_reason=response.get("Judgment_Reason"),
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.markdown(format_message(msg["text"], False), unsafe_allow_html=True)

        # LLM分析ボタン
        # === chat.py:1095-1106 ===

    # 入力フォーム
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("質問を入力してください", key="query", label_visibility="collapsed", placeholder="質問を入力...")
        submit_button = st.form_submit_button("送信", use_container_width=True)
    if submit_button and query.strip():
        st.session_state.chat_history.append({"type": "user", "text": query})
        process_query(query.strip())
        st.rerun()

if __name__ == '__main__':
    run_streamlit_ui()
```

> **ポイント**: `scenario_id`/`is_correct`/`correct_ids` パラメータは全て除去。回答支援AIに正解バッジは不要。

**Step 2: `__init__.py` 作成**

```bash
touch rag-local/apps/answer-support/ui/__init__.py
```

**Step 3: main.py の subprocess パスを最終パスに更新（C2）**

Task 2 で暫定パス（旧 `ui/chat.py`）にしていた箇所を、新ファイルのパスに更新:

```python
# 変更前（Task 2 の暫定パス）:
process = subprocess.Popen([python_executable, "-m", "streamlit", "run",
    str(PROJECT_ROOT / "ui" / "chat.py")])

# 変更後（最終パス）:
ui_chat_path = str(Path(__file__).parent / "ui" / "chat.py")
process = subprocess.Popen([python_executable, "-m", "streamlit", "run", ui_chat_path])
```

**Step 4: コミット**

```bash
git add rag-local/apps/answer-support/ui/ rag-local/apps/answer-support/main.py
git commit -m "refactor: 回答支援UI を apps/answer-support/ui/chat.py に分離

chat.py の回答支援モード（業務分野選択・Processor検索・LLM分析）のみを抽出。
事務改定評価モード（dual_provider, 正解バッジ）は除去。
main.py の subprocess パスを最終パスに更新。
get_available_business_areas の base_dir を PROJECT_ROOT に修正。"
```

---

### Task 5: 事務改定評価UI → apps/revision-eval/ui/eval_ui.py

**Files:**
- Create: `rag-local/apps/revision-eval/ui/eval_ui.py`（事務改定評価モードのみ）
- Create: `rag-local/apps/revision-eval/ui/__init__.py`

**Step 1: eval_ui.py を作成**

chat.py の事務改定評価モード部分で構成:

```python
"""事務改定評価AI — 評価 Streamlit UI

改定番号を選択し、Azure OpenAI / VertexAI 両方で検索して正解IDとのマッチを評価する。
バッチ版は evaluate_revisions.py（Excel出力）。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
import copy
import html
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import SearchConfig, load_settings
from src.utils.logger import setup_logger

# 共通UI部品
from ui.shared import (
    format_message, format_response_card, render_vector_weight_slider,
    apply_common_styles, initialize_common_session_state,
)

logger = setup_logger(__name__)

# 評価設定
_eval_settings = load_settings("evaluation")
AREA_TO_BOT = _eval_settings.get("area_to_bot", {})
AREA_TO_CATEGORY = _eval_settings.get("area_to_category", {})
CATEGORY_TO_AREA = {v: k for k, v in AREA_TO_CATEGORY.items()}

# === chat.py:34-49（改定番号設定）===
_raw_revision_areas = _eval_settings.get("revision_areas", {})
REVISION_TO_AREAS = {}
REVISION_VECTOR_WEIGHTS = {}
REVISION_SEARCH_TYPES = {}
DEFAULT_VECTOR_WEIGHT = _eval_settings.get("vector_weight", 0.9)
# ... for ループでパース ...

INPUT_FILE = PROJECT_ROOT / "data" / "input" / "multi_stage_input.xlsx"

# === chat.py:60-183（評価専用関数群）===
@st.cache_data(ttl=300)
def load_revision_correct_ids(): ...
def extract_bot_name_from_category(category): ...
def extract_bot_name_from_area(area): ...
def build_scenario_id(result): ...
def build_scenario_id_from_area(result, area): ...
def check_if_correct(result, correct_ids, area=None): ...

# === chat.py:200-246（W1: 共通部分は initialize_common_session_state() に委譲）===
def initialize_session_state():
    initialize_common_session_state()  # chat_history, processing_query, config の共通初期化
    if "correct_ids" not in st.session_state:
        st.session_state.correct_ids = []
    if "selected_revision" not in st.session_state:
        st.session_state.selected_revision = None
    if "azure_results" not in st.session_state:
        st.session_state.azure_results = []
    if "vertex_results" not in st.session_state:
        st.session_state.vertex_results = []
    # 注: config 初期化は initialize_common_session_state() で完了済み

# === chat.py:743-775（W2: shared.py ではなくここに配置。eval_ui 版）===
def save_chat_history():
    # ... そのまま移植（eval_ui 版: dual_provider 結果の保存に対応）...

# === chat.py:440-589（評価検索エンジン）===
def execute_dual_provider_search(query, revision): ...
def _search_with_provider(query, revision, provider, areas, vector_weight): ...

# === chat.py:617-643（評価パスのみ）===
def process_query(query):
    st.session_state.processing_query = True
    try:
        query_number = len(st.session_state.chat_history) // 2 + 1
        revision = st.session_state.selected_revision
        azure_results, vertex_results, llm_query = execute_dual_provider_search(query, revision)
        st.session_state.azure_results = azure_results
        st.session_state.vertex_results = vertex_results
        st.session_state.chat_history.append({
            "type": "bot",
            "text": {"mode": "dual_provider", "azure": azure_results, "vertex": vertex_results, "llm_query": llm_query}
        })
    except Exception as e:
        # ... エラー処理 ...
    finally:
        st.session_state.processing_query = False

# === chat.py:778-809 ===
def _render_provider_results(results, correct_ids, is_vertex=False): ...

# === run_streamlit_ui（評価モードのみ）===
def run_streamlit_ui():
    st.set_page_config(page_title="事務改定評価", layout="wide", initial_sidebar_state="expanded")
    apply_common_styles()
    initialize_session_state()

    with st.sidebar:
        st.title("事務改定評価設定")
        # === chat.py:861-943（評価サイドバー）===
        # 改定番号選択、検索タイプ、候補数

        # W3: render_vector_weight_slider は値を返すのみ。config更新は呼び出し側で行う
        weight = render_vector_weight_slider(
            st.session_state.config.vector_weight, key="vector_weight_slider"
        )
        st.session_state.config.vector_weight = weight
        st.session_state.config.keyword_weight = 1.0 - weight

        st.markdown("---")
        if st.button("💾 チャット履歴を保存", use_container_width=True, key="save_chat_history_button"):
            save_chat_history()

    if st.session_state.selected_revision:
        st.title(f"事務改定評価【改定{st.session_state.selected_revision}】")
    else:
        st.title("事務改定評価")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["type"] == "user":
                st.markdown(format_message(msg["text"], True), unsafe_allow_html=True)
            else:
                # === chat.py:1056-1073（評価結果: Azure/VertexAIタブ表示）===
                if isinstance(msg["text"], dict) and msg["text"].get("mode") == "dual_provider":
                    azure_results = msg["text"].get("azure", [])
                    vertex_results = msg["text"].get("vertex", [])
                    correct_ids = st.session_state.correct_ids
                    tab_azure, tab_vertex = st.tabs(["Azure", "VertexAI"])
                    with tab_azure:
                        _render_provider_results(azure_results, correct_ids)
                    with tab_vertex:
                        _render_provider_results(vertex_results, correct_ids, is_vertex=True)
                else:
                    st.markdown(format_message(str(msg["text"]), False), unsafe_allow_html=True)

    # 入力フォーム
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("検索クエリを入力", key="query", label_visibility="collapsed", placeholder="改定影響を検索...")
        submit_button = st.form_submit_button("検索", use_container_width=True)
    if submit_button and query.strip():
        st.session_state.chat_history.append({"type": "user", "text": query})
        process_query(query.strip())
        st.rerun()

if __name__ == '__main__':
    run_streamlit_ui()
```

**Step 2: `__init__.py` 作成 + 旧 chat.py 削除**

```bash
touch rag-local/apps/revision-eval/ui/__init__.py
git rm rag-local/ui/chat.py
```

> `ui/` ディレクトリには `shared.py` と `__init__.py` が残る。
> `ui/__init__.py` は shared.py のパッケージとして必要なため残す（N3: 設計書の `git mv` 方式ではなく新規作成で統一）。

**Step 3: main.py の subprocess パス確認**

Task 4 で修正済みの `main.py` が新しい chat.py を指していることを確認:
```python
ui_chat_path = str(Path(__file__).parent / "ui" / "chat.py")
# → apps/answer-support/ui/chat.py を指す ✓（Task 4 Step 3 で更新済み）
```

**Step 4: コミット**

```bash
git add rag-local/apps/revision-eval/ui/ rag-local/ui/chat.py
git commit -m "refactor: 事務改定評価UI を apps/revision-eval/ui/eval_ui.py に分離

chat.py の事務改定評価モード（改定番号選択・両プロバイダー検索・正解バッジ）を抽出。
旧 ui/chat.py を削除（shared.py + 2つの分割UIに置換完了）。
save_chat_history を eval_ui 側にローカル配置（Processor 依存のため shared.py には含めない）。"
```

---

### Task 6: 非移動ファイルのパス修正

**Files:**
- Modify: `rag-local/scripts/remove_empty_dirs.py:42`
- Modify: `rag-local/scripts/rebuild_faq_db.py:68`
- Modify: `rag-maintenance/scripts/convert-excel-to-json.py:2-3,22`

**Step 1: `remove_empty_dirs.py:42`**

```python
# 変更前:
base_dir = r"C:\VSCode\rag\rag-gemini"
# 変更後:
base_dir = str(Path(__file__).parent.parent)
```
（先頭に `from pathlib import Path` 追加）

**Step 2: `rebuild_faq_db.py:68`**

```python
# 変更前:
print("  streamlit run ui/chat.py")
# 変更後:
print("  streamlit run apps/answer-support/ui/chat.py")
```

**Step 3: `convert-excel-to-json.py:22`**

```python
# 変更前:
RAG_GEMINI_BASE = Path(r"C:\VSCode\rag\rag-gemini\data\source")
# 変更後:
RAG_LOCAL_BASE = Path(r"C:\VSCode\rag\rag-local\data\source")
```
`RAG_GEMINI_BASE` → `RAG_LOCAL_BASE` 全置換 + docstring 更新。

**Step 4: コミット**

```bash
git add rag-local/scripts/remove_empty_dirs.py rag-local/scripts/rebuild_faq_db.py rag-maintenance/scripts/convert-excel-to-json.py
git commit -m "fix: 非移動ファイルのハードコードパス修正"
```

---

### Task 7: アーカイブ移動

**Step 1: 移動 + DEPRECATED.md 削除**

```bash
mkdir -p archive
git mv rag-batch archive/rag-batch
git mv rag-streamlit archive/rag-streamlit
git mv rag-reranker archive/rag-reranker
git rm DEPRECATED.md
```

**Step 2: コミット**

```bash
git add archive DEPRECATED.md
git commit -m "refactor: 非推奨プロジェクトを archive/ に移動"
```

---

### Task 8: 全ドキュメント更新

**Files:**
- Modify: `rag-local/CLAUDE.md`（RAG-Gemini → RAG-Local + ディレクトリ構造図）
- Modify: `rag-local/docs/ARCHITECTURE.md`（パス更新）
- Modify: `rag-local/docs/CONFIGURATION.md`（コマンド例更新）
- Modify: `rag-local/docs/REVISION_EVALUATION.md`（コマンド例更新）
- Modify: `rag-local/docs/TROUBLESHOOTING.md`（コマンド例更新）
- Modify: `rag-local/config/settings.yaml:2`（コメント `RAG-Gemini` → `RAG-Local`）
- Modify: `rag-local/Dockerfile`（新構造に全面更新）
- Modify: `README.md`（ルート引き継ぎ資料）
- Modify: `docs/DOCKER.md`, `docs/TROUBLESHOOTING.md`, `docs/SECURITY.md`
- Modify: `rag-maintenance/docs/導入手順書.md`
- Modify: `rag-maintenance/docs/検索ロジック比較_Phase1_vs_Phase2.md`
- Modify: `rag-maintenance/CLAUDE.md`

**Step 1: rag-local 内**

全ファイルで:
- `rag-gemini` → `rag-local`
- `RAG-Gemini` → `RAG-Local`
- `python main.py` → `python apps/answer-support/main.py`
- `streamlit run ui/chat.py` → `streamlit run apps/answer-support/ui/chat.py`（回答支援UI）
- `python scripts/evaluate_revisions.py` → `python apps/revision-eval/evaluate_revisions.py`
- 「通常検索モード」「通常検索UI」→「回答支援AI」「回答支援UI」に用語統一（N1）

CLAUDE.md ディレクトリ構造図に `apps/` + `ui/shared.py` を反映。

事務改定評価UIの起動コマンドを追加:
```bash
streamlit run apps/revision-eval/ui/eval_ui.py  # 事務改定評価UI
```

Dockerfile を新構造に全面更新（N4: 具体的な COPY 命令を明示）:
```dockerfile
COPY apps/ apps/
COPY ui/ ui/
COPY src/ src/
COPY config.py config.py
COPY config/ config/
COPY prompt/ prompt/
COPY data/ data/
COPY requirements.txt .
```
> 注: 現 Dockerfile はリファクタリング前のモノリシック構造（`processor.py`, `search.py` 等）を参照しており既に壊れている。この機に正しく更新する。

**Step 2: ルート + rag-maintenance**

全ファイルで `rag-gemini` → `rag-local` 置換。
ルート README.md にアプリ一覧テーブルを更新:

```markdown
| AI | エントリポイント | UI |
|----|-----------------|-----|
| 回答支援AI | `apps/answer-support/main.py` | `apps/answer-support/ui/chat.py` |
| 事務改定評価AI | `apps/revision-eval/evaluate_revisions.py` | `apps/revision-eval/ui/eval_ui.py` |
```

**Step 3: コミット**

```bash
git add rag-local/CLAUDE.md rag-local/docs/ rag-local/config/settings.yaml rag-local/Dockerfile README.md docs/ rag-maintenance/
git commit -m "docs: 全ドキュメントを新構造に更新

- rag-gemini → rag-local 全置換
- コマンド例を apps/ パスに更新
- 事務改定評価UIの起動コマンド追加
- Dockerfile 全面更新"
```

---

### Task 9: rag-local README.md 更新

**Files:**
- Modify: `rag-local/README.md`

**Step 1: 既存 README を新構造に書き換え**

```markdown
# RAG-Local（ローカル検証・評価基盤）

> 旧名: rag-gemini

## 概要

2つのAIアプリケーション + 共有コアで構成:

| AI | バッチ | UI | 用途 |
|----|-------|-----|------|
| **回答支援AI** | `apps/answer-support/main.py` | `apps/answer-support/ui/chat.py` | FAQ/シナリオ検索 |
| **事務改定評価AI** | `apps/revision-eval/evaluate_revisions.py` | `apps/revision-eval/ui/eval_ui.py` | 改定影響候補の評価 |

## 前提条件（W4: 環境構築手順）

### 環境構築
cd rag-local
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 環境変数
cp .env.example .env       # .env.example をコピーして必要な値を設定
# 必須: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, GOOGLE_APPLICATION_CREDENTIALS
# 詳細は docs/CONFIGURATION.md 参照

### データ配置
data/source/scenarios/ と data/source/faq/ に検索対象のExcelファイルを配置。
詳細な配置ルールは docs/REVISION_EVALUATION.md 参照。

## クイックスタート

### 回答支援AI
cd rag-local
python apps/answer-support/main.py                  # バッチ
python apps/answer-support/main.py interactive       # UI起動
streamlit run apps/answer-support/ui/chat.py         # UI直接起動

### 事務改定評価AI
python apps/revision-eval/evaluate_revisions.py      # バッチ（Excel出力）
streamlit run apps/revision-eval/ui/eval_ui.py       # 評価UI

## ディレクトリ構造
rag-local/
├── apps/
│   ├── answer-support/          # 回答支援AI
│   │   ├── main.py              # バッチ処理エントリポイント
│   │   └── ui/chat.py           # 回答支援 Streamlit UI
│   └── revision-eval/           # 事務改定評価AI
│       ├── evaluate_revisions.py # バッチExcel出力
│       └── ui/eval_ui.py        # 評価 Streamlit UI
├── ui/
│   └── shared.py                # 共通UI部品（apps/*/ui/ から import される）
├── src/                         # 共有コア（検索エンジン、DB管理等）
├── config.py + config/          # 設定
├── scripts/                     # ユーティリティ
├── data/                        # データ（ベクトルDB、入出力）
└── prompt/                      # プロンプトテンプレート

## rag-maintenance との関係
- rag-local: 検索ロジックのローカル開発・検証・評価に使用
- rag-maintenance: 本番 Teams Bot（M365 Agents SDK + Azure AI Search）
- rag-local で検証した検索パラメータを rag-maintenance に反映するワークフロー
```

**Step 2: コミット**

```bash
git add rag-local/README.md
git commit -m "docs: rag-local README を新構造（apps/ + UI分割）に更新"
```

---

### Task 10: 最終検証 + クリーンアップ

**Step 1: rag-gemini 参照の残存チェック**

```bash
grep -r "rag-gemini" --include="*.py" --include="*.md" --include="*.yaml" --include="*.yml" --include="*.json" --include="Dockerfile" . 2>/dev/null | grep -v ".git/" | grep -v "archive/" | grep -v ".plans/" | grep -v "node_modules/" | grep -v "__pycache__/"
```

Expected: 0件

**Step 2: import スモークテスト**

```bash
cd rag-local
python apps/answer-support/main.py --help
python apps/revision-eval/evaluate_revisions.py --help
python -c "from ui.shared import format_response_card; print('shared OK')"
```

Expected: 全て成功

**Step 3: SearchConfig インスタンス化テスト（N7: --help だけでは不十分なため追加）**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from config import SearchConfig
c = SearchConfig(base_dir='.')
print(f'SearchConfig OK: base_dir={c.base_dir}')
"
```

```bash
python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from ui.shared import initialize_common_session_state
print('initialize_common_session_state import OK')
"
```

Expected: パス解決と環境変数読み込みが正常に動作

**Step 4: Streamlit import チェック**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from apps.answer_support.ui.chat import run_streamlit_ui
print('chat.py imports OK')
" 2>&1 || echo "Note: Streamlit runtime required — import-level check only"
```

```bash
python -c "
import sys; sys.path.insert(0, '.')
from apps.revision_eval.ui.eval_ui import run_streamlit_ui
print('eval_ui.py imports OK')
" 2>&1 || echo "Note: Streamlit runtime required — import-level check only"
```

> 注: Streamlit デコレータ (`@st.cache_data` 等) はランタイム外ではエラーになる場合がある。
> その場合は `streamlit run apps/answer-support/ui/chat.py --server.headless true` で起動確認を行う。

**Step 5: __pycache__ クリーンアップ**

```bash
find rag-local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

**Step 6: ブランチ完了確認**

```bash
git log --oneline refactor/rag-reorganization --not master
```

Expected: 9コミット（Task 1〜9）

---

## コミット一覧（予定）

| # | メッセージ | Task | 種別 |
|---|-----------|------|------|
| 1 | `refactor: rag-gemini → rag-local リネーム` | Task 1 | 構造 |
| 2 | `refactor: main.py + evaluate_revisions.py を apps/ に移動（subprocess パスは暫定）` | Task 2 | 構造 + コード |
| 3 | `refactor: 共通UI部品を ui/shared.py に抽出（純粋関数・Processor非依存）` | Task 3 | **コード** |
| 4 | `refactor: 回答支援UI を apps/answer-support/ui/chat.py に分離` | Task 4 | **コード** |
| 5 | `refactor: 事務改定評価UI を apps/revision-eval/ui/eval_ui.py に分離` | Task 5 | **コード** |
| 6 | `fix: 非移動ファイルのハードコードパス修正` | Task 6 | コード |
| 7 | `refactor: 非推奨プロジェクトを archive/ に移動` | Task 7 | 構造 |
| 8 | `docs: 全ドキュメントを新構造に更新（用語統一・Dockerfile修正）` | Task 8 | ドキュメント |
| 9 | `docs: rag-local README を新構造（apps/ + 環境構築手順）に更新` | Task 9 | ドキュメント |

## v2 → v3 レビュー反映事項

### 必須修正（C: Critical）
- **C1**: `get_available_business_areas()` の `SearchConfig(base_dir=".")` → `str(PROJECT_ROOT)` に変更（Task 4）
  - main.py 経由起動で作業ディレクトリが異なる場合、検索DB参照が失敗するバグ
- **C2**: Task 2 の subprocess パス修正を暫定パス（旧 `ui/chat.py`）にし、Task 4 完了後に最終パスへ更新
  - Task 2 完了時点で `apps/answer-support/ui/chat.py` がまだ存在しないため、`main.py interactive` が壊れるリスクを回避

### 強く推奨（W: Warning）
- **W1**: `initialize_session_state()` の SearchConfig 初期化25行を `initialize_common_session_state()` として shared.py に抽出（Task 3）
  - chat.py と eval_ui.py で重複していた部分の一元化。片方のみ修正する乖離バグを防止
- **W2**: `save_chat_history()` を shared.py から除外し各UI側に配置（Task 3/4/5）
  - Processor への直接依存を shared.py（純粋UI部品）に持ち込まない設計
- **W3**: `render_vector_weight_slider()` を純粋関数化（値を返すのみ、config更新は呼び出し側）（Task 3/4/5）
  - shared.py の「副作用なし」契約を維持し、将来のメンテナンス性を確保
- **W4**: Task 9 README に環境構築手順（venv, .env, データ配置）を追加（Task 9）
  - 引き継ぎ先が Day 2 で環境構築に詰まらないよう、最低限の手順を明示

### 推奨（N: Notice）
- **N1**: 「通常検索」用語を廃止し「回答支援AI」「回答支援UI」に統一（Task 8 全体）
- **N2**: `ui/shared.py` の docstring に「apps/*/ui/ から import される共通モジュール」と明記（Task 3）
- **N3**: 旧 `ui/__init__.py` は shared.py のパッケージとして残す（削除不要）。設計書の `git mv` 方式ではなく実装計画の新規作成方式に統一（Task 5）
- **N4**: Dockerfile の具体的な COPY 命令を計画に追記（Task 8）
- **N5**: `format_response_card` に型ヒント追加（Task 3）
- **N6**: Architecture 注記に「将来 `pyproject.toml` + `pip install -e .` への移行を推奨」を追記（ヘッダー）
- **N7**: Task 10 のテストを `--help` スモーク + SearchConfig インスタンス化テストに強化（Task 10）

## 注意事項

- **git 履歴**: chat.py → shared.py + 2ファイルの分割では履歴追跡は途切れる。git blame で元の行を辿るには `git log --all -p -S "関数名"` を使う
- **venv 再作成**: `rag-local/venv/` は絶対パスが壊れるため再作成推奨
- **中間状態**: Task 3 完了後〜Task 5 完了前は旧 `ui/chat.py` と新 `apps/.../chat.py` が共存する。これは意図的（Task 5 Step 2 の `git rm` で解消）
- **Windows nul ファイル**: 今回のスコープ外
- **.plans/** と **archive/**: 歴史的記録として更新不要
- **settings.yaml の `evaluation` セクション**: eval_ui.py が `load_settings("evaluation")` で読み込む。構造変更なし
- **sys.path 操作箇所**: 合計5箇所（main.py, evaluate_revisions.py, shared.py, chat.py, eval_ui.py）。将来 `pyproject.toml` 導入で一括削除可能（N6）
