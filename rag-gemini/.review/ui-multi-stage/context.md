# UI Multi-Stage Search Support

## 変更概要

`ui/chat.py` に多段階検索（multi_stage）モードのサポートを追加。

## 変更内容

1. **SearchConfig初期化**: `search_mode="original"` パラメータを追加
2. **サイドバー**: 検索モード選択ドロップダウンを追加（原文検索/LLMクエリ検索/多段階OR検索）
3. **多段階検索パラメータ**: multi_stage選択時のみ、しきい値スライダーとLLM影響分析チェックボックスを表示
4. **Processor再初期化**: search_mode変更時も再初期化するよう条件を追加
5. **結果表示**: Search_Categoryバッジ表示対応（両方/原文のみ/LLMのみ）

## レビュー観点

1. **Streamlit状態管理**: session_stateの使用方法は適切か
2. **UI/UX**: 検索モード切り替えのユーザー体験は良好か
3. **エラーハンドリング**: search_modeが不正な値の場合の処理
4. **後方互換性**: 既存の検索機能に影響はないか
