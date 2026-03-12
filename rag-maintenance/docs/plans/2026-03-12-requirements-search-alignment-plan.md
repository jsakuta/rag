# 要件定義書・検索設計書 整合化 修正計画

**作成日:** 2026-03-12  
**対象:** `docs/要件定義書.md`, `docs/検索設計書.md`, 関連実装

---

## 1. 目的

要件定義書、検索設計書、関連実装の間に残っている不整合を解消し、以下を満たす状態にする。

- 現行スコープが初学者にも誤解なく読める
- 将来構想が現行要件に混入しない
- 検索仕様、保存仕様、POC制約が文書とコードで一致する
- API制約と実装上の独自制約を区別して記述する

---

## 2. 今回の確定方針

今回の修正では、以下を正とする。

1. Week 3 の画像ベクトル化関連は、今回の要件定義書・検索設計書から**完全に外す**
2. `rerankerScore` は **Cosmos DB に保存する前提**で、文書とコードを合わせる
3. `searchSessionId` は **POC の一時セッションキー**として明記する
4. Excel 出力サイズ上限は、**現行実装の 4MB 制限**と**Graph API の公式上限**を区別して記載する
5. データ件数は、**概算**と**検証環境の実測**を分けて書く
6. 修正対象は **文書 + 関連コードの是正** とする

---

## 3. 現時点の主要ギャップ

### 3.1 スコープ

- `docs/要件定義書.md` に FR-007、Week 3、`imageVector`、Blob Storage、Foundry、画像試験項目が残っている
- 現行実装には画像ベクトル化の実体がない

### 3.2 検索仕様

- キーワード検索の表現は改善済みだが、要件定義書と検索設計書の関連記述を再統一する必要がある
- semantic モードでは `rerankerScore` を優先して順位付けしている

### 3.3 保存仕様

- 実装では `rerankerScore` を検索結果として保持しているが、`impactAssessments` には保存していない
- `searchSessionId` は UI 継続用の一時キーだが、要件書で十分に定義されていない

### 3.4 制約・根拠

- Graph API の公式上限と、実装上の 4MB 制限が混在している
- POC 制約である単一プロセス・メモリキャッシュ依存が文書化されていない

### 3.5 データ件数

- 要件定義書では概算の「シナリオ数百件 + FAQ 約3,000件」
- 検索設計書では検証環境の実測「シナリオ 2,313 件 + FAQ 18,734 件 = 合計 21,047 件」
- 両方とも意味はあるが、役割を分けて書く必要がある

---

## 4. 修正対象ファイル

### 文書

- `docs/要件定義書.md`
- `docs/検索設計書.md`
- 必要に応じて `docs/データベース設計書.md`

### 実装

- `maintenance-bot/src/agent.ts`
- `maintenance-bot/src/cards.ts`
- `maintenance-bot/src/cosmos.ts`
- `maintenance-bot/src/config.ts`
- `maintenance-bot/src/sharepoint.ts`

### 参照定義

- `scripts/index-definition.json`
- `scripts/skillset.json`
- `scripts/indexer-scenarios.json`
- `scripts/indexer-faqs.json`

---

## 5. 実行順序

修正は以下の順序で行う。

1. Week 3 関連を現行要件から除外する
2. 検索仕様と保存仕様の文書を現行実装に合わせる
3. `rerankerScore` 保存のコード修正を行う
4. POC 制約と API 根拠を文書へ明記する
5. データ件数と indexer 間隔の記載を統一する
6. 最後に全体整合性レビューを行う

この順序にする理由は、スコープ確定前に細部を直すと再修正が発生しやすいため。

---

## 6. タスク詳細

### Task 1: Week 3 関連の除外

**目的:** 現行要件から未実装の画像ベクトル化関連を切り離す

**対象:**

- `docs/要件定義書.md`

**修正内容:**

- FR-007 を現行要件一覧から削除
- 画像ベクトル化、`imageVector`、Blob Storage、Foundry、Week 3 試験項目を削除
- Step 1 現行スコープに関係ない図、構成図、コスト、データ構成、付録を整理
- 画像未検知の説明は「今回スコープ外」として簡潔に触れるか、別文書に移す

**完了条件:**

- 要件定義書を読んだとき、画像ベクトル化が現行スコープに見えない

---

### Task 2: キーワード検索仕様の最終統一

**目的:** FR-004 の説明を要件定義書・検索設計書・実装で一致させる

**対象:**

- `docs/要件定義書.md`
- `docs/検索設計書.md`
- `maintenance-bot/src/agent.ts`

**修正内容:**

- 「完全一致検索」というニュアンスが残っていないか全検索
- `queryType: "full"`、`title` / `content` / `keywords` を対象とした語句ベース全文検索であることを明記
- substring 一致ではないことを検索設計書の説明に合わせる
- 意味検索との使い分けを初学者向けに簡潔化する

**完了条件:**

- FR-004 を読んで、実装以上に強いことを言っていない

---

### Task 3: `rerankerScore` 保存仕様の反映

**目的:** semantic モードの関連度を文書・コードで揃える

**対象:**

- `maintenance-bot/src/cards.ts`
- `maintenance-bot/src/agent.ts`
- `maintenance-bot/src/cosmos.ts`
- `docs/要件定義書.md`
- 必要に応じて `docs/データベース設計書.md`

**修正内容:**

- `saveNeedsUpdate` に `rerankerScore` を保存できるよう設計する
- 保存対象を「チェックされたシナリオごとの `rerankerScore`」として定義する
- `impactAssessments` の JSON 例、項目説明、用途を更新する
- 「検索結果の表示用スコア」と「DB保存値」の意味を明確化する

**実装上の論点:**

- 複数選択時に各シナリオの `rerankerScore` をどこから渡すか
- 現状の `saveNeedsUpdate` では `scenarioIds` と `searchQuery` しか渡していないため、カードデータかキャッシュから復元する必要がある

**完了条件:**

- 文書で保存すると書いた項目が、実際に保存される

---

### Task 4: `searchSessionId` の定義と POC 制約の明記

**目的:** UI 継続用セッションと DB 上の検索記録 ID を区別する

**対象:**

- `docs/要件定義書.md`
- `docs/検索設計書.md`
- 必要に応じて `docs/データベース設計書.md`

**修正内容:**

- `searchSessionId` を「検索結果カードの継続操作に使う一時セッションキー」と定義
- `impactAssessments.searchId` とは別物であることを明記
- メモリキャッシュ依存、TTL 30 分、最大 50 セッション、再起動・スケールアウトで消えることを制約へ追加
- 「POC のため単一プロセス前提」であることを明記

**初学者向け説明方針:**

- `searchSessionId` は「検索結果画面の作業を続けるための整理番号」と表現する

**完了条件:**

- 読者が `searchSessionId` を監査用 ID と誤解しない

---

### Task 5: Graph API 上限と実装制限の切り分け

**目的:** 製品制約と POC 実装制約を分けて記載する

**対象:**

- `docs/要件定義書.md`
- `maintenance-bot/src/config.ts`
- `maintenance-bot/src/sharepoint.ts`

**修正内容:**

- 文書には次の 2 つを分けて書く
  - Graph API `/content` の公式上限
  - 本実装の安全側制限 4MB
- 4MB が `SPO_SIMPLE_UPLOAD_LIMIT` によるアプリ制約であることを明記
- 将来は公式上限ベースに見直せることも注記
- 可能であれば `config.ts` のコメントも「Graph API 制約」ではなく「POC 実装制約」に直す

**根拠リンク:**

- Microsoft Graph `driveItem-put-content`

**完了条件:**

- 仕様書を読んだ人が「Graph API 自体が 4MB 制限」と誤解しない

---

### Task 6: データ件数の書き分け整理

**目的:** 概算と実測を同居させつつ、意味を分ける

**対象:**

- `docs/要件定義書.md`
- `docs/検索設計書.md`

**修正内容:**

- 要件定義書の業務背景では概算を使う
- 検索設計書の現行パラメータでは検証環境実測を使う
- 必要であれば要件定義書にも注記として実測を追記する

**実測値:**

- シナリオ: 2,313 件
- FAQ（問い合わせ履歴）: 18,734 件
- 合計: 21,047 件

**完了条件:**

- 概算と実測のどちらを書いているかが読者に分かる

---

### Task 7: Indexer 間隔と運用記述の統一

**目的:** 要件定義書・検索設計書・定義ファイルの間でスケジュール記述を一致させる

**対象:**

- `docs/要件定義書.md`
- `docs/検索設計書.md`
- `scripts/indexer-scenarios.json`
- `scripts/indexer-faqs.json`

**修正内容:**

- 実設定が `PT10M` であることを基準に記述を統一
- 検索設計書の「1時間ごと」を修正
- Azure AI Search Basic SKU の最短 5 分という製品制約と、現設定 10 分を分けて書く

**完了条件:**

- 文書間で「1時間」と「10分」が混在しない

---

### Task 8: 最終整合チェック

**目的:** 修正後の文書とコードの整合性を最終確認する

**確認観点:**

- 要件定義書に Week 3 の現行要件混入がない
- FR-004 の説明が実装と一致している
- `rerankerScore` の保存仕様が文書とコードで一致している
- `searchSessionId` の扱いが明確
- Graph API 公式上限と実装制限が分離されている
- 件数、indexer 間隔、制約が文書間で一致している

**検証:**

- `npm run build`
- 変更箇所の grep 確認
- 文書横断レビュー

---

## 7. 想定コード修正の最小範囲

文書だけではなく、以下のコード修正を想定する。

### 必須

- `maintenance-bot/src/cosmos.ts`
  - `impactAssessments` に `rerankerScore` を保存できるよう変更

### 条件付き

- `maintenance-bot/src/agent.ts`
  - `saveNeedsUpdate` 呼び出し時に対象シナリオの `rerankerScore` を渡す
  - キャッシュまたはカードデータから対象スコアを復元する

- `maintenance-bot/src/config.ts`
  - 4MB 制限コメントの見直し

- `maintenance-bot/src/sharepoint.ts`
  - 4MB が実装制約であることが分かるコメントに修正

---

## 8. リスクと注意点

1. `rerankerScore` 保存対応は、選択済みシナリオごとのスコア受け渡し設計が必要
2. `searchSessionId` と `searchId` を混同すると文書が再び崩れる
3. Week 3 削除時に、付録・図・コスト・用語定義の残骸が出やすい
4. 概算件数と実測件数の両立は、書き分けルールを明文化しないと再発する

---

## 9. 完了の定義

以下を満たしたら完了とする。

- 要件定義書から Week 3 現行要件が除去されている
- 検索設計書が現行実装に一致している
- `rerankerScore` が DB 保存まで含めて整合している
- `searchSessionId` が POC 一時キーとして定義されている
- Graph API 公式制約と POC 実装制約が分離記載されている
- データ件数が概算と実測で整理されている
- `npm run build` が通る

