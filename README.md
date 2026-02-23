# MathResearch — Math Daily Digest

数学・専門情報を毎日自動収集し、Discord に日次ダイジェストとして配信するシステム。

## セットアップ

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. Discord Webhook の設定
1. Discord で配信先チャンネルを選択
2. チャンネル設定 → 連携サービス → Webhook → 新しいウェブフック
3. Webhook URL をコピー
4. 環境変数に設定:
```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

### 3. ローカル実行
```bash
python daily_math_digest.py
```

### 4. GitHub Actions 定時実行
1. GitHub リポジトリの Settings → Secrets and variables → Actions
2. `DISCORD_WEBHOOK_URL` を Repository secret として追加
3. 毎日 JST 7:00 に自動実行されます（手動実行も可）

## 情報ソース

| カテゴリ | ソース |
|:--|:--|
| 論文 | arXiv (math, math.AG, math.NT, math.CA, math.CT, cs.LG) |
| ニュース | Quanta Magazine, ScienceDaily, AMS |
| ブログ | Terence Tao, n-Category Café, Math3ma, Math∩Programming, Gödel's Lost Letter, Joel David Hamkins |
| YouTube | 3Blue1Brown, Numberphile, Mathologer, Stand-up Maths, Veritasium |
