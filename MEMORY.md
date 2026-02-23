# MEMORY.md — MathResearch

## プロジェクト概要
数学・専門情報を毎日自動収集し、Discord に配信するダイジェストシステム。

## 技術スタック
- Python 3.12 + feedparser + requests + pyyaml + beautifulsoup4
- Discord Webhook（Embed形式）
- GitHub Actions cron（JST 7:00 定時実行）

## 情報ソース (21フィード)
- arXiv: math全般, math.AG, math.NT, math.CA, math.CT, cs.LG
- ニュース: Quanta Magazine, ScienceDaily, AMS (2フィード)
- ブログ: Terence Tao, n-Category Café, Math3ma, Math∩Programming, Gödel's Lost Letter, Joel David Hamkins
- YouTube: 3Blue1Brown, Numberphile, Mathologer, Stand-up Maths, Veritasium

## 現在のステータス
- 初期実装完了
- Discord Webhook URL 未設定（ユーザー設定待ち）

## 前回のセッション
- 2026-02-23: 初期設計・実装
