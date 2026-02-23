"""
Math Daily Digest — 数学・専門情報 日次配信スクリプト

arXiv, Quanta Magazine, 数学ブログ, YouTube数学チャンネル等から
毎日の新着記事を取得し、Discord Webhook で配信する。

Phase A: feedparserタイムアウト対策 + Discordレート制限リトライ
Phase B: AI要約(Gemini) + 分野フィルター
"""

import os
import sys
import json
import re
import time
import logging
from datetime import datetime, timedelta, timezone
from html import unescape

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

# ── ログ設定 ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 定数 ──────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
MAX_DESCRIPTION_LENGTH = 200
DISCORD_EMBED_LIMIT = 6000      # Discord embed 合計文字数制限
DISCORD_FIELD_LIMIT = 1024       # Discord embed field value 上限
MAX_EMBEDS_PER_MESSAGE = 10      # Discord は1メッセージ最大10 embed
FEED_READ_DEADLINE_SEC = 15      # フィード取得のハードデッドライン
FEED_MAX_BYTES = 1 * 1024 * 1024 # フィード最大サイズ (1MB)

# Gemini API
GEMINI_MODELS = [  # フォールバック順（クォータ超過時に次を試行）
    "gemini-2.5-flash-lite",    # 最も安定（クォータ残量が多い）
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]
GEMINI_BATCH_SIZE = 8            # 1リクエストあたりの記事数（大きすぎるとトークン制限に抵触）


# ═══════════════════════════════════════════════════
# 設定読み込み
# ═══════════════════════════════════════════════════

def load_config(path: str = CONFIG_PATH) -> dict:
    """config.yaml を読み込む。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 環境変数展開
    webhook = cfg.get("discord", {}).get("webhook_url", "")
    if webhook.startswith("${") and webhook.endswith("}"):
        env_key = webhook[2:-1]
        cfg["discord"]["webhook_url"] = os.environ.get(env_key, "")

    return cfg


# ═══════════════════════════════════════════════════
# RSS フィード取得 (Phase A-1: タイムアウト対策済み)
# ═══════════════════════════════════════════════════

def clean_html(raw_html: str) -> str:
    """HTML タグを除去し、プレーンテキストを返す。"""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_category(entry: dict, feed_name: str) -> str:
    """記事のカテゴリ/分野を推定する。"""
    tags = entry.get("tags", [])
    if tags:
        terms = [t.get("term", "") for t in tags if t.get("term")]
        arxiv_cats = [t for t in terms if re.match(r"^[a-z]+\.[A-Z]{2}$", t)]
        if arxiv_cats:
            return ", ".join(arxiv_cats[:3])
        if terms:
            return terms[0][:50]
    return ""


def truncate(text: str, max_len: int = MAX_DESCRIPTION_LENGTH) -> str:
    """テキストを指定長で切り詰める。"""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _fetch_feed_content(url: str, name: str) -> bytes:
    """requests + iter_content でフィードを安全に取得する。

    feedparser.parse(url) は内部で urllib を使い、TCPチャンク間で
    read timeout がリセットされるため無限にハングする可能性がある。
    代わりに requests + ハードデッドラインで安全に取得する。
    (GameResearch Bug #2 のバックポート)
    """
    resp = requests.get(
        url,
        timeout=(5, 10),  # (connect, read) timeout
        headers={"User-Agent": "MathDailyDigest/1.0 (RSS Reader)"},
        stream=True,
    )
    resp.raise_for_status()

    chunks = []
    bytes_read = 0
    deadline = time.time() + FEED_READ_DEADLINE_SEC

    for chunk in resp.iter_content(chunk_size=8192):
        chunks.append(chunk)
        bytes_read += len(chunk)
        if bytes_read >= FEED_MAX_BYTES:
            log.warning(f"    ⚠ サイズ上限到達 ({name}: {bytes_read} bytes)")
            break
        if time.time() > deadline:
            log.warning(f"    ⚠ 読み取りデッドライン到達 ({name}: {bytes_read} bytes)")
            break

    resp.close()
    return b"".join(chunks)


def fetch_feed(feed_cfg: dict, max_age_hours: int, global_max: int) -> list[dict]:
    """単一フィードから記事を取得し、構造化して返す。"""
    url = feed_cfg["url"]
    name = feed_cfg["name"]
    per_feed_max = feed_cfg.get("max_articles", global_max)

    log.info(f"  フィード取得中: {name}")

    try:
        content = _fetch_feed_content(url, name)
        parsed = feedparser.parse(content)
    except requests.Timeout:
        log.warning(f"  ⚠ 接続タイムアウト ({name})")
        return []
    except requests.RequestException as e:
        log.warning(f"  ⚠ フィード取得失敗 ({name}): {e}")
        return []
    except Exception as e:
        log.warning(f"  ⚠ フィード解析失敗 ({name}): {e}")
        return []

    if parsed.bozo and not parsed.entries:
        log.warning(f"  ⚠ フィード解析エラー ({name}): {parsed.bozo_exception}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    articles = []

    for entry in parsed.entries:
        # 日時の解析
        pub_date = None
        for date_field in ("published", "updated", "created"):
            raw = entry.get(date_field)
            if raw:
                try:
                    pub_date = date_parser.parse(raw)
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    break
                except (ValueError, TypeError):
                    continue

        # 日時フィルタ（日時が取れない場合は含める）
        if pub_date and pub_date < cutoff:
            continue

        title = clean_html(entry.get("title", "（タイトルなし）"))
        if not title:
            continue

        desc_raw = entry.get("summary", "") or entry.get("description", "")
        description = truncate(clean_html(desc_raw))
        link = entry.get("link", "")
        category = extract_category(entry, name)

        articles.append({
            "title": title,
            "description": description,
            "url": link,
            "category": category,
            "source_name": name,
            "source_emoji": feed_cfg.get("emoji", "📌"),
            "source_category": feed_cfg.get("category", ""),
            "published": pub_date.isoformat() if pub_date else "",
        })

        if len(articles) >= per_feed_max:
            break

    log.info(f"    → {len(articles)} 件取得")
    return articles


def fetch_all_feeds(config: dict) -> dict[str, list[dict]]:
    """全フィードを取得し、ソース名ごとにグループ化して返す。"""
    schedule = config.get("schedule", {})
    max_age = schedule.get("max_age_hours", 24)
    global_max = schedule.get("max_articles_per_feed", 5)

    feeds = config.get("feeds", [])
    results: dict[str, list[dict]] = {}

    log.info(f"📡 {len(feeds)} フィードの取得を開始...")

    for feed_cfg in feeds:
        name = feed_cfg["name"]
        articles = fetch_feed(feed_cfg, max_age, global_max)
        if articles:
            results[name] = articles

    total = sum(len(v) for v in results.values())
    log.info(f"✅ 取得完了: {len(results)} ソース / {total} 記事")

    return results


# ═══════════════════════════════════════════════════
# Phase B-2: 分野フィルター
# ═══════════════════════════════════════════════════

def _apply_category_filter(
    grouped: dict[str, list[dict]],
    config: dict,
) -> dict[str, list[dict]]:
    """config の include_categories / exclude_categories に基づいて記事をフィルタ。"""
    filters = config.get("filters", {})
    include = filters.get("include_categories", [])
    exclude = filters.get("exclude_categories", [])

    if not include and not exclude:
        return grouped

    filtered: dict[str, list[dict]] = {}
    removed_count = 0

    for source_name, articles in grouped.items():
        kept = []
        for art in articles:
            cats = [c.strip() for c in art.get("category", "").split(",")]

            # include が指定されている場合: いずれかのカテゴリに一致する記事のみ
            if include:
                if not any(c in include for c in cats):
                    removed_count += 1
                    continue

            # exclude が指定されている場合: いずれかのカテゴリに一致する記事を除外
            if exclude:
                if any(c in exclude for c in cats):
                    removed_count += 1
                    continue

            kept.append(art)

        if kept:
            filtered[source_name] = kept

    if removed_count:
        log.info(f"🔍 分野フィルター: {removed_count} 件除外")

    return filtered


# ═══════════════════════════════════════════════════
# Phase B-1: AI要約 (Gemini API)
# ═══════════════════════════════════════════════════

def _call_gemini(api_key: str, prompt: str) -> dict | None:
    """Gemini API を呼び出す（モデルフォールバック付き）。

    クォータ超過(429/503)時に次のモデルを自動試行する。
    """
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            resp = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 2048,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=30,
            )

            result = resp.json()

            # クォータ超過チェック
            if resp.status_code == 429 or resp.status_code == 503:
                log.warning(f"    ⚠ {model}: クォータ超過 — 次のモデルを試行")
                continue

            # エラーレスポンスチェック (candidates がない場合)
            if "candidates" not in result:
                error_msg = result.get("error", {}).get("message", "不明なエラー")
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    log.warning(f"    ⚠ {model}: {error_msg} — 次のモデルを試行")
                    continue
                log.warning(f"    ⚠ {model}: {error_msg}")
                continue

            log.info(f"    📡 使用モデル: {model}")
            return result

        except requests.RequestException as e:
            log.warning(f"    ⚠ {model}: 通信エラー ({e}) — 次のモデルを試行")
            continue

    log.error("    ❌ 全モデルでAI要約に失敗")
    return None


def _extract_json(text: str) -> list | None:
    """Gemini レスポンスからJSON配列を安全に抽出する。

    Gemini は時々 markdown コードブロックでラップしたり
    余分なテキストを付加することがあるため、堅牢に抽出する。
    """
    # 直接パースを試みる
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # markdown コードブロックを除去して再試行
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # テキスト中の [...] を正規表現で抽出
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _summarize_articles(grouped: dict[str, list[dict]]) -> None:
    """Gemini API で記事を日本語要約する (in-place)。

    モデルフォールバック付きで英語記事の要約+翻訳を同時に行う。
    バッチ処理で API コールを最小化。GEMINI_API_KEY 未設定時はスキップ。
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        log.info("ℹ️  GEMINI_API_KEY 未設定 — AI要約をスキップ")
        return

    # 全記事をフラットに集める
    all_articles: list[dict] = []
    for articles in grouped.values():
        all_articles.extend(articles)

    if not all_articles:
        return

    log.info(f"🤖 AI要約を開始... ({len(all_articles)} 記事)")

    # バッチ処理
    batch_size = GEMINI_BATCH_SIZE

    for i in range(0, len(all_articles), batch_size):
        batch = all_articles[i: i + batch_size]

        # プロンプト構築
        prompt_lines = [
            "以下の数学・科学記事のリストについて、それぞれ日本語で1-2行の簡潔な要約を作成してください。",
            "数学用語は正確に訳してください（例: ring→環, field→体, category→圏, manifold→多様体）。",
            'JSONの配列形式で返してください。各要素は {"index": 番号, "summary": "要約"} の形式です。',
            "",
        ]
        for idx, art in enumerate(batch):
            prompt_lines.append(
                f"[{idx}] タイトル: {art['title']}"
                f"\n    説明: {art['description'][:300]}"
                f"\n    分野: {art.get('category', 'N/A')}"
            )

        prompt = "\n".join(prompt_lines)

        try:
            result = _call_gemini(api_key, prompt)
            if result is None:
                continue

            # レスポンス解析（堅牢なJSON抽出）
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            summaries = _extract_json(text)

            if summaries is None:
                log.warning(f"    ⚠ バッチ {i // batch_size + 1}: JSON抽出失敗")
                log.debug(f"    Raw response: {text[:200]}")
                continue

            applied = 0
            for item in summaries:
                idx = item.get("index", -1)
                summary = item.get("summary", "")
                if 0 <= idx < len(batch) and summary:
                    batch[idx]["ai_summary"] = f"🤖 {summary}"
                    applied += 1

            log.info(f"    ✅ バッチ {i // batch_size + 1} 完了 ({applied}/{len(batch)} 件)")

        except Exception as e:
            log.warning(f"    ⚠ AI要約エラー (バッチ {i // batch_size + 1}): {e}")

    summarized = sum(1 for a in all_articles if "ai_summary" in a)
    log.info(f"🤖 AI要約完了: {summarized}/{len(all_articles)} 件")


# ═══════════════════════════════════════════════════
# Discord 送信 (Phase A-2: レート制限リトライ対応)
# ═══════════════════════════════════════════════════

def build_embed_fields(articles: list[dict]) -> list[dict]:
    """記事リストを Discord embed の fields に変換する。"""
    fields = []
    for art in articles:
        cat_str = f" ({art['category']})" if art["category"] else ""
        name = f"📝 {art['title']}"
        if len(name) > 256:
            name = name[:255] + "…"

        value_parts = []
        # AI要約がある場合はそちらを優先表示
        if art.get("ai_summary"):
            value_parts.append(art["ai_summary"])
        elif art["description"]:
            value_parts.append(art["description"])
        if art["url"]:
            value_parts.append(f"🔗 [記事を読む]({art['url']})")
        if cat_str:
            value_parts.append(f"🏷️{cat_str}")

        value = "\n".join(value_parts) if value_parts else "—"
        if len(value) > DISCORD_FIELD_LIMIT:
            value = value[: DISCORD_FIELD_LIMIT - 1] + "…"

        fields.append({"name": name, "value": value, "inline": False})

    return fields


def _embed_char_count(embed: dict) -> int:
    """embed 内の文字数を計算する（Discord API の制限チェック用）。"""
    count = 0
    count += len(embed.get("title", ""))
    count += len(embed.get("description", ""))
    for field in embed.get("fields", []):
        count += len(field.get("name", ""))
        count += len(field.get("value", ""))
    footer = embed.get("footer", {})
    count += len(footer.get("text", ""))
    count += len(embed.get("author", {}).get("name", ""))
    return count


def build_discord_payloads(grouped: dict[str, list[dict]], date_str: str) -> list[dict]:
    """Discord に送信する payload のリストを構築する。

    Discord API の制限:
    - 1メッセージあたり最大 10 embed
    - 1メッセージのembed合計文字数が 6000 文字以内
    """
    today = date_str
    embeds = []

    # ヘッダー embed
    has_ai = any(
        art.get("ai_summary")
        for arts in grouped.values()
        for art in arts
    )
    desc = "数学・専門情報の日次ダイジェスト"
    if has_ai:
        desc += " (🤖 AI要約付き)"

    embeds.append({
        "title": f"📐 Math Daily Digest ({today})",
        "description": desc,
        "color": 0x4A90D9,
    })

    # カテゴリ順に並べる
    category_order = ["論文", "ニュース", "ブログ", "YouTube"]
    category_colors = {
        "論文": 0xE74C3C,
        "ニュース": 0x2ECC71,
        "ブログ": 0xF39C12,
        "YouTube": 0x9B59B6,
    }

    for cat in category_order:
        sources_in_cat = {
            name: arts
            for name, arts in grouped.items()
            if arts and arts[0].get("source_category") == cat
        }
        if not sources_in_cat:
            continue

        for source_name, articles in sources_in_cat.items():
            emoji = articles[0].get("source_emoji", "📌") if articles else "📌"
            fields = build_embed_fields(articles)

            embed = {
                "title": f"{emoji} {source_name}",
                "color": category_colors.get(cat, 0x95A5A6),
                "fields": fields,
            }
            embeds.append(embed)

    # フッター embed
    total_sources = len(grouped)
    total_articles = sum(len(v) for v in grouped.values())
    embeds.append({
        "title": f"📊 本日の集計: {total_sources} ソース / {total_articles} 記事",
        "color": 0x4A90D9,
    })

    # サイズと個数の制限を考慮して分割
    payloads = []
    current_chunk: list[dict] = []
    current_chars = 0

    for embed in embeds:
        embed_size = _embed_char_count(embed)
        would_exceed_chars = (current_chars + embed_size) > DISCORD_EMBED_LIMIT
        would_exceed_count = len(current_chunk) >= MAX_EMBEDS_PER_MESSAGE

        if current_chunk and (would_exceed_chars or would_exceed_count):
            payloads.append({"embeds": current_chunk})
            current_chunk = []
            current_chars = 0

        current_chunk.append(embed)
        current_chars += embed_size

    if current_chunk:
        payloads.append({"embeds": current_chunk})

    return payloads


def _post_webhook(url: str, payload: dict, max_retries: int = 3) -> bool:
    """Discord Webhook に送信する（レート制限リトライ付き）。

    GameResearch Bug #1 のバックポート:
    429 (Rate Limited) 応答時に retry_after 秒待機してリトライ。
    """
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 2)
                log.warning(f"    ⏳ レート制限 — {retry_after}秒待機...")
                time.sleep(retry_after)
                continue
            if resp.status_code == 204:
                return True
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            log.error(f"    ❌ Discord送信エラー (試行 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return False


def send_to_discord(webhook_url: str, payloads: list[dict]) -> bool:
    """Discord Webhook にメッセージを送信する。"""
    if not webhook_url:
        log.error("❌ DISCORD_WEBHOOK_URL が設定されていません")
        return False

    success = True
    for idx, payload in enumerate(payloads):
        log.info(f"  Discord 送信中... ({idx + 1}/{len(payloads)})")
        if _post_webhook(webhook_url, payload):
            log.info(f"    ✅ 送信成功")
        else:
            log.error(f"    ❌ 送信失敗 (リトライ上限)")
            success = False

    return success


# ═══════════════════════════════════════════════════
# メイン処理
# ═══════════════════════════════════════════════════

def main():
    log.info("=" * 50)
    log.info("📐 Math Daily Digest — 開始")
    log.info("=" * 50)

    # 設定読み込み
    config = load_config()
    webhook_url = config.get("discord", {}).get("webhook_url", "")

    # フィード取得
    grouped = fetch_all_feeds(config)

    if not grouped:
        log.warning("⚠ 新着記事がありません。Discord への送信をスキップします。")
        sys.exit(0)

    # Phase B-2: 分野フィルター
    grouped = _apply_category_filter(grouped, config)

    if not grouped:
        log.warning("⚠ フィルター適用後、該当記事がありません。")
        sys.exit(0)

    # Phase B-1: AI要約 (Gemini API)
    if config.get("schedule", {}).get("summarize", False):
        _summarize_articles(grouped)

    # Discord 送信
    jst = timezone(timedelta(hours=9))
    today = datetime.now(jst).strftime("%Y/%m/%d")
    payloads = build_discord_payloads(grouped, today)

    log.info(f"📤 Discord に送信中... ({len(payloads)} メッセージ)")
    ok = send_to_discord(webhook_url, payloads)

    if ok:
        log.info("✅ Math Daily Digest — 完了")
    else:
        log.error("❌ 一部の送信に失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
