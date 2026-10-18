"""Single-file Telethon RAG bot for multiple Telegram groups.

Stores messages from allowed groups in SQLite FTS5 (per chat), retrieves the
most relevant ones for /ask, and sends them as context to an OpenAI-compatible
chat completions endpoint. Configuration comes from environment variables only.

The bot must be a group admin, or have privacy mode disabled via @BotFather,
to see regular group messages.
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.request

from telethon import TelegramClient, events

log = logging.getLogger("rag_bot")

HELP_TEXT = (
    "/ask <question> - answer using this group's history and notes\n"
    "/addnote <text> - add a note to this group's knowledge base (admins)\n"
    "/help - show this message"
)
SYSTEM_PROMPT = (
    "You answer questions for a Telegram group. Use only the provided context. "
    "If the context does not contain the answer, say you don't know."
)


def env_ids(name):
    return {int(x) for x in os.environ.get(name, "").replace(" ", "").split(",") if x}


def load_config():
    return {
        "api_id": int(os.environ["TG_API_ID"]),
        "api_hash": os.environ["TG_API_HASH"],
        "bot_token": os.environ["TG_BOT_TOKEN"],
        "allowed_groups": env_ids("ALLOWED_GROUPS"),
        "admin_ids": env_ids("ADMIN_IDS"),
        "llm_url": os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        "llm_key": os.environ.get("LLM_API_KEY", ""),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "db_path": os.environ.get("DB_PATH", "rag_bot.db"),
        "session": os.environ.get("TG_SESSION_NAME", "rag_bot"),
        "top_k": int(os.environ.get("RAG_TOP_K", "8")),
    }


def open_db(path):
    db = sqlite3.connect(path)
    db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
        "text, chat_id UNINDEXED, sender_id UNINDEXED, kind UNINDEXED, date UNINDEXED)"
    )
    return db


def add_doc(db, chat_id, sender_id, text, kind="msg", date=""):
    db.execute(
        "INSERT INTO docs (text, chat_id, sender_id, kind, date) VALUES (?, ?, ?, ?, ?)",
        (text, chat_id, sender_id, kind, date),
    )
    db.commit()


def fts_query(question):
    terms = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    return " OR ".join(f'"{t}"' for t in terms)


def search(db, chat_id, question, limit=8):
    query = fts_query(question)
    if not query:
        return []
    rows = db.execute(
        "SELECT text, kind, date FROM docs WHERE docs MATCH ? AND chat_id = ? "
        "ORDER BY bm25(docs) - (kind = 'note') * 5 LIMIT ?",
        (query, chat_id, limit),
    ).fetchall()
    return rows


def build_messages(question, rows):
    context = "\n".join(f"[{kind} {date}] {text}" for text, kind, date in rows) or "(no context found)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def call_llm(cfg, messages):
    body = json.dumps({"model": cfg["llm_model"], "messages": messages}).encode()
    req = urllib.request.Request(
        cfg["llm_url"],
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['llm_key']}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


async def is_admin(client, cfg, event):
    if event.sender_id in cfg["admin_ids"]:
        return True
    perms = await client.get_permissions(event.chat_id, event.sender_id)
    return perms.is_admin


def register_handlers(client, cfg, db):
    groups = cfg["allowed_groups"] or None

    @client.on(
        events.NewMessage(chats=groups, func=lambda e: e.is_group and e.raw_text and not e.raw_text.startswith("/"))
    )
    async def on_group_message(event):
        add_doc(db, event.chat_id, event.sender_id, event.raw_text, "msg", event.date.isoformat())

    @client.on(events.NewMessage(chats=groups, pattern=r"^/ask(?:@\w+)?(?:\s+(.+))?$"))
    async def on_ask(event):
        question = (event.pattern_match.group(1) or "").strip()
        if not question:
            await event.reply("Usage: /ask <question>")
            return
        rows = search(db, event.chat_id, question, cfg["top_k"])
        try:
            answer = await asyncio.to_thread(call_llm, cfg, build_messages(question, rows))
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            log.warning("LLM call failed: %s", exc)
            answer = "Sorry, the language model is unavailable right now."
        await event.reply(answer)

    @client.on(events.NewMessage(chats=groups, pattern=r"^/addnote(?:@\w+)?(?:\s+(.+))?$"))
    async def on_addnote(event):
        text = (event.pattern_match.group(1) or "").strip()
        if not text:
            await event.reply("Usage: /addnote <text>")
            return
        if not await is_admin(client, cfg, event):
            await event.reply("Only admins can add notes.")
            return
        add_doc(db, event.chat_id, event.sender_id, text, "note", event.date.isoformat())
        await event.reply("Note saved.")

    @client.on(events.NewMessage(pattern=r"^/(?:help|start)(?:@\w+)?$"))
    async def on_help(event):
        await event.reply(HELP_TEXT)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    db = open_db(cfg["db_path"])
    client = TelegramClient(cfg["session"], cfg["api_id"], cfg["api_hash"])
    register_handlers(client, cfg, db)
    client.start(bot_token=cfg["bot_token"])
    log.info("Bot started; allowed groups: %s", sorted(cfg["allowed_groups"]) or "all")
    client.run_until_disconnected()


if __name__ == "__main__":
    main()
