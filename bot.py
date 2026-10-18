"""Single-file Telethon RAG bot for multiple Telegram groups.

Stores messages from allowed groups in SQLite FTS5 (per chat), retrieves the
most relevant ones for questions, and sends them as context to an
OpenAI-compatible chat completions endpoint. Everything is driven by inline
buttons with callbacks; commands remain as shortcuts. Configuration comes from
environment variables only.

The bot must be a group admin, or have privacy mode disabled via @BotFather,
to see regular group messages.
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
import urllib.request

from telethon import Button, TelegramClient, events
from telethon.errors import MessageNotModifiedError

log = logging.getLogger("rag_bot")

MAX_TEXT = 4000
NOTES_PER_PAGE = 5
PENDING_TTL = 120
ADMIN_CACHE_TTL = 300

SYSTEM_PROMPT = (
    "You answer questions for a Telegram group. Use only the provided context. "
    "If the context does not contain the answer, say you don't know."
)
HELP_TEXT = (
    "❓ **How to use me**\n\n"
    "🔎 **Ask** – tap the button (or `/ask <question>`) and I answer from this group's history and notes.\n"
    "📝 **Add note** – admins add facts to the group knowledge base (`/addnote <text>`).\n"
    "📚 **Notes** – browse and 🗑 delete notes.\n"
    "📊 **Stats** – what I know about this group.\n"
    "⚙️ **Settings** – admins toggle learning or 🧹 clear history.\n\n"
    "Under each answer: 🔁 regenerate · 📎 sources · 👍/👎 feedback."
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
        "cooldown": float(os.environ.get("ASK_COOLDOWN", "5")),
    }


# --- storage -----------------------------------------------------------------


def open_db(path):
    db = sqlite3.connect(path)
    db.executescript(
        "PRAGMA journal_mode=WAL;"
        "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
        "text, chat_id UNINDEXED, sender_id UNINDEXED, kind UNINDEXED, date UNINDEXED);"
        "CREATE TABLE IF NOT EXISTS asks (id INTEGER PRIMARY KEY, chat_id INTEGER, user_id INTEGER,"
        " question TEXT, answer TEXT, rating INTEGER DEFAULT 0);"
        "CREATE TABLE IF NOT EXISTS settings (chat_id INTEGER PRIMARY KEY, learn INTEGER DEFAULT 1);"
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
    return db.execute(
        "SELECT text, kind, date FROM docs WHERE docs MATCH ? AND chat_id = ? "
        "ORDER BY bm25(docs) - (kind = 'note') * 5 LIMIT ?",
        (query, chat_id, limit),
    ).fetchall()


def list_notes(db, chat_id, page):
    total = db.execute("SELECT count(*) FROM docs WHERE chat_id = ? AND kind = 'note'", (chat_id,)).fetchone()[0]
    rows = db.execute(
        "SELECT rowid, text FROM docs WHERE chat_id = ? AND kind = 'note' ORDER BY rowid DESC LIMIT ? OFFSET ?",
        (chat_id, NOTES_PER_PAGE, page * NOTES_PER_PAGE),
    ).fetchall()
    return rows, total


def delete_note(db, chat_id, rowid):
    db.execute("DELETE FROM docs WHERE rowid = ? AND chat_id = ? AND kind = 'note'", (rowid, chat_id))
    db.commit()


def clear_history(db, chat_id):
    db.execute("DELETE FROM docs WHERE chat_id = ? AND kind = 'msg'", (chat_id,))
    db.commit()


def chat_stats(db, chat_id):
    msgs, notes, last = db.execute(
        "SELECT sum(kind = 'msg'), sum(kind = 'note'), max(date) FROM docs WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    asks, up, down = db.execute(
        "SELECT count(*), sum(rating = 1), sum(rating = -1) FROM asks WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return {"msgs": msgs or 0, "notes": notes or 0, "last": last or "–", "asks": asks, "up": up or 0, "down": down or 0}


def save_ask(db, chat_id, user_id, question, answer):
    cur = db.execute(
        "INSERT INTO asks (chat_id, user_id, question, answer) VALUES (?, ?, ?, ?)",
        (chat_id, user_id, question, answer),
    )
    db.commit()
    return cur.lastrowid


def get_ask(db, ask_id):
    return db.execute("SELECT chat_id, question, answer, rating FROM asks WHERE id = ?", (ask_id,)).fetchone()


def update_ask(db, ask_id, **fields):
    cols = ", ".join(f"{k} = ?" for k in fields)
    db.execute(f"UPDATE asks SET {cols} WHERE id = ?", (*fields.values(), ask_id))
    db.commit()


def learning_enabled(db, chat_id):
    row = db.execute("SELECT learn FROM settings WHERE chat_id = ?", (chat_id,)).fetchone()
    return True if row is None else bool(row[0])


def toggle_learning(db, chat_id):
    value = not learning_enabled(db, chat_id)
    db.execute("INSERT OR REPLACE INTO settings (chat_id, learn) VALUES (?, ?)", (chat_id, int(value)))
    db.commit()
    return value


# --- llm ---------------------------------------------------------------------


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


async def generate_answer(cfg, db, chat_id, question):
    rows = search(db, chat_id, question, cfg["top_k"])
    try:
        return await asyncio.to_thread(call_llm, cfg, build_messages(question, rows))
    except (OSError, KeyError, IndexError, ValueError) as exc:
        log.warning("LLM call failed: %s", exc)
        return None


# --- views (text + buttons) --------------------------------------------------


def clip(text, limit=MAX_TEXT):
    return text if len(text) <= limit else text[: limit - 1] + "…"


def home_view(bot_username, private):
    if private:
        text = (
            "👋 **Hi! I'm a group knowledge bot.**\n\nAdd me to a group, make me admin, and I'll learn and answer. 🧠"
        )
        buttons = [
            [Button.url("➕ Add me to a group", f"https://t.me/{bot_username}?startgroup=true", style="primary")],
            [Button.inline("❓ Help", b"m:help")],
        ]
        return text, buttons
    text = "🤖 **Group Knowledge Bot**\n\nPick an option below 👇"
    buttons = [
        [Button.inline("🔎 Ask", b"m:ask", style="primary"), Button.inline("📝 Add note", b"m:note")],
        [Button.inline("📚 Notes", b"m:notes:0"), Button.inline("📊 Stats", b"m:stats")],
        [Button.inline("⚙️ Settings", b"m:set"), Button.inline("❓ Help", b"m:help")],
        [Button.inline("✖️ Close", b"x")],
    ]
    return text, buttons


def back_row():
    return [Button.inline("🏠 Menu", b"m:home"), Button.inline("✖️ Close", b"x")]


def answer_view(ask_id, question, answer, rating=0):
    if answer is None:
        text = f"❓ **{clip(question, 300)}**\n\n⚠️ The language model is unavailable right now. Try 🔁 again."
    else:
        text = clip(f"❓ **{clip(question, 300)}**\n\n💡 {answer}")
    up = "👍✅" if rating == 1 else "👍"
    down = "👎✅" if rating == -1 else "👎"
    buttons = [
        [
            Button.inline("🔁 Regenerate", f"q:re:{ask_id}".encode(), style="primary"),
            Button.inline("📎 Sources", f"q:src:{ask_id}".encode()),
        ],
        [
            Button.inline(up, f"q:up:{ask_id}".encode(), style="success" if rating == 1 else None),
            Button.inline(down, f"q:dn:{ask_id}".encode(), style="danger" if rating == -1 else None),
        ],
    ]
    return text, buttons


def sources_view(ask_id, rows):
    if rows:
        icons = {"note": "📝", "msg": "💬"}
        lines = [f"{icons.get(kind, '•')} {clip(text, 300)}" for text, kind, _ in rows]
        text = clip("📎 **Sources used**\n\n" + "\n\n".join(lines))
    else:
        text = "📎 **Sources used**\n\n🤷 No matching messages or notes."
    return text, [[Button.inline("⬅️ Back to answer", f"q:ans:{ask_id}".encode())]]


def notes_view(rows, total, page):
    pages = max(1, -(-total // NOTES_PER_PAGE))
    if not rows:
        text = "📚 **Notes**\n\n🫙 No notes yet. Tap 📝 Add note to create one."
        return text, [back_row()]
    start = page * NOTES_PER_PAGE
    lines = [f"**{start + i + 1}.** {clip(text, 400)}" for i, (_, text) in enumerate(rows)]
    text = clip(f"📚 **Notes** ({total})\n\n" + "\n\n".join(lines))
    delete_row = [
        Button.inline(f"🗑 {start + i + 1}", f"n:del:{rowid}:{page}".encode(), style="danger")
        for i, (rowid, _) in enumerate(rows)
    ]
    nav = [
        Button.inline("◀️", f"m:notes:{page - 1}".encode()) if page > 0 else Button.inline("⏺", b"noop"),
        Button.inline(f"📄 {page + 1}/{pages}", b"noop"),
        Button.inline("▶️", f"m:notes:{page + 1}".encode()) if page + 1 < pages else Button.inline("⏺", b"noop"),
    ]
    return text, [delete_row, nav, back_row()]


def stats_view(stats):
    text = (
        "📊 **Group stats**\n\n"
        f"💬 Messages learned: **{stats['msgs']}**\n"
        f"📝 Notes: **{stats['notes']}**\n"
        f"🔎 Questions answered: **{stats['asks']}**\n"
        f"👍 {stats['up']}  ·  👎 {stats['down']}\n"
        f"🕒 Last activity: `{stats['last'][:16]}`"
    )
    return text, [[Button.inline("🔄 Refresh", b"m:stats")], back_row()]


def settings_view(learn):
    status = "✅ On" if learn else "⏸ Paused"
    text = f"⚙️ **Settings**\n\n🧠 Learning from chat: **{status}**"
    buttons = [
        [Button.inline("⏸ Pause learning" if learn else "▶️ Resume learning", b"s:learn")],
        [Button.inline("🧹 Clear chat history", b"s:clear", style="danger")],
        back_row(),
    ]
    return text, buttons


def confirm_clear_view():
    text = "🧹 **Clear chat history?**\n\n⚠️ Learned messages are deleted. Notes are kept."
    return text, [
        [
            Button.inline("✅ Yes, clear", b"s:clear!", style="danger"),
            Button.inline("❌ Cancel", b"m:set", style="success"),
        ]
    ]


# --- handlers ----------------------------------------------------------------


def register_handlers(client, cfg, db):
    groups = cfg["allowed_groups"] or None
    pending = {}
    admin_cache = {}
    last_ask = {}
    me = {"username": ""}

    def allowed(chat_id):
        return not cfg["allowed_groups"] or chat_id in cfg["allowed_groups"]

    async def is_admin(chat_id, user_id):
        if user_id in cfg["admin_ids"]:
            return True
        key = (chat_id, user_id)
        cached = admin_cache.get(key)
        if cached and cached[1] > time.monotonic():
            return cached[0]
        perms = await client.get_permissions(chat_id, user_id)
        admin_cache[key] = (perms.is_admin, time.monotonic() + ADMIN_CACHE_TTL)
        return perms.is_admin

    def on_cooldown(chat_id, user_id):
        now = time.monotonic()
        key = (chat_id, user_id)
        if key in last_ask and now - last_ask[key] < cfg["cooldown"]:
            return True
        last_ask[key] = now
        return False

    async def ask(event, question):
        if on_cooldown(event.chat_id, event.sender_id):
            await event.reply("⏳ Slow down a little, try again in a few seconds.")
            return
        placeholder = await event.reply("🤔 Thinking…")
        answer = await generate_answer(cfg, db, event.chat_id, question)
        ask_id = save_ask(db, event.chat_id, event.sender_id, question, answer)
        text, buttons = answer_view(ask_id, question, answer)
        await placeholder.edit(text, buttons=buttons)

    async def save_note(event, text):
        add_doc(db, event.chat_id, event.sender_id, text, "note", event.date.isoformat())
        await event.reply("📝✅ Note saved.", buttons=[[Button.inline("📚 View notes", b"m:notes:0")]])

    @client.on(
        events.NewMessage(chats=groups, func=lambda e: e.is_group and e.raw_text and not e.raw_text.startswith("/"))
    )
    async def on_group_message(event):
        key = (event.chat_id, event.sender_id)
        action = pending.pop(key, None)
        if action and action[1] > time.monotonic():
            if action[0] == "ask":
                await ask(event, event.raw_text)
            else:
                await save_note(event, event.raw_text)
            return
        if learning_enabled(db, event.chat_id):
            add_doc(db, event.chat_id, event.sender_id, event.raw_text, "msg", event.date.isoformat())

    @client.on(events.NewMessage(chats=groups, pattern=r"^/ask(?:@\w+)?(?:\s+(.+))?$"))
    async def on_ask(event):
        question = (event.pattern_match.group(1) or "").strip()
        if not question:
            await event.reply("🔎 Usage: `/ask <question>`, or tap 🔎 Ask in /menu.")
            return
        await ask(event, question)

    @client.on(events.NewMessage(chats=groups, pattern=r"^/addnote(?:@\w+)?(?:\s+(.+))?$"))
    async def on_addnote(event):
        text = (event.pattern_match.group(1) or "").strip()
        if not text:
            await event.reply("📝 Usage: `/addnote <text>`")
            return
        if not await is_admin(event.chat_id, event.sender_id):
            await event.reply("🔒 Only admins can add notes.")
            return
        await save_note(event, text)

    @client.on(events.NewMessage(pattern=r"^/(?:start|menu)(?:@\w+)?(?:\s.*)?$"))
    async def on_menu(event):
        if not event.is_private and not allowed(event.chat_id):
            return
        text, buttons = home_view(me["username"], event.is_private)
        await event.reply(text, buttons=buttons)

    @client.on(events.NewMessage(pattern=r"^/help(?:@\w+)?$"))
    async def on_help(event):
        await event.reply(HELP_TEXT, buttons=[back_row()] if not event.is_private else None)

    @client.on(events.CallbackQuery())
    async def on_callback(event):
        data = event.data.decode()
        chat_id, user_id = event.chat_id, event.sender_id

        async def show(view, toast=None):
            text, buttons = view
            try:
                await event.edit(text, buttons=buttons)
            except MessageNotModifiedError:
                pass
            await event.answer(toast)

        if data == "noop":
            await event.answer()
            return
        if data == "x":
            await event.delete()
            return
        if data == "m:help":
            await show((HELP_TEXT, [back_row()] if not event.is_private else None))
            return
        if event.is_private or not allowed(chat_id):
            await event.answer("🚫 This only works in enabled groups.", alert=True)
            return

        if data == "m:home":
            await show(home_view(me["username"], False))
        elif data in ("m:ask", "m:note"):
            if data == "m:note" and not await is_admin(chat_id, user_id):
                await event.answer("🔒 Only admins can add notes.", alert=True)
                return
            pending[(chat_id, user_id)] = ("ask" if data == "m:ask" else "note", time.monotonic() + PENDING_TTL)
            prompt = "✍️ Send your question now" if data == "m:ask" else "✍️ Send the note text now"
            await event.answer(f"{prompt} (2 min).", alert=True)
        elif data.startswith("m:notes:"):
            page = max(0, int(data.split(":")[2]))
            rows, total = list_notes(db, chat_id, page)
            if not rows and page:
                page = 0
                rows, total = list_notes(db, chat_id, page)
            await show(notes_view(rows, total, page))
        elif data == "m:stats":
            await show(stats_view(chat_stats(db, chat_id)), "📊 Updated")
        elif data == "m:set" or data.startswith("s:"):
            if not await is_admin(chat_id, user_id):
                await event.answer("🔒 Admins only.", alert=True)
                return
            if data == "s:learn":
                learn = toggle_learning(db, chat_id)
                await show(settings_view(learn), "✅ Learning on" if learn else "⏸ Learning paused")
            elif data == "s:clear":
                await show(confirm_clear_view())
            elif data == "s:clear!":
                clear_history(db, chat_id)
                await show(settings_view(learning_enabled(db, chat_id)), "🧹 History cleared")
            else:
                await show(settings_view(learning_enabled(db, chat_id)))
        elif data.startswith("n:del:"):
            if not await is_admin(chat_id, user_id):
                await event.answer("🔒 Admins only.", alert=True)
                return
            _, _, rowid, page = data.split(":")
            delete_note(db, chat_id, int(rowid))
            rows, total = list_notes(db, chat_id, int(page))
            if not rows and int(page):
                page = int(page) - 1
                rows, total = list_notes(db, chat_id, page)
            await show(notes_view(rows, total, int(page)), "🗑 Note deleted")
        elif data.startswith("q:"):
            _, action, ask_id = data.split(":")
            ask_id = int(ask_id)
            row = get_ask(db, ask_id)
            if not row or row[0] != chat_id:
                await event.answer("⌛ This answer has expired.", alert=True)
                return
            _, question, answer, rating = row
            if action == "re":
                if on_cooldown(chat_id, user_id):
                    await event.answer("⏳ Slow down a little.")
                    return
                await event.answer("🔁 Regenerating…")
                await event.edit(f"❓ **{clip(question, 300)}**\n\n🤔 Thinking…")
                answer = await generate_answer(cfg, db, chat_id, question)
                update_ask(db, ask_id, answer=answer, rating=0)
                text, buttons = answer_view(ask_id, question, answer)
                await event.edit(text, buttons=buttons)
            elif action == "src":
                await show(sources_view(ask_id, search(db, chat_id, question, cfg["top_k"])))
            elif action == "ans":
                await show(answer_view(ask_id, question, answer, rating))
            elif action in ("up", "dn"):
                new = 1 if action == "up" else -1
                new = 0 if rating == new else new
                update_ask(db, ask_id, rating=new)
                toast = {1: "👍 Thanks for the feedback!", -1: "👎 Noted, thanks!", 0: "↩️ Feedback removed"}[new]
                await show(answer_view(ask_id, question, answer, new), toast)
        else:
            await event.answer()

    async def load_me():
        me["username"] = (await client.get_me()).username or ""

    return load_me


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    db = open_db(cfg["db_path"])
    client = TelegramClient(cfg["session"], cfg["api_id"], cfg["api_hash"])
    load_me = register_handlers(client, cfg, db)
    client.start(bot_token=cfg["bot_token"])
    client.loop.run_until_complete(load_me())
    log.info("Bot started; allowed groups: %s", sorted(cfg["allowed_groups"]) or "all")
    client.run_until_disconnected()


if __name__ == "__main__":
    main()
