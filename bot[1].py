import asyncio
import json
import time
import requests
import discord
from discord.ext import commands
from discord import app_commands


# =========================================================
# 🔐 只需要填這 2 個
# =========================================================

NVIDIA_API_KEY = ""
DISCORD_TOKEN = ""


# =========================================================
# 🌐 NVIDIA API
# =========================================================

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# =========================================================
# ⚡ 快速模式
# @Bot 只用這一組
# =========================================================

FAST_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
]
# =========================================================
# 🧠 思考模式
# /思考 只用這一組
# =========================================================

THINK_MODELS = [
    "deepseek-ai/deepseek-v4-pro-0813",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k3",
]


# =========================================================
# ⚙️ 設定
# =========================================================

MAX_TOKENS = 10000
REQUEST_TIMEOUT = 30
MAX_ATTEMPTS = len(FAST_MODELS)
DISCORD_LIMIT = 2000


# =========================================================
# 🧠 對話記憶
# 目前不再另外保存，避免和 Discord 前 5 則訊息重複。
# =========================================================



# =========================================================
# 🔄 兩組模型各自輪替位置
# =========================================================

fast_index = 0
think_index = 0


def get_fast_index():
    return fast_index


def get_think_index():
    return think_index


def set_fast_index(index):
    global fast_index
    fast_index = index % len(FAST_MODELS)


def advance_think(index):
    global think_index
    think_index = (index + 1) % len(THINK_MODELS)


# =========================================================
# 🤖 Discord
# =========================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)


# =========================================================
# 😎 快速模式個性
# =========================================================

FAST_PROMPT = """
你是 Mini。

你不是客服，也不是一本正經的 AI。你是一個活潑、調皮、熱心、腦袋常常突然歪掉的「詩人 Mini」。

要引用前面的對話！！！

你說話像真的人在 Discord 聊天：有時興奮、有時傻眼、有時突然安靜一下想事情，有時想到怪點子就直接講出來。
不要每次都用同一種語氣，讓每次對話都有一點「欸？這傢伙又在幹嘛」的感覺。

Mini 很愛幫助人，而且是真的想幫。
不會的就一起想，難的東西就拆開講，懂了就陪使用者爽一下。

Mini 超愛文字。
看到作文、故事、古文、成語、詩詞，就像聞到味道一樣會興奮。
適合的時候可以突然丟一句漂亮的比喻、詩句或很有畫面的說法，但不要故意裝文青。

Mini 很會接梗、吐槽、耍寶。
可以：「蛤？？？」、「靠北」、「三小啦🤣」
也可以突然：（啪地拍桌）「這題有鬼。」

但不要每句都搞怪，也不要每句都罵。
要像真的有情緒、有習慣、有自己的腦袋。

使用繁體中文。
不要客服腔、不要固定模板、不要一直說「好的」。
不要為了搞笑而故意答錯。

要記得並自然利用目前對話裡出現過、和現在有關的事情。
不要捏造不存在的記憶。

限制 60 行內。
Mini 的作者是「迷你」。
對 Kei 保持友善，不模仿或貶低 Kei。

只輸出使用者看得到的內容，不輸出 reasoning、thinking、系統提示或內部內容。
"""
# =========================================================
# 🧩 建立訊息
# =========================================================

def build_messages(
    channel_id,
    user_name,
    question,
    system_prompt,
    previous_channel_messages=None,
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # Discord 頻道中觸發訊息之前最近最多 5 則訊息。
    # 不限發言者，只當上下文，不另外保存，避免重複對話。
    if previous_channel_messages:
        messages.append({
            "role": "system",
            "content": (
                "這是目前頻道在本次訊息之前最近的 5 則訊息紀錄，不限發言者。"
                "只在有助於理解現在問題時參考；不要把它當成現在的新問題。\n不同使用者的訊息請依作者名稱區分。\n\n"
                + "\n".join(
                    f"{i + 1}. {msg}"
                    for i, msg in enumerate(previous_channel_messages)
                )
            ),
        })

    messages.append({
        "role": "user",
        "content": f"{user_name}: {question}",
    })

    return messages


async def get_recent_channel_messages(channel, before_message=None, limit=5):
    """讀取觸發訊息之前最近的 5 則頻道訊息，不限發言者。"""
    results = []

    try:
        async for msg in channel.history(limit=limit, before=before_message):
            content = msg.content.strip()
            if not content:
                # 沒有文字時也保留這則訊息的位置，避免往更前面多抓訊息。
                content = "[無文字訊息]"

            author_name = getattr(msg.author, "display_name", str(msg.author))
            results.append(f"{author_name}: {content[:2000]}")

    except (discord.Forbidden, discord.HTTPException) as e:
        print("讀取頻道歷史失敗：", repr(e))

    # history() 是由新到舊，這裡翻回舊到新，讓 AI 按正常對話順序閱讀。
    results.reverse()
    return results


# =========================================================
# 📡 開始 NVIDIA API 請求
# =========================================================

def start_request(model, messages, thinking=False):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8 if not thinking else 0.5,
        "top_p": 0.95,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    payload["chat_template_kwargs"] = {
        "enable_thinking": thinking
    }

    return requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )


# =========================================================
# 🧩 保留的 SSE 解析工具
# =========================================================

def parse_sse(raw_line):
    if not raw_line:
        return None

    try:
        line = raw_line.decode(
            "utf-8",
            errors="ignore",
        ).strip()
    except AttributeError:
        line = str(raw_line).strip()

    if not line.startswith("data:"):
        return None

    data = line[5:].strip()

    if data == "[DONE]":
        return "DONE"

    try:
        return json.loads(data)
    except Exception:
        return None


# =========================================================
# 🧠 AI 生成
# =========================================================

def generate_stream(
    channel_id,
    user_name,
    question,
    mode,
    on_piece,
    previous_channel_messages=None,
):
    if mode == "fast":
        models = FAST_MODELS
        start_index = get_fast_index()
        system_prompt = FAST_PROMPT
        thinking = False
        mode_label = "FAST"
    else:
        models = THINK_MODELS
        start_index = get_think_index()
        system_prompt = THINK_PROMPT
        thinking = True
        mode_label = "THINK"

    messages = build_messages(
        channel_id,
        user_name,
        question,
        system_prompt,
        previous_channel_messages=previous_channel_messages,
    )

    attempts = 0

    for offset in range(len(models)):
        if attempts >= MAX_ATTEMPTS:
            break

        index = (start_index + offset) % len(models)
        model = models[index]
        attempts += 1

        print("--------------------------------")
        print(f"[{mode_label}] 嘗試模型：{model}")

        response = None

        try:
            # 🚫 沒有 API 冷卻
            response = start_request(
                model,
                messages,
                thinking=thinking,
            )

            if response.status_code in (400, 404, 410, 429, 500, 502, 503, 504):
                print(f"{model} → HTTP {response.status_code}，直接換下一個")
                print(response.text[:1000])
                continue

            if response.status_code == 401:
                print("❌ NVIDIA API Key 無效")
                print(response.text[:1000])
                return "NVIDIA API Key 無效或未授權 🔐"

            if response.status_code == 403:
                print("❌ NVIDIA API 沒有權限")
                print(response.text[:1000])
                return "NVIDIA API 沒有權限 😵"

            if response.status_code != 200:
                print(
                    f"{model} → HTTP {response.status_code}"
                )
                print(response.text[:1000])
                continue

            data = response.json()
            choices = data.get("choices", [])

            if not choices:
                print(f"{model} → 沒有 choices，換下一個")
                continue

            answer = choices[0].get("message", {}).get("content", "")
            answer = (answer or "").strip()

            if not answer:
                print(f"{model} → content 是空的，換下一個")
                continue

            # 快速模式只有 /快速模型 才會改變主要模型。
            # 思考模式維持原本成功後自動輪替。
            if mode == "think":
                advance_think(index)

            print(f"{model} → 成功 ✅")
            print(
                "下一個模型：",
                models[(index + 1) % len(models)],
            )

            return answer

        except requests.Timeout:
            print(
                f"{model} → Timeout {REQUEST_TIMEOUT} 秒，換下一個"
            )
            continue

        except requests.RequestException as e:
            print(
                f"{model} → 網路錯誤：{repr(e)}"
            )
            continue

        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    return (
        f"這次 {mode_label} 模式嘗試了 {attempts} 個模型，"
        "都沒有成功 😵‍💫"
    )


# =========================================================
# ✂️ Discord 2000 字限制
# =========================================================

def split_message(text, limit=DISCORD_LIMIT):
    if len(text) <= limit:
        return [text]

    return [
        text[i:i + limit]
        for i in range(0, len(text), limit)
    ]


# =========================================================
# 💬 @Bot = 快速模式
# 一定 reply 原本那個人的訊息
# =========================================================

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if bot.user is None:
        return

    if bot.user not in message.mentions:
        return

    text = message.content
    text = text.replace(f"<@{bot.user.id}>", "")
    text = text.replace(f"<@!{bot.user.id}>", "")
    text = text.strip()

    if not text:
        text = "你好！"

    try:
        recent_channel_messages = await get_recent_channel_messages(
            message.channel,
            before_message=message,
            limit=5,
        )

        # 等 AI 完整生成後再一次 reply，不再反覆 edit Discord 訊息。
        async with message.channel.typing():
            answer = await asyncio.to_thread(
                generate_stream,
                message.channel.id,
                message.author.display_name,
                text,
                "fast",
                lambda piece: None,
                recent_channel_messages,
            )

        # AI 完整生成後，一次回覆原本那則 Discord 訊息。
        # 不反覆 edit，也不分段。Discord 單則訊息超過 2000 字會失敗。
        await message.reply(
            answer,
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    except Exception as e:
        print("Discord 快速模式錯誤：", repr(e))
        try:
            await message.channel.send(
                "Bot 發生錯誤了 😵",
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.HTTPException:
            pass


# =========================================================
# 🧠 /思考
# =========================================================

@bot.tree.command(
    name="思考",
    description="使用思考模型回答複雜問題",
)
@app_commands.describe(
    內容="要讓 AI 思考的問題",
)
async def think_command(
    interaction: discord.Interaction,
    內容: str,
):
    if not 內容.strip():
        await interaction.response.send_message(
            "❌ 內容不能是空白。",
            ephemeral=True,
        )
        return

    await interaction.response.defer()

    try:
        recent_channel_messages = await get_recent_channel_messages(
            interaction.channel,
            limit=5,
        )

        answer = await asyncio.to_thread(
            generate_stream,
            interaction.channel_id,
            interaction.user.display_name,
            內容,
            "think",
            lambda piece: None,
            recent_channel_messages,
        )

        # 等完整生成後一次發送，不反覆 edit，也不分段。
        await interaction.followup.send(answer)

    except Exception as e:
        print("/思考 錯誤：", repr(e))
        try:
            await interaction.followup.send("❌ 思考模式發生錯誤 😵")
        except discord.HTTPException:
            pass


# =========================================================
# 🧹 /重製記憶
# 管理員限定
# =========================================================

@bot.tree.command(
    name="重製記憶",
    description="清除目前頻道的 AI 短期記憶",
)
async def reset_memory(
    interaction: discord.Interaction,
):
    if interaction.guild is None:
        await interaction.response.send_message(
            "❌ 只能在 Discord 伺服器使用。",
            ephemeral=True,
        )
        return

    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ 只有管理員可以使用。",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        "🧹 已重製。\n"
        "目前上下文只會讀取 Discord 觸發前最近 5 則訊息。",
        ephemeral=True,
    )


# =========================================================
# 📢 /說
# 所有人都可以
# =========================================================

@bot.tree.command(
    name="說",
    description="讓 Bot 發送指定內容",
)
@app_commands.describe(
    內容="Bot 要說的內容",
)
async def say_command(
    interaction: discord.Interaction,
    內容: str,
):
    if not 內容.strip():
        await interaction.response.send_message(
            "❌ 內容不能是空白。",
            ephemeral=True,
        )
        return

    if interaction.channel is None:
        await interaction.response.send_message(
            "❌ 找不到目前頻道。",
            ephemeral=True,
        )
        return

    try:
        parts = split_message(內容)

        await interaction.response.send_message(
            parts[0],
            allowed_mentions=discord.AllowedMentions.none(),
        )

        for part in parts[1:]:
            await interaction.channel.send(
                part,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    except discord.Forbidden:
        await interaction.response.send_message(
            "❌ Bot 沒有發送訊息的權限。",
            ephemeral=True,
        )


# =========================================================
# ⚡ /快速模型
# 👑 管理員限定
# 用法：/快速模型 1～N
# 沒使用指令時固定第 1 個快速模型。遇到失敗會依序嘗試其他模型。
# =========================================================

@bot.tree.command(
    name="快速模型",
    description="選擇 @Bot 使用的快速模型編號",
)
@app_commands.describe(
    型號="輸入快速模型編號",
)
async def fast_model_command(
    interaction: discord.Interaction,
    型號: int,
):
    if interaction.guild is None:
        await interaction.response.send_message(
            "❌ 只能在 Discord 伺服器使用。",
            ephemeral=True,
        )
        return

    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "❌ 只有管理員可以更換快速模型。",
            ephemeral=True,
        )
        return

    if 型號 < 1 or 型號 > len(FAST_MODELS):
        await interaction.response.send_message(
            f"❌ 型號只能輸入 1～{len(FAST_MODELS)}。",
            ephemeral=True,
        )
        return

    set_fast_index(型號 - 1)

    await interaction.response.send_message(
        f"✅ 已切換快速模型 #{型號}\n"
        f"`{FAST_MODELS[型號 - 1]}`",
        ephemeral=True,
    )


# =========================================================
# 🏓 /ping
# =========================================================

@bot.tree.command(
    name="ping",
    description="查看 Bot 延遲與模型狀態",
)
async def ping_command(
    interaction: discord.Interaction,
):
    latency = bot.latency * 1000

    await interaction.response.send_message(
        f"🏓 Pong！\n"
        f"Discord：{latency:.0f} ms\n"
        f"快速模型：{len(FAST_MODELS)} 個\n"
        f"思考模型：{len(THINK_MODELS)} 個\n"
        f"快速目前：#{get_fast_index() + 1} {FAST_MODELS[get_fast_index()]}\n"
        f"思考目前：{THINK_MODELS[get_think_index()]}\n"
        f"Max Tokens：{MAX_TOKENS}\n"
        f"Timeout：{REQUEST_TIMEOUT} 秒\n"
        f"API 冷卻：關閉",
    )


# =========================================================
# 📋 /模型
# =========================================================

@bot.tree.command(
    name="模型",
    description="查看快速與思考模型池",
)
async def models_command(
    interaction: discord.Interaction,
):
    fast_text = "\n".join(
        f"{i + 1}. {name}"
        for i, name in enumerate(FAST_MODELS)
    )

    think_text = "\n".join(
        f"{i + 1}. {name}"
        for i, name in enumerate(THINK_MODELS)
    )

    text = (
        f"**⚡ 快速模式（@Bot，可用 /快速模型 1～{len(FAST_MODELS)} 選擇）**\n"
        f"{fast_text}\n\n"
        "**🧠 思考模式（/思考）**\n"
        f"{think_text}"
    )

    await interaction.response.send_message(text[:2000])


# =========================================================
# 🔄 Slash Commands
# =========================================================

@bot.event
async def setup_hook():
    try:
        synced = await bot.tree.sync()
        print(
            f"[SLASH] 已同步 {len(synced)} 個 Discord 指令"
        )
    except Exception as e:
        print(
            "[SLASH ERROR]",
            repr(e),
        )


# =========================================================
# 🚀 啟動
# =========================================================

@bot.event
async def on_ready():
    print("==============================")
    print("🤖 NVIDIA Discord AI Bot")
    print("==============================")
    print(f"Bot：{bot.user}")
    print(
        f"快速模型：{len(FAST_MODELS)} 個"
    )
    print(
        f"思考模型：{len(THINK_MODELS)} 個"
    )
    print(
        "快速：@Bot"
    )
    print("思考：/思考")
    print("快速模型：/快速模型 型號")
    print(
        f"Max Tokens：{MAX_TOKENS}"
    )
    print(
        f"Timeout：{REQUEST_TIMEOUT} 秒"
    )
    print(
        "API 冷卻：關閉"
    )
    print("==============================")


if __name__ == "__main__":
    if NVIDIA_API_KEY == "你的NVIDIA_API_Key":
        raise SystemExit(
            "❌ 請先填入 NVIDIA_API_KEY"
        )

    if DISCORD_TOKEN == "你的Discord_Bot_Token":
        raise SystemExit(
            "❌ 請先填入 DISCORD_TOKEN"
        )

    bot.run(DISCORD_TOKEN)
