import discord
import requests
import asyncio
import json
from gtts import gTTS
import os
import random
from collections import defaultdict
from bs4 import BeautifulSoup
import re

# --- CONFIG ---
DISCORD_TOKEN = ""
GOOGLE_API_KEY = ""
#MODEL = "gemini-2.5-flash"
SPECIAL_CHANNEL_ID = 467495464678195213
FFMPEG_PATH = os.path.join(os.getcwd(), 'ffmpeg.exe')
AVAILABLE_MODELS = ["gemini-1.5-flash", "gemini-2.5-flash"]

print("Select AI model:")
for idx, m in enumerate(AVAILABLE_MODELS, 1):
    print(f"{idx}. {m}")
choice = input("Enter number (1 or 2): ").strip()

try:
    MODEL = AVAILABLE_MODELS[int(choice)-1]
except (IndexError, ValueError):
    print("Invalid choice, defaulting to gemini-1.5-flash")
    MODEL = "gemini-1.5-flash"

# --- DISCORD CLIENT ---
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)

# --- Conversation memory ---
# Keep last N messages per user (or per channel if you prefer)
conversation_memory = defaultdict(list)
MEMORY_LIMIT = 5  # number of messages to keep per user

# --- Joke & Quote lists ---
JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I told my computer I needed a break, and it said 'No problem — I'll go to sleep.'",
    "Why did the scarecrow win an award? Because he was outstanding in his field!"
]

QUOTES = [
    "The best way to get started is to quit talking and begin doing. – Walt Disney",
    "Don't let yesterday take up too much of today. – Will Rogers",
    "The only limit to our realization of tomorrow is our doubts of today. – Franklin D. Roosevelt"
]

# --- New function to get a random quote from Goodreads ---
def get_goodreads_quote():
    try:
        url = "https://www.goodreads.com/quotes"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            quote_divs = soup.find_all("div", class_="quoteText")
            quotes_list = []
            for q in quote_divs:
                full_text = q.get_text(separator=" ").strip()
                
                # Split into quote text and author
                parts = full_text.split("―")
                if len(parts) >= 2:
                    quote_text = parts[0].strip()
                    author = parts[1].strip()
                    if quote_text and author:
                        quotes_list.append(f'"{quote_text}" — {author}')
            
            if quotes_list:
                return random.choice(quotes_list)
        return "⚠️ Could not fetch a quote from Goodreads."
    except Exception as e:
        print(f"Error fetching Goodreads quote: {e}")
        return "⚠️ Error fetching quote."

# --- Google API endpoint ---
GOOGLE_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GOOGLE_API_KEY}"

def query_google(prompt, context=None):
    # Add conversation context if provided
    if context:
        full_prompt = "\n".join(context) + f"\nUser: {prompt}\nAssistant:"
    else:
        full_prompt = prompt

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 512
        }
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(GOOGLE_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "⚠️ No response from Google API"
    else:
        return f"⚠️ API request failed with code {response.status_code}: {response.text}"

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # --- Commands first ---

    # --- Join voice channel ---
    if message.content.startswith("!joinvc"):
        if message.author.voice:
            channel = message.author.voice.channel
            await channel.connect()
            await message.reply(f"🔊 Joined voice channel: {channel.name}")
        else:
            await message.reply("You need to be in a voice channel for me to join!")
        return

    # --- Leave voice channel ---
    if message.content.startswith("!leavevc"):
        if message.guild.voice_client:
            await message.guild.voice_client.disconnect()
            await message.reply("👋 Left the voice channel.")
        else:
            await message.reply("I'm not in a voice channel.")
        return

    # --- Random pre-defined jokes ---
    if message.content.startswith("!joke"):
        await message.reply(random.choice(JOKES))
        return

    # --- Random pre-defined quotes ---
    if message.content.startswith("!quote"):
        await message.reply(random.choice(QUOTES))
        return
    
    # --- Pull quote from Goodreads.com ---
    if message.content.startswith("!goodreads"):
        quote2 = get_goodreads_quote()
        await message.reply(quote2)
        return

    # --- Dynamic bot name mention detection ---
    bot_names = [client.user.name]
    if message.guild:
        bot_names.append(message.guild.me.display_name)  # server nickname

    is_named = any(re.search(rf'\b{name}\b', message.content, re.IGNORECASE) for name in bot_names)
    is_tagged = client.user.mentioned_in(message)
    is_in_special_channel = message.channel.id == SPECIAL_CHANNEL_ID

    if is_named or is_tagged or is_in_special_channel:
        user_input = message.content

        # Remove bot mentions
        if is_tagged:
            user_input = re.sub(f'<@!?{client.user.id}>', '', user_input).strip()
        
        # Remove bot name mentions
        for name in bot_names:
            user_input = re.sub(rf'\b{re.escape(name)}\b', '', user_input, flags=re.IGNORECASE).strip()

        if not user_input:
            return

        # --- Conversation memory & AI response ---
        user_history = conversation_memory[message.author.id]
        bot_reply = query_google(user_input, context=user_history)

        user_history.append(f"User: {user_input}")
        user_history.append(f"Assistant: {bot_reply}")
        if len(user_history) > MEMORY_LIMIT * 2:
            user_history = user_history[-MEMORY_LIMIT*2:]
        conversation_memory[message.author.id] = user_history

        async with message.channel.typing():
            await asyncio.sleep(1)
            await message.reply(bot_reply)

        # --- Voice output ---
        if message.guild.voice_client:
            try:
                tts = gTTS(text=bot_reply, lang='en')
                audio_file = f"{message.id}.mp3"
                tts.save(audio_file)

                source = discord.FFmpegPCMAudio(source=audio_file, executable=FFMPEG_PATH)
                message.guild.voice_client.play(source)

                while message.guild.voice_client.is_playing():
                    await asyncio.sleep(1)
                
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"Error playing audio: {e}")
                await message.channel.send("Failed to play the audio response.")
            finally:
                if os.path.exists(audio_file):
                    os.remove(audio_file)

client.run(DISCORD_TOKEN)
input("Press Enter to exit...")
