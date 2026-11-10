# Discord-Gemini-Bot

A Google Gemini Discord chatbot that **talks back out loud**. Mention it (or say its
name) in text chat and it answers in Discord *and* speaks the reply in your voice
channel with a natural neural voice.

## What's new

* **Current models.** The old `gemini-1.5-*` models are retired. This bot defaults to
  `gemini-3.8-flash` and lets you switch models at runtime.
* **Natural voice.** `edge-tts` neural voices replace the robotic gTTS output, and
  **Gemini 3.8 Flash TTS** is available for the most expressive delivery.
* **Options everywhere.** Persona, temperature, reply length, memory, voice, rate,
  pitch and speaking style can all be changed per server with commands, and persist
  in `settings.json`.
* **No more stuttering voice.** Replies go through a playback queue, so two answers
  never fight over the voice client, and the bot leaves when the channel empties.
* **Async.** Gemini calls, TTS and the Goodreads scrape no longer block the event loop.
* **Vision.** Attach an image with your mention and Gemini looks at it.
* **Clean speech.** Markdown, URLs, custom emoji and code blocks are stripped before
  text is spoken.

## Install

1. Install Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html).
   Put `ffmpeg.exe` next to the bot script, or install it on your PATH.

2. Install the dependencies:

   ```
   python -m pip install -r requirements.txt
   ```

3. Create your credentials:
   * Discord: <https://discord.com/developers/applications> → your app → **Bot** →
     enable **Message Content Intent** → copy the token.
   * Gemini: <https://aistudio.google.com/apikey> → create an API key.
   * Optional, for the other free models: OpenRouter
     (<https://openrouter.ai/keys>) or Groq (<https://console.groq.com/keys>).

4. Copy `.env.example` to `.env` and fill in `DISCORD_TOKEN` and `GEMINI_API_KEY`.
   (If you skip this step, the bot creates `.env`, opens it in a text editor and
   stops, so you can paste the keys there and start it again.)

5. Start it with `Run.bat` (Windows) or `python Discord_bot_google_gtts.py`.

Invite the bot with the `bot` and `applications.commands` scopes plus the
*View Channels*, *Send Messages*, *Read Message History*, *Connect* and *Speak*
permissions.

## Commands

Prefix is `!` by default (`PREFIX` in `.env`). Settings commands need *Manage Server*
unless you set `ALLOW_EVERYONE_TO_CONFIGURE=true`.

### Chat
| Command | What it does |
| --- | --- |
| `!ask <text>` | Ask Gemini without mentioning the bot |
| `!reset` | Forget this user's conversation history |
| `!reset settings` | Reset the server's options to the defaults |
| `!help` | Full command list |

### Models
| Command | What it does |
| --- | --- |
| `!model` | Show the model menu |
| `!model 2` / `!model gemini-3.5-flash-lite` | Switch model |
| `!models` | List every text model your API key can use, live from the API |
| `!provider` | Show the free providers; `!provider groq` switches (and resets the model) |

### Voice
| Command | What it does |
| --- | --- |
| `!joinvc` / `!leavevc` | Join or leave your voice channel |
| `!tts` | Show the engine; `!tts edge`, `!tts gemini`, `!tts gtts`, `!tts off` to change it |
| `!voice` | List voices; `!voice ava`, `!voice Kore` to switch |
| `!voices british` | Search the full ~400 voice Edge catalogue |
| `!rate +15%`, `!pitch -10Hz` | Speed and pitch (Edge engine) |
| `!style warm and playful` | Acting style (Gemini engine) |
| `!say hello there` | Speak text without asking the model |
| `!skip` | Skip the line currently playing |

### Options
| Command | What it does |
| --- | --- |
| `!persona <text>` / `!persona reset` | Change the personality prompt |
| `!temp 1.2` | Creativity, 0 to 2 |
| `!tokens 1024` | Reply budget |
| `!memory 12` | Past exchanges remembered per user (0 = off) |
| `!settings` | Show every option for this server |

### Fun
`!joke`, `!quote`, `!goodreads`

## Other free providers

You are not limited to Google. `!provider` switches between Gemini, **OpenRouter**
and **Groq**, each with its own free tier, and `!model` / `!models` follow whichever
provider the server is on.

| Provider | Key | Free models |
| --- | --- | --- |
| `gemini` | `GEMINI_API_KEY` | Gemini 3.8 Flash and friends |
| `openrouter` | `OPENROUTER_API_KEY` | `openrouter/free` (auto-picks a working free model) plus many `:free` models: Gemma 4, Llama 3.3/4, Qwen, DeepSeek, Nemotron |
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, served very fast |

`PROVIDER=auto` (the default) uses the first provider that has a key, in that order.
Each server can switch with `!provider`, which also selects that provider's default
model. Free limits are real, so a busy server may want to rotate providers:
OpenRouter allows 20 requests/minute and 50/day without credit, Groq 30/minute and
1000/day per model. `!models` reads the live roster, so new free models appear
without touching this file.

## Voices

* **edge** (default, free, no API cost): Microsoft neural voices. `en-US-AvaNeural`
  is the warm default; `andrew`, `emma`, `brian`, `sonia`, `ryan`, `natasha` and more
  are aliased, and any Edge `ShortName` works too.
* **gemini**: Gemini 3.8 Flash TTS. The most expressive option and the only one that
  supports `!style`. It is billed to your Gemini API key and needs a paid tier, so
  it is opt-in.
* **gtts**: the original Google Translate voice, kept as a fallback. If Edge TTS
  fails, the bot automatically retries with gTTS instead of going silent.

## Configuration

Everything in `.env.example` is optional except the two credentials. Useful ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Default model |
| `PROVIDER` | `auto` | `auto`, `gemini`, `openrouter` or `groq` |
| `OPENROUTER_API_KEY` | – | Enables the OpenRouter free models |
| `GROQ_API_KEY` | – | Enables Groq's free tier |
| `OPENROUTER_MODEL` | `openrouter/free` | Default OpenRouter model |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Default Groq model |
| `TTS_ENGINE` | `auto` | `auto`, `edge`, `gemini`, `gtts` or `off` |
| `TTS_VOICE` | `en-US-AvaNeural` | Edge voice |
| `TEMPERATURE` | `0.8` | Creativity |
| `MEMORY_TURNS` | `8` | Exchanges remembered per user |
| `SPECIAL_CHANNEL_ID` | `0` | Channel where every message gets a reply |
| `IDLE_LEAVE_SECONDS` | `300` | Leave an empty voice channel after this long |
| `FFMPEG_PATH` | auto | Explicit ffmpeg location |

Per-server changes live in `settings.json` (git-ignored). Delete it or run
`!reset settings` to start over.

## Troubleshooting

* **"ffmpeg was not found"** – install ffmpeg or set `FFMPEG_PATH`; text replies
  still work without it, voice does not.
* **"Voice output failed"** – usually a stale voice name or throttling. Try
  `!tts gtts` to confirm playback works, then `!tts edge` again.
* **Robotic voice, or `!tts` says `gtts`** – edge-tts is not installed. Run `Run.bat`
  again so it installs the full dependency list; the startup lines report exactly
  which TTS libraries loaded.
* **Voice joins then drops, log says `4017`** – Discord requires the DAVE/E2EE
  protocol for voice since March 2026, which needs discord.py 2.7 or newer. `Run.bat`
  installs it automatically; to force it, run
  `python -m pip install --upgrade "discord.py[voice]"`.
* **404 on a model** – model IDs change. Run `!models` and pick a current one.
* **No reply at all** – the bot needs the *Message Content Intent* enabled in the
  developer portal, otherwise it never sees your messages.

## License

See [LICENSE](LICENSE).
