import discord
from discord.ext import commands
from openai import OpenAI
import subprocess
import io
import os
import asyncio
import json
from pathlib import Path

# --- STARTUP CONFIGURATION ---
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
OWNER_USERS_STR = os.getenv("OWNER_USERS", "")
OWNER_IDS = {int(x.strip()) for x in OWNER_USERS_STR.split(",") if x.strip().isdigit()}

# Repo & API Config
REPO_PATH = '/app/repo'
REPO_URL = "github.com/ocf/waddles.git"

# Split URLs: Proxy for OpenCode agent, Direct SGLang for OpenAI commit generation
PROXY_API_URL = "http://127.0.0.1:4000/v1"
SGLANG_API_URL = "http://127.0.0.1:30000/v1"
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"

# Git Identity
GIT_USER = "ocfbot"
GIT_EMAIL = "ocfbot@ocf.berkeley.edu"

# Initialize Client directly to SGLang for basic completions (like commit messages)
client = OpenAI(base_url=SGLANG_API_URL, api_key="local")
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='?', intents=intents, owner_ids=OWNER_IDS, help_command=None)

def run_cmd(cmd, cwd=REPO_PATH):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)

def clone_if_not_exists():
    """Clones the repository if the target folder is empty or doesn't exist."""
    path = Path(REPO_PATH)
    if not path.exists() or not any(path.iterdir()):
        print(f"🚚 Repo not found at {REPO_PATH}. Cloning now...")
        auth_url = f"https://{GITHUB_TOKEN}@{REPO_URL}"
        result = subprocess.run(f"git clone {auth_url} {REPO_PATH}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Clone successful.")
        else:
            print(f"❌ Clone failed: {result.stderr}")
    else:
        print(f"📂 Repo already exists at {REPO_PATH}. Skipping clone.")

def configure_opencode_json():
    """Writes the OpenCode configuration file."""
    opencode_dir = Path.home() / ".config/opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)
    config_path = opencode_dir / "opencode.json"

    # Strict OpenCode specific JSON configuration
    config_content = {
      "$schema": "https://opencode.ai/config.json",
      "model": "local-qwen/qwen",
      "provider": {
        "local-qwen": {
          "npm": "@ai-sdk/openai-compatible",
          "options": {
            "baseURL": PROXY_API_URL,
            "apiKey": "not-needed"
          },
          "models": {
            "qwen": {
              "id": MODEL_NAME,
              "tool_call": True,
              "limit": {
                "context": 32768,
                "output": 4096
              }
            }
          }
        }
      }
    }

    with open(config_path, "w") as f:
        json.dump(config_content, f, indent=2)
    print(f"⚙️ OpenCode config updated at {config_path}")

def setup_environment():
    """Configures Git identity and auth for the existing repo."""
    print("🔧 Setting up Git identity...")
    run_cmd(f'git config user.name "{GIT_USER}"')
    run_cmd(f'git config user.email "{GIT_EMAIL}"')

    auth_url = f"https://{GITHUB_TOKEN}@{REPO_URL}"
    run_cmd(f"git remote set-url origin {auth_url}")

@bot.event
async def on_ready():
    # 1. Ensure repo exists
    clone_if_not_exists()
    # 2. Configure OpenCode
    configure_opencode_json()
    # 3. Configure Git within that repo
    setup_environment()
    print(f'🚀 {bot.user} is active. Prefix: ?')

# --- BOT COMMANDS ---

@bot.command(name="explain")
@commands.guild_only()
@commands.is_owner()
async def explain(ctx, *, question: str):
    async with ctx.typing():
        # Instruct OpenCode explicitly not to edit files for an explain command
        safe_prompt = f"Explain the following. Do not modify or create any files: {question}"
        res = run_cmd(f'opencode run "{safe_prompt}"')

    content = res.stdout or "OpenCode found no answer."

    if len(content) > 2000:
        with io.BytesIO(content.encode()) as buf:
            await ctx.reply("📄 Explanation:", file=discord.File(fp=buf, filename="explanation.txt"))
    else:
        await ctx.reply(content)

@bot.command(name="change")
@commands.guild_only()
@commands.is_owner()
async def change(ctx, *, prompt: str):
    await ctx.send("🔄 **Syncing and planning...**")

    async with ctx.typing():
        run_cmd("git pull --rebase")

        # Run OpenCode modification
        run_cmd(f'opencode run "{prompt}"')

    diff = run_cmd("git diff").stdout
    if not diff.strip():
        return await ctx.send("⚠️ No changes detected.")

    with io.BytesIO(diff.encode()) as buf:
        review_msg = await ctx.send(
            "📝 **Review changes.** React ✅ to Push or ❌ to Discard.",
            file=discord.File(fp=buf, filename="review.diff")
        )

    await review_msg.add_reaction("✅")
    await review_msg.add_reaction("❌")

    def check(reaction, user):
        return user == ctx.author and str(reaction.emoji) in ["✅", "❌"] and reaction.message.id == review_msg.id

    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=180.0, check=check)

        if str(reaction.emoji) == "✅":
            await ctx.send("🚀 **Pushing...**")
            msg_res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Write a 1-line commit message for:\n{diff[:2000]}"}]
            )
            commit_msg = msg_res.choices[0].message.content.strip().strip('"')

            run_cmd("git add .")
            run_cmd(f'git commit -m "{commit_msg}"')
            push_res = run_cmd("git push")

            if push_res.returncode == 0:
                await ctx.send(f"✅ **Pushed!**\n`{commit_msg}`")
            else:
                await ctx.send(f"❌ **Push failed:**\n```{push_res.stderr}```")
        else:
            run_cmd("git reset --hard HEAD")
            await ctx.send("🗑️ **Discarded.**")

    except asyncio.TimeoutError:
        run_cmd("git reset --hard HEAD")
        await ctx.send("⏰ **Timed out.** Resetting.")

@bot.command(name="revert")
@commands.guild_only()
@commands.is_owner()
async def revert(ctx):
    await ctx.send("⏪ **Reverting...**")
    run_cmd("git pull --rebase")
    if run_cmd("git revert HEAD --no-edit").returncode == 0:
        run_cmd("git push")
        await ctx.send("✅ **Revert successful.**")
    else:
        await ctx.send("❌ **Revert failed.**")

bot.run(DISCORD_TOKEN)
