import discord
from discord.ext import commands
from openai import AsyncOpenAI
import asyncio
import io
import os
import json
import types
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

# Initialize Async Client directly to SGLang for basic completions
client = AsyncOpenAI(base_url=SGLANG_API_URL, api_key="local")

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents, owner_ids=OWNER_IDS, help_command=None)

# --- ASYNC SHELL HELPER ---
async def run_cmd(cmd, cwd=REPO_PATH):
    """Runs shell commands asynchronously without blocking the Discord event loop."""
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd
    )
    stdout, stderr = await process.communicate()
    
    # Return a namespace mimicking subprocess.CompletedProcess
    return types.SimpleNamespace(
        stdout=stdout.decode('utf-8', errors='replace'),
        stderr=stderr.decode('utf-8', errors='replace'),
        returncode=process.returncode
    )

# --- SETUP FUNCTIONS ---
async def clone_if_not_exists():
    """Clones the repository if the target folder is empty or doesn't exist."""
    path = Path(REPO_PATH)
    if not path.exists() or not any(path.iterdir()):
        print(f"🚚 Repo not found at {REPO_PATH}. Cloning now...")
        auth_url = f"https://{GITHUB_TOKEN}@{REPO_URL}"
        
        # Temporarily run clone from parent directory
        process = await asyncio.create_subprocess_shell(
            f"git clone {auth_url} {REPO_PATH}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await process.communicate()
        
        if process.returncode == 0:
            print("✅ Clone successful.")
        else:
            print(f"❌ Clone failed: {stderr.decode()}")
    else:
        print(f"📂 Repo already exists at {REPO_PATH}. Skipping clone.")

def configure_opencode_json():
    """Writes the OpenCode configuration file (File I/O is fast enough to remain sync here)."""
    opencode_dir = Path.home() / ".config/opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)
    config_path = opencode_dir / "opencode.json"

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

async def setup_environment():
    """Configures Git identity and auth for the existing repo."""
    print("🔧 Setting up Git identity...")
    await run_cmd(f'git config user.name "{GIT_USER}"')
    await run_cmd(f'git config user.email "{GIT_EMAIL}"')

    auth_url = f"https://{GITHUB_TOKEN}@{REPO_URL}"
    await run_cmd(f"git remote set-url origin {auth_url}")

@bot.event
async def on_ready():
    await clone_if_not_exists()
    configure_opencode_json()
    await setup_environment()
    print(f'🚀 {bot.user} is active. Prefix: !')

# --- BOT COMMANDS ---

@bot.command(name="ask")
@commands.guild_only()
@commands.is_owner()
async def ask(ctx, *, question: str):
    async with ctx.typing():
        safe_prompt = f"Explain the following. Do not modify or create any files: {question}"
        res = await run_cmd(f'opencode run "{safe_prompt}"')

    content = res.stdout or "OpenCode found no answer."

    if len(content) > 2000:
        with io.BytesIO(content.encode()) as buf:
            await ctx.reply("📄 Explanation:", file=discord.File(fp=buf, filename="explanation.txt"))
    else:
        await ctx.reply(content)

@bot.command(name="update")
@commands.guild_only()
@commands.is_owner()
async def update(ctx, *, prompt: str):
    await ctx.send("🔄 **Syncing and planning...**")

    async with ctx.typing():
        await run_cmd("git pull --rebase")
        await run_cmd(f'opencode run "{prompt}"')

    diff_res = await run_cmd("git diff")
    diff = diff_res.stdout
    
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
            
            # Awaiting the AsyncOpenAI completion
            msg_res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Write a 1-line commit message for:\n{diff[:2000]}"}]
            )
            commit_msg = msg_res.choices[0].message.content.strip().strip('"')

            await run_cmd("git add .")
            await run_cmd(f'git commit -m "{commit_msg}"')
            push_res = await run_cmd("git push")

            if push_res.returncode == 0:
                await ctx.send(f"✅ **Pushed!**\n`{commit_msg}`")
            else:
                await ctx.send(f"❌ **Push failed:**\n```{push_res.stderr}```")
        else:
            await run_cmd("git reset --hard HEAD")
            await ctx.send("🗑️ **Discarded.**")

    except asyncio.TimeoutError:
        await run_cmd("git reset --hard HEAD")
        await ctx.send("⏰ **Timed out.** Resetting.")

@bot.command(name="revert")
@commands.guild_only()
@commands.is_owner()
async def revert(ctx):
    await run_cmd("git pull --rebase")
    
    last_commit_res = await run_cmd("git log -1 --oneline")
    last_commit = last_commit_res.stdout

    confirm_msg = await ctx.send(f"⚠️ **Are you sure you want to revert the last commit?**\n`{last_commit}`\n\nReact ✅ to Revert or ❌ to Cancel.")
    await confirm_msg.add_reaction("✅")
    await confirm_msg.add_reaction("❌")

    def check(reaction, user):
        return user == ctx.author and str(reaction.emoji) in ["✅", "❌"] and reaction.message.id == confirm_msg.id

    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=60.0, check=check)

        if str(reaction.emoji) == "✅":
            await ctx.send("⏪ **Reverting...**")
            
            revert_res = await run_cmd("git revert HEAD --no-edit")
            if revert_res.returncode == 0:
                push_res = await run_cmd("git push")
                if push_res.returncode == 0:
                    await ctx.send("✅ **Revert successful and pushed.**")
                else:
                    await ctx.send(f"❌ **Revert committed locally, but push failed:**\n```{push_res.stderr}```")
            else:
                await ctx.send("❌ **Revert failed (likely a conflict).**")
        else:
            await ctx.send("👍 **Revert cancelled.**")

    except asyncio.TimeoutError:
        await ctx.send("⏰ **Revert timed out.** No action taken.")

bot.run(DISCORD_TOKEN)
