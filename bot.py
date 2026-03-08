"""OCF Discord Bot - Waddles

A RAG-powered Discord bot using LlamaIndex Workflows for function calling.
"""

import io
import os
import time
import asyncio
import textwrap
import traceback
from contextlib import redirect_stdout
from typing import Dict, Any

import discord
from discord.ext import commands, tasks

# Local imports
from config import (
    TOKEN,
    PREFIX,
    OWNER_IDS,
    ADMIN_ROLE_ID,
    DOCS_DIR,
    SGLANG_URL,
)
from index_manager import (
    get_llm,
    setup_settings,
    build_or_load_index,
    update_existing_index,
)
from prompts import (
    is_valid_persona_name,
    get_user_default_persona,
    set_user_default_persona,
    get_persona_prompt,
    format_persona_prompt,
    persona_exists,
    get_persona_data,
    save_persona,
    delete_persona,
    list_personas,
)
from agent_workflows import OCFAgentWorkflow, run_query_workflow
from events import ResponseCompleteEvent


# --- 1. SETUP LLAMAINDEX ---
print(f"Connecting to SGLang at {SGLANG_URL}...")

llm_standard = get_llm(thinking=False)
llm_thinking = get_llm(thinking=True)

# Configure global settings
setup_settings(llm_standard)


# --- 2. DISCORD BOT SETUP ---
class OCFBot(commands.Bot):
    """Discord bot with LlamaIndex Workflow integration."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=PREFIX, intents=intents, owner_ids=OWNER_IDS)
        self.index: Any = None
        self.workflow: Any = None
        # Track active workflow contexts for cancellation: {user_id: {query_id: Context}}
        self.active_workflows: Dict[int, Dict[int, Any]] = {}
        # For eval command
        self._last_result: Any = None

    async def setup_hook(self):
        """Initialize the index and workflow on bot startup."""
        self.index = await asyncio.to_thread(build_or_load_index)
        self._setup_workflow()
        self.update_docs_loop.start()

    def _setup_workflow(self):
        """Create the agent workflow with current index."""
        self.workflow = OCFAgentWorkflow(
            llm_standard=llm_standard,
            llm_thinking=llm_thinking,
            index=self.index,
            timeout=300.0,
        )

    @tasks.loop(hours=1.0)
    async def update_docs_loop(self):
        """Hourly task to update the document index."""
        if self.update_docs_loop.current_loop == 0:
            return
        print("⏰ Running scheduled hourly docs update...")
        try:
            await asyncio.to_thread(update_existing_index, self.index)
            self._setup_workflow()  # Refresh workflow with updated index
            print("✅ Hourly smart-update complete.")
        except Exception as e:
            print(f"❌ Failed to update index: {e}")


bot = OCFBot()


# --- 3. CORE QUERY PROCESSING ---
async def process_query(
    ctx: commands.Context,
    question: str,
    prompt_template_str: str,
    use_thinking: bool
) -> None:
    """Process a query using the workflow system.

    Args:
        ctx: The Discord command context.
        question: The user's question.
        prompt_template_str: The persona prompt template.
        use_thinking: Whether to use thinking mode.
    """
    if not bot.index:
        await ctx.reply("I'm still warming up my brain, try again in a sec!")
        return

    user_id = ctx.author.id
    query_id = ctx.message.id

    # Register this query as active
    if user_id not in bot.active_workflows:
        bot.active_workflows[user_id] = {}

    msg = await ctx.reply("💭 Thinking deeply..." if use_thinking else "Processing...")

    # Create a message callback for status updates
    async def message_callback(text: str) -> None:
        try:
            await msg.edit(content=text[:2000])
        except discord.HTTPException:
            pass  # Ignore rate limits and other HTTP errors

    # Create workflow context for this specific query
    workflow_ctx = None

    async with ctx.typing():
        try:
            # Create a fresh workflow instance for this query
            workflow = OCFAgentWorkflow(
                llm_standard=llm_standard,
                llm_thinking=llm_thinking,
                index=bot.index,
                timeout=300.0,
            )

            # Store reference to workflow for cancellation
            bot.active_workflows[user_id][query_id] = workflow

            # Run the workflow
            result = await workflow.run(
                question=question,
                user_name=ctx.author.name,
                persona_prompt=prompt_template_str,
                use_thinking=use_thinking,
                message_callback=message_callback,
            )

            # Handle the result
            if isinstance(result, ResponseCompleteEvent):
                final_text = result.final_text
            else:
                final_text = str(result) if result else "I couldn't generate a response."

            await msg.edit(content=final_text[:2000])

        except Exception as e:
            await msg.edit(content=f"My circuits fried trying to answer that: {e}")
        finally:
            # Cleanup: Remove this query from active workflows
            if user_id in bot.active_workflows:
                bot.active_workflows[user_id].pop(query_id, None)
                if not bot.active_workflows[user_id]:
                    del bot.active_workflows[user_id]


# --- 4. COMMANDS ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} - ready to answer OCF questions!")

@bot.command(name="ping")
@commands.guild_only()
async def ping(ctx):
    start_time = time.time()
    message = await ctx.reply("Pinging...")
    end_time = time.time()
    api_latency = round(bot.latency * 1000)
    round_trip = round((end_time - start_time) * 1000)
    await message.edit(content=f"🏓 **Pong!**\nWebsocket: `{api_latency}ms`\nRound-trip: `{round_trip}ms`")

@bot.command(name="ask")
@commands.guild_only()
async def ask(ctx, *, question: str):
    """Ask Waddles a question using your default persona."""
    persona_name = get_user_default_persona(ctx.author.id)
    prompt_str = get_persona_prompt(persona_name)
    await process_query(ctx, question, prompt_str, use_thinking=False)

@bot.command(name="think")
@commands.guild_only()
async def think(ctx, *, question: str):
    """Ask Waddles a question using thinking mode and your default persona."""
    persona_name = get_user_default_persona(ctx.author.id)
    prompt_str = get_persona_prompt(persona_name)
    await process_query(ctx, question, prompt_str, use_thinking=True)

@bot.command(name="askas")
@commands.guild_only()
async def askas(ctx, name: str, *, question: str):
    """Ask a custom persona a question normally."""
    name = name.lower()
    if not is_valid_persona_name(name):
        return await ctx.reply("❌ Persona names can only contain lowercase letters and numbers (a-z0-9).")

    if not persona_exists(name):
        return await ctx.reply(f"❌ Persona `{name}` not found. Check `?persona list`.")

    prompt = get_persona_prompt(name)
    await process_query(ctx, question, prompt, use_thinking=False)

@bot.command(name="thinkas")
@commands.guild_only()
async def thinkas(ctx, name: str, *, question: str):
    """Ask a custom persona a question using thinking mode."""
    name = name.lower()
    if not is_valid_persona_name(name):
        return await ctx.reply("❌ Persona names can only contain lowercase letters and numbers (a-z0-9).")

    if not persona_exists(name):
        return await ctx.reply(f"❌ Persona `{name}` not found. Check `?persona list`.")

    prompt = get_persona_prompt(name)
    await process_query(ctx, question, prompt, use_thinking=True)

@bot.command(name="stop")
@commands.guild_only()
async def stop(ctx):
    """Stops all of your ongoing response generations."""
    user_id = ctx.author.id

    if user_id in bot.active_workflows and bot.active_workflows[user_id]:
        # Request cancellation for all active workflows
        for workflow in bot.active_workflows[user_id].values():
            workflow.cancel()  # Set the cancelled flag
        bot.active_workflows[user_id].clear()
        await ctx.reply("🛑 Stopping all your active generations...")
    else:
        await ctx.reply("You don't have any active queries to stop.")

# --- 7. PERSONA MANAGEMENT ---
@bot.group(name="persona", invoke_without_command=True)
@commands.guild_only()
async def persona(ctx):
    """Manage custom bot personas."""
    await ctx.reply(
        "⚙️ **Persona Commands:**\n"
        "`?persona default <name>` (Set your default)\n"
        "`?persona set <name> <prompt>`\n"
        "`?persona delete <name>`\n"
        "`?persona list`\n"
        "`?persona view <name>`"
    )

@persona.command(name="default")
@commands.guild_only()
async def persona_default(ctx, name: str):
    """Sets your default persona for ?ask and ?think."""
    name = name.lower()
    if not is_valid_persona_name(name):
        return await ctx.reply("❌ Persona names can only contain lowercase letters and numbers (a-z0-9).")

    # Validate the persona actually exists
    if not persona_exists(name):
        return await ctx.reply(f"❌ Persona `{name}` not found. Check `?persona list`.")

    set_user_default_persona(ctx.author.id, name)
    await ctx.reply(f"✅ Your default persona has been set to `{name}`!")

@persona.command(name="set")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_set(ctx, name: str, *, prompt: str):
    """Creates or updates a persona."""
    name = name.lower()
    if not is_valid_persona_name(name):
        return await ctx.reply("❌ Persona names can only contain lowercase letters and numbers (a-z0-9).")

    # Check permissions if overwriting
    existing_data = get_persona_data(name)
    if existing_data:
        if ctx.author.id != existing_data.get("creator_id") and ctx.author.id not in OWNER_IDS:
            return await ctx.reply("❌ You didn't create this persona and you aren't an owner. You cannot overwrite it.")

    save_persona(name, ctx.author.id, prompt)
    await ctx.reply(f"✅ Persona `{name}` saved! Test it with `?askas {name} hi`.")

@persona.command(name="delete")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_delete(ctx, name: str):
    """Deletes a persona."""
    name = name.lower()
    if not is_valid_persona_name(name):
        return await ctx.reply("❌ Persona names can only contain lowercase letters and numbers (a-z0-9).")

    existing_data = get_persona_data(name)
    if not existing_data:
        return await ctx.reply(f"❌ Persona `{name}` not found.")

    if ctx.author.id != existing_data.get("creator_id") and ctx.author.id not in OWNER_IDS:
        return await ctx.reply("❌ You didn't create this persona and you aren't an owner. You cannot delete it.")

    delete_persona(name)
    await ctx.reply(f"🗑️ Persona `{name}` has been deleted.")

@persona.command(name="list")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_list(ctx):
    """Lists all available personas."""
    files = list_personas()
    if not files:
        return await ctx.reply("No custom personas exist yet.")
    await ctx.reply(f"👥 **Available Personas:**\n" + "\n".join([f"- `{f}`" for f in files]))

@persona.command(name="view")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_view(ctx, name: str):
    """Shows the prompt configuration for a given persona."""
    name = name.lower()
    if not is_valid_persona_name(name):
        return await ctx.reply("❌ Persona names can only contain lowercase letters and numbers (a-z0-9).")

    data = get_persona_data(name)
    if not data:
        return await ctx.reply(f"❌ Persona `{name}` not found.")

    prompt_text = data.get('prompt', 'No prompt found.')
    creator = f"<@{data.get('creator_id')}>" if data.get('creator_id') else "Unknown"

    await ctx.reply(f"**Persona:** `{name}`\n**Creator:** {creator}\n**Prompt:**\n```text\n{prompt_text}\n```")

# --- 8. ADMIN UTILITIES ---
@bot.command(name="note", hidden=True)
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def note(ctx, *, content: str):
    file_path = os.path.join(DOCS_DIR, "others.txt")
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\nNote from {ctx.author.name}: {content}\n")
        await ctx.reply("✅ Note saved! Waddles will learn this on the next `?reload` or hourly sync.")
    except Exception as e:
        await ctx.reply(f"❌ Failed to save note: {e}")

@bot.command(name="reload", hidden=True)
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def reload(ctx):
    msg = await ctx.reply("🔄 Checking for changed documents and updating the index...")
    try:
        await asyncio.to_thread(update_existing_index, bot.index, False)
        bot._setup_workflow()
        await msg.edit(content="✅ Successfully smartly updated the index!")
    except Exception as e:
        await msg.edit(content=f"❌ Failed to update: {e}")

@bot.command(name="reloadfull", hidden=True)
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def reloadfull(ctx):
    msg = await ctx.reply("🔄 Checking for changed documents and updating the index...")
    try:
        await asyncio.to_thread(update_existing_index, bot.index)
        bot._setup_workflow()
        await msg.edit(content="✅ Successfully synced and smartly updated the index!")
    except Exception as e:
        await msg.edit(content=f"❌ Failed to sync or update: {e}")

@bot.command(name="eval", hidden=True)
@commands.guild_only()
@commands.is_owner()
async def _eval(ctx, *, body: str):
    """Evaluates python code."""
    # Prepare the environment with useful variables
    env = {
        'bot': bot,
        'ctx': ctx,
        'channel': ctx.channel,
        'author': ctx.author,
        'guild': ctx.guild,
        'message': ctx.message,
        '_': bot._last_result
    }

    # Clean up the input (remove code blocks if present)
    if body.startswith("```") and body.endswith("```"):
        body = "\n".join(body.split("\n")[1:-1])
    else:
        body = body.strip("` \n")

    stdout = io.StringIO()

    # Wrap code in an async function to allow 'await'
    to_compile = f'async def func():\n{textwrap.indent(body, "  ")}'

    try:
        exec(to_compile, env)
    except Exception as e:
        return await ctx.send(f'```py\n{e.__class__.__name__}: {e}\n```')

    func = env['func']
    try:
        with redirect_stdout(stdout):
            ret = await func()
    except Exception as e:
        value = stdout.getvalue()
        await ctx.send(f'```py\n{value}{traceback.format_exc()}\n```')
    else:
        value = stdout.getvalue()
        try:
            await ctx.message.add_reaction('\u2705') # Checkmark
        except:
            pass

        if ret is None:
            if value:
                await ctx.send(f'```py\n{value}\n```')
        else:
            bot._last_result = ret
            await ctx.send(f'```py\n{value}{ret}\n```')

@bot.command(name="shell", hidden=True)
@commands.guild_only()
@commands.is_owner()
async def shell(ctx, *, command: str):
    """Executes a shell command and returns the output."""
    # Strip code blocks if the user wrapped the command in markdown
    if command.startswith("```") and command.endswith("```"):
        command = "\n".join(command.split("\n")[1:-1])
    else:
        command = command.strip("` \n")

    msg = await ctx.reply(f"💻 Executing: `{command}`")

    try:
        # Run the command asynchronously
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        # Prepare result strings
        out = stdout.decode().strip()
        err = stderr.decode().strip()

        # Construct the response
        response = ""
        if out:
            response += f"**STDOUT**\n```bash\n{out}\n```"
        if err:
            response += f"**STDERR**\n```bash\n{err}\n```"
        if not out and not err:
            response = "✅ Command executed with no output."

        # Handle Discord's 2000 character limit
        if len(response) > 2000:
            # If too long, upload as a file instead
            with io.BytesIO(response.encode()) as file_ptr:
                await ctx.send(
                    "Output too long, sending as file:",
                    file=discord.File(file_ptr, filename="output.txt")
                )
        else:
            await msg.edit(content=response)

    except Exception as e:
        await msg.edit(content=f"❌ Error: `{e}`")

bot.run(TOKEN or "")
