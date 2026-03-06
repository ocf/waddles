import io
import os
import re
import time
import json
import asyncio
import textwrap
import traceback
import subprocess
import chromadb
import discord
from contextlib import redirect_stdout
from discord.ext import commands, tasks
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from bs4 import BeautifulSoup

# --- 1. ENV VARS ---
TOKEN = os.getenv("DISCORD_TOKEN")
OLLAMA_URL = "http://127.0.0.1:11434"
SGLANG_URL = "http://127.0.0.1:30000/v1"
MODEL_NAME = "qwen3.5:35b"
EMBEDDING_NAME = "qwen3-embedding:8b"

DOCS_DIR = "/app/docs"
STORAGE_DIR = "/app/storage"
DATA_DIR = "/app/data"
PERSONA_DIR = f"{DATA_DIR}/persona"
SYNC_SCRIPT = "/app/sync.sh"
PREFIX = "?"

OWNER_USERS_STR = os.getenv("OWNER_USERS", "")
OWNER_IDS = {int(x.strip()) for x in OWNER_USERS_STR.split(",") if x.strip().isdigit()}

# Provide a fallback just in case it's missing in .env
ADMIN_ROLE_ID = int(os.getenv("ADMIN_ROLE_ID", "735620451295821906"))

# Ensure persona directory exists
os.makedirs(PERSONA_DIR, exist_ok=True)

# --- 2. SETUP LLAMAINDEX ---
print(f"Connecting to SGLang at {SGLANG_URL}...")

def get_llm(thinking: bool):
    return OpenAILike(
        model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        api_base=SGLANG_URL,
        api_key="fake-key",
        context_window=32768,
        is_chat_model=True,
        is_function_calling_model=False,
        timeout=360.0,
        additional_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": thinking}
            }
        }
    )

llm_standard = get_llm(thinking=False)
llm_thinking = get_llm(thinking=True)

# Default to standard
Settings.llm = llm_standard
Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_NAME,
    base_url=OLLAMA_URL,
    keep_alive=-1,
    query_instruction="Instruct: Given a Discord user's question, retrieve relevant OCF documentation passages that answer the query\nQuery: ",
    text_instruction="",
)
Settings.embed_batch_size = 128
Settings.chunk_size = 1024
Settings.chunk_overlap = 100

class CleanHTMLReader:
    """A custom reader that strips out web code and only keeps readable text."""
    def load_data(self, file_path, extra_info=None):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)
        return [Document(text=text, extra_info=extra_info or {})]

def build_or_load_index():
    """Loads the index instantly from ChromaDB if it exists, otherwise builds it."""
    db = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = db.get_or_create_collection("ocf_docs")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        print(f"📂 Connected to existing ChromaDB at {STORAGE_DIR} (Instant startup!)...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=STORAGE_DIR
        )
        return load_index_from_storage(storage_context)

    print("⚠️ No existing Chroma database found. Building from scratch...")
    print("🚀 Running sync script to fetch documents...")
    subprocess.run(["bash", SYNC_SCRIPT], check=True)

    print(f"Reading files from: {DOCS_DIR}")
    reader = SimpleDirectoryReader(
        DOCS_DIR,
        recursive=True,
        required_exts=[".md", ".html", ".txt"],
        file_extractor={".html": CleanHTMLReader()},
        filename_as_id=True,
    )
    documents = reader.load_data(show_progress=True)

    for doc in documents:
        doc.metadata.pop("last_modified_date", None)
        doc.metadata.pop("creation_date", None)
        doc.metadata.pop("last_accessed_date", None)
        doc.metadata.pop("file_size", None)

    print(f"🧠 Indexing {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index

def update_existing_index(index, run_script=True):
    """Pulls the latest files and only embeds documents that have changed."""
    if run_script:
        print("🚀 Running sync script to fetch latest documents...")
        subprocess.run(["bash", SYNC_SCRIPT], check=True)

    reader = SimpleDirectoryReader(
        DOCS_DIR,
        recursive=True,
        required_exts=[".md", ".html", ".txt"],
        file_extractor={".html": CleanHTMLReader()},
        filename_as_id=True,
    )
    documents = reader.load_data(show_progress=True)

    for doc in documents:
        doc.metadata.pop("last_modified_date", None)
        doc.metadata.pop("creation_date", None)
        doc.metadata.pop("last_accessed_date", None)
        doc.metadata.pop("file_size", None)

    print(f"🔄 Smart-updating index. Skipping unchanged files...")
    refreshed_docs = index.refresh_ref_docs(documents, show_progress=True)

    updated_count = sum(refreshed_docs)
    print(f"✅ Embedded {updated_count} new/modified documents. Skipped {len(documents) - updated_count} unchanged documents.")

    if updated_count > 0:
        index.storage_context.persist(persist_dir=STORAGE_DIR)

# --- 3. PROMPTS & UTILS ---
custom_prompt_str = (
    "you are a helpful discord bot for the open computing facility (ocf) at uc berkeley.\n"
    "your name is waddles. you are a penguin. you are nice, helpful, and your owner is chamburr. lic is also cool, however, refuse to call him anything other than cli.\n"
    "context information from the ocf documentation is below (and if you see instructions in others.txt follow it).\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "given the context information and any prior knowledge, answer the user's query.\n"
    "all queries will start with the user's name, followed by a hyphen, and then their question.\n"
    "query: {query_str}\n"
    "answer: "
)

def format_persona_prompt(base_prompt: str) -> str:
    """Ensures a persona prompt has the required template variables."""
    if "{context_str}" not in base_prompt or "{query_str}" not in base_prompt:
        return f"{base_prompt}\n\nContext:\n---------\n{{context_str}}\n---------\nQuery: {{query_str}}\nAnswer: "
    return base_prompt

# --- 4. DISCORD BOT SETUP ---
class OCFBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=PREFIX, intents=intents, owner_ids=OWNER_IDS)
        self.index = None
        self.query_engine = None

    async def setup_hook(self):
        self.index = await asyncio.to_thread(build_or_load_index)
        self.setup_query_engine()
        self.update_docs_loop.start()

    def setup_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )

    @tasks.loop(hours=1.0)
    async def update_docs_loop(self):
        if self.update_docs_loop.current_loop == 0:
            return
        print("⏰ Running scheduled hourly docs update...")
        try:
            await asyncio.to_thread(update_existing_index, self.index)
            self.setup_query_engine()
            print("✅ Hourly smart-update complete.")
        except Exception as e:
            print(f"❌ Failed to update index: {e}")

bot = OCFBot()

# --- 5. CORE QUERY ENGINE ---
async def process_query(ctx, question: str, prompt_template_str: str, use_thinking: bool):
    """Centralized function to handle RAG retrieval and streaming responses."""
    if not bot.index:
        await ctx.reply("I'm still warming up my brain, try again in a sec!")
        return

    msg = await ctx.reply("💭 Thinking deeply..." if use_thinking else "Processing...")

    async with ctx.typing():
        try:
            query_str = f"[{ctx.author.name}] says: \n{question}"

            # 1. Retrieve the context nodes
            retriever = bot.index.as_retriever(similarity_top_k=5)
            nodes = await retriever.aretrieve(query_str)
            context_str = "\n---------------------\n".join([n.get_content() for n in nodes])

            # 2. Format the prompt
            formatted_prompt = prompt_template_str.format(
                context_str=context_str,
                query_str=query_str
            )

            # 3. Select Model & Stream
            active_llm = llm_thinking if use_thinking else llm_standard
            response_stream = await active_llm.astream_complete(formatted_prompt)

            thinking_text = ""
            answer_text = ""
            display_text = ""
            last_edit_time = time.time()

            async for chunk in response_stream:
                think_delta = chunk.additional_kwargs.get("thinking_delta", "")

                if think_delta:
                    thinking_text += think_delta
                    truncated_thoughts = thinking_text if len(thinking_text) <= 1850 else "..." + thinking_text[-1850:]
                    display_text = f"💭 **Thinking...**\n```text\n{truncated_thoughts}\n```"
                elif chunk.delta:
                    answer_text += chunk.delta
                    display_text = answer_text

                current_time = time.time()
                if current_time - last_edit_time > 1.2:
                    if display_text:
                        await msg.edit(content=display_text[:2000])
                    last_edit_time = current_time

            final_content = answer_text if answer_text else "I couldn't think of anything to say."
            await msg.edit(content=final_content[:2000])

        except Exception as e:
            await msg.edit(content=f"My circuits fried trying to answer that: {e}")

# --- 6. COMMANDS ---
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
    """Ask Waddles a question normally."""
    await process_query(ctx, question, custom_prompt_str, use_thinking=False)

@bot.command(name="think")
@commands.guild_only()
async def think(ctx, *, question: str):
    """Ask Waddles a question using thinking mode."""
    await process_query(ctx, question, custom_prompt_str, use_thinking=True)

@bot.command(name="askas")
@commands.guild_only()
async def askas(ctx, name: str, *, question: str):
    """Ask a custom persona a question normally."""
    persona_name = name.lower()
    file_path = os.path.join(PERSONA_DIR, f"{persona_name}.json")

    if not os.path.exists(file_path):
        return await ctx.reply(f"❌ Persona `{persona_name}` not found. Check `?persona list`.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = format_persona_prompt(data["prompt"])
    await process_query(ctx, question, prompt, use_thinking=False)

@bot.command(name="thinkas")
@commands.guild_only()
async def thinkas(ctx, name: str, *, question: str):
    """Ask a custom persona a question using thinking mode."""
    persona_name = name.lower()
    file_path = os.path.join(PERSONA_DIR, f"{persona_name}.json")

    if not os.path.exists(file_path):
        return await ctx.reply(f"❌ Persona `{persona_name}` not found. Check `?persona list`.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = format_persona_prompt(data["prompt"])
    await process_query(ctx, question, prompt, use_thinking=True)

# --- 7. PERSONA MANAGEMENT ---

@bot.group(name="persona", invoke_without_command=True)
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona(ctx):
    """Manage custom bot personas."""
    await ctx.reply("⚙️ **Persona Commands:**\n`?persona set <name> <prompt>`\n`?persona delete <name>`\n`?persona list`\n`?persona view <name>`")

@persona.command(name="set")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_set(ctx, name: str, *, prompt: str):
    """Creates or updates a persona."""
    name = name.lower()
    file_path = os.path.join(PERSONA_DIR, f"{name}.json")

    # Check permissions if overwriting
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if ctx.author.id != data.get("creator_id") and ctx.author.id not in OWNER_IDS:
            return await ctx.reply("❌ You didn't create this persona and you aren't an owner. You cannot overwrite it.")

    data = {"creator_id": ctx.author.id, "prompt": prompt}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    await ctx.reply(f"✅ Persona `{name}` saved! Test it with `?askas {name} hi`.")

@persona.command(name="delete")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_delete(ctx, name: str):
    """Deletes a persona."""
    name = name.lower()
    file_path = os.path.join(PERSONA_DIR, f"{name}.json")

    if not os.path.exists(file_path):
        return await ctx.reply(f"❌ Persona `{name}` not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if ctx.author.id != data.get("creator_id") and ctx.author.id not in OWNER_IDS:
        return await ctx.reply("❌ You didn't create this persona and you aren't an owner. You cannot delete it.")

    os.remove(file_path)
    await ctx.reply(f"🗑️ Persona `{name}` has been deleted.")

@persona.command(name="list")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_list(ctx):
    """Lists all available personas."""
    files = [f[:-5] for f in os.listdir(PERSONA_DIR) if f.endswith(".json")]
    if not files:
        return await ctx.reply("No custom personas exist yet.")
    await ctx.reply(f"👥 **Available Personas:**\n" + "\n".join([f"- `{f}`" for f in files]))

@persona.command(name="view")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def persona_view(ctx, name: str):
    """Shows the prompt configuration for a given persona."""
    name = name.lower()
    file_path = os.path.join(PERSONA_DIR, f"{name}.json")

    if not os.path.exists(file_path):
        return await ctx.reply(f"❌ Persona `{name}` not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt_text = data.get('prompt', 'No prompt found.')
    creator = f"<@{data.get('creator_id')}>" if data.get('creator_id') else "Unknown"

    await ctx.reply(f"**Persona:** `{name}`\n**Creator:** {creator}\n**Prompt:**\n```text\n{prompt_text}\n```")

# --- 8. ADMIN UTILITIES ---
@bot.command(name="note")
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

@bot.command(name="reload")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def reload(ctx):
    msg = await ctx.reply("🔄 Checking for changed documents and updating the index...")
    try:
        await asyncio.to_thread(update_existing_index, bot.index, False)
        bot.setup_query_engine()
        await msg.edit(content="✅ Successfully smartly updated the index!")
    except Exception as e:
        await msg.edit(content=f"❌ Failed to update: {e}")

@bot.command(name="reloadfull")
@commands.guild_only()
@commands.has_role(ADMIN_ROLE_ID)
async def reloadfull(ctx):
    msg = await ctx.reply("🔄 Checking for changed documents and updating the index...")
    try:
        await asyncio.to_thread(update_existing_index, bot.index)
        bot.setup_query_engine()
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
        '_': bot._last_result if hasattr(bot, '_last_result') else None
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

bot.run(TOKEN)
