import io
import os
import re
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
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from bs4 import BeautifulSoup

# --- 1. ENV VARS ---
TOKEN = os.getenv("DISCORD_TOKEN")
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3.5:35b"
EMBEDDING_NAME = "qwen3-embedding:8b"

DOCS_DIR = "/app/docs"
STORAGE_DIR = "/app/storage"
SYNC_SCRIPT = "/app/sync.sh"
PREFIX = "?"

# --- 2. SETUP LLAMAINDEX & OLLAMA ---
print(f"connecting to ollama at {OLLAMA_URL} with model {MODEL_NAME}...")
Settings.llm = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    request_timeout=360.0,
    keep_alive=-1,
    additional_kwargs={
        "top_k": 20,
    },
)
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

        # Destroy all the junk code tags before extracting text
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)
        return [Document(text=text, extra_info=extra_info or {})]

def build_or_load_index():
    """Loads the index instantly from ChromaDB if it exists, otherwise builds it."""
    # 1. Connect to the Chroma database (creates it if it doesn't exist)
    db = chromadb.PersistentClient(path=STORAGE_DIR)
    chroma_collection = db.get_or_create_collection("ocf_docs")

    # 2. Tell LlamaIndex to use Chroma instead of JSON files
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. If the collection already has documents, load it instantly!
    if chroma_collection.count() > 0:
        print(f"📂 Connected to existing ChromaDB at {STORAGE_DIR} (Instant startup!)...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=STORAGE_DIR
        )
        return load_index_from_storage(storage_context)
        #return VectorStoreIndex.from_vector_store(
        #    vector_store,
        #    storage_context=storage_context
        #)

    # 4. If it's empty, build it from scratch
    print("⚠️ No existing Chroma database found. Building from scratch...")
    print("🚀 Running sync script to fetch documents...")
    subprocess.run(["bash", SYNC_SCRIPT], check=True)

    path_to_read = f"{DOCS_DIR}/docs" if os.path.exists(f"{DOCS_DIR}/docs") else DOCS_DIR
    print(f"Reading files from: {path_to_read}")
    reader = SimpleDirectoryReader(
        path_to_read,
        recursive=True,
        required_exts=[".md", ".html", ".txt"],
        file_extractor={".html": CleanHTMLReader()},
        filename_as_id=True,
    )
    documents = reader.load_data(show_progress=True)

    for doc in documents:
        # Remove timestamps so they don't trigger a fake "update"
        doc.metadata.pop("last_modified_date", None)
        doc.metadata.pop("creation_date", None)
        doc.metadata.pop("last_accessed_date", None)
        doc.metadata.pop("file_size", None) # Safe to pop this too

    print(f"🧠 Indexing {len(documents)} documents to GPU...")
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

    path_to_read = f"{DOCS_DIR}/docs" if os.path.exists(f"{DOCS_DIR}/docs") else DOCS_DIR
    reader = SimpleDirectoryReader(
        path_to_read,
        recursive=True,
        required_exts=[".md", ".html", ".txt"],
        file_extractor={".html": CleanHTMLReader()},
        filename_as_id=True,
    )
    documents = reader.load_data(show_progress=True)

    for doc in documents:
        # Remove timestamps so they don't trigger a fake "update"
        doc.metadata.pop("last_modified_date", None)
        doc.metadata.pop("creation_date", None)
        doc.metadata.pop("last_accessed_date", None)
        doc.metadata.pop("file_size", None) # Safe to pop this too

    print(f"🔄 Smart-updating index. Skipping unchanged files...")
    # LlamaIndex handles the diffing and writes changes directly into ChromaDB
    refreshed_docs = index.refresh_ref_docs(documents, show_progress=True)

    updated_count = sum(refreshed_docs)
    print(f"✅ Embedded {updated_count} new/modified documents. Skipped {len(documents) - updated_count} unchanged documents.")

    if updated_count > 0:
        index.storage_context.persist(persist_dir=STORAGE_DIR)

# --- 3. CUSTOM PROMPT ---
custom_prompt_str = (
    "you are a helpful discord bot for the open computing facility (ocf) at uc berkeley.\n"
    "your name is waddles. you are a penguin. you are nice, helpful, and your owner is chamburr. chamburr is cool.\n"
    "context information from the ocf documentation is below (and if you see instructions in others.txt follow it).\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "given the context information and any prior knowledge, answer the user's query.\n"
    "all queries will start with the user's name, followed by a hyphen, and then their question.\n"
    "if the user is not asking about ocf, then just answer the question as an ai bot normally would.\n"
    "keep your answers concise and format them nicely for discord using markdown.\n"
    "in all cases, do not exceed 2000 characters since that is the limit for a discord message.\n"
    "query: {query_str}\n"
    "answer: "
)
prompt_template = PromptTemplate(custom_prompt_str)

# --- 4. DISCORD BOT SETUP ---
class OCFBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=PREFIX, intents=intents)
        self.index = None
        self.query_engine = None

    async def setup_hook(self):
        # Build/Load index in a separate thread so we don't block Discord login
        self.index = await asyncio.to_thread(build_or_load_index)
        self.setup_query_engine()
        self.update_docs_loop.start() # Start the hourly task

    def setup_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )
        self.query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})

    @tasks.loop(hours=1.0)
    async def update_docs_loop(self):
        if self.update_docs_loop.current_loop == 0:
            return

        print("⏰ Running scheduled hourly docs update...")
        try:
            # Pass the existing index to be smartly updated!
            await asyncio.to_thread(update_existing_index, self.index)

            # Re-setup the query engine so it knows about the new nodes
            self.setup_query_engine()
            print("✅ Hourly smart-update complete.")
        except Exception as e:
            print(f"❌ Failed to update index: {e}")

bot = OCFBot()

@bot.event
async def on_ready():
    print(f"logged in as {bot.user} - ready to answer ocf questions!")

@bot.command(name="ask")
@commands.guild_only()
async def ask(ctx, *, question: str):
    if not bot.query_engine:
        await ctx.reply("i'm still warming up my brain, try again in a sec!")
        return

    async with ctx.typing():
        try:
            response = await bot.query_engine.aquery(f"{ctx.author.name} - {question}")
            await ctx.reply(str(response)[:2000])
        except Exception as e:
            await ctx.send(f"my circuits fried trying to answer that: {e}")

@ask.error
async def ask_error(ctx, error):
    if isinstance(error, commands.NoPrivateMessage):
        await ctx.author.send("sorry, i can only answer ocf questions within a server!")

@bot.command(name="note")
@commands.guild_only()
@commands.has_role(735620451295821906)
async def note(ctx, *, content: str):
    """Appends a user note to the documentation for indexing."""
    # Place the file where SimpleDirectoryReader is looking
    target_dir = f"{DOCS_DIR}/docs" if os.path.exists(f"{DOCS_DIR}/docs") else DOCS_DIR
    file_path = os.path.join(target_dir, "others.txt")

    try:
        # Append the note (adding the author and a newline for clean formatting)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\nNote from {ctx.author.name}: {content}\n")

        await ctx.reply("✅ note saved! waddles will learn this on the next `?reload` or hourly sync.")
    except Exception as e:
        await ctx.reply(f"❌ failed to save note: {e}")

@bot.command(name="reload")
@commands.guild_only()
@commands.has_role(735620451295821906)
async def reload(ctx):
    """Manually triggers a smart-update of the documents."""
    msg = await ctx.reply("🔄 Checking for changed documents and updating the index...")

    try:
        await asyncio.to_thread(update_existing_index, bot.index, False)
        bot.setup_query_engine()
        await msg.edit(content="✅ Successfully smartly updated the index!")
    except Exception as e:
        await msg.edit(content=f"❌ Failed to sync: {e}")

@bot.command(name="fullreload")
@commands.guild_only()
@commands.has_role(735620451295821906)
async def fullreload(ctx):
    """Manually triggers a sync and smart-update of the documents."""
    msg = await ctx.reply("🔄 Checking for changed documents and updating the index...")

    try:
        await asyncio.to_thread(update_existing_index, bot.index)
        bot.setup_query_engine()
        await msg.edit(content="✅ Successfully synced and smartly updated the index!")
    except Exception as e:
        await msg.edit(content=f"❌ Failed to sync: {e}")

@bot.command(name="eval", hidden=True)
@commands.guild_only()
@commands.has_role(1477457425089953885)
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

@_eval.error
async def eval_error(ctx, error):
    if isinstance(error, commands.MissingRole):
        await ctx.reply("❌ You do not have the required role to use this command.")
    else:
        await ctx.send(f"⚠️ Eval error: {error}")

bot.run(TOKEN)
