import io
import os
import re
import time
import asyncio
import textwrap
import traceback
import subprocess
import discord
from contextlib import redirect_stdout
from discord.ext import commands, tasks
from bs4 import BeautifulSoup

# --- LANGCHAIN IMPORTS ---
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager, index
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. ENV VARS ---
TOKEN = os.getenv("DISCORD_TOKEN")
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3:8b"
EMBEDDING_NAME = "nomic-embed-text"

DOCS_DIR = "/app/docs"
STORAGE_DIR = "/app/storage"
SYNC_SCRIPT = "/app/sync.sh"
PREFIX = "?"

# --- 2. SETUP LANGCHAIN & OLLAMA ---
print(f"Connecting to Ollama at {OLLAMA_URL} with model {MODEL_NAME}...")

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    temperature=0,
    top_k=20,
    streaming=True,
    # keep_alive is handled by your ollama server config, but you can pass kwargs if needed
)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_NAME,
    base_url=OLLAMA_URL,
)

# Set up the Vector Store (Chroma) and the Record Manager (for smart updates)
vectorstore = Chroma(
    collection_name="ocf_docs",
    embedding_function=embeddings,
    persist_directory=STORAGE_DIR
)

namespace = f"chroma/ocf_docs"
record_manager = SQLRecordManager(
    namespace, db_url=f"sqlite:///{STORAGE_DIR}/record_manager_cache.sql"
)
record_manager.create_schema()

def load_and_clean_documents(directory):
    """Walks the directory, cleans HTML, and returns LangChain Documents."""
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".md", ".html", ".txt")):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    if file.endswith(".html"):
                        soup = BeautifulSoup(content, "html.parser")
                        # Destroy all the junk code tags before extracting text
                        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                            tag.extract()
                        text = soup.get_text(separator="\n", strip=True)
                    else:
                        text = content

                    # "source" metadata is required by the Indexing API to track file changes
                    docs.append(Document(page_content=text, metadata={"source": filepath}))
                except Exception as e:
                    print(f"⚠️ Failed to read {filepath}: {e}")
    return docs

def update_index():
    """Pulls the latest files, chunks them, and smartly updates the vector store."""
    print("🚀 Running sync script to fetch documents...")
    subprocess.run(["bash", SYNC_SCRIPT], check=True)

    path_to_read = f"{DOCS_DIR}/docs" if os.path.exists(f"{DOCS_DIR}/docs") else DOCS_DIR
    print(f"Reading files from: {path_to_read}")

    raw_docs = load_and_clean_documents(path_to_read)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(raw_docs)

    print(f"🔄 Smart-updating index. Skipping unchanged files...")

    # --- THE MAGIC HAPPENS HERE ---
    # The LangChain indexing API uses the record manager to hash doc contents.
    # 'incremental' means it will add new, update changed, and delete removed docs.
    index_result = index(
        split_docs,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source"
    )

    print(f"✅ Sync result: {index_result}")

# --- 3. CUSTOM PROMPT & CHAIN SETUP ---
system_prompt = (
    "you are a helpful discord bot for the open computing facility (ocf) at uc berkeley.\n"
    "your name is waddles. you are nice, helpful, and your owner is chamburr. chamburr is cool.\n"
    "context information from the ocf documentation is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "given the context information and any prior knowledge, answer the user's query.\n"
    "if the user is not asking about ocf, then just answer the question as an ai bot normally would.\n"
    "keep your answers concise and format them nicely for discord using markdown."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def create_bot_qa_chain():
    """Creates the LangChain retrieval chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, question_answer_chain)

# --- 4. DISCORD BOT SETUP ---
class OCFBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=PREFIX, intents=intents)
        self.qa_chain = None

    async def setup_hook(self):
        # Build/Load index in a separate thread so we don't block Discord login
        await asyncio.to_thread(update_index)
        self.qa_chain = create_bot_qa_chain()
        self.update_docs_loop.start()

    @tasks.loop(hours=1.0)
    async def update_docs_loop(self):
        if self.update_docs_loop.current_loop == 0:
            return

        print("⏰ Running scheduled hourly docs update...")
        try:
            await asyncio.to_thread(update_index)
            # Chain automatically uses the updated vectorstore under the hood
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
    if not bot.qa_chain:
        await ctx.reply("i'm still warming up my brain, try again in a sec!")
        return

    current_message = await ctx.reply("🤔 Thinking...")
    current_text = ""
    last_edit_time = time.time()

    try:
        async with ctx.typing():
            async for event in bot.qa_chain.astream_events({"input": question}, version="v2"):

                if event["event"] == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content

                    if chunk_content:
                        # 1. PRINT TO CONSOLE: This lets you see if Ollama is actually streaming
                        print(chunk_content, end="", flush=True)

                        current_text += chunk_content

                        if len(current_text) > 1900:
                            await current_message.edit(content=current_text)
                            current_text = ""
                            current_message = await ctx.send("...")
                            last_edit_time = time.time()

                        # 2. FASTER UPDATES: Lowered from 1.5s to 0.75s
                        elif time.time() - last_edit_time > 0.75:
                            # We use try/except here just in case Discord complains about editing too fast
                            try:
                                await current_message.edit(content=current_text + " ▌")
                                last_edit_time = time.time()
                            except discord.errors.HTTPException:
                                pass # Ignore minor rate limits and keep going

            print() # Print a newline in your console when it finishes

            # Final cleanup
            if current_text.strip():
                await current_message.edit(content=current_text)
            elif current_message.content == "🤔 Thinking...":
                await current_message.edit(content="I couldn't generate an answer.")

    except Exception as e:
        traceback.print_exc()
        await ctx.send(f"my circuits fried trying to answer that: {e}")

@ask.error
async def ask_error(ctx, error):
    if isinstance(error, commands.NoPrivateMessage):
        await ctx.author.send("sorry, i can only answer ocf questions within a server!")

@bot.command(name="reload")
@commands.guild_only()
@commands.has_permissions(administrator=True)
async def reload(ctx):
    """Manually triggers a smart-update of the documents."""
    msg = await ctx.reply("🔄 Checking for changed documents and updating the index...")

    try:
        await asyncio.to_thread(update_index)
        await msg.edit(content="✅ Successfully synced and smartly updated the index!")
    except Exception as e:
        await msg.edit(content=f"❌ Failed to sync: {e}")

@bot.command(name="eval", hidden=True)
@commands.guild_only()
@commands.has_role(1477457425089953885)
async def _eval(ctx, *, body: str):
    """Evaluates python code."""
    env = {
        'bot': bot,
        'ctx': ctx,
        'channel': ctx.channel,
        'author': ctx.author,
        'guild': ctx.guild,
        'message': ctx.message,
        '_': bot._last_result if hasattr(bot, '_last_result') else None
    }

    if body.startswith("```") and body.endswith("```"):
        body = "\n".join(body.split("\n")[1:-1])
    else:
        body = body.strip("` \n")

    stdout = io.StringIO()
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
            await ctx.message.add_reaction('\u2705')
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
