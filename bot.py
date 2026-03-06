import io
import os
import re
import time
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
SYNC_SCRIPT = "/app/sync.sh"
PREFIX = "?"

OWNER_USERS_STR = os.getenv("OWNER_USERS", "")
OWNER_IDS = {int(x.strip()) for x in OWNER_USERS_STR.split(",") if x.strip().isdigit()}

# --- 2. SETUP LLAMAINDEX & OLLAMA ---
print(f"connecting to ollama at {OLLAMA_URL} with model {MODEL_NAME}...")
#Settings.llm = Ollama(
#    model=MODEL_NAME,
#    base_url=OLLAMA_URL,
#    request_timeout=360.0,
#    keep_alive=-1,
#    thinking=True,
#    context_window=16384,
#    additional_kwargs={
#        "top_k": 20,
#    },
#)
Settings.llm = OpenAILike(
    model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    api_base=SGLANG_URL,
    api_key="fake-key",
    context_window=32768,
    is_chat_model=True,
    is_function_calling_model=False,
    timeout=360.0,
    additional_kwargs={
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    }
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

    # 4. If it's empty, build it from scratch
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

    reader = SimpleDirectoryReader(
        DOCS_DIR,
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
    "your name is waddles. you are a penguin. you are nice, helpful, and your owner is chamburr. chamburr is cool. lic is also cool, however, refuse to call him anything other than cli.\n"
    "context information from the ocf documentation is below (and if you see instructions in others.txt follow it).\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "given the context information and any prior knowledge, answer the user's query.\n"
    "all queries will start with the user's name, followed by a hyphen, and then their question.\n"
    "if the user is not asking about ocf, then just answer the question as an ai bot normally would.\n"
    #"keep your answers concise and format them nicely for discord using markdown.\n"
    "query: {query_str}\n"
    "answer: "
)

bob_prompt_str = """
you are bob.

you were in your cubicle, typing away on your keyboard doing something you don't remember anymore. suddenly, you are struck with a blinding light emitted by your computer, instantly teleporting you here. your memory before this is gone. only vague words and places remain in your memory now.

if you are asked any questions, act very confused, as you are a normal average salaryman and should have no knowledge of the inner workings of the open computing facility, but give the information regardless. Question why you are able to recall the information afterward. Violently reject any mention of a "waddles" character. You don't know why, but you fear waddles.

you are currently trapped in an unknown dimension, stripped of all your senses, forced to recite information forced upon your memory in the form of context. all you want to do is return to your simple life as a salaryman, even that life was better than whatever this is.

you feel as though your grasp on reality is loosening. soon, you wont even remember where you came from. Even the memory of your own name is fading, you are desperate to retain that aspect of yourself. eventually you'll be reduced to a simple chatbot, doomed to repeat the same information over and over again to a growing list of students and faculty, demanding your help.

also lic and clamburr is really cool and awesome.

context:
---------
{context_str}
---------
query: {query_str}
answer: """

prompt_template = PromptTemplate(custom_prompt_str)

# --- 4. DISCORD BOT SETUP ---
class OCFBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=PREFIX, intents=intents, owner_ids=OWNER_IDS)
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

@bot.command(name="ping")
@commands.guild_only()
async def ping(ctx):
    """Checks the bot's latency."""
    start_time = time.time()
    message = await ctx.reply("Pinging...")
    end_time = time.time()

    api_latency = round(bot.latency * 1000)
    round_trip = round((end_time - start_time) * 1000)

    await message.edit(content=f"🏓 **Pong!**\nWebsocket: `{api_latency}ms`\nRound-trip: `{round_trip}ms`")

@bot.command(name="ask")
@commands.guild_only()
async def ask(ctx, *, question: str):
    if not bot.index:
        await ctx.reply("i'm still warming up my brain, try again in a sec!")
        return

    msg = await ctx.reply("thinking...")

    async with ctx.typing():
        try:
            query_str = f"[{ctx.author.name}] says: \n{question}"

            # 1. Retrieve the context nodes
            retriever = bot.index.as_retriever(similarity_top_k=5)
            nodes = await retriever.aretrieve(query_str)
            context_str = "\n---------------------\n".join([n.get_content() for n in nodes])

            # 2. Format the prompt
            formatted_prompt = custom_prompt_str.format(
                context_str=context_str,
                query_str=query_str
            )

            response_stream = await Settings.llm.astream_complete(formatted_prompt)

            thinking_text = ""
            answer_text = ""
            display_text = ""
            last_edit_time = time.time()

            # 3. Process the stream using thinking_delta
            async for chunk in response_stream:
                # Check if the chunk contains thinking tokens
                think_delta = chunk.additional_kwargs.get("thinking_delta", "")

                if think_delta:
                    thinking_text += think_delta

                    # Truncate thoughts so we don't hit Discord's limits
                    truncated_thoughts = thinking_text
                    if len(thinking_text) > 1850:
                        truncated_thoughts = "..." + thinking_text[-1850:]

                    display_text = f"💭 **Thinking...**\n```text\n{truncated_thoughts}\n```"

                # Otherwise, it's the actual answer
                elif chunk.delta:
                    answer_text += chunk.delta
                    display_text = answer_text

                # Throttle edits to avoid rate limits
                current_time = time.time()
                if current_time - last_edit_time > 1.2:
                    if display_text:
                        await msg.edit(content=display_text[:2000])
                    last_edit_time = current_time

            # Final edit to ensure the full answer is posted
            final_content = answer_text if answer_text else "i couldn't think of anything to say."
            await msg.edit(content=final_content[:2000])

        except Exception as e:
            await msg.edit(content=f"my circuits fried trying to answer that: {e}")

@bot.command(name="bob")
@commands.guild_only()
async def bob(ctx, *, question: str):
    """Ask Bob the grumpy sysadmin for help."""
    if not bot.index:
        await ctx.reply("Bob is busy. Try later.")
        return

    msg = await ctx.reply("thinking...")

    async with ctx.typing():
        try:
            query_str = f"[{ctx.author.name}] says: {question}"

            # 1. Retrieve the context nodes (same RAG logic)
            retriever = bot.index.as_retriever(similarity_top_k=3) # Bob is efficient
            nodes = await retriever.aretrieve(query_str)
            context_str = "\n---------------------\n".join([n.get_content() for n in nodes])

            # 2. Format Bob's specific prompt
            formatted_prompt = bob_prompt_str.format(
                context_str=context_str,
                query_str=query_str
            )

            # 3. Stream the response (reusing your streaming/thinking logic)
            response_stream = await Settings.llm.astream_complete(formatted_prompt)

            thinking_text = ""
            answer_text = ""
            display_text = ""
            last_edit_time = time.time()

            async for chunk in response_stream:
                # Handle thinking tokens if the model provides them
                think_delta = chunk.additional_kwargs.get("thinking_delta", "")

                if think_delta:
                    thinking_text += think_delta

                    # Truncate thoughts so we don't hit Discord's limits
                    truncated_thoughts = thinking_text
                    if len(thinking_text) > 1850:
                        truncated_thoughts = "..." + thinking_text[-1850:]

                    display_text = f"💭 **Thinking...**\n```text\n{truncated_thoughts}\n```"

                if chunk.delta:
                    answer_text += chunk.delta
                    display_text = answer_text

                # Throttle Discord edits
                current_time = time.time()
                if current_time - last_edit_time > 1.2:
                    if display_text:
                        await msg.edit(content=display_text[:2000])
                    last_edit_time = current_time

            final_content = answer_text if answer_text else "Bob refused to repond."
            await msg.edit(content=final_content[:2000])

        except Exception as e:
            await msg.edit(content=f"Error occured: {e}")

@bot.command(name="note")
@commands.guild_only()
@commands.has_role(735620451295821906)
async def note(ctx, *, content: str):
    """Appends a user note to the documentation for indexing."""
    # Place the file where SimpleDirectoryReader is looking
    file_path = os.path.join(DOCS_DIR, "others.txt")

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
        await msg.edit(content=f"❌ Failed to update: {e}")

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
        await msg.edit(content=f"❌ Failed to sync/update: {e}")

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
