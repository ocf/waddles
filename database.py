"""Index management for ChromaDB and document embedding."""

import os
import subprocess
from typing import List, cast
import chromadb
from bs4 import BeautifulSoup

from llama_index.core.memory import (
    Memory,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
)
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.schema import TextNode
from llama_index.core.readers.base import BaseReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    OLLAMA_URL,
    EMBEDDING_NAME,
    DOCS_DIR,
    STORAGE_DIR,
    MEMORY_DIR,
    SYNC_SCRIPT,
    EMBED_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MEMORY_MAX_FACTS,
    MEMORY_TOKEN_LIMIT,
    MEMORY_CHAT_HISTORY_TOKEN_RATIO,
    MEMORY_DB_URI,
)

# Patch TextNode.hash to ignore date metadata to ensure consistent hashing
# during index.refresh_ref_docs() even if file timestamps change.
_original_hash = TextNode.hash.fget


def _custom_hash(self) -> str:
    # Temporarily remove volatile keys to ensure consistent hashing
    val = self.metadata.pop("last_modified_date", None)
    try:
        return _original_hash(self)
    finally:
        # Restore it for use in retrieval/reranking
        if val is not None:
            self.metadata["last_modified_date"] = val


TextNode.hash = property(_custom_hash)


class CleanHTMLReader(BaseReader):
    """A custom reader that strips out web code and only keeps readable text."""

    def load_data(self, file_path, extra_info=None) -> List[Document]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)
        return [Document(text=text, extra_info=extra_info or {})]


def setup_llm(llm: OpenAILike) -> None:
    """Configure global LlamaIndex settings.

    Args:
        llm: The LLM to use as default.
    """
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_NAME,
        base_url=OLLAMA_URL,
        keep_alive=-
        1,
        query_instruction="Instruct: Given a Discord user's question, retrieve relevant OCF documentation passages that answer the query\nQuery: ",
        text_instruction="",
    )
    Settings.embed_batch_size = EMBED_BATCH_SIZE  # type: ignore[attr-defined]
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def _clean_document_metadata(documents: list[Document]) -> None:
    """Remove date metadata from documents to ensure consistent hashing.

    Args:
        documents: List of documents to clean.
    """
    for doc in documents:
        # Preserve last_modified_date as a fallback for non-Git files
        doc.metadata.pop("creation_date", None)
        doc.metadata.pop("last_accessed_date", None)
        doc.metadata.pop("file_size", None)


def _get_file_metadata(file_path: str) -> dict:
    """Extract true last modified date from Git for OCF docs."""
    # Only applies to docs synced from git
    if "/ocf/" not in file_path:
        return {}
    try:
        rel_path = file_path.split("/ocf/", 1)[1]
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", f"docs/{rel_path}"],
            cwd="/app/cache/ocf_mkdocs",
            capture_output=True,
            text=True,
            check=True
        )
        git_date = result.stdout.strip()
        if git_date:
            # Return just YYYY-MM-DD
            return {"last_modified_date": git_date[:10]}
    except Exception:
        pass


def _load_documents() -> list[Document]:
    """Load documents from the docs directory.

    Returns:
        List of loaded documents.
    """
    reader = SimpleDirectoryReader(
        DOCS_DIR,
        recursive=True,
        required_exts=[".md", ".html", ".txt"],
        file_extractor={".html": CleanHTMLReader()},
        filename_as_id=True,
        file_metadata=_get_file_metadata,
    )
    documents = reader.load_data(show_progress=True)
    _clean_document_metadata(documents)
    return documents


def get_index() -> VectorStoreIndex:
    """Loads the index instantly from ChromaDB if it exists, otherwise builds it.

    Returns:
        The VectorStoreIndex instance.
    """
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
        return cast(VectorStoreIndex, load_index_from_storage(storage_context))

    print("⚠️ No existing Chroma database found. Building from scratch...")
    print("🚀 Running sync script to fetch documents...")
    subprocess.run(["bash", SYNC_SCRIPT], check=True)

    print(f"Reading files from: {DOCS_DIR}")
    documents = _load_documents()

    print(f"🧠 Indexing {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index


def update_index(index: VectorStoreIndex, run_script: bool = True) -> int:
    """Pulls the latest files and only embeds documents that have changed.

    Args:
        index: The existing index to update.
        run_script: Whether to run the sync script first.

    Returns:
        Number of documents that were updated.
    """
    if run_script:
        print("🚀 Running sync script to fetch latest documents...")
        subprocess.run(["bash", SYNC_SCRIPT], check=True)

    documents = _load_documents()

    print(f"🔄 Smart-updating index. Skipping unchanged files...")
    refreshed_docs = index.refresh_ref_docs(documents, show_progress=True)

    updated_count = sum(refreshed_docs)
    print(f"✅ Embedded {updated_count} new/modified documents. Skipped {len(documents) - updated_count} unchanged documents.")

    if updated_count > 0:
        index.storage_context.persist(persist_dir=STORAGE_DIR)

    return updated_count


def get_user_memory(user_id: int, llm: OpenAILike) -> Memory:
    """Gets or creates a persistent, block-based memory for a Discord user.

    Args:
        user_id: The Discord user ID.
        llm: The LLM to use for fact extraction.

    Returns:
        A Memory instance for the user.
    """
    db = chromadb.PersistentClient(path=MEMORY_DIR)
    # Unique collection name per user for long-term vector memory
    collection_name = f"user_memory_{user_id}"
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Memory Blocks
    blocks = [
        # Extracts and maintains persistent facts
        FactExtractionMemoryBlock(llm=llm, max_facts=MEMORY_MAX_FACTS),
        # Stores and retrieves semantic conversation chunks
        VectorMemoryBlock(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )
    ]

    return Memory.from_defaults(
        memory_blocks=blocks,
        async_database_uri=MEMORY_DB_URI,
        session_id=str(user_id),
        token_limit=MEMORY_TOKEN_LIMIT,
        chat_history_token_ratio=MEMORY_CHAT_HISTORY_TOKEN_RATIO
    )
