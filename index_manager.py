"""Index management for ChromaDB and document embedding."""

import subprocess
from typing import List, cast
import chromadb
from bs4 import BeautifulSoup

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.core.readers.base import BaseReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    LLM_REPETITION_PENALTY,
    SGLANG_URL,
    OLLAMA_URL,
    MODEL_NAME,
    EMBEDDING_NAME,
    DOCS_DIR,
    STORAGE_DIR,
    SYNC_SCRIPT,
    LLM_CONTEXT_WINDOW,
    LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_FREQUENCY_PENALTY,
    LLM_PRESENCE_PENALTY,
    LLM_MIN_P,
    EMBED_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class CleanHTMLReader(BaseReader):
    """A custom reader that strips out web code and only keeps readable text."""

    def load_data(self, file_path, extra_info=None) -> List[Document]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)
        return [Document(text=text, extra_info=extra_info or {})]


def get_llm(thinking: bool) -> OpenAILike:
    """Create an LLM instance with optional thinking mode.

    Args:
        thinking: Whether to enable thinking mode.

    Returns:
        Configured OpenAILike LLM instance.
    """
    return OpenAILike(
        model=MODEL_NAME,
        api_base=SGLANG_URL,
        api_key="fake-key",
        context_window=LLM_CONTEXT_WINDOW,
        is_chat_model=True,
        is_function_calling_model=True,
        timeout=LLM_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        additional_kwargs={
            "stop": ["<|im_end|>", "<|im_start|>", "\nUser:"],
            "frequency_penalty": LLM_FREQUENCY_PENALTY,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": thinking},
                "repetition_penalty": LLM_REPETITION_PENALTY,
                "min_p": LLM_MIN_P
            }
        }
    )


def setup_settings(llm: OpenAILike) -> None:
    """Configure global LlamaIndex settings.

    Args:
        llm: The LLM to use as default.
    """
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_NAME,
        base_url=OLLAMA_URL,
        keep_alive=-1,
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
        doc.metadata.pop("last_modified_date", None)
        doc.metadata.pop("creation_date", None)
        doc.metadata.pop("last_accessed_date", None)
        doc.metadata.pop("file_size", None)


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
    )
    documents = reader.load_data(show_progress=True)
    _clean_document_metadata(documents)
    return documents


def build_or_load_index() -> VectorStoreIndex:
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


def update_existing_index(index: VectorStoreIndex, run_script: bool = True) -> int:
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
