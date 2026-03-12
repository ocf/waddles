import asyncio
import trafilatura
from bs4 import BeautifulSoup
from llama_index.core.tools import FunctionTool

def create_web_scrape_tool() -> FunctionTool:
    """Create a tool to fetch and scrape text content from a webpage."""

    def sync_scrape_url(url: str) -> str:
        """Synchronously fetch and parse a URL using trafilatura."""
        try:
            # Trafilatura's built-in downloader is robust and handles user-agents/timeouts
            downloaded = trafilatura.fetch_url(url)
            
            if downloaded is None:
                return "Error: Could not fetch the webpage. The site may be blocking scrapers or is currently down."

            # Attempt professional extraction (removes boilerplate, ads, nav)
            text = trafilatura.extract(
                downloaded, 
                include_comments=False,
                include_tables=True,
                no_fallback=False # Let trafilatura use its own internal fallbacks
            )

            # Fallback to BeautifulSoup if trafilatura extraction is too aggressive (returns None)
            # or if the page is not an "article" (e.g. a simple landing page)
            if not text:
                soup = BeautifulSoup(downloaded, "html.parser")
                
                # Remove common navigational and boilerplate elements
                for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)

            if not text:
                return "The page was successfully fetched but no readable text was found."

            # Truncate to avoid blowing up LLM context (10k chars is usually enough for a summary)
            if len(text) > 10000:
                text = text[:10000] + "\n... (Content truncated)"
                
            return text

        except Exception as e:
            return f"Error scraping URL: {e}"

    async def scrape_url(url: str) -> str:
        """Fetch a webpage and return its text content."""
        if not url.startswith(("http://", "https://")):
            return "Error: Invalid URL. Must start with http:// or https://"
            
        try:
            return await asyncio.to_thread(sync_scrape_url, url)
        except Exception as e:
            return f"Unexpected error during scraping: {e}"

    return FunctionTool.from_defaults(
        async_fn=scrape_url,
        name="scrape_url",
        description=(
            "Fetches a webpage from a given URL and returns its visible text content. "
            "Use this when you are given a specific link or need to read the details of an external webpage."
        ),
    )
