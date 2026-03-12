import asyncio
import re
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from llama_index.core.tools import FunctionTool

from config import WEB_SCRAPE_MAX_LEN, WEB_SCRAPE_MAX_LINKS


def create_web_scrape_tool() -> FunctionTool:
    """Create a tool to fetch and scrape text content from a webpage."""

    def sync_scrape_url(url: str) -> str:
        """Synchronously fetch and parse a URL using trafilatura and BeautifulSoup for links."""
        try:
            # 1. Fetch URL and handle failures immediately
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return f"Error: Could not fetch content from {url}. The site may be blocking scrapers."

            # 2. Extract content in Markdown format for better LLM comprehension
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                output_format="markdown"  # <-- CRITICAL: Keeps headers, lists, and structure
            )

            # 3. Parse with BeautifulSoup ONCE for both fallback and link extraction
            soup = BeautifulSoup(downloaded, "html.parser")

            # 4. Improved Fallback Logic
            if not text:
                # Remove useless tags
                for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript", "iframe", "form"]):
                    tag.decompose()
                text = soup.get_text(separator="\n\n", strip=True)
                # Clean up excessive blank lines
                text = re.sub(r'\n{3,}', '\n\n', text)

            if not text:
                return f"--- Content of {url} ---\n\nNo readable text could be extracted."

            # 5. Extract Links Cleanly
            links = []
            seen_hrefs = set()

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()

                # Ignore anchor links, javascript, and mailto
                if href.startswith(("javascript:", "#", "mailto:", "tel:")):
                    continue

                # Clean up link text (removes internal newlines/tabs)
                raw_text = a.get_text(strip=True)
                link_text = " ".join(raw_text.split())

                absolute_url = urljoin(url, href)

                if absolute_url.startswith(("http://", "https://")) and absolute_url not in seen_hrefs:
                    if 2 < len(link_text) < 100:
                        links.append(f"- [{link_text}]({absolute_url})")
                        seen_hrefs.add(absolute_url)

                if len(links) >= WEB_SCRAPE_MAX_LINKS:
                    break

            # 6. Smarter Truncation (tries not to cut off mid-word)
            if len(text) > WEB_SCRAPE_MAX_LEN:
                text = text[:WEB_SCRAPE_MAX_LEN]
                # Try to snap back to the last newline so we don't cut a sentence in half
                last_newline = text.rfind("\n")
                if last_newline > (WEB_SCRAPE_MAX_LEN * 0.75):
                    text = text[:last_newline]
                text += "\n\n... (Content truncated due to length limits)"

            # 7. Assemble Output
            output = f"--- Content of {url} ---\n\n{text}\n\n--- Links Found ---\n"
            output += "\n".join(links) if links else "No significant internal or external links discovered."

            return output

        except Exception as e:
            return f"Error scraping URL '{url}': {str(e)}"

    async def scrape_url(url: str) -> str:
        """Fetch a webpage and return its cleaned text content and discovered links."""
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
            "Fetches a webpage from a given URL and returns its visible text content plus a list of discovered links. "
            "Use this when you are given a specific link or need to explore a website by following its links."
        ),
    )
