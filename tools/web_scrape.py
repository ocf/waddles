import asyncio
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
            # Trafilatura's built-in downloader is robust and handles user-agents/timeouts
            downloaded = trafilatura.fetch_url(url)
            
            if downloaded is None:
                return "Error: Could not fetch the webpage. The site may be blocking scrapers or is currently down."

            # Extract Main Text Content
            text = trafilatura.extract(
                downloaded, 
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )

            # Fallback to BeautifulSoup if trafilatura extraction fails (e.g., non-article pages)
            soup = BeautifulSoup(downloaded, "html.parser")
            if not text:
                # Remove common navigational and boilerplate elements for fallback text
                for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)

            if not text:
                return "The page was successfully fetched but no readable text was found."

            # Extract Links for Traversal (Navigational Discovery)
            links = []
            seen_hrefs = set()
            
            # Re-parse to ensure we have the full document for link discovery
            link_soup = BeautifulSoup(downloaded, "html.parser")
            for a in link_soup.find_all("a", href=True):
                href = a["href"].strip()
                link_text = a.get_text(strip=True)
                
                # Normalize relative URLs to absolute URLs
                absolute_url = urljoin(url, href)
                
                # Quality filtering for links
                if absolute_url.startswith(("http://", "https://")) and absolute_url not in seen_hrefs:
                    # Filter out noise: very short text, empty links, or extremely long text
                    if 2 < len(link_text) < 100:
                        links.append(f"- [{link_text}]({absolute_url})")
                        seen_hrefs.add(absolute_url)
                
                if len(links) >= WEB_SCRAPE_MAX_LINKS:
                    break

            # Final Formatting and Context Management
            if len(text) > WEB_SCRAPE_MAX_LEN:
                text = text[:WEB_SCRAPE_MAX_LEN] + "\n... (Content truncated)"
            
            output = f"--- Content of {url} ---\n\n{text}\n\n--- Links Found ---\n"
            if links:
                output += "\n".join(links)
            else:
                output += "No significant internal or external links discovered on this page."
                
            return output

        except Exception as e:
            return f"Error scraping URL: {e}"

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
