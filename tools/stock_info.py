import asyncio
from typing import Optional

import yfinance as yf
from ddgs import DDGS
from llama_index.core.tools import FunctionTool


def create_stock_info_tool() -> FunctionTool:
    """Create a tool to fetch stock information and news from Yahoo Finance and DuckDuckGo."""

    def get_historical_data(
        ticker: yf.Ticker, start_date: Optional[str], end_date: Optional[str]
    ) -> str:
        if not start_date or not end_date:
            return ""
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            return ""
        output = []
        output.append("\n**Historical Data:**")
        for date, row in hist.iterrows():
            output.append(
                f"- {date}: Close = {row['Close']:.2f}, Volume = {row['Volume']}"
            )
        return "".join(output)

    def get_news_data(sym: str) -> str:
        try:
            output = []
            output.append("\n**Recent News:**")
            with DDGS() as ddgs:
                news_results = list(ddgs.news(f"{sym} stock", max_results=5))
                if not news_results:
                    return "No recent news found."
                for idx, news in enumerate(news_results, 1):
                    title = news.get("title", "No Title")
                    url = news.get("url", "#")
                    source = news.get("source", "Unknown")
                    date_pub = news.get("date", "")
                    output.append(f"{idx}. {title} ({source}, {date_pub})\n   {url}")
        except Exception as e:
            return f"Error fetching news: {e}"
        return "".join(output)

    async def get_stock_info(
        sym: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> str:
        def fetch_data() -> str:
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info

                if not info or (
                    "regularMarketPrice" not in info and "currentPrice" not in info
                ):
                    return f"Could not find data for symbol '{sym}'"

                name = info.get("shortName") or info.get("longName") or sym
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                currency = info.get("currency", "USD")

                output = [
                    f"**Stock Info for:** {name} ({sym.upper()})",
                    f"- **Current Price:** {price} {currency}",
                    f"- **Day Range:** {info.get('dayLow')} - {info.get('dayHigh')}",
                    f"- **52-Week Range:** {info.get('fiftyTwoWeekLow')} - {info.get('fiftyTwoWeekHigh')}",
                    f"- **Market Cap:** {info.get('marketCap')}",
                    f"- **Volume:** {info.get('volume')}",
                    f"- **P/E Ratio (Trailing):** {info.get('trailingPE', 'N/A')}",
                    f"- **Dividend Yield:** {info.get('dividendYield', 'N/A')}",
                ]

                output.append(get_historical_data(ticker, start_date, end_date))
                output.append(get_news_data(sym))

                formatted_output = [line for line in output if "None" not in line]
                return "\n".join(formatted_output)

            except Exception as e:
                return f"Error fetching data for symbol '{sym}': {e}"

        return await asyncio.to_thread(fetch_data)

    return FunctionTool.from_defaults(
        async_fn=get_stock_info,
        name="get_stock_info",
        description=(
            "Fetches current stock information, key metrics, and recent news for a given ticker symbol. "
            "Optionally accepts 'start_date' and 'end_date' (format: YYYY-MM-DD) to retrieve historical "
            "closing prices and volume. Use this for stock price queries, financial health analysis, "
            "or checking recent company developments."
        ),
    )
