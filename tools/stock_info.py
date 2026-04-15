import asyncio

import yfinance as yf
from llama_index.core.tools import FunctionTool


def create_stock_info_tool() -> FunctionTool:
    """Create a tool to fetch stock information from Yahoo Finance."""

    async def get_stock_info(sym: str) -> str:
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

                formatted_output = [line for line in output if "None" not in line]
                return "\n".join(formatted_output)

            except Exception as e:
                return f"Error fetching data for symbol '{sym}': {e}"

        return await asyncio.to_thread(fetch_data)

    return FunctionTool.from_defaults(
        async_fn=get_stock_info,
        name="get_stock_info",
        description=(
            "Fetches current stock information and key metrics for a given ticker symbol "
            "from Yahoo Finance. Use this when you need to answer questions about a company's "
            "stock price, market cap, P/E ratio, or other financial data."
        ),
    )
