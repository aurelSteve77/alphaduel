from pydantic import BaseModel, Field
from datetime import datetime



class TickerData(BaseModel):

    ticker: str = Field(..., description="The ticker symbol of the asset")
    price: float = Field(..., description="The price of the asset")
    volume: int = Field(..., description="The volume of the asset")
    date: datetime = Field(..., description="The date of the asset")


class NewsItem(BaseModel):

    ticker: str = Field(..., description="The ticker the text item refers to")
    published_at: datetime = Field(..., description="Publication timestamp (UTC), point-in-time")
    source: str = Field(..., description="Origin of the text, e.g. finnhub, sec-edgar")
    category: str = Field("", description="Item type, e.g. news, 8-K, 10-Q")
    headline: str = Field("", description="Short title / headline of the item")
    summary: str = Field("", description="Body or summary text of the item")
    url: str = Field("", description="Canonical URL of the item")