from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from enum import Enum
from rss_core import fetch_tribune_news, fetch_news_by_category
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import asyncio
from time import time

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Initialize LLM
model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=groq_api)

# Summarization prompt - Three lines
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following news content in three concise lines, capturing the main points clearly. Each line should be a key aspect or development. Avoid HTML tags or boilerplate text. Return only the summary text, with lines separated by newlines.\n\n{content}"
)
summarizer = summary_prompt | model | StrOutputParser()

# In-memory caches
# raw_cache: key -> (raw_items, timestamp)
# details_cache: key -> (details_list, timestamp)
CACHE_TTL = 1800  # 30 minutes in seconds
raw_cache = {}
details_cache = {}

def get_cache_key(region: str, query: str, days_back: int) -> str:
    return f"{region}_{query}_{days_back}"

def is_cache_valid(timestamp: float) -> bool:
    return time() - timestamp < CACHE_TTL

# FastAPI app
app = FastAPI(title="Tribune News API", description="Structured APIs for region-specific news from Express Tribune with LLM summaries")

# Pydantic models
class Top3TitlesResponse(BaseModel):
    titles: List[str] = Field(..., description="Top 3 news titles for the region, sorted by recency")
    region: str = Field(..., description="The queried region")
    total_available: int = Field(..., description="Total news items available for the region")

class NewsDetail(BaseModel):
    title: str = Field(..., description="News title")
    link: str = Field(..., description="Full article link")
    brief_summary: str = Field(..., description="LLM-generated summary from full_content (three lines)")
    img: Optional[str] = Field(None, description="Link to the news image")
    published: Optional[str] = Field(None, description="Publication date")

class CategoryNewsItem(BaseModel):
    title: str = Field(..., description="News title")
    link: str = Field(..., description="Full article link")
    img: Optional[str] = Field(None, description="Link to the news image")
    full_content: str = Field(..., description="Full content of the news article")
    published: Optional[str] = Field(None, description="Publication date")

class CategoryNewsResponse(BaseModel):
    news_items: List[CategoryNewsItem] = Field(..., description="List of news items for the category, sorted by recency")
    category: str = Field(..., description="The queried category")
    total_available: int = Field(..., description="Total news items available for the category")
class Top10DetailsResponse(BaseModel):
    details: List[NewsDetail] = Field(..., description="Top 10 news details for the region, sorted by recency")
    region: str = Field(..., description="The queried region")
    total_available: int = Field(..., description="Total news items available for the region")

class Region(str, Enum):
    Pakistan = "Pakistan"
    Punjab = "Punjab"
    Sindh = "Sindh"
    Balochistan = "Balochistan"
    Khyber_Pakhtunkhwa = "Khyber Pakhtunkhwa"
    Jammu_Kashmir = "Jammu & Kashmir"
    Gilgit_Baltistan = "Gilgit-Baltistan"

class Category(str, Enum):
    Politics = "Politics"
    Technology = "Technology"
    Sports = "Sports"
    Movies = "Movies"
    Music = "Music"
    Health = "Health"
    Business = "Business"
    World = "World"


class Location(BaseModel):
    region: Region = Field(..., description="Valid regions in Pakistan")

class NewsCategory(BaseModel):
    category: Category = Field(..., description="Valid news categories")

@app.get("/")
async def home():
    return {"message": "Welcome to AI News Simulator"}

async def fetch_top_items(region: str, query: str, days_back: int, max_items: int):
    """Helper to fetch news items asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fetch_tribune_news(region=region, query=query, days_back=days_back, max_items=max_items))

async def generate_top_10_details(region: str, query: str, days_back: int, key: str):
    """Generate top 10 news details with summaries for a region."""
    if key in details_cache:
        return  # Already done

    if key not in raw_cache:
        # Raw not available, fetch it
        raw_items = await fetch_top_items(region, query, days_back, 10)
        raw_cache[key] = (raw_items, time())
    else:
        cached = raw_cache.get(key)
        if not is_cache_valid(cached[1]):
            # Expired, refetch
            raw_items = await fetch_top_items(region, query, days_back, 10)
            raw_cache[key] = (raw_items, time())
        else:
            raw_items = cached[0]

    if not raw_items:
        return

    # Generate summaries in parallel
    summary_tasks = []
    for item in raw_items:
        content = item['full_content'] or item['summary']
        summary_tasks.append(summarizer.ainvoke({"content": content[:2000]}))

    try:
        summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)
    except Exception as e:
        print(f"Error in parallel summaries: {e}")
        return

    details = []
    for i, item in enumerate(raw_items):
        if isinstance(summaries[i], Exception):
            summary = (item['full_content'] or item['summary'])[:300]
        else:
            summary = summaries[i]
        brief_summary = str(summary)[:300] + "..." if len(str(summary)) > 300 else str(summary)
        details.append(
            NewsDetail(
                title=item['title'],
                link=item['link'],
                brief_summary=brief_summary,
                img = item['img'],
                published=item.get('published')
            )
        )

    details_cache[key] = (details, time())

@app.get("/top_3_titles", response_model=Top3TitlesResponse)
async def top_3_titles(
    region: Region = Query(..., description="Region to fetch news for"),
    query: str = Query("", description="Optional keywords to filter (e.g., 'flood')"),
    days_back: int = Query(7, ge=1, le=30, description="Days to look back")
):
    """
    Fetch top 3 news titles for a region from live RSS, sorted by recency.
    Caches raw top 10 and starts background task for LLM summaries.
    """
    try:
        key = get_cache_key(region, query, days_back)
        
        # Check raw cache first
        fetch_raw = True
        total_available = 10
        if key in raw_cache:
            cached_items, timestamp = raw_cache[key]
            if is_cache_valid(timestamp):
                news_items = cached_items[:3]
                total_available = len(cached_items)
                fetch_raw = False
            else:
                del raw_cache[key]
                if key in details_cache:
                    del details_cache[key]

        if fetch_raw:
            # Fetch raw top 10
            full_items = await fetch_top_items(region, query, days_back, 10)
            if not full_items:
                raise HTTPException(status_code=404, detail=f"No recent news found for region '{region}' with query '{query}'")
            raw_cache[key] = (full_items, time())
            news_items = full_items[:3]
            total_available = len(full_items)

        # Start background task for details if not already cached
        if key not in details_cache:
            asyncio.create_task(generate_top_10_details(region, query, days_back, key))

        titles = [item['title'] for item in news_items]
        
        return Top3TitlesResponse(
            titles=titles,
            region=region,
            total_available=total_available
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching titles: {str(e)}")

@app.get("/see_more_details", response_model=Top10DetailsResponse)
async def see_more_details(
    region: Region = Query(..., description="Region to fetch full details for"),
    query: str = Query("", description="Optional keywords to filter (e.g., 'flood')"),
    days_back: int = Query(7, ge=1, le=30, description="Days to look back")
):
    """
    Retrieve top 10 news details from cache (with LLM summaries). Falls back to generating if raw cached but details not ready.
    """
    try:
        key = get_cache_key(region, query, days_back)
        if key in details_cache:
            details_list, timestamp = details_cache[key]
            if is_cache_valid(timestamp):
                return Top10DetailsResponse(
                    details=details_list,
                    region=region,
                    total_available=len(details_list)
                )

        # Check raw cache
        if key in raw_cache:
            cached_items, timestamp = raw_cache[key]
            if is_cache_valid(timestamp):
                # Generate details on the fly
                await generate_top_10_details(region, query, days_back, key)
                # After generation, it should be in details_cache
                if key in details_cache:
                    details_list, _ = details_cache[key]
                    return Top10DetailsResponse(
                        details=details_list,
                        region=region,
                        total_available=len(details_list)
                    )

        raise HTTPException(status_code=404, detail=f"No cached data found for region '{region}'. Call /top_3_titles first.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching details: {str(e)}")

@app.get("/category_news", response_model=CategoryNewsResponse)
async def category_news(
    category: Category = Query(..., description="Category to fetch news for"),
    query: str = Query("", description="Optional keywords to filter (e.g., 'cricket')"),
    days_back: int = Query(7, ge=1, le=30, description="Days to look back"),
    max_items: int = Query(10, ge=1, le=50, description="Maximum number of news items to return")
):
    """
    Fetch news items for a category from live RSS, sorted by recency.
    Returns raw data (title, link, img, full_content, published) without LLM summaries.
    """
    try:
        
        news_items = await asyncio.to_thread(
            fetch_news_by_category,
            category=category, query=query, days_back=days_back, max_items=max_items
        )
        if not news_items:
            raise HTTPException(status_code=404, detail=f"No recent news found for category '{category}' with query '{query}'")

        # Prepare response
        response_items = [
            CategoryNewsItem(
                title=item['title'],
                link=item['link'],
                img=item['img'],
                full_content=item['full_content'] or item['summary'],
                published=item.get('published')
            ) for item in news_items
        ]

        return CategoryNewsResponse(
            news_items=response_items,
            category=category,
            total_available=len(news_items)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching category news: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)