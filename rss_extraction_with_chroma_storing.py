import chromadb
import feedparser
import hashlib
import time
from typing import List, Dict
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from dateutil.parser import parse as parse_date
import pytz

# Chroma setup (persistent)
dbpath = r"Chromadb"
client = chromadb.PersistentClient(path=dbpath)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, local embeddings

def fetch_tribune_news(days_back: int = 7, max_items: int = 100) -> List[Dict]:
    """
    Fetch Express Tribune news from the past `days_back` days.
    Returns unique items with full content and metadata, up to `max_items`.
    """
    rss_url = "https://tribune.com.pk/feed/home"
    all_headlines = []
    seen_hashes = set()
    # Use Pakistan timezone (+05:00) for offset-aware datetime
    pk_timezone = pytz.timezone("Asia/Karachi")
    cutoff_date = datetime.now(pk_timezone) - timedelta(days=days_back)

    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            print("Warning: No entries from Tribune RSS")
            return []

        for entry in feed.entries:
            # Parse pubDate
            pub_date_str = entry.get("published", datetime.now(pk_timezone).isoformat())
            try:
                pub_date = parse_date(pub_date_str)
            except (ValueError, TypeError):
                print(f"Invalid date format for {entry.get('title', 'unknown')}: {pub_date_str}")
                pub_date = datetime.now(pk_timezone)

            # Filter by date
            if pub_date < cutoff_date:
                continue  # Skip older than 7 days

            news_item = {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": pub_date.isoformat(),
                "summary": entry.get("description", "").strip(),
                "full_content": entry.get("content", [{}])[0].get("value", "").strip(),
                "category": entry.get("category", ""),
                "source_url": rss_url,
                "fetched_at": datetime.now(pk_timezone).isoformat()
            }

            # Dedup hash
            content_hash = hashlib.md5(f"{news_item['title']}{news_item['published']}".encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            all_headlines.append(news_item)

            if len(all_headlines) >= max_items:
                break

        time.sleep(1)  # Polite delay

    except Exception as e:
        print(f"Error fetching Tribune RSS: {e}")
        return []

    return sorted(all_headlines, key=lambda x: x["published"], reverse=True)

def store_in_chroma(items: List[Dict], collection_name: str = "tribune_news"):
    """
    Embed and store only new Tribune news items in Chroma.
    Embeds title + summary + full_content; ID = content_hash.
    Skips items already in the collection.
    """
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # Get all existing IDs in the collection
    existing_ids = set(collection.get(include=[])['ids'])

    inserted = 0
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for item in items:
        content_hash = hashlib.md5(f"{item['title']}{item['published']}".encode()).hexdigest()
        
        # Skip if already in collection
        if content_hash in existing_ids:
            print(f"Skipping duplicate: {item['title'][:50]}...")
            continue

        # Embed: Use full content for richer context
        text_to_embed = f"{item['title']} {item['summary']} {item['full_content']}"[:5000]
        embedding = model.encode(text_to_embed).tolist()

        ids.append(content_hash)
        documents.append(text_to_embed)
        metadatas.append({**item, "content_hash": content_hash})
        embeddings.append(embedding)
        inserted += 1

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Inserted {inserted} new items into Chroma.")
    else:
        print("No new items to insert into Chroma.")

    return inserted

if __name__ == "__main__":
    news = fetch_tribune_news(days_back=7, max_items=100)
    print(f"Fetched {len(news)} unique items from Tribune (past 7 days):\n")
    for i, item in enumerate(news, 1):
        print(f"{i}. {item['title']}")
        print(f"   📰 {item['summary'][:150]}...")
        print(f"   📅 {item['published']} | 🔗 {item['link']} | 🏷️ {item['category']}\n")
    
    inserted = store_in_chroma(news)
    print(f"\nTotal inserted to Chroma: {inserted}")