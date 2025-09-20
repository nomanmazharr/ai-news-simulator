import requests
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import pytz
import hashlib
import time
from typing import List, Dict

def fetch_tribune_news(region: str, query: str = "", days_back: int = 7, max_items: int = 10) -> List[Dict]:
    """
    Fetch Express Tribune news for a specific region from RSS feed, optionally filtered by query.
    Returns unique items with full content and metadata, up to `max_items`, within `days_back` days.
    """
    rss_feeds = {
        "Pakistan": "https://tribune.com.pk/feed/home",
        "Punjab": "https://tribune.com.pk/feed/punjab",
        "Sindh": "https://tribune.com.pk/feed/sindh",
        "Balochistan": "https://tribune.com.pk/feed/balochistan",
        "Khyber Pakhtunkhwa": "https://tribune.com.pk/feed/khyber-pakhtunkhwa",
        "Jammu & Kashmir": "https://tribune.com.pk/feed/jammu-kashmir",
        "Gilgit-Baltistan": "https://tribune.com.pk/feed/gilgit-baltistan"
    }
    
    if region not in rss_feeds:
        raise ValueError(f"Unknown region: {region}. Available: {list(rss_feeds.keys())}")
    
    rss_url = rss_feeds[region]
    all_headlines = []
    seen_hashes = set()
    pk_timezone = pytz.timezone("Asia/Karachi")
    cutoff_date = datetime.now(pk_timezone) - timedelta(days=days_back)
    query_lower = query.lower().strip() if query else ""
    
    try:
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        namespace = {'content': 'http://purl.org/rss/1.0/modules/content/'}
        items = root.findall('.//item')
        print(f"Raw RSS entries for {region}: {len(items)}")  # Debug
        
        for item in items[:max_items * 2]:  # Fetch extra for filtering
            title_elem = item.find('title')
            title = title_elem.text.strip() if title_elem is not None else ""
            
            link_elem = item.find('link')
            link = link_elem.text.strip() if link_elem is not None else ""
            
            img_elem = item.find('.//image/img')
            img = img_elem.get("src") if img_elem is not None else ""
            pub_date_elem = item.find('pubDate')
            pub_date_str = pub_date_elem.text.strip() if pub_date_elem is not None else datetime.now(pk_timezone).isoformat()
            try:
                pub_date = parse_date(pub_date_str)
            except (ValueError, TypeError):
                print(f"Invalid date for '{title[:50]}' in {region}: {pub_date_str}")
                pub_date = datetime.now(pk_timezone)
            
            if pub_date < cutoff_date:
                continue
            
            desc_elem = item.find('description')
            summary = ""
            if desc_elem is not None:
                summary = desc_elem.text.strip() if desc_elem.text else ""
            
            content_elem = item.find('content:encoded', namespace)
            full_content = content_elem.text.strip() if content_elem is not None and content_elem.text else ""
            
            cat_elem = item.find('category')
            category = cat_elem.text.strip() if cat_elem is not None else ""
            
            # Simple keyword filter for query
            if query_lower and not (
                query_lower in title.lower() or
                query_lower in summary.lower() or
                query_lower in full_content.lower()
            ):
                continue
            
            news_item = {
                "title": title,
                "link": link,
                "img": img,
                "published": pub_date.isoformat(),
                "summary": summary,
                "full_content": full_content,
                "category": category,
                "region": region,
                "source_url": rss_url,
                "fetched_at": datetime.now(pk_timezone).isoformat()
            }
            
            content_hash = hashlib.md5(f"{news_item['title']}{news_item['published']}".encode()).hexdigest()
            if content_hash in seen_hashes:
                print(f"Skipping duplicate in {region}: {title[:50]}...")
                continue
            seen_hashes.add(content_hash)
            all_headlines.append(news_item)
            
            if len(all_headlines) >= max_items:
                break

        time.sleep(1)  # Polite delay

    except requests.RequestException as e:
        print(f"Network error fetching {region} RSS: {e}")
        return []
    except ET.ParseError as e:
        print(f"XML parse error for {region} RSS: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching {region} RSS: {e}")
        return []

    return sorted(all_headlines, key=lambda x: x["published"], reverse=True)


def fetch_news_by_category(category: str, query: str = "", days_back: int = 7, max_items: int = 10) -> List[Dict]:
    """
    Fetch Express Tribune news for a specific category from RSS feed, optionally filtered by query.
    Returns unique items with full content and metadata, up to `max_items`, within `days_back` days.
    """
    rss_feeds = {
        "Politics": "https://tribune.com.pk/feed/politics",
        "Technology": "https://tribune.com.pk/feed/technology",
        "Sports": "https://tribune.com.pk/feed/sports",
        "Movies": "https://tribune.com.pk/feed/movies",
        "Music": "https://tribune.com.pk/feed/music",
        "Health": "https://tribune.com.pk/feed/health",
        "Business": "https://tribune.com.pk/feed/business",
        "World": "https://tribune.com.pk/feed/world"
    }
    
    if category not in rss_feeds:
        raise ValueError(f"Unknown category: {category}. Available: {list(rss_feeds.keys())}")
    
    rss_url = rss_feeds[category]
    all_headlines = []
    seen_hashes = set()
    pk_timezone = pytz.timezone("Asia/Karachi")
    cutoff_date = datetime.now(pk_timezone) - timedelta(days=days_back)
    query_lower = query.lower().strip() if query else ""
    
    try:
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        namespace = {'content': 'http://purl.org/rss/1.0/modules/content/'}
        items = root.findall('.//item')
        print(f"Raw RSS entries for {category}: {len(items)}")  # Debug
        
        for item in items[:max_items * 2]:  # Fetch extra for filtering
            title_elem = item.find('title')
            title = title_elem.text.strip() if title_elem is not None else ""
            
            link_elem = item.find('link')
            link = link_elem.text.strip() if link_elem is not None else ""
            
            img_elem = item.find('.//image/img')
            img = img_elem.get("src") if img_elem is not None else ""
            
            pub_date_elem = item.find('pubDate')
            pub_date_str = pub_date_elem.text.strip() if pub_date_elem is not None else datetime.now(pk_timezone).isoformat()
            try:
                pub_date = parse_date(pub_date_str)
            except (ValueError, TypeError):
                print(f"Invalid date for '{title[:50]}' in {category}: {pub_date_str}")
                pub_date = datetime.now(pk_timezone)
            
            if pub_date < cutoff_date:
                continue
            
            desc_elem = item.find('description')
            summary = ""
            if desc_elem is not None:
                summary = desc_elem.text.strip() if desc_elem.text else ""
            
            content_elem = item.find('content:encoded', namespace)
            full_content = content_elem.text.strip() if content_elem is not None and content_elem.text else ""
            
            cat_elem = item.find('category')
            item_category = cat_elem.text.strip() if cat_elem is not None else category
            
            # Simple keyword filter for query
            if query_lower and not (
                query_lower in title.lower() or
                query_lower in summary.lower() or
                query_lower in full_content.lower()
            ):
                continue
            
            news_item = {
                "title": title,
                "link": link,
                "img": img,
                "published": pub_date.isoformat(),
                "summary": summary,
                "full_content": full_content,
                "category": item_category,
                "source_url": rss_url,
                "fetched_at": datetime.now(pk_timezone).isoformat()
            }
            
            content_hash = hashlib.md5(f"{news_item['title']}{news_item['published']}".encode()).hexdigest()
            if content_hash in seen_hashes:
                print(f"Skipping duplicate in {category}: {title[:50]}...")
                continue
            seen_hashes.add(content_hash)
            all_headlines.append(news_item)
            
            if len(all_headlines) >= max_items:
                break

        time.sleep(1)  # Polite delay

    except requests.RequestException as e:
        print(f"Network error fetching {category} RSS: {e}")
        return []
    except ET.ParseError as e:
        print(f"XML parse error for {category} RSS: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching {category} RSS: {e}")
        return []

    return sorted(all_headlines, key=lambda x: x["published"], reverse=True)

# fetch_news_by_category(category="Sports")