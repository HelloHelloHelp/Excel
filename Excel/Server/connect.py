import os
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("connect")

BING_API_KEY = os.environ.get("BING_API_KEY")

# Session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)


def check_internet(timeout=3):
    """Check outbound connectivity to the internet. Try Bing, Google, then Wikipedia."""
    urls = [
        "https://www.bing.com",
        "https://www.google.com",
        "https://en.wikipedia.org",
    ]
    for url in urls:
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code < 400:
                return {"ok": True, "url": url, "status_code": r.status_code}
        except Exception as e:
            logger.debug("Connectivity check failed for %s: %s", url, e)
    return {"ok": False, "error": "No outbound connection to common sites"}


def bing_visual_search(image_bytes, top_k=5):
    """Call Bing Visual Search API. Returns list of results dict or empty list on failure."""
    if not BING_API_KEY or not image_bytes:
        logger.info("Bing API key or image bytes missing for visual search")
        return []
    try:
        url = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        files = {"image": ("image.jpg", image_bytes, "application/octet-stream")}
        r = session.post(url, headers=headers, files=files, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = []
        for tag in data.get("tags", []):
            for action in tag.get("actions", []):
                data_nodes = action.get("data", {})
                values = data_nodes.get("value", []) if isinstance(data_nodes, dict) else []
                for v in values:
                    name = v.get("name") or v.get("hostPageDisplayUrl") or v.get("accentColor")
                    snippet = v.get("snippet") if isinstance(v.get("snippet"), str) else ""
                    urlv = v.get("hostPageDisplayUrl") or v.get("contentUrl") or v.get("webSearchUrl")
                    results.append({"name": name, "snippet": snippet, "url": urlv})
                    if len(results) >= top_k:
                        return results
        return results
    except Exception as e:
        logger.exception("Bing visual search failed: %s", e)
        return []


def bing_web_search(query, top_k=3):
    if not BING_API_KEY or not query:
        return []
    try:
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        params = {"q": query, "count": top_k}
        r = session.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        docs = []
        for item in data.get("webPages", {}).get("value", [])[:top_k]:
            docs.append({"name": item.get("name"), "snippet": item.get("snippet"), "url": item.get("url")})
        return docs
    except Exception as e:
        logger.exception("Bing web search failed: %s", e)
        return []


def wikipedia_search(query, top_k=3):
    if not query:
        return []
    try:
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": top_k}
        r = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        docs = []
        for s in data.get("query", {}).get("search", [])[:top_k]:
            title = s.get("title")
            snippet = s.get("snippet")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            docs.append({"name": title, "snippet": snippet, "url": url})
        return docs
    except Exception as e:
        logger.exception("Wikipedia search failed: %s", e)
        return []


def search(query=None, image_bytes=None, top_k=3):
    """Unified search: prefer visual search (image_bytes + Bing), then Bing web search, then Wikipedia."""
    # Quick connectivity check
    c = check_internet()
    if not c.get("ok"):
        return {"success": False, "error": "no_internet", "detail": c}

    # Try visual
    if image_bytes and BING_API_KEY:
        results = bing_visual_search(image_bytes, top_k=top_k)
        if results:
            return {"success": True, "source": "bing_visual", "results": results}

    # Try web text search
    if query and BING_API_KEY:
        results = bing_web_search(query, top_k=top_k)
        if results:
            return {"success": True, "source": "bing_web", "results": results}

    # Wikipedia fallback
    if query:
        results = wikipedia_search(query, top_k=top_k)
        if results:
            return {"success": True, "source": "wikipedia", "results": results}

    return {"success": False, "error": "no_results"}
