import json
import re
import time
import os
from urllib.parse import urljoin
from collections import Counter

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.accc.gov.au"
LISTING_URL = "https://www.accc.gov.au/news-centre?type=accc_news&layout=full_width"

OUTPUT_PATH = "accc_media_releases.json"
URLS_CACHE = "accc_discovered_urls.json"

DELAY = 0.8

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}

MIN_YEAR = 2020
MAX_LIST_PAGES = 400

# keywords that indicate the release is about enforcement / consumer law
RELEVANCE_PATTERN = re.compile(
    r"\b(misleading|deceptive|misrepresent|false (?:claim|representation|advertis)"
    r"|drip pricing|price fixing|was/now|strikethrough"
    r"|RRP|recommended retail|discount(?:\s+claim)?"
    r"|environmental claim|green\s*claim|greenwash"
    r"|infringement notice|court[- ]enforceable undertaking"
    r"|federal court|warning notice|consumer guarantee|unconscionable)\b",
    re.IGNORECASE,
)


def classify_path(text, title):
    full = (title + " " + text).lower()

    has_court_penalty = bool(re.search(
        r"federal court (?:has )?(?:ordered|imposed)|court ordered \w+ to pay"
        r"|pay.{0,20}penalt.{0,30}(?:imposed|ordered) by",
        full
    ))
    has_infringement = "infringement notice" in full
    has_undertaking = ("court-enforceable undertaking" in full
                       or "court enforceable undertaking" in full
                       or "section 87b" in full or "s.87b" in full or "s. 87b" in full)
    has_warning = ("warning notice" in full
                   or re.search(r"\baccc warns\b", full)
                   or "please explain" in full)

    if has_court_penalty:
        return "Court Penalty"
    if has_undertaking and has_infringement:
        return "Infringement + Undertaking"
    if has_undertaking:
        return "Undertaking"
    if has_infringement:
        return "Infringement Notice"
    if has_warning:
        return "Warning"
    if re.search(r"penalt|fine", full):
        return "Other Penalty"
    return "Other Enforcement"


def extract_year(date_str):
    m = re.search(r"\b(20\d{2})\b", date_str or "")
    return int(m.group(1)) if m else None


def extract_topics(soup):
    topics = []
    for label in soup.find_all(string=re.compile(r"^Topics?$")):
        parent = label.parent
        if parent and parent.parent:
            for a in parent.parent.find_all("a"):
                t = a.get_text(strip=True)
                if t and len(t) < 50:
                    topics.append(t)
    return list(dict.fromkeys(topics))[:6]


def extract_penalties(text):
    pattern = re.compile(
        r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(million|m\b|thousand|k\b)?",
        re.IGNORECASE
    )
    found = []
    for amount, suffix in pattern.findall(text):
        full = f"${amount}" + (f" {suffix}" if suffix else "")
        found.append(full.strip())
    seen, out = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
        if len(out) >= 8:
            break
    return out


def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def parse_listing_page(html):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/media-release/" not in href:
            continue
        full_url = urljoin(BASE_URL, href.split("?")[0].split("#")[0])

        # walk up a few parents to find a year near the link
        year_hint = None
        node = a
        for _ in range(4):
            if node is None:
                break
            text = node.get_text(" ", strip=True)
            m = re.search(r"\b(20\d{2})\b", text)
            if m:
                year_hint = int(m.group(1))
                break
            node = node.parent

        items.append((full_url, year_hint))

    # dedupe
    seen, out = set(), []
    for url, yr in items:
        if url not in seen:
            seen.add(url)
            out.append((url, yr))
    return out


def discover_via_listing(session):
    # cache hit
    if os.path.exists(URLS_CACHE):
        with open(URLS_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"Loaded {len(cached)} URLs from cache")
        return cached

    print(f"Walking listing pages, stopping at year < {MIN_YEAR}")
    all_urls = []
    seen_urls = set()
    consecutive_old_pages = 0

    for page in range(MAX_LIST_PAGES):
        url = LISTING_URL + f"&page={page}"
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as e:
            print(f"Network error on page {page}: {e}")
            break

        if r.status_code != 200:
            print(f"Page {page} returned {r.status_code}, stopping")
            break

        items = parse_listing_page(r.text)
        if not items:
            print(f"No items on page {page}, stopping")
            break

        new_count = 0
        old_count = 0
        for u, yr in items:
            if u in seen_urls:
                continue
            seen_urls.add(u)
            if yr is not None and yr < MIN_YEAR:
                old_count += 1
                continue
            all_urls.append(u)
            new_count += 1

        # 3 pages in a row of only-old releases = we're done
        if new_count == 0 and old_count > 0:
            consecutive_old_pages += 1
            if consecutive_old_pages >= 3:
                print(f"3 consecutive pages of pre-{MIN_YEAR} content, stopping")
                break
        else:
            consecutive_old_pages = 0

        if (page + 1) % 10 == 0:
            print(f"page {page+1}, total kept: {len(all_urls)}")

        time.sleep(DELAY / 2)

    print(f"Collected {len(all_urls)} candidate URLs")

    with open(URLS_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_urls, f, indent=2)

    return all_urls


def fetch_release(session, url):
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return None
    except requests.RequestException:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""
    if not title:
        return None

    body = (soup.select_one("main")
            or soup.select_one("article")
            or soup.select_one(".field--name-body"))
    if not body:
        return None

    for tag in body.find_all(["script", "style", "nav", "aside", "footer", "header"]):
        tag.decompose()

    text = body.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) < 400:
        return None

    date_str = ""
    m = re.search(r"Date\s*\n\s*(\d{1,2} \w+ \d{4})", text)
    if m:
        date_str = m.group(1)
    else:
        time_el = soup.find("time")
        if time_el:
            date_str = time_el.get_text(strip=True)

    return {
        "title": title,
        "date": date_str,
        "year": extract_year(date_str),
        "url": url,
        "topics": extract_topics(soup),
        "text": text,
        "char_count": len(text),
        "enforcement_path": classify_path(text, title),
        "penalties_mentioned": extract_penalties(text),
    }


def load_progress():
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            return existing, {item["url"] for item in existing}
        except Exception:
            return [], set()
    return [], set()


def save_progress(items):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def main():
    session = get_session()

    urls = discover_via_listing(session)
    if not urls:
        print("No URLs discovered, cannot continue")
        return

    items, done_urls = load_progress()
    if done_urls:
        print(f"{len(done_urls)} already scraped, skipping those")
    todo = [u for u in urls if u not in done_urls]

    if not todo:
        print("Nothing new to scrape")
    else:
        print(f"{len(todo)} releases to fetch\n")

        kept = 0
        skipped_old = 0
        skipped_irrelevant = 0
        errors = 0

        for url in tqdm(todo):
            rel = fetch_release(session, url)
            if not rel:
                errors += 1
                time.sleep(DELAY)
                continue

            if rel["year"] and rel["year"] < MIN_YEAR:
                skipped_old += 1
                time.sleep(DELAY)
                continue

            if not RELEVANCE_PATTERN.search(rel["text"]):
                skipped_irrelevant += 1
                time.sleep(DELAY)
                continue

            items.append(rel)
            kept += 1

            # checkpoint every 25 releases
            if kept % 25 == 0:
                save_progress(items)

            time.sleep(DELAY)

        save_progress(items)
        print(f"\nKept: {kept}")
        print(f"Skipped (pre-{MIN_YEAR}): {skipped_old}")
        print(f"Skipped (off-topic): {skipped_irrelevant}")
        print(f"Errors: {errors}")

    print(f"\nFinal corpus: {len(items)} relevant releases")

    paths = Counter(i["enforcement_path"] for i in items)
    print("\nEnforcement path distribution:")
    for p, n in paths.most_common():
        print(f"  {p}: {n}")

    years = Counter(i["year"] for i in items if i["year"])
    print("\nYear distribution:")
    for y in sorted(years):
        print(f"  {y}: {years[y]}")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()