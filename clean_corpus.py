import json
import re
import unicodedata
from datetime import datetime
from collections import Counter

INPUT_PATH = "data/accc_raw.json"
OUTPUT_PATH = "data/accc_corpus_clean.json"
CURRENT_YEAR = datetime.now().year


def clean_text(text):
    # smart quotes, em dashes, non-breaking spaces, bullets
    text = (text
            .replace('\u2019', "'").replace('\u2018', "'")
            .replace('\u201C', '"').replace('\u201D', '"')
            .replace('\u2013', '-').replace('\u2014', '-').replace('\u2011', '-')
            .replace('\u00A0', ' ').replace('\u2022', '*'))
    text = unicodedata.normalize('NFKC', text)

    # remove "(PDF31.01 MB)" style markers
    text = re.sub(r'\(PDF\s*\d+\.?\d*\s*[KM]B\)', '', text)

    # drop consecutive duplicate lines (scraper artifact)
    lines = text.split('\n')
    deduped, prev = [], None
    for line in lines:
        s = line.strip()
        if s != prev or not s:
            deduped.append(line)
        prev = s
    text = '\n'.join(deduped)

    # headers with no value next to them
    HEADERS_TO_DROP = {
        'Date lodged', 'Status', 'Type', 'Outcome',
        'Undertaking date', 'Description of Conduct',
        'Date commenced', 'Indicative date',
        'Document title', 'Date',
        'Applications', 'Consultations', 'Determinations',
        'Company or individual details', 'Name', 'ACN',
    }
    POTENTIALLY_ORPHAN_LABELS = {'Industry', 'Commenced', 'Total review days'}
    lines = text.split('\n')
    kept = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s in HEADERS_TO_DROP:
            i += 1
            continue
        if s in POTENTIALLY_ORPHAN_LABELS:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or lines[j].strip() in (
                POTENTIALLY_ORPHAN_LABELS | HEADERS_TO_DROP |
                {'Acquirer(s)', 'Target(s)', 'Summary',
                 'Applicant(s)', 'Authorisation number(s)', 'Notification number(s)'}
            ):
                i += 1
                continue
        kept.append(line)
        i += 1
    text = '\n'.join(kept)

    # merge label/value pairs into one line so embeddings see them together
    LABELS_TO_MERGE = {
        'Industry', 'Commenced', 'Total review days',
        'Acquirer(s)', 'Target(s)', 'Summary',
        'Applicant(s)', 'Authorisation number(s)', 'Notification number(s)',
    }
    lines = text.split('\n')
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped in LABELS_TO_MERGE and i + 1 < len(lines):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                value = lines[j].strip()
                if value and value not in LABELS_TO_MERGE and value not in HEADERS_TO_DROP:
                    merged.append(f"{stripped}: {value}")
                    i = j + 1
                    continue
        merged.append(line)
        i += 1
    text = '\n'.join(merged)

    # whitespace cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def recategorize(case):
    url = case['case_url'].lower()
    text = case['text'].lower()

    if 'mergers-register' in url or 'informal-merger' in url:
        return 'Merger & Acquisition'

    if ('authorisations-register' in url
            or 'notifications-register' in url
            or 'class-exemptions-register' in url
            or 'authorisations-and-notifications' in url):
        return 'Competition Law (Authorisation)'

    if ('court-enforcement' in url
            or 'court-proceedings' in url
            or 'enforceable-undertaking' in url
            or 'undertakings-register' in url
            or 'infringement-notices' in url
            or '87b' in url):
        return 'ACL Compliance (Enforcement)'

    if 'telecommunications-registers' in url:
        return 'Telecommunications Regulation'

    if any(k in url for k in ['energy', 'water-charge', 'fuel', 'nbn', 'wheat-port']):
        return 'Infrastructure Regulation'

    # fallback: keyword search
    if re.search(r'\bmisleading\b|\bdeceptive\b|consumer guarantee', text):
        return 'ACL Compliance (Enforcement)'
    if re.search(r'\bauthorisation\b|\bcollective bargain\b', text):
        return 'Competition Law (Authorisation)'

    return 'Other Regulatory'


def fix_dates(case):
    # the original 'year' field is extracted from body text and not reliable
    case['extracted_year'] = case.pop('year', None)
    try:
        case['extracted_year'] = int(case['extracted_year']) if case['extracted_year'] else None
    except (ValueError, TypeError):
        case['extracted_year'] = None

    # take year from the date field instead
    m = re.search(r'\b(19|20)\d{2}\b', case.get('date', ''))
    case['case_year'] = int(m.group(0)) if m else None

    # some entries have future dates (these are expiration dates, not case dates)
    case['is_future_date'] = bool(
        case['case_year'] and case['case_year'] > CURRENT_YEAR
    )
    return case


def dedupe_by_text(cases):
    seen, out = set(), []
    removed = 0
    for c in cases:
        h = hash(c['text'])
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        out.append(c)
    return out, removed


def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n_in = len(data)
    print(f"Loaded {n_in} cases from {INPUT_PATH}")

    original_lengths = [len(c['text']) for c in data]
    original_cats = Counter(c['category'] for c in data)

    # clean text
    for c in data:
        c['text'] = clean_text(c['text'])
    print("Text cleaning done")

    # recategorize
    cat_changes = 0
    for c in data:
        old = c['category']
        new = recategorize(c)
        if old != new:
            cat_changes += 1
        c['original_category'] = old
        c['category'] = new
    print(f"Re-categorization: {cat_changes} cases changed")

    # fix dates
    for c in data:
        fix_dates(c)
    n_future = sum(1 for c in data if c['is_future_date'])
    print(f"Date normalization: {n_future} future dates flagged")

    # dedupe
    data, n_removed = dedupe_by_text(data)
    print(f"Content dedup: {n_removed} duplicates removed")

    # drop cases that became too small after cleaning
    before_empty = len(data)
    data = [c for c in data if len(c['text']) >= 100]
    n_empty_filtered = before_empty - len(data)
    if n_empty_filtered:
        print(f"Removed {n_empty_filtered} cases with <100 chars after cleaning")

    # save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # stats
    new_lengths = [len(c['text']) for c in data]
    new_cats = Counter(c['category'] for c in data)
    pct_shrink = 100 * (1 - sum(new_lengths) / sum(original_lengths))

    print("\nResults")
    print(f"Cases: {n_in} -> {len(data)} ({n_in - len(data)} removed)")
    print(f"Text size: {sum(original_lengths):,} -> {sum(new_lengths):,} chars ({pct_shrink:+.1f}%)")
    print(f"Avg length: {sum(original_lengths)//n_in:,} -> {sum(new_lengths)//len(data):,} chars")

    print("\nCategories (before -> after):")
    all_cats = set(original_cats) | set(new_cats)
    for cat in sorted(all_cats):
        before = original_cats.get(cat, 0)
        after = new_cats.get(cat, 0)
        print(f"  {cat}: {before} -> {after}")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()