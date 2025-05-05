import re

ioc_patterns = {
    "date": re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b"),
    "ip": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    "domain": re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}\b"),
    "url": re.compile(r"\b(?:https?://|www\.)[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}\S*\b"),
    "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}\b"),
    "hash_md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "hash_sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "hash_sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "hash_sha512": re.compile(r"\b[a-fA-F0-9]{128}\b"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b"),
    "cvss": re.compile(r"\bCVSS\d\.\d\b"),
    "yara": re.compile(r"\bYARA\d{4}\b"),
    "money": re.compile(r"[€£\$]\d+(?:\.\d+)?\s(?:million|billion)\b"),
    "os": re.compile(r"\b(?:Windows|Linux|MacOS|Android|iOS|Unix)\soperating\s(?:system|systems)\b"),
    "sector": re.compile(r"\b[A-Za-z]+(?:\s[A-Za-z]+)*\ssector\b"),
    "version": re.compile(r"\b(?:v|version)\s\d+(?:\.\d+){1,3}\b")
}


def IOC_detect(mention_merged, mention_text):

    iocs = set()
    mention_list = mention_merged + [mention_text] 

    # Pre-filter mentions to reduce unnecessary regex checks
    ioc_keywords = r"(?:CVE|CVSS|YARA|Windows|Linux|MacOS|Android|iOS|Unix|sector|million|billion)"
    ioc_symbols = r"[0-9@:/\-\.]"
    potential_ioc_indicators = re.compile(f"{ioc_symbols}|{ioc_keywords}")
    filtered_mentions = [mention for mention in mention_list if potential_ioc_indicators.search(mention)]

    # Match filtered mentions against IOC patterns
    for mention in filtered_mentions:
        for pattern_name, pattern in ioc_patterns.items():
            match = pattern.search(mention)
            if match:
                iocs.add(match.group())
    return iocs
    # Return True if more than one unique IOC is detected
    # return len(iocs) > 1