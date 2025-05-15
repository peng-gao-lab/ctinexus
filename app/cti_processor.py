import copy
import logging
import re

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


ioc_patterns = {
    "date": re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b"
    ),
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
    "os": re.compile(
        r"\b(?:Windows|Linux|MacOS|Android|iOS|Unix)\soperating\s(?:system|systems)\b"
    ),
    "sector": re.compile(r"\b[A-Za-z]+(?:\s[A-Za-z]+)*\ssector\b"),
    "version": re.compile(r"\b(?:v|version)\s\d+(?:\.\d+){1,3}\b"),
}


def IOC_detect(mention_merged, mention_text):
    iocs = set()
    mention_list = mention_merged + [mention_text]

    # Pre-filter mentions to reduce unnecessary regex checks
    ioc_keywords = (
        r"(?:CVE|CVSS|YARA|Windows|Linux|MacOS|Android|iOS|Unix|sector|million|billion)"
    )
    ioc_symbols = r"[0-9@:/\-\.]"
    potential_ioc_indicators = re.compile(f"{ioc_symbols}|{ioc_keywords}")
    filtered_mentions = [
        mention for mention in mention_list if potential_ioc_indicators.search(mention)
    ]

    # Match filtered mentions against IOC patterns
    for mention in filtered_mentions:
        for pattern_name, pattern in ioc_patterns.items():
            match = pattern.search(mention)
            if match:
                iocs.add(match.group())
    return iocs
    # Return True if more than one unique IOC is detected
    # return len(iocs) > 1


class PostProcessor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.mention_dict = {}  # key is mention_text, value is mention_id
        self.node_dict = {}  # key is mention_id, value is a list of nodes

    def call(self, result: dict) -> dict:
        self.js = result
        self.entity_idx = self.js["EA"]["entity_num"]

        for triple in self.js["EA"]["aligned_triplets"]:
            for key, node in triple.items():
                if key in ["subject", "object"]:
                    if node["mention_text"] not in self.mention_dict:
                        self.mention_dict[node["mention_text"]] = node["mention_id"]

                    if node["mention_id"] not in self.node_dict:
                        self.node_dict[node["mention_id"]] = []
                    self.node_dict[node["mention_id"]].append(
                        node
                    )  # with reference to the original node

        for triple in self.js["EA"]["aligned_triplets"]:
            for key, node in triple.items():
                if key in ["subject", "object"]:
                    if node["mention_merged"] == []:
                        continue

                    else:
                        iocs = IOC_detect(node["mention_merged"], node["mention_text"])

                        if len(iocs):
                            if len(iocs) < len(node["mention_merged"]) + 1:
                                # This means not all the merged mentions are IOC
                                # TODO: Need to call LLM to check if the non-IOC mentions should be merged with the IOC mentions
                                pass

                            else:
                                # This means all the merged mentions are IOC
                                for m_text in iocs:
                                    # Skip if this IOC wasn't in the original mentions
                                    if m_text not in self.mention_dict:
                                        continue

                                    m_id = self.mention_dict[m_text]
                                    node_list = self.node_dict[m_id]
                                    entity_id = self.entity_idx

                                    if node_list[0]["entity_text"] != m_text:
                                        # This is the new entity
                                        self.entity_idx += 1
                                        self.js["EA"]["entity_num"] += 1

                                    for node in node_list:
                                        if node["mention_text"] == node["entity_text"]:
                                            # This is the original mention
                                            node["mention_merged"] = []
                                            continue

                                        else:
                                            node["mention_merged"] = []
                                            node["entity_id"] = entity_id
                                            node["entity_text"] = m_text

        return self.js


def preprocessor(result: dict) -> dict:
    # Dictionary to track mention_text to mention_id mapping
    mention_id_map = {}
    current_id = 0

    jsr = copy.deepcopy(result)
    jsr["EA"] = {}
    jsr["EA"]["aligned_triplets"] = result["ET"]["typed_triplets"]

    for triple in jsr["EA"]["aligned_triplets"]:
        for key, entity in triple.items():
            if key in ["subject", "object"]:
                mention_text = entity["text"]

                # Check if mention_text already has an ID
                if mention_text not in mention_id_map:
                    mention_id_map[mention_text] = current_id
                    current_id += 1

                # Assign the same mention_id for identical mention_text
                entity["mention_id"] = mention_id_map[mention_text]
                entity["mention_text"] = entity.pop("text")
                entity["mention_class"] = entity.pop("class")

                # Handle mention_class if it's a dictionary
                if isinstance(entity["mention_class"], dict):
                    entity["mention_class"] = list(entity["mention_class"].keys())[0]

    jsr["EA"]["mentions_num"] = current_id

    return jsr
