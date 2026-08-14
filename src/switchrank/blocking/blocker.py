import time
from typing import List, Dict, Tuple, Set, Any
import pandas as pd
from switchrank.normalize.cleaner import normalize_text

class CandidateBlocker:
    """Multi-pass blocking engine for candidate pair reduction."""

    def __init__(self, prefix_len: int = 4, ngram_size: int = 3, window_size: int = 5):
        self.prefix_len = prefix_len
        self.ngram_size = ngram_size
        self.window_size = window_size

    def block_by_brand(self, records: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
        """Block pairs by exact normalized brand agreement."""
        brand_map: Dict[str, List[str]] = {}
        for r in records:
            rid = str(r["id"])
            brand = normalize_text(r.get("brand", ""))
            if brand:
                brand_map.setdefault(brand, []).append(rid)

        candidate_pairs: Set[Tuple[str, str]] = set()
        for b, ids in brand_map.items():
            if 1 < len(ids) <= 1000:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                        candidate_pairs.add(pair)
        return candidate_pairs

    def block_by_prefix(self, records: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
        """Block pairs by first token prefix."""
        prefix_map: Dict[str, List[str]] = {}
        for r in records:
            rid = str(r["id"])
            title = normalize_text(r.get("title", ""))
            tokens = title.split()
            if tokens:
                prefix = tokens[0][: self.prefix_len]
                prefix_map.setdefault(prefix, []).append(rid)

        candidate_pairs: Set[Tuple[str, str]] = set()
        for p, ids in prefix_map.items():
            if 1 < len(ids) <= 1000:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                        candidate_pairs.add(pair)
        return candidate_pairs

    def block_sorted_neighborhood(self, records: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
        """Sorted neighborhood blocking across titles."""
        sorted_recs = sorted(records, key=lambda r: normalize_text(r.get("title", "")))
        candidate_pairs: Set[Tuple[str, str]] = set()

        for i in range(len(sorted_recs)):
            for j in range(i + 1, min(i + self.window_size + 1, len(sorted_recs))):
                id1 = str(sorted_recs[i]["id"])
                id2 = str(sorted_recs[j]["id"])
                if id1 != id2:
                    candidate_pairs.add((min(id1, id2), max(id1, id2)))
        return candidate_pairs

    def evaluate_blocking(
        self, candidate_pairs: Set[Tuple[str, str]], ground_truth_matches: Set[Tuple[str, str]], total_universe_size: int
    ) -> Dict[str, Any]:
        """Compute Pair Reduction Ratio (PRR) and Pair Completeness (Recall)."""
        start_time = time.time()
        n_candidates = len(candidate_pairs)
        n_total_possible = (total_universe_size * (total_universe_size - 1)) // 2 if total_universe_size > 1 else 1

        reduction_ratio = 1.0 - (n_candidates / max(n_total_possible, 1))

        # True matches captured
        true_matches_captured = ground_truth_matches.intersection(candidate_pairs)
        recall = len(true_matches_captured) / max(len(ground_truth_matches), 1)

        return {
            "candidate_count": n_candidates,
            "total_possible": n_total_possible,
            "pair_reduction_ratio": float(reduction_ratio),
            "pair_completeness_recall": float(recall),
            "true_matches_found": len(true_matches_captured),
            "total_ground_truth_matches": len(ground_truth_matches),
            "runtime_seconds": time.time() - start_time,
        }
