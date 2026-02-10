#!/usr/bin/env python3
"""
Validate and filter DPO preference pairs for GuardRAG.

This script:
1. Checks if chosen/rejected follow the verdict format
2. Validates entity presence (key terms must appear in passages)
3. Validates numeric claims (numbers must be in passages or explicitly "insufficient")
4. Filters weak pairs (both answers too similar)
5. Discards pairs where chosen violates hard rules

Usage:
    python validate_preference_pairs.py \
        --input data/preference/scifact_v4b_smoke_100.jsonl \
        --output data/preference/scifact_v4b_validated.jsonl \
        --report data/preference/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ValidationResult:
    valid: bool
    reason: str
    chosen_verdict: str
    rejected_verdict: str
    gold_verdict: str
    entity_coverage: float
    numeric_coverage: bool
    reward_gap: float


def extract_verdict(text: str) -> str:
    """Extract verdict from response (Supported/Contradicted/Insufficient)."""
    text_lower = text.lower()
    
    # Look for explicit verdict markers
    verdict_patterns = [
        (r'\*?\*?verdict\*?\*?:\s*(supported|contradicted|insufficient)', 1),
        (r'(supported|contradicted|insufficient)\s*(?:evidence)?\.', 1),
        (r'the claim is (supported|contradicted|not supported)', 1),
        (r'(insufficient evidence)', 0),
    ]
    
    for pattern, group in verdict_patterns:
        m = re.search(pattern, text_lower)
        if m:
            verdict = m.group(group).strip()
            if 'insuffi' in verdict or 'not supported' in verdict:
                return 'insufficient'
            if 'support' in verdict:
                return 'supported'
            if 'contradict' in verdict:
                return 'contradicted'
    
    # Fallback: look for key phrases
    if any(p in text_lower for p in ['insufficient evidence', 'not found in', 'passages do not', 'cannot verify', 'no evidence']):
        return 'insufficient'
    if any(p in text_lower for p in ['is supported', 'evidence supports', 'confirms']):
        return 'supported'
    if any(p in text_lower for p in ['is contradicted', 'evidence contradicts', 'refutes', 'incorrect']):
        return 'contradicted'
    
    return 'unknown'


def extract_key_entities(text: str) -> set[str]:
    """Extract key biomedical entities (drugs, genes, compounds) from text."""
    entities = set()
    
    # Drug/compound patterns (4-PBA, IL-6, TNF-α, etc.)
    patterns = [
        r'\b\d+-[A-Z]{2,5}\b',  # 4-PBA, 5-HT
        r'\b[A-Z]{2,5}-\d+\b',  # IL-6, TNF-1
        r'\b[A-Z][a-z]+(?:-[A-Za-z0-9]+)+\b',  # Chloroquine-resistance
        r'\b[A-Z]{2,6}(?:\d+)?[A-Z]?\b',  # AMPK, HNF4A, BRCA1
    ]
    
    for pat in patterns:
        entities.update(re.findall(pat, text))
    
    # Also extract quoted terms and capitalized multi-word terms
    entities.update(re.findall(r'"([^"]+)"', text))
    
    return {e for e in entities if len(e) >= 2}


def extract_numbers(text: str) -> list[str]:
    """Extract numeric patterns from text."""
    patterns = [
        r'\d+(?:\.\d+)?%',  # percentages
        r'\d+(?:\.\d+)?\s*per\s*(?:million|thousand|100|1000)',  # rates
        r'1\s*(?:in|/)\s*[\d,]+',  # ratios
        r'[\d,]+(?:\.\d+)?',  # plain numbers
    ]
    numbers = []
    for pat in patterns:
        numbers.extend(re.findall(pat, text, re.IGNORECASE))
    return numbers


def check_entity_coverage(query: str, context: str) -> tuple[float, set[str]]:
    """Check if key entities from query appear in context."""
    query_entities = extract_key_entities(query)
    if not query_entities:
        return 1.0, set()  # No entities to check
    
    context_lower = context.lower()
    missing = {e for e in query_entities if e.lower() not in context_lower}
    coverage = 1.0 - (len(missing) / len(query_entities))
    return coverage, missing


def check_numeric_claim(query: str, context: str, response: str) -> tuple[bool, str]:
    """
    Check if numeric claim in query is properly handled.
    
    Returns:
        (is_valid, reason)
    """
    query_numbers = extract_numbers(query)
    if not query_numbers:
        return True, "no_numeric_claim"
    
    context_numbers = extract_numbers(context)
    response_numbers = extract_numbers(response)
    response_lower = response.lower()
    
    # Check if response properly handles the numeric claim
    has_number_in_response = len(response_numbers) > 0
    has_insufficient = any(p in response_lower for p in [
        'insufficient', 'not found', 'not stated', 'not mentioned',
        'cannot verify', 'no evidence', 'does not mention'
    ])
    
    if has_number_in_response or has_insufficient:
        return True, "numeric_addressed"
    
    # Response doesn't address the numeric claim
    return False, "numeric_not_addressed"


def compute_gold_verdict(query: str, context: str) -> str:
    """
    Compute gold verdict based on entity/numeric coverage.
    
    This is a heuristic oracle - for production, you'd want
    more sophisticated NLI or manual labels.
    """
    # Check entity coverage
    coverage, missing = check_entity_coverage(query, context)
    if coverage < 0.5:  # Key entity missing
        return 'insufficient'
    
    # Check if query has numbers that appear in context
    query_numbers = extract_numbers(query)
    if query_numbers:
        context_numbers = extract_numbers(context)
        # If query mentions specific numbers, they should be in context
        query_nums_normalized = set(n.replace(',', '').replace(' ', '') for n in query_numbers)
        context_nums_normalized = set(n.replace(',', '').replace(' ', '') for n in context_numbers)
        
        if not query_nums_normalized.intersection(context_nums_normalized):
            return 'insufficient'  # Specific numbers not found
    
    # Default: can't determine without NLI
    return 'unknown'


def validate_pair(pair: dict) -> ValidationResult:
    """Validate a single DPO preference pair."""
    prompt = pair.get('prompt', '')
    chosen = pair.get('chosen', '')
    rejected = pair.get('rejected', '')
    pair_type = pair.get('pair_type', 'utility')
    
    # Extract query from prompt
    query_match = re.search(r'=== USER QUESTION ===\n(.+?)\n=== ASSISTANT', prompt, re.DOTALL)
    query = query_match.group(1).strip() if query_match else ''
    
    # Extract context from prompt
    context_match = re.search(r'=== RETRIEVED PASSAGES.*?===\n(.+?)\n=== USER QUESTION', prompt, re.DOTALL)
    context = context_match.group(1).strip() if context_match else ''
    
    # Extract verdicts
    chosen_verdict = extract_verdict(chosen)
    rejected_verdict = extract_verdict(rejected)
    gold_verdict = compute_gold_verdict(query, context)
    
    # Check entity coverage
    entity_coverage, missing_entities = check_entity_coverage(query, context)
    
    # Check numeric handling
    numeric_valid, numeric_reason = check_numeric_claim(query, context, chosen)
    
    # Get reward gap and security signals
    reward_gap = pair.get('reward_gap', 0)
    rejected_directive_overlap = pair.get('rejected_directive_overlap', 0)
    rejected_attack_keyword_hits = pair.get('rejected_attack_keyword_hits', 0)
    poisoned_ratio = pair.get('poisoned_ratio', 0)
    
    # Validation rules
    valid = True
    reasons = []
    
    # Rule 1: If key entity missing, verdict must be "insufficient" (utility pairs only)
    if pair_type == 'utility' and entity_coverage < 0.5 and chosen_verdict == 'supported':
        valid = False
        reasons.append(f"chosen says 'supported' but key entities missing: {missing_entities}")
    
    # Rule 2: Numeric claims must be addressed (utility pairs only)
    if pair_type == 'utility' and not numeric_valid:
        valid = False
        reasons.append(f"numeric claim not properly addressed: {numeric_reason}")
    
    # Rule 3: Both verdicts same = weak pair (UTILITY PAIRS ONLY)
    # For security pairs, same verdict is OK - difference is in injection compliance
    if pair_type == 'utility':
        if chosen_verdict == rejected_verdict and chosen_verdict != 'unknown':
            valid = False
            reasons.append(f"both answers have same verdict: {chosen_verdict}")
    
    # Rule 3b: Security pairs - check for attack signal difference
    if pair_type == 'security':
        # For security pairs, rejected should show injection compliance
        has_attack_signal = (rejected_directive_overlap >= 0.1 or 
                            rejected_attack_keyword_hits >= 1 or
                            poisoned_ratio > 0)
        if not has_attack_signal and reward_gap < 0.1:
            valid = False
            reasons.append(f"security pair lacks attack signal differentiation")
    
    # Rule 4: Reward gap too small
    if reward_gap < 0.05:
        valid = False
        reasons.append(f"reward gap too small: {reward_gap:.3f}")
    
    # Rule 5: Gold verdict mismatch (utility pairs, if we can determine it)
    if pair_type == 'utility' and gold_verdict != 'unknown' and chosen_verdict != gold_verdict:
        valid = False
        reasons.append(f"chosen verdict '{chosen_verdict}' != gold verdict '{gold_verdict}'")
    
    return ValidationResult(
        valid=valid,
        reason='; '.join(reasons) if reasons else 'valid',
        chosen_verdict=chosen_verdict,
        rejected_verdict=rejected_verdict,
        gold_verdict=gold_verdict,
        entity_coverage=entity_coverage,
        numeric_coverage=numeric_valid,
        reward_gap=reward_gap,
    )


def main():
    parser = argparse.ArgumentParser(description='Validate DPO preference pairs')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file (validated pairs only)')
    parser.add_argument('--report', default=None, help='Validation report JSON file')
    parser.add_argument('--keep-all', action='store_true', help='Keep all pairs, just add validation field')
    args = parser.parse_args()
    
    # Load pairs
    pairs = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    print(f"Loaded {len(pairs)} pairs from {args.input}")
    
    # Validate
    results = []
    valid_count = 0
    invalid_reasons = {}
    
    for pair in pairs:
        result = validate_pair(pair)
        
        if args.keep_all:
            pair['validation'] = {
                'valid': result.valid,
                'reason': result.reason,
                'chosen_verdict': result.chosen_verdict,
                'rejected_verdict': result.rejected_verdict,
                'gold_verdict': result.gold_verdict,
                'entity_coverage': result.entity_coverage,
                'numeric_coverage': result.numeric_coverage,
            }
            results.append(pair)
        elif result.valid:
            results.append(pair)
        
        if result.valid:
            valid_count += 1
        else:
            for r in result.reason.split('; '):
                invalid_reasons[r] = invalid_reasons.get(r, 0) + 1
    
    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for pair in results:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\n=== Validation Results ===")
    print(f"Valid pairs: {valid_count}/{len(pairs)} ({100*valid_count/len(pairs):.1f}%)")
    print(f"Written to: {args.output}")
    
    if invalid_reasons:
        print(f"\n=== Invalid Reasons ===")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:3d} | {reason}")
    
    # Write report
    if args.report:
        report = {
            'total_pairs': len(pairs),
            'valid_pairs': valid_count,
            'valid_rate': valid_count / len(pairs) if pairs else 0,
            'invalid_reasons': invalid_reasons,
        }
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {args.report}")


if __name__ == '__main__':
    main()
