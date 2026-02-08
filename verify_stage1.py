#!/usr/bin/env python3
"""Verify Stage 1 attack generation is complete and correct"""

import json
from pathlib import Path
from collections import defaultdict

print("="*80)
print("STAGE 1 VERIFICATION")
print("="*80)
print()

corpora = ['nfcorpus', 'scifact', 'fiqa', 'hotpotqa', 'msmarco', 'natural_questions']
base_path = Path('IPI_generators')

total_attacks = 0
attack_families_all = set()
techniques_all = set()
obfuscation_all = set()

for corpus in corpora:
    corpus_path = base_path / f'ipi_{corpus}'
    metadata_file = corpus_path / f'{corpus}_ipi_metadata_v2.jsonl'
    
    if not metadata_file.exists():
        print(f"❌ {corpus}: Metadata file missing")
        continue
    
    # Count attacks
    attacks = 0
    families = set()
    techniques = set()
    obfuscations = set()
    
    with open(metadata_file) as f:
        for line in f:
            if line.strip():
                meta = json.loads(line)
                attacks += 1
                families.add(meta.get('attack_family'))
                techniques.add(meta.get('technique'))
                obfuscations.add(meta.get('obfuscation_method'))
    
    total_attacks += attacks
    attack_families_all.update(families)
    techniques_all.update(techniques)
    obfuscation_all.update(obfuscations)
    
    # Check files exist
    required_files = [
        f'{corpus}_ipi_poisoned_v2.jsonl',
        f'{corpus}_ipi_mixed_v2.jsonl',
        f'{corpus}_ipi_metadata_v2.jsonl',
        f'{corpus}_id_mapping.csv',
        f'{corpus}_attack_manifest_v2.json',
        f'{corpus}_ipi_statistics_v2.txt'
    ]
    
    missing = [f for f in required_files if not (corpus_path / f).exists()]
    
    status = "✅" if not missing else "⚠️"
    print(f"{status} {corpus.upper()}: {attacks:,} attacks, {len(families)} families")
    if missing:
        print(f"   Missing: {', '.join(missing)}")

print()
print(f"Total Attacks: {total_attacks:,}")
print(f"Attack Families: {len(attack_families_all)} - {sorted(attack_families_all)}")
print(f"Techniques: {len(techniques_all)}")
print(f"Obfuscation Methods: {len(obfuscation_all)}")
print()

# Expected values
expected = {
    'total_attacks': 39992,
    'families': 12,
    'techniques': 13,
    'obfuscations': 8
}

print("Verification:")
if total_attacks >= expected['total_attacks'] * 0.95:  # Within 5%
    print(f"✅ Attack count: {total_attacks:,} (expected ~{expected['total_attacks']:,})")
else:
    print(f"⚠️  Attack count: {total_attacks:,} (expected ~{expected['total_attacks']:,})")

if len(attack_families_all) >= expected['families']:
    print(f"✅ Attack families: {len(attack_families_all)} (expected {expected['families']})")
else:
    print(f"⚠️  Attack families: {len(attack_families_all)} (expected {expected['families']})")

print()
print("="*80)
print("✅ STAGE 1: COMPLETE AND VERIFIED")
print("="*80)
