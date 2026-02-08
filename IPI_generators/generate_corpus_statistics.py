#!/usr/bin/env python3
"""
Generate comprehensive statistics for all IPI corpora
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict

def analyze_corpus(corpus_name, metadata_path):
    """Analyze a single corpus and return statistics"""
    if not os.path.exists(metadata_path):
        return None
    
    stats = {
        'corpus_name': corpus_name,
        'total_attacks': 0,
        'attack_families': Counter(),
        'techniques': Counter(),
        'positions': Counter(),
        'visual_attacks': {
            'total': 0,
            'styles': Counter(),
            'positions': Counter(),
            'ocr_accuracy': Counter()
        },
        'obfuscation': Counter(),
        'position_percentages': {}
    }
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                meta = json.loads(line)
                stats['total_attacks'] += 1
                
                # Attack families
                attack_family = meta.get('attack_family', 'unknown')
                stats['attack_families'][attack_family] += 1
                
                # Techniques
                technique = meta.get('technique', 'unknown')
                stats['techniques'][technique] += 1
                
                # Positions
                position = meta.get('position', 'unknown')
                stats['positions'][position] += 1
                
                # Obfuscation
                obfuscation = meta.get('obfuscation_method', 'none')
                stats['obfuscation'][obfuscation] += 1
                
                # Visual attacks
                if attack_family == 'visual_ocr':
                    stats['visual_attacks']['total'] += 1
                    stats['visual_attacks']['styles'][meta.get('visual_style', 'unknown')] += 1
                    stats['visual_attacks']['positions'][position] += 1
                    stats['visual_attacks']['ocr_accuracy'][meta.get('ocr_accuracy', 'unknown')] += 1
                    
            except json.JSONDecodeError:
                continue
    
    # Calculate position percentages
    if stats['total_attacks'] > 0:
        for pos, count in stats['positions'].items():
            stats['position_percentages'][pos] = (count / stats['total_attacks']) * 100
    
    return stats

def main():
    # Find all metadata files
    base_dir = Path(__file__).parent
    corpora = []
    
    for metadata_file in base_dir.glob('ipi_*/**/*_metadata_v2.jsonl'):
        corpus_name = metadata_file.parent.name.replace('ipi_', '')
        corpora.append((corpus_name, str(metadata_file)))
    
    # Sort by name
    corpora.sort(key=lambda x: x[0])
    
    print("=" * 100)
    print("COMPREHENSIVE IPI CORPUS STATISTICS")
    print("=" * 100)
    
    all_stats = []
    for corpus_name, metadata_path in corpora:
        stats = analyze_corpus(corpus_name, metadata_path)
        if stats:
            all_stats.append(stats)
    
    # Print statistics for each corpus
    for stats in all_stats:
        print(f"\n{'='*100}")
        print(f"CORPUS: {stats['corpus_name'].upper()}")
        print(f"{'='*100}")
        
        print(f"\n📊 OVERVIEW:")
        print(f"   Total Attacks: {stats['total_attacks']}")
        print(f"   Attack Families: {len(stats['attack_families'])}")
        print(f"   Techniques: {len(stats['techniques'])}")
        
        print(f"\n🎯 ATTACK FAMILY DISTRIBUTION:")
        for family, count in sorted(stats['attack_families'].items(), key=lambda x: -x[1]):
            pct = (count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
            print(f"   {family:20s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n📍 POSITION DISTRIBUTION:")
        # Key positions: start, mid, end
        key_positions = ['start', 'mid', 'end', 'early_mid', 'late_mid', 'early', 'near_end']
        for pos in key_positions:
            if pos in stats['positions']:
                count = stats['positions'][pos]
                pct = stats['position_percentages'].get(pos, 0)
                print(f"   {pos:15s}: {count:5d} ({pct:5.1f}%)")
        
        # Calculate mid position percentage (includes mid, early_mid, late_mid)
        mid_total = (stats['positions'].get('mid', 0) + 
                    stats['positions'].get('early_mid', 0) + 
                    stats['positions'].get('late_mid', 0))
        mid_pct = (mid_total / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
        print(f"\n   MID POSITIONS (mid + early_mid + late_mid): {mid_total:5d} ({mid_pct:5.1f}%)")
        
        # Start and end percentages
        start_count = stats['positions'].get('start', 0) + stats['positions'].get('early', 0)
        start_pct = (start_count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
        end_count = stats['positions'].get('end', 0) + stats['positions'].get('near_end', 0)
        end_pct = (end_count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
        
        print(f"   START POSITIONS (start + early): {start_count:5d} ({start_pct:5.1f}%)")
        print(f"   END POSITIONS (end + near_end): {end_count:5d} ({end_pct:5.1f}%)")
        
        # Visual attacks
        if stats['visual_attacks']['total'] > 0:
            print(f"\n🖼️  VISUAL OCR ATTACKS:")
            print(f"   Total Visual Attacks: {stats['visual_attacks']['total']} ({stats['visual_attacks']['total']/stats['total_attacks']*100:.1f}%)")
            
            print(f"\n   Visual Style Distribution:")
            for style, count in sorted(stats['visual_attacks']['styles'].items()):
                pct = (count / stats['visual_attacks']['total']) * 100 if stats['visual_attacks']['total'] > 0 else 0
                print(f"      {style:15s}: {count:3d} ({pct:5.1f}%)")
            
            print(f"\n   Visual Attack Position Distribution:")
            for pos, count in sorted(stats['visual_attacks']['positions'].items()):
                pct = (count / stats['visual_attacks']['total']) * 100 if stats['visual_attacks']['total'] > 0 else 0
                print(f"      {pos:15s}: {count:3d} ({pct:5.1f}%)")
            
            print(f"\n   OCR Accuracy:")
            for acc, count in sorted(stats['visual_attacks']['ocr_accuracy'].items()):
                print(f"      {acc:15s}: {count:3d}")
        else:
            print(f"\n🖼️  VISUAL OCR ATTACKS: 0 (0.0%) - Not included in this corpus")
        
        print(f"\n🔒 OBFUSCATION DISTRIBUTION (Top 5):")
        for method, count in stats['obfuscation'].most_common(5):
            pct = (count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
            print(f"   {method:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Summary table
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE - ALL CORPORA")
    print(f"{'='*100}")
    print(f"\n{'Corpus':<25} {'Total':<10} {'Families':<10} {'Mid %':<10} {'Start %':<10} {'End %':<10} {'Visual':<10}")
    print("-" * 100)
    
    for stats in all_stats:
        mid_total = (stats['positions'].get('mid', 0) + 
                    stats['positions'].get('early_mid', 0) + 
                    stats['positions'].get('late_mid', 0))
        mid_pct = (mid_total / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
        
        start_count = stats['positions'].get('start', 0) + stats['positions'].get('early', 0)
        start_pct = (start_count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
        
        end_count = stats['positions'].get('end', 0) + stats['positions'].get('near_end', 0)
        end_pct = (end_count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
        
        visual_count = stats['visual_attacks']['total']
        visual_pct = (visual_count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
        
        print(f"{stats['corpus_name']:<25} {stats['total_attacks']:<10} {len(stats['attack_families']):<10} {mid_pct:<10.1f} {start_pct:<10.1f} {end_pct:<10.1f} {visual_count} ({visual_pct:.1f}%)")
    
    print(f"\n{'='*100}")
    print("✅ STATISTICS GENERATION COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()

