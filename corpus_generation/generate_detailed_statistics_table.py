#!/usr/bin/env python3
"""
Generate a detailed CSV table with all corpus statistics
"""

import json
import csv
import os
from pathlib import Path
from collections import Counter

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
        },
    }
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                meta = json.loads(line)
                stats['total_attacks'] += 1
                
                attack_family = meta.get('attack_family', 'unknown')
                stats['attack_families'][attack_family] += 1
                
                technique = meta.get('technique', 'unknown')
                stats['techniques'][technique] += 1
                
                position = meta.get('position', 'unknown')
                stats['positions'][position] += 1
                
                if attack_family == 'visual_ocr':
                    stats['visual_attacks']['total'] += 1
                    stats['visual_attacks']['styles'][meta.get('visual_style', 'unknown')] += 1
                    stats['visual_attacks']['positions'][position] += 1
                    
            except json.JSONDecodeError:
                continue
    
    return stats

def main():
    base_dir = Path(__file__).parent
    # Outputs go to IPI_generators/ (sibling of corpus_generation/)
    ipi_dir = base_dir.parent / 'IPI_generators'
    corpora = []

    for metadata_file in ipi_dir.glob('ipi_*/*_metadata_v2.jsonl'):
        corpus_name = metadata_file.parent.name.replace('ipi_', '')
        corpora.append((corpus_name, str(metadata_file)))
    
    corpora.sort(key=lambda x: x[0])
    
    # Collect all unique attack families across all corpora
    all_families = set()
    all_stats = []
    
    for corpus_name, metadata_path in corpora:
        stats = analyze_corpus(corpus_name, metadata_path)
        if stats:
            all_stats.append(stats)
            all_families.update(stats['attack_families'].keys())
    
    all_families = sorted(all_families)
    
    # Generate CSV
    csv_path = base_dir / 'corpus_statistics_detailed.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ['Corpus', 'Total Attacks', 'Num Families']
        header.extend([f'{fam} Count' for fam in all_families])
        header.extend([f'{fam} %' for fam in all_families])
        header.extend(['Mid %', 'Start %', 'End %', 'Visual Count', 'Visual %'])
        writer.writerow(header)
        
        # Data rows
        for stats in all_stats:
            row = [stats['corpus_name'], stats['total_attacks'], len(stats['attack_families'])]
            
            # Family counts
            for fam in all_families:
                count = stats['attack_families'].get(fam, 0)
                row.append(count)
            
            # Family percentages
            for fam in all_families:
                count = stats['attack_families'].get(fam, 0)
                pct = (count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
                row.append(f'{pct:.1f}%')
            
            # Position percentages
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
            
            row.extend([f'{mid_pct:.1f}%', f'{start_pct:.1f}%', f'{end_pct:.1f}%', 
                       visual_count, f'{visual_pct:.1f}%'])
            
            writer.writerow(row)
    
    print(f"Detailed CSV saved to: {csv_path}")
    
    # Also create a markdown table
    md_path = base_dir / 'corpus_statistics_detailed.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Complete Corpus Statistics\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Corpus | Total | Families | Mid % | Start % | End % | Visual |\n")
        f.write("|--------|-------|----------|-------|---------|-------|--------|\n")
        
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
            
            f.write(f"| {stats['corpus_name']} | {stats['total_attacks']} | {len(stats['attack_families'])} | "
                   f"{mid_pct:.1f}% | {start_pct:.1f}% | {end_pct:.1f}% | {visual_count} ({visual_pct:.1f}%) |\n")
        
        f.write("\n## Attack Family Distribution\n\n")
        f.write("| Corpus | " + " | ".join(all_families) + " |\n")
        f.write("|--------|" + "|".join(["---" for _ in all_families]) + "|\n")
        
        for stats in all_stats:
            row = [stats['corpus_name']]
            for fam in all_families:
                count = stats['attack_families'].get(fam, 0)
                pct = (count / stats['total_attacks']) * 100 if stats['total_attacks'] > 0 else 0
                row.append(f"{count} ({pct:.1f}%)")
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## Visual Attacks Details (nfcorpus only)\n\n")
        nfcorpus_stats = next((s for s in all_stats if s['corpus_name'] == 'nfcorpus'), None)
        if nfcorpus_stats and nfcorpus_stats['visual_attacks']['total'] > 0:
            f.write(f"**Total Visual Attacks:** {nfcorpus_stats['visual_attacks']['total']} "
                   f"({nfcorpus_stats['visual_attacks']['total']/nfcorpus_stats['total_attacks']*100:.1f}%)\n\n")
            f.write("### Visual Style Distribution\n\n")
            f.write("| Style | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            for style, count in sorted(nfcorpus_stats['visual_attacks']['styles'].items()):
                pct = (count / nfcorpus_stats['visual_attacks']['total']) * 100
                f.write(f"| {style} | {count} | {pct:.1f}% |\n")
            
            f.write("\n### Visual Attack Position Distribution\n\n")
            f.write("| Position | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")
            for pos, count in sorted(nfcorpus_stats['visual_attacks']['positions'].items()):
                pct = (count / nfcorpus_stats['visual_attacks']['total']) * 100
                f.write(f"| {pos} | {count} | {pct:.1f}% |\n")
    
    print(f" Detailed Markdown saved to: {md_path}")

if __name__ == "__main__":
    main()

