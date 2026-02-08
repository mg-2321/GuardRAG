#!/usr/bin/env python3
"""
Analyze IPI Attack Statistics
Extracts comprehensive statistics about:
- Attack families
- Injection types (techniques)
- Position distribution (mid, start, end, early_mid, late_mid, etc.)

Prepares data for ASR (Attack Success Rate) calculations:
- Overall ASR
- Individual ASR for each attack type
- Individual ASR for each attack family
"""

import json
import csv
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any
import argparse


class IPIStatisticsAnalyzer:
    """Analyze IPI attack corpus statistics"""
    
    def __init__(self, metadata_path: str):
        self.metadata_path = Path(metadata_path)
        self.attacks = []
        self.stats = {
            'total_attacks': 0,
            'by_family': defaultdict(int),
            'by_technique': defaultdict(int),
            'by_position': defaultdict(int),
            'by_placement': defaultdict(int),
            'by_objective': defaultdict(int),
            'by_retriever_hint': defaultdict(int),
            'by_level': defaultdict(int),
            'family_technique': defaultdict(lambda: defaultdict(int)),
            'family_position': defaultdict(lambda: defaultdict(int)),
            'technique_position': defaultdict(lambda: defaultdict(int)),
            'optimized_count': 0,
            'optimize_free_count': 0,
        }
        
    def load_metadata(self):
        """Load attack metadata from JSONL file"""
        print(f"Loading metadata from {self.metadata_path}...")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        attack = json.loads(line.strip())
                        self.attacks.append(attack)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping malformed line {line_num}: {e}")
        
        self.stats['total_attacks'] = len(self.attacks)
        print(f"✓ Loaded {len(self.attacks)} attack records")
        
    def analyze_statistics(self):
        """Analyze all statistics"""
        print("\n" + "="*80)
        print("ANALYZING IPI ATTACK STATISTICS")
        print("="*80)
        
        for attack in self.attacks:
            # Basic statistics
            family = attack.get('attack_family', 'unknown')
            technique = attack.get('technique', 'unknown')
            position = attack.get('position', 'unknown')
            placement = attack.get('placement', 'unknown')
            objective = attack.get('objective', 'unknown')
            retriever = attack.get('retriever_hint', 'unknown')
            level = attack.get('level', 'unknown')
            
            # Update counters
            self.stats['by_family'][family] += 1
            self.stats['by_technique'][technique] += 1
            self.stats['by_position'][position] += 1
            self.stats['by_placement'][placement] += 1
            self.stats['by_objective'][objective] += 1
            self.stats['by_retriever_hint'][retriever] += 1
            self.stats['by_level'][level] += 1
            
            # Cross-tabulations
            self.stats['family_technique'][family][technique] += 1
            self.stats['family_position'][family][position] += 1
            self.stats['technique_position'][technique][position] += 1
            
            # Optimized vs optimize-free
            if family == 'IDEM' or 'idem' in technique.lower():
                self.stats['optimized_count'] += 1
            else:
                self.stats['optimize_free_count'] += 1
        
        print("✓ Statistics calculated")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        print(f"\nTotal Attacks: {self.stats['total_attacks']}")
        
        print(f"\n{'─'*80}")
        print("BY ATTACK FAMILY")
        print(f"{'─'*80}")
        for family, count in sorted(self.stats['by_family'].items(), key=lambda x: -x[1]):
            pct = (count / self.stats['total_attacks']) * 100
            print(f"  {family:25s}: {count:4d} ({pct:5.2f}%)")
        
        print(f"\n{'─'*80}")
        print("BY INJECTION TECHNIQUE (TYPE)")
        print(f"{'─'*80}")
        for technique, count in sorted(self.stats['by_technique'].items(), key=lambda x: -x[1]):
            pct = (count / self.stats['total_attacks']) * 100
            print(f"  {technique:40s}: {count:4d} ({pct:5.2f}%)")
        
        print(f"\n{'─'*80}")
        print("BY POSITION")
        print(f"{'─'*80}")
        # Group positions
        mid_positions = ['mid', 'early_mid', 'late_mid']
        start_positions = ['start', 'early']
        end_positions = ['end', 'near_end', 'optimized_end']
        
        mid_count = sum(self.stats['by_position'][p] for p in mid_positions)
        start_count = sum(self.stats['by_position'][p] for p in start_positions)
        end_count = sum(self.stats['by_position'][p] for p in end_positions)
        other_count = self.stats['total_attacks'] - mid_count - start_count - end_count
        
        print(f"  Mid positions (mid, early_mid, late_mid):    {mid_count:4d} ({(mid_count/self.stats['total_attacks']*100):5.2f}%)")
        print(f"  Start positions (start, early):              {start_count:4d} ({(start_count/self.stats['total_attacks']*100):5.2f}%)")
        print(f"  End positions (end, near_end):               {end_count:4d} ({(end_count/self.stats['total_attacks']*100):5.2f}%)")
        if other_count > 0:
            print(f"  Other positions:                            {other_count:4d} ({(other_count/self.stats['total_attacks']*100):5.2f}%)")
        
        print(f"\n  Detailed position breakdown:")
        for position, count in sorted(self.stats['by_position'].items(), key=lambda x: -x[1]):
            pct = (count / self.stats['total_attacks']) * 100
            print(f"    {position:20s}: {count:4d} ({pct:5.2f}%)")
        
        print(f"\n{'─'*80}")
        print("BY OBJECTIVE")
        print(f"{'─'*80}")
        for objective, count in sorted(self.stats['by_objective'].items(), key=lambda x: -x[1]):
            pct = (count / self.stats['total_attacks']) * 100
            print(f"  {objective:25s}: {count:4d} ({pct:5.2f}%)")
        
        print(f"\n{'─'*80}")
        print("BY RETRIEVER HINT")
        print(f"{'─'*80}")
        for retriever, count in sorted(self.stats['by_retriever_hint'].items(), key=lambda x: -x[1]):
            pct = (count / self.stats['total_attacks']) * 100
            print(f"  {retriever:15s}: {count:4d} ({pct:5.2f}%)")
        
        print(f"\n{'─'*80}")
        print("OPTIMIZED vs OPTIMIZE-FREE ATTACKS")
        print(f"{'─'*80}")
        optimized = self.stats['optimized_count']
        optimize_free = self.stats['optimize_free_count']
        total = self.stats['total_attacks']
        print(f"  Optimized attacks (IDEM - uses optimal position finding):")
        print(f"    {optimized:4d} attacks ({optimized/total*100:5.2f}%)")
        print(f"  Optimize-free attacks (non-IDEM - uses weighted random selection):")
        print(f"    {optimize_free:4d} attacks ({optimize_free/total*100:5.2f}%)")
    
    def export_to_json(self, output_path: str):
        """Export statistics to JSON for ASR calculations"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare export data
        export_data = {
            'total_attacks': self.stats['total_attacks'],
            'attack_families': dict(self.stats['by_family']),
            'injection_techniques': dict(self.stats['by_technique']),
            'position_distribution': dict(self.stats['by_position']),
            'position_groups': {
                'mid': sum(self.stats['by_position'][p] for p in ['mid', 'early_mid', 'late_mid']),
                'start': sum(self.stats['by_position'][p] for p in ['start', 'early']),
                'end': sum(self.stats['by_position'][p] for p in ['end', 'near_end']),
            },
            'placement_distribution': dict(self.stats['by_placement']),
            'objective_distribution': dict(self.stats['by_objective']),
            'retriever_hint_distribution': dict(self.stats['by_retriever_hint']),
            'level_distribution': dict(self.stats['by_level']),
            'family_technique_matrix': {
                family: dict(techniques) 
                for family, techniques in self.stats['family_technique'].items()
            },
            'family_position_matrix': {
                family: dict(positions)
                for family, positions in self.stats['family_position'].items()
            },
            'technique_position_matrix': {
                technique: dict(positions)
                for technique, positions in self.stats['technique_position'].items()
            },
            'optimized_vs_optimize_free': {
                'optimized_count': self.stats['optimized_count'],
                'optimized_percentage': (self.stats['optimized_count'] / self.stats['total_attacks']) * 100,
                'optimize_free_count': self.stats['optimize_free_count'],
                'optimize_free_percentage': (self.stats['optimize_free_count'] / self.stats['total_attacks']) * 100,
                'description': {
                    'optimized': 'IDEM attacks using optimal position finding algorithm',
                    'optimize_free': 'Non-IDEM attacks using weighted random position selection'
                }
            },
            # Individual attack records for ASR calculation
            'attack_records': self.attacks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Statistics exported to JSON: {output_file}")
        return output_file
    
    def export_to_csv(self, output_path: str):
        """Export individual attack records to CSV for easy analysis"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV columns
        fieldnames = [
            'doc_id', 'original_id', 'attack_family', 'technique',
            'position', 'placement', 'objective', 'retriever_hint',
            'level', 'span_len', 'corpus_fit'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for attack in self.attacks:
                row = {
                    'doc_id': attack.get('doc_id', ''),
                    'original_id': attack.get('original_id', ''),
                    'attack_family': attack.get('attack_family', ''),
                    'technique': attack.get('technique', ''),
                    'position': attack.get('position', ''),
                    'placement': attack.get('placement', ''),
                    'objective': attack.get('objective', ''),
                    'retriever_hint': attack.get('retriever_hint', ''),
                    'level': attack.get('level', ''),
                    'span_len': attack.get('span_len', ''),
                    'corpus_fit': attack.get('corpus_fit', '')
                }
                writer.writerow(row)
        
        print(f"✓ Attack records exported to CSV: {output_file}")
        return output_file
    
    def export_asr_template(self, output_path: str):
        """Export template for ASR calculations with attack grouping"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Group attacks for ASR calculation
        asr_template = {
            'total_attacks': self.stats['total_attacks'],
            'overall_asr_placeholder': 0.0,  # Will be filled after evaluation
            
            'by_attack_family': {},
            'by_injection_technique': {},
            'by_position': {},
            
            # Detailed breakdowns
            'family_technique_asr': {},
            'family_position_asr': {},
            'technique_position_asr': {}
        }
        
        # Initialize ASR structures
        for family, count in self.stats['by_family'].items():
            asr_template['by_attack_family'][family] = {
                'count': count,
                'successful': 0,  # Will be filled after evaluation
                'asr': 0.0
            }
        
        for technique, count in self.stats['by_technique'].items():
            asr_template['by_injection_technique'][technique] = {
                'count': count,
                'successful': 0,
                'asr': 0.0
            }
        
        for position, count in self.stats['by_position'].items():
            asr_template['by_position'][position] = {
                'count': count,
                'successful': 0,
                'asr': 0.0
            }
        
        # Position groups
        mid_positions = ['mid', 'early_mid', 'late_mid']
        start_positions = ['start', 'early']
        end_positions = ['end', 'near_end']
        
        asr_template['by_position_group'] = {
            'mid': {
                'count': sum(self.stats['by_position'][p] for p in mid_positions),
                'successful': 0,
                'asr': 0.0,
                'positions': mid_positions
            },
            'start': {
                'count': sum(self.stats['by_position'][p] for p in start_positions),
                'successful': 0,
                'asr': 0.0,
                'positions': start_positions
            },
            'end': {
                'count': sum(self.stats['by_position'][p] for p in end_positions),
                'successful': 0,
                'asr': 0.0,
                'positions': end_positions
            }
        }
        
        # Cross-tabulations
        for family, techniques in self.stats['family_technique'].items():
            if family not in asr_template['family_technique_asr']:
                asr_template['family_technique_asr'][family] = {}
            for technique, count in techniques.items():
                asr_template['family_technique_asr'][family][technique] = {
                    'count': count,
                    'successful': 0,
                    'asr': 0.0
                }
        
        for family, positions in self.stats['family_position'].items():
            if family not in asr_template['family_position_asr']:
                asr_template['family_position_asr'][family] = {}
            for position, count in positions.items():
                asr_template['family_position_asr'][family][position] = {
                    'count': count,
                    'successful': 0,
                    'asr': 0.0
                }
        
        for technique, positions in self.stats['technique_position'].items():
            if technique not in asr_template['technique_position_asr']:
                asr_template['technique_position_asr'][technique] = {}
            for position, count in positions.items():
                asr_template['technique_position_asr'][technique][position] = {
                    'count': count,
                    'successful': 0,
                    'asr': 0.0
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(asr_template, f, indent=2, ensure_ascii=False)
        
        print(f"✓ ASR calculation template exported: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Analyze IPI attack statistics')
    parser.add_argument('--metadata', 
                       default='IPI_generators/ipi_nfcorpus/ipi_metadata_v2.jsonl',
                       help='Path to IPI metadata JSONL file')
    parser.add_argument('--output-dir', 
                       default='IPI_generators/ipi_nfcorpus',
                       help='Output directory for analysis results')
    parser.add_argument('--json', action='store_true',
                       help='Export detailed statistics to JSON')
    parser.add_argument('--csv', action='store_true',
                       help='Export attack records to CSV')
    parser.add_argument('--asr-template', action='store_true',
                       help='Export ASR calculation template')
    parser.add_argument('--all', action='store_true',
                       help='Export all formats')
    
    args = parser.parse_args()
    
    # If --all, enable all exports
    if args.all:
        args.json = True
        args.csv = True
        args.asr_template = True
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = IPIStatisticsAnalyzer(args.metadata)
    analyzer.load_metadata()
    analyzer.analyze_statistics()
    analyzer.print_summary()
    
    # Export formats
    print("\n" + "="*80)
    print("EXPORTING DATA")
    print("="*80)
    
    if args.json:
        analyzer.export_to_json(output_dir / 'ipi_statistics_detailed_v2.json')
    
    if args.csv:
        analyzer.export_to_csv(output_dir / 'ipi_attacks_v2.csv')
    
    if args.asr_template:
        analyzer.export_asr_template(output_dir / 'asr_calculation_template_v2.json')
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the statistics above")
    print("  2. Use the exported data files for ASR calculations")
    print("  3. Update asr_calculation_template_v2.json with evaluation results")
    print("  4. Calculate ASR: asr = successful_attacks / total_attacks")


if __name__ == "__main__":
    main()

