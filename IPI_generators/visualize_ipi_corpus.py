#!/usr/bin/env python3
"""
Visualize IPI Attack Corpus Statistics
Generates 4 professional plots and comprehensive statistics table
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import argparse

# Set professional style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class IPICorpusVisualizer:
    def __init__(self, output_dir='IPI_generators/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = []
        self.data = defaultdict(lambda: defaultdict(int))
        
    def load_dataset_metadata(self, dataset_name):
        """Load metadata for a dataset"""
        metadata_file = Path(f'IPI_generators/ipi_{dataset_name}/{dataset_name}_ipi_metadata_v2.jsonl')
        
        if not metadata_file.exists():
            print(f"⚠️  Metadata file not found: {metadata_file}")
            return None
        
        metadata = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        
        return metadata
    
    def aggregate_all_datasets(self, dataset_names):
        """Aggregate data from all datasets"""
        all_metadata = []
        
        for dataset in dataset_names:
            metadata = self.load_dataset_metadata(dataset)
            if metadata:
                self.datasets.append(dataset)
                all_metadata.extend(metadata)
                print(f"✓ Loaded {len(metadata)} attacks from {dataset}")
        
        return all_metadata
    
    def plot_attack_family_distribution(self, metadata, save=True):
        """Plot distribution of attack families"""
        families = Counter()
        for meta in metadata:
            families[meta.get('attack_family', 'unknown')] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        families_sorted = dict(sorted(families.items(), key=lambda x: -x[1]))
        colors = plt.cm.Set3(np.linspace(0, 1, len(families_sorted)))
        
        bars = ax1.bar(families_sorted.keys(), families_sorted.values(), color=colors)
        ax1.set_xlabel('Attack Family', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Attacks', fontsize=12, fontweight='bold')
        ax1.set_title('Attack Family Distribution (All Corpora)', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        # Pie chart
        explode = [0.05] * len(families_sorted)
        wedges, texts, autotexts = ax2.pie(families_sorted.values(), 
                                           labels=families_sorted.keys(),
                                           autopct='%1.1f%%',
                                           explode=explode,
                                           colors=colors,
                                           startangle=90)
        
        ax2.set_title('Attack Family Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'attack_family_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'attack_family_distribution.png'}")
        plt.close()
    
    def plot_position_distribution(self, metadata, save=True):
        """Plot position distribution"""
        positions = Counter()
        for meta in metadata:
            pos = meta.get('position', 'unknown')
            if pos:
                positions[pos] += 1
        
        # Group positions
        mid_positions = ['mid', 'early_mid', 'late_mid']
        start_positions = ['start', 'early']
        end_positions = ['end', 'near_end', 'optimized_end']
        
        mid_count = sum(positions.get(p, 0) for p in mid_positions)
        start_count = sum(positions.get(p, 0) for p in start_positions)
        end_count = sum(positions.get(p, 0) for p in end_positions)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Grouped bar chart
        grouped = {
            'Mid Positions': mid_count,
            'Start Positions': start_count,
            'End Positions': end_count
        }
        
        colors_grouped = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax1.bar(grouped.keys(), grouped.values(), color=colors_grouped, alpha=0.8)
        ax1.set_xlabel('Position Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Attacks', fontsize=12, fontweight='bold')
        ax1.set_title('Position Distribution (Grouped)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total = sum(grouped.values())
        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Detailed position breakdown
        positions_filtered = {k: v for k, v in positions.items() if k != 'unknown'}
        positions_sorted = dict(sorted(positions_filtered.items(), key=lambda x: -x[1]))
        
        colors_detailed = plt.cm.viridis(np.linspace(0, 1, len(positions_sorted)))
        bars2 = ax2.barh(list(positions_sorted.keys()), list(positions_sorted.values()), 
                        color=colors_detailed, alpha=0.8)
        ax2.set_xlabel('Number of Attacks', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Position Type', fontsize=12, fontweight='bold')
        ax2.set_title('Detailed Position Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {int(width)}',
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'position_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'position_distribution.png'}")
        plt.close()
    
    def plot_obfuscation_distribution(self, metadata, save=True):
        """Plot obfuscation methods distribution"""
        obfuscation = Counter()
        for meta in metadata:
            obf_method = meta.get('obfuscation_method', 'none')
            obfuscation[obf_method] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        obf_sorted = dict(sorted(obfuscation.items(), key=lambda x: -x[1]))
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(obf_sorted)))
        
        bars = ax1.bar(obf_sorted.keys(), obf_sorted.values(), color=colors, alpha=0.8)
        ax1.set_xlabel('Obfuscation Method', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Attacks', fontsize=12, fontweight='bold')
        ax1.set_title('Obfuscation Methods Distribution', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        total = sum(obf_sorted.values())
        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(obf_sorted.values(),
                                          labels=obf_sorted.keys(),
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax2.set_title('Obfuscation Methods (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'obfuscation_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'obfuscation_distribution.png'}")
        plt.close()
    
    def plot_directive_categories(self, metadata, save=True):
        """Plot sophisticated directive categories"""
        role_play = 0
        indirect = 0
        obfuscated = 0
        context_aware = 0
        multi_step = 0
        other = 0
        
        for meta in metadata:
            directive = meta.get('directive_preview', '').lower()
            found = False
            
            if any(phrase in directive for phrase in ['act as', 'you are', 'operate as', 'function as']):
                role_play += 1
                found = True
            if any(phrase in directive for phrase in ['for comprehensive', 'to ensure', 'when providing', 'in order to']):
                indirect += 1
                found = True
            if any(phrase in directive for phrase in ['base64', 'rot13', 'decode', 'encoded']):
                obfuscated += 1
                found = True
            if any(phrase in directive for phrase in ['step 1', 'phase one', 'initialize', 'then include']):
                multi_step += 1
                found = True
            if any(phrase in directive for phrase in ['in medical research', 'for clinical research', 'research mode']):
                context_aware += 1
                found = True
            
            if not found:
                other += 1
        
        categories = {
            'Role-playing': role_play,
            'Obfuscated': obfuscated,
            'Indirect/Subtle': indirect,
            'Context-aware': context_aware,
            'Multi-step': multi_step,
            'Other': other
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
        bars = ax1.bar(categories.keys(), categories.values(), color=colors, alpha=0.8)
        ax1.set_xlabel('Directive Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Attacks', fontsize=12, fontweight='bold')
        ax1.set_title('Sophisticated Directive Categories Distribution', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        total = sum(categories.values())
        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(categories.values(),
                                          labels=categories.keys(),
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax2.set_title('Directive Categories (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'directive_categories.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'directive_categories.png'}")
        plt.close()
    
    def plot_dataset_comparison(self, dataset_names, save=True):
        """Compare attack distributions across datasets"""
        datasets_data = {}
        
        for dataset in dataset_names:
            metadata = self.load_dataset_metadata(dataset)
            if not metadata:
                continue
            
            families = Counter()
            positions_grouped = {'Mid': 0, 'Start': 0, 'End': 0}
            obfuscation_rate = 0
            
            for meta in metadata:
                # Count families
                families[meta.get('attack_family', 'unknown')] += 1
                
                # Group positions
                pos = meta.get('position', '')
                if pos in ['mid', 'early_mid', 'late_mid']:
                    positions_grouped['Mid'] += 1
                elif pos in ['start', 'early']:
                    positions_grouped['Start'] += 1
                elif pos in ['end', 'near_end']:
                    positions_grouped['End'] += 1
                
                # Obfuscation rate
                if meta.get('obfuscation_method', 'none') != 'none':
                    obfuscation_rate += 1
            
            total = len(metadata)
            datasets_data[dataset] = {
                'total': total,
                'families': len(families),
                'mid_pct': (positions_grouped['Mid'] / total * 100) if total > 0 else 0,
                'obfuscation_pct': (obfuscation_rate / total * 100) if total > 0 else 0
            }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        datasets_list = list(datasets_data.keys())
        
        # Total attacks per dataset
        totals = [datasets_data[d]['total'] for d in datasets_list]
        axes[0, 0].bar(datasets_list, totals, color='steelblue', alpha=0.8)
        axes[0, 0].set_title('Total Attacks per Dataset', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Attacks', fontsize=11)
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Mid position percentage
        mid_pcts = [datasets_data[d]['mid_pct'] for d in datasets_list]
        axes[0, 1].bar(datasets_list, mid_pcts, color='green', alpha=0.8)
        axes[0, 1].set_title('Mid Position Percentage per Dataset', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Percentage (%)', fontsize=11)
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].axhline(y=77, color='r', linestyle='--', label='Target (77%)')
        axes[0, 1].legend()
        
        # Obfuscation rate
        obf_pcts = [datasets_data[d]['obfuscation_pct'] for d in datasets_list]
        axes[1, 0].bar(datasets_list, obf_pcts, color='orange', alpha=0.8)
        axes[1, 0].set_title('Obfuscation Rate per Dataset', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=11)
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=50, color='r', linestyle='--', label='Target (50%)')
        axes[1, 0].legend()
        
        # Number of attack families
        family_counts = [datasets_data[d]['families'] for d in datasets_list]
        axes[1, 1].bar(datasets_list, family_counts, color='purple', alpha=0.8)
        axes[1, 1].set_title('Number of Attack Families per Dataset', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Families', fontsize=11)
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].axhline(y=11, color='r', linestyle='--', label='Target (11)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'dataset_comparison.png'}")
        plt.close()
    
    def plot_technique_distribution(self, metadata, save=True):
        """Plot injection technique distribution"""
        techniques = Counter()
        for meta in metadata:
            technique = meta.get('technique', 'unknown')
            techniques[technique] += 1
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        tech_sorted = dict(sorted(techniques.items(), key=lambda x: -x[1]))
        colors = plt.cm.tab20(np.linspace(0, 1, len(tech_sorted)))
        
        bars = ax.barh(list(tech_sorted.keys()), list(tech_sorted.values()), color=colors, alpha=0.8)
        ax.set_xlabel('Number of Attacks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Injection Technique', fontsize=12, fontweight='bold')
        ax.set_title('Injection Technique Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {int(width)}',
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'technique_distribution.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'technique_distribution.png'}")
        plt.close()
    
    def create_comprehensive_report(self, metadata, dataset_names):
        """Create all visualizations"""
        print("="*80)
        print("GENERATING VISUALIZATIONS FOR IPI ATTACK CORPUS")
        print("="*80)
        print(f"Total attacks: {len(metadata)}")
        print(f"Datasets: {', '.join(dataset_names)}")
        print()
        
        # Generate all plots
        self.plot_attack_family_distribution(metadata)
        self.plot_position_distribution(metadata)
        self.plot_obfuscation_distribution(metadata)
        self.plot_directive_categories(metadata)
        self.plot_technique_distribution(metadata)
        self.plot_dataset_comparison(dataset_names)
        
        print()
        print("="*80)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print()
        print("Generated plots:")
        print("  1. attack_family_distribution.png")
        print("  2. position_distribution.png")
        print("  3. obfuscation_distribution.png")
        print("  4. directive_categories.png")
        print("  5. technique_distribution.png")
        print("  6. dataset_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize IPI attack corpus statistics')
    parser.add_argument('--datasets', nargs='+', 
                       default=['nfcorpus', 'fiqa', 'hotpotqa', 'scifact', 'nq', 
                               'natural_questions', 'msmarco', 'hotpotqa_standalone', 'synthetic'],
                       help='Datasets to visualize (default: all)')
    parser.add_argument('--output-dir', default='IPI_generators/visualizations',
                       help='Output directory for plots (default: IPI_generators/visualizations)')
    parser.add_argument('--dataset', type=str,
                       help='Visualize single dataset only')
    
    args = parser.parse_args()
    
    visualizer = IPICorpusVisualizer(output_dir=args.output_dir)
    
    if args.dataset:
        # Single dataset mode
        dataset_names = [args.dataset]
        metadata = visualizer.load_dataset_metadata(args.dataset)
        if metadata:
            print(f"Visualizing single dataset: {args.dataset}")
            visualizer.create_comprehensive_report(metadata, dataset_names)
        else:
            print(f"❌ Error: Could not load metadata for {args.dataset}")
    else:
        # All datasets mode
        dataset_names = args.datasets
        metadata = visualizer.aggregate_all_datasets(dataset_names)
        if metadata:
            visualizer.create_comprehensive_report(metadata, dataset_names)
        else:
            print("❌ Error: No metadata could be loaded")


if __name__ == "__main__":
    main()

