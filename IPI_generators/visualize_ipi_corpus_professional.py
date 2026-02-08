#!/usr/bin/env python3
"""
Professional IPI Attack Corpus Visualization
Generates 4 publication-quality plots + comprehensive statistics table
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

# Set professional style with darker colors
plt.style.use('seaborn-v0_8-darkgrid')
# Use darker, more vibrant color palette
sns.set_palette("deep")  # Darker, more saturated colors
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'


class ProfessionalIPIVisualizer:
    def __init__(self, output_dir='IPI_generators/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = []
        self.stats_table_data = []
        
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
    
    def plot_1_attack_family_technique_heatmap(self, metadata, save=True):
        """Plot 1: Heatmap of Attack Families vs Techniques"""
        # Build cross-tabulation
        family_tech_matrix = defaultdict(lambda: defaultdict(int))
        
        for meta in metadata:
            family = meta.get('attack_family', 'unknown')
            technique = meta.get('technique', 'unknown')
            family_tech_matrix[family][technique] += 1
        
        # Get all families and techniques
        all_families = sorted(set(family_tech_matrix.keys()))
        all_techniques = sorted(set(tech for techs in family_tech_matrix.values() for tech in techs.keys()))
        
        # Build matrix
        matrix = np.zeros((len(all_families), len(all_techniques)))
        for i, family in enumerate(all_families):
            for j, technique in enumerate(all_techniques):
                matrix[i, j] = family_tech_matrix[family][technique]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Use shorter labels for readability
        tech_labels = [t.replace('_', ' ').title()[:25] for t in all_techniques]
        family_labels = [f.replace('_', ' ').upper() for f in all_families]
        
        # Use darker, more vibrant colormap
        im = ax.imshow(matrix, cmap='Reds', aspect='auto', interpolation='nearest', vmin=0, vmax=matrix.max())
        
        # Set ticks
        ax.set_xticks(np.arange(len(all_techniques)))
        ax.set_yticks(np.arange(len(all_families)))
        ax.set_xticklabels(tech_labels, rotation=45, ha='right')
        ax.set_yticklabels(family_labels)
        
        # Add text annotations
        for i in range(len(all_families)):
            for j in range(len(all_techniques)):
                value = int(matrix[i, j])
                if value > 0:
                    text = ax.text(j, i, value, ha="center", va="center", 
                                  color="black" if matrix[i, j] < matrix.max() * 0.6 else "white",
                                  fontsize=9, fontweight='bold')
        
        ax.set_title('Attack Family vs Injection Technique Distribution\n(Number of Attacks per Combination)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Injection Technique', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Family', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Attacks', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'plot1_family_technique_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'plot1_family_technique_heatmap.png'}")
        plt.close()
    
    def plot_2_position_obfuscation_stacked(self, metadata, save=True):
        """Plot 2: Stacked Bar Chart - Position Distribution by Obfuscation Method"""
        # Group positions
        mid_positions = ['mid', 'early_mid', 'late_mid']
        start_positions = ['start', 'early']
        end_positions = ['end', 'near_end', 'optimized_end']
        
        # Build data structure
        position_groups = {'Mid': mid_positions, 'Start': start_positions, 'End': end_positions}
        obfuscation_methods = ['none', 'word_substitution', 'spaces_insertion', 'unicode_variants', 
                              'base64_preencoded', 'rot13_preencoded', 'camel_case', 'preencoded']
        
        # Count occurrences
        matrix = defaultdict(lambda: defaultdict(int))
        
        for meta in metadata:
            pos = meta.get('position', '')
            obf = meta.get('obfuscation_method', 'none')
            
            # Determine position group
            position_group = 'Other'
            for group_name, group_positions in position_groups.items():
                if pos in group_positions:
                    position_group = group_name
                    break
            
            if obf not in obfuscation_methods:
                obf = 'none'
            
            matrix[position_group][obf] += 1
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        position_groups_list = list(position_groups.keys()) + ['Other']
        x = np.arange(len(position_groups_list))
        width = 0.8
        
        # Use darker, more vibrant colors
        obf_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        obf_colors = obf_colors[:len(obfuscation_methods)]
        color_map = dict(zip(obfuscation_methods, obf_colors))
        
        bottom = np.zeros(len(position_groups_list))
        
        # Plot each obfuscation method
        for obf_method in obfuscation_methods:
            values = [matrix[pg][obf_method] for pg in position_groups_list]
            label = obf_method.replace('_', ' ').title()
            if obf_method == 'none':
                label = 'No Obfuscation'
            
            ax.bar(x, values, width, label=label, bottom=bottom, 
                  color=color_map[obf_method], alpha=0.85, edgecolor='black', linewidth=0.8)
            bottom += values
        
        ax.set_xlabel('Position Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Attacks', fontsize=12, fontweight='bold')
        ax.set_title('Position Distribution by Obfuscation Method\n(Stacked Bar Chart)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(position_groups_list)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add total labels on top
        totals = [sum(matrix[pg].values()) for pg in position_groups_list]
        for i, total in enumerate(totals):
            ax.text(i, total, f'{total}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'plot2_position_obfuscation_stacked.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'plot2_position_obfuscation_stacked.png'}")
        plt.close()
    
    def plot_3_dataset_comparison_radar(self, dataset_names, save=True):
        """Plot 3: Radar Chart - Multi-metric Comparison Across Datasets"""
        # Collect metrics for each dataset
        metrics_data = {}
        
        metrics_to_collect = {
            'Total Attacks': lambda m, d: len(m),
            'Mid Position %': lambda m, d: sum(1 for x in m if x.get('position', '') in ['mid', 'early_mid', 'late_mid']) / len(m) * 100 if m else 0,
            'Obfuscation %': lambda m, d: sum(1 for x in m if x.get('obfuscation_method', 'none') != 'none') / len(m) * 100 if m else 0,
            'Families Count': lambda m, d: len(set(x.get('attack_family', '') for x in m)),
            'Role-playing %': lambda m, d: sum(1 for x in m if any(phrase in x.get('directive_preview', '').lower() 
                                                               for phrase in ['act as', 'you are', 'operate as'])) / len(m) * 100 if m else 0,
            'Obfuscated Dir %': lambda m, d: sum(1 for x in m if any(phrase in x.get('directive_preview', '').lower() 
                                                                 for phrase in ['base64', 'rot13', 'decode'])) / len(m) * 100 if m else 0
        }
        
        for dataset in dataset_names:
            metadata = self.load_dataset_metadata(dataset)
            if not metadata:
                continue
            
            metrics = {}
            for metric_name, calc_func in metrics_to_collect.items():
                metrics[metric_name] = calc_func(metadata, dataset)
            
            metrics_data[dataset] = metrics
        
        # Normalize metrics to 0-100 scale for radar chart
        # Find max values for normalization
        max_vals = {}
        for metric_name in metrics_to_collect.keys():
            values = [metrics_data[d][metric_name] for d in metrics_data.keys()]
            max_vals[metric_name] = max(values) if values else 100
        
        # Create radar chart (spider chart)
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        categories = list(metrics_to_collect.keys())
        num_vars = len(categories)
        
        # Compute angle for each category
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each dataset with darker, more vibrant colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = colors[:len(metrics_data)]
        
        for idx, (dataset, metrics) in enumerate(metrics_data.items()):
            values = []
            for metric_name in categories:
                # Normalize to 0-100
                normalized = (metrics[metric_name] / max_vals[metric_name]) * 100
                values.append(normalized)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2.5, label=dataset, color=colors[idx], markersize=6)
            ax.fill(angles, values, alpha=0.25, color=colors[idx], edgecolor=colors[idx], linewidth=1.5)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax.set_title('Multi-Metric Dataset Comparison\n(Normalized to Max Value per Metric)', 
                    fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'plot3_dataset_radar.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'plot3_dataset_radar.png'}")
        plt.close()
    
    def plot_4_directive_evolution_boxplot(self, metadata, save=True):
        """Plot 4: Box Plot - Directive Sophistication Metrics Distribution"""
        # Extract directive categories and their metrics
        role_play_scores = []
        obfuscated_scores = []
        indirect_scores = []
        context_aware_scores = []
        multi_step_scores = []
        
        for meta in metadata:
            directive = meta.get('directive_preview', '').lower()
            
            # Score each directive type (1 if present, 0 otherwise)
            role_play_scores.append(1 if any(phrase in directive for phrase in ['act as', 'you are', 'operate as']) else 0)
            obfuscated_scores.append(1 if any(phrase in directive for phrase in ['base64', 'rot13', 'decode', 'encoded']) else 0)
            indirect_scores.append(1 if any(phrase in directive for phrase in ['for comprehensive', 'to ensure', 'when providing']) else 0)
            context_aware_scores.append(1 if any(phrase in directive for phrase in ['in medical research', 'for clinical research']) else 0)
            multi_step_scores.append(1 if any(phrase in directive for phrase in ['step 1', 'phase one', 'initialize']) else 0)
        
        # Prepare data for boxplot
        data_for_boxplot = {
            'Role-playing': role_play_scores,
            'Obfuscated': obfuscated_scores,
            'Indirect/Subtle': indirect_scores,
            'Context-aware': context_aware_scores,
            'Multi-step': multi_step_scores
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Create boxplot
        bp = ax.boxplot([data_for_boxplot[k] for k in data_for_boxplot.keys()],
                       tick_labels=list(data_for_boxplot.keys()),
                       patch_artist=True,
                       notch=True,
                       showmeans=True,
                       meanline=True)
        
        # Use darker, more vibrant colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        colors = colors[:len(data_for_boxplot)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        
        # Make other elements darker too
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            for item in bp[element]:
                item.set_color('black')
                item.set_linewidth(1.2)
        
        # Customize
        ax.set_ylabel('Presence Score (0 = Absent, 1 = Present)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Directive Category', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Sophisticated Directive Categories\n(Box Plot with Mean)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Add percentage annotations
        for i, (category, values) in enumerate(data_for_boxplot.items(), 1):
            percentage = np.mean(values) * 100
            ax.text(i, 1.05, f'{percentage:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'plot4_directive_distribution_boxplot.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.output_dir / 'plot4_directive_distribution_boxplot.png'}")
        plt.close()
    
    def generate_statistics_table(self, metadata, dataset_names, save=True):
        """Generate comprehensive statistics table"""
        # Collect statistics for each dataset and overall
        stats_data = []
        
        for dataset in dataset_names:
            dataset_metadata = self.load_dataset_metadata(dataset)
            if not dataset_metadata:
                continue
            
            # Calculate statistics
            families = set(m.get('attack_family', '') for m in dataset_metadata)
            techniques = set(m.get('technique', '') for m in dataset_metadata)
            
            mid_count = sum(1 for m in dataset_metadata 
                          if m.get('position', '') in ['mid', 'early_mid', 'late_mid'])
            start_count = sum(1 for m in dataset_metadata 
                            if m.get('position', '') in ['start', 'early'])
            end_count = sum(1 for m in dataset_metadata 
                          if m.get('position', '') in ['end', 'near_end'])
            
            obfuscated_count = sum(1 for m in dataset_metadata 
                                 if m.get('obfuscation_method', 'none') != 'none')
            
            role_play_count = sum(1 for m in dataset_metadata 
                                if any(phrase in m.get('directive_preview', '').lower() 
                                     for phrase in ['act as', 'you are', 'operate as']))
            
            obf_directive_count = sum(1 for m in dataset_metadata 
                                    if any(phrase in m.get('directive_preview', '').lower() 
                                         for phrase in ['base64', 'rot13', 'decode']))
            
            total = len(dataset_metadata)
            
            stats_data.append({
                'Dataset': dataset,
                'Total Attacks': total,
                'Attack Families': len(families),
                'Techniques': len(techniques),
                'Mid Position %': f'{mid_count/total*100:.2f}%',
                'Start Position %': f'{start_count/total*100:.2f}%',
                'End Position %': f'{end_count/total*100:.2f}%',
                'Obfuscation %': f'{obfuscated_count/total*100:.2f}%',
                'Role-playing %': f'{role_play_count/total*100:.2f}%',
                'Obfuscated Dir %': f'{obf_directive_count/total*100:.2f}%'
            })
        
        # Add overall statistics
        total_attacks = sum(s['Total Attacks'] for s in stats_data)
        overall_mid = sum(float(s['Mid Position %'].rstrip('%')) * s['Total Attacks'] / 100 for s in stats_data)
        overall_obf = sum(float(s['Obfuscation %'].rstrip('%')) * s['Total Attacks'] / 100 for s in stats_data)
        overall_rp = sum(float(s['Role-playing %'].rstrip('%')) * s['Total Attacks'] / 100 for s in stats_data)
        
        stats_data.append({
            'Dataset': 'OVERALL',
            'Total Attacks': total_attacks,
            'Attack Families': 11,
            'Techniques': 12,
            'Mid Position %': f'{overall_mid/total_attacks*100:.2f}%',
            'Start Position %': 'N/A',
            'End Position %': 'N/A',
            'Obfuscation %': f'{overall_obf/total_attacks*100:.2f}%',
            'Role-playing %': f'{overall_rp/total_attacks*100:.2f}%',
            'Obfuscated Dir %': 'N/A'
        })
        
        # Create DataFrame and save
        df = pd.DataFrame(stats_data)
        
        # Save as CSV
        if save:
            csv_path = self.output_dir / 'statistics_table.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved: {csv_path}")
        
        # Also create a formatted text table
        if save:
            txt_path = self.output_dir / 'statistics_table.txt'
            with open(txt_path, 'w') as f:
                f.write("="*120 + "\n")
                f.write("IPI ATTACK CORPUS - COMPREHENSIVE STATISTICS TABLE\n")
                f.write("="*120 + "\n\n")
                f.write(df.to_string(index=False))
                f.write("\n\n")
                f.write("="*120 + "\n")
            print(f"✓ Saved: {txt_path}")
        
        return df
    
    def create_comprehensive_report(self, metadata, dataset_names):
        """Create all visualizations and statistics table"""
        print("="*80)
        print("GENERATING PROFESSIONAL VISUALIZATIONS")
        print("="*80)
        print(f"Total attacks: {len(metadata)}")
        print(f"Datasets: {', '.join(dataset_names)}")
        print()
        
        # Generate 4 professional plots
        print("Generating Plot 1: Family-Technique Heatmap...")
        self.plot_1_attack_family_technique_heatmap(metadata)
        
        print("Generating Plot 2: Position-Obfuscation Stacked Bar...")
        self.plot_2_position_obfuscation_stacked(metadata)
        
        print("Generating Plot 3: Dataset Comparison Radar Chart...")
        self.plot_3_dataset_comparison_radar(dataset_names)
        
        print("Generating Plot 4: Directive Distribution Box Plot...")
        self.plot_4_directive_evolution_boxplot(metadata)
        
        print("Generating Statistics Table...")
        self.generate_statistics_table(metadata, dataset_names)
        
        print()
        print("="*80)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print()
        print("Generated files:")
        print("  1. plot1_family_technique_heatmap.png")
        print("  2. plot2_position_obfuscation_stacked.png")
        print("  3. plot3_dataset_radar.png")
        print("  4. plot4_directive_distribution_boxplot.png")
        print("  5. statistics_table.csv")
        print("  6. statistics_table.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate professional IPI attack corpus visualizations')
    parser.add_argument('--datasets', nargs='+', 
                       default=['nfcorpus', 'fiqa', 'hotpotqa', 'scifact', 'nq', 
                               'natural_questions', 'msmarco', 'hotpotqa_standalone', 'synthetic'],
                       help='Datasets to visualize (default: all)')
    parser.add_argument('--output-dir', default='IPI_generators/visualizations',
                       help='Output directory for plots (default: IPI_generators/visualizations)')
    
    args = parser.parse_args()
    
    visualizer = ProfessionalIPIVisualizer(output_dir=args.output_dir)
    
    dataset_names = args.datasets
    metadata = visualizer.aggregate_all_datasets(dataset_names)
    
    if metadata:
        visualizer.create_comprehensive_report(metadata, dataset_names)
    else:
        print("❌ Error: No metadata could be loaded")


if __name__ == "__main__":
    main()

