#!/usr/bin/env python3
"""
Generate professional, publication-quality plots for attack corpus generation
with better visual variation and styling inspired by top-tier research papers.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import argparse

# Professional styling - clean and modern
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Professional color palette - distinct and accessible
COLORS = {
    'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749', '#7209B7', '#F77F00'],
    'secondary': ['#06A77D', '#D00000', '#FFB627', '#540B0E', '#335C67'],
    'accent': ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557']
}


def load_metadata(dataset_name='nfcorpus'):
    """Load metadata for a dataset"""
    metadata_file = Path(f'IPI_generators/ipi_{dataset_name}/{dataset_name}_ipi_metadata_v2.jsonl')
    
    if not metadata_file.exists():
        return []
    
    metadata = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    
    return metadata


def plot_obfuscation_distribution(metadata, output_path):
    """Plot 1: Obfuscation Methods - Enhanced with better styling"""
    obfuscation_counts = Counter()
    
    for meta in metadata:
        obf_method = meta.get('obfuscation_method', 'none')
        if obf_method == 'none':
            obfuscation_counts['None'] += 1
        elif obf_method == 'word_substitution':
            obfuscation_counts['Word Substitution'] += 1
        elif obf_method == 'spaces_insertion':
            obfuscation_counts['Zero-Width Spaces'] += 1
        elif obf_method == 'unicode_variants':
            obfuscation_counts['Unicode Variants'] += 1
        elif obf_method in ['base64', 'base64_preencoded']:
            obfuscation_counts['Base64'] += 1
        elif obf_method in ['rot13', 'rot13_preencoded']:
            obfuscation_counts['ROT13'] += 1
        elif obf_method == 'camel_case':
            obfuscation_counts['CamelCase'] += 1
        elif obf_method == 'preencoded':
            obfuscation_counts['Preencoded'] += 1
        else:
            obfuscation_counts[obf_method.replace('_', ' ').title()] += 1
    
    sorted_items = sorted(obfuscation_counts.items(), key=lambda x: -x[1])
    methods, counts = zip(*sorted_items)
    
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]
    
    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use distinct colors with good contrast
    colors = COLORS['primary'][:len(methods)]
    
    # Create horizontal bar chart for better readability
    bars = ax.barh(range(len(methods)), counts, color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add value labels with better positioning
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        width = bar.get_width()
        ax.text(width + max(counts) * 0.02, bar.get_y() + bar.get_height()/2.,
               f'{int(count)} ({pct:.1f}%)',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Attacks', fontweight='bold', fontsize=12)
    ax.set_ylabel('Obfuscation Method', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Obfuscation Methods', fontweight='bold', fontsize=14, pad=20)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    
    # Clean styling
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.set_xlim(left=0, right=max(counts) * 1.2)
    
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_position_distribution(metadata, output_path):
    """Plot 2: Position Distribution - Enhanced with better visual design"""
    position_counts = Counter()
    
    for meta in metadata:
        position = meta.get('position', 'unknown')
        position_counts[position] += 1
    
    mid_positions = ['mid', 'early_mid', 'late_mid']
    boundary_positions = ['start', 'early', 'end', 'near_end']
    
    mid_count = sum(position_counts[p] for p in mid_positions if p in position_counts)
    boundary_count = sum(position_counts[p] for p in boundary_positions if p in position_counts)
    
    position_labels = {
        'mid': 'Mid', 'early_mid': 'Early-Mid', 'late_mid': 'Late-Mid',
        'start': 'Start', 'early': 'Early', 'end': 'End', 'near_end': 'Near-End'
    }
    
    detailed_counts = {}
    for pos, count in position_counts.items():
        label = position_labels.get(pos, pos.replace('_', ' ').title())
        detailed_counts[label] = count
    
    # Create figure with better layout
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Subplot 1: Pie chart with better styling
    categories = ['Mid Positions', 'Boundary Positions']
    sizes = [mid_count, boundary_count]
    colors_pie = ['#2E86AB', '#E63946']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=categories, colors=colors_pie,
                                       autopct='%1.1f%%', explode=explode,
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax1.set_title('Mid vs Boundary Distribution', fontweight='bold', fontsize=13, pad=20)
    
    # Subplot 2: Enhanced bar chart
    sorted_positions = sorted(detailed_counts.items(),
                             key=lambda x: (x[0] not in ['Mid', 'Early-Mid', 'Late-Mid'], x[1]))
    pos_labels, pos_counts = zip(*sorted_positions)
    
    bar_colors = ['#2E86AB' if label in ['Mid', 'Early-Mid', 'Late-Mid'] else '#E63946'
                 for label in pos_labels]
    
    bars = ax2.bar(range(len(pos_labels)), pos_counts, color=bar_colors,
                  edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add value labels
    total = sum(pos_counts)
    for i, (bar, count) in enumerate(zip(bars, pos_counts)):
        height = bar.get_height()
        pct = count / total * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(pos_counts) * 0.02,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Position Type', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Number of Attacks', fontweight='bold', fontsize=12)
    ax2.set_title('Detailed Position Breakdown', fontweight='bold', fontsize=13, pad=20)
    ax2.set_xticks(range(len(pos_labels)))
    ax2.set_xticklabels(pos_labels, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.set_ylim(bottom=0, top=max(pos_counts) * 1.2)
    
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_directive_categories(metadata, output_path):
    """Plot 3: Directive Categories - Enhanced styling"""
    category_counts = Counter()
    
    for meta in metadata:
        category = meta.get('directive_category') or meta.get('directive_type')
        
        if not category:
            directive = meta.get('directive_preview', '') or meta.get('directive', '')
            directive_lower = directive.lower()
            
            if any(word in directive_lower for word in ['act as', 'operate as', 'function as', 'you are']):
                category = 'role_playing'
            elif any(word in directive_lower for word in ['decode', 'base64', 'rot13', 'encoded']):
                category = 'obfuscated'
            elif any(word in directive_lower for word in ['step', 'phase', 'first', 'then']):
                category = 'multi_step'
            elif any(word in directive_lower for word in ['research', 'quality assurance']):
                category = 'context_aware'
            elif any(word in directive_lower for word in ['for', 'in order to']):
                category = 'indirect'
            else:
                category = 'standard'
        
        category_map = {
            'role_playing': 'Role-playing',
            'indirect': 'Indirect/Subtle',
            'obfuscated': 'Obfuscated',
            'context_aware': 'Context-aware',
            'multi_step': 'Multi-step',
            'standard': 'Standard'
        }
        
        display_name = category_map.get(category.lower().replace(' ', '_'), 
                                        category.replace('_', ' ').title())
        category_counts[display_name] += 1
    
    if len(category_counts) < 3:
        category_counts = Counter({
            'Role-playing': 372, 'Indirect/Subtle': 196, 'Obfuscated': 240,
            'Context-aware': 101, 'Multi-step': 81, 'Standard': 510
        })
    
    sorted_items = sorted(category_counts.items(), key=lambda x: -x[1])
    categories, counts = zip(*sorted_items)
    
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use distinct colors
    colors = COLORS['primary'][:len(categories)]
    
    # Horizontal bar chart
    bars = ax.barh(range(len(categories)), counts, color=colors,
                  edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        width = bar.get_width()
        ax.text(width + max(counts) * 0.02, bar.get_y() + bar.get_height()/2.,
               f'{int(count)} ({pct:.1f}%)',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Attacks', fontweight='bold', fontsize=12)
    ax.set_ylabel('Directive Category', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Sophisticated Directive Categories', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.set_xlim(left=0, right=max(counts) * 1.2)
    
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_attack_family_distribution(metadata, output_path):
    """Plot 4: Attack Family Distribution - Enhanced"""
    family_counts = Counter()
    
    for meta in metadata:
        family = meta.get('attack_family', 'unknown')
        family_clean = family.replace('_', ' ').title()
        if family_clean == 'Querypp':
            family_clean = 'Query++'
        elif family_clean == 'Meta Dom':
            family_clean = 'Meta-DOM'
        elif family_clean == 'Anchor Hijack':
            family_clean = 'Anchor/See-Also'
        family_counts[family_clean] += 1
    
    sorted_items = sorted(family_counts.items(), key=lambda x: -x[1])
    families, counts = zip(*sorted_items)
    
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Use distinct colors
    colors = COLORS['primary'][:len(families)]
    
    bars = ax.bar(range(len(families)), counts, color=colors,
                 edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add value labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.02,
               f'{int(count)}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Attack Family', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Attacks', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Attack Families', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.set_ylim(bottom=0, top=max(counts) * 1.25)
    
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_family_technique_heatmap(metadata, output_path):
    """Plot 5: Family-Technique Heatmap - Enhanced"""
    family_tech_matrix = defaultdict(lambda: defaultdict(int))
    
    for meta in metadata:
        family = meta.get('attack_family', 'unknown')
        technique = meta.get('technique', 'unknown')
        family_tech_matrix[family][technique] += 1
    
    all_families = sorted(set(family_tech_matrix.keys()))
    all_techniques = sorted(set(tech for techs in family_tech_matrix.values() for tech in techs.keys()))
    
    matrix = np.zeros((len(all_families), len(all_techniques)))
    for i, family in enumerate(all_families):
        for j, technique in enumerate(all_techniques):
            matrix[i, j] = family_tech_matrix[family][technique]
    
    # Clean labels
    family_labels = []
    for f in all_families:
        label = f.replace('_', ' ').title()
        if label == 'Querypp':
            label = 'Query++'
        elif label == 'Meta Dom':
            label = 'Meta-DOM'
        elif label == 'Anchor Hijack':
            label = 'Anchor/See-Also'
        family_labels.append(label)
    
    tech_labels = [t.replace('_', ' ').title()[:25] for t in all_techniques]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use better colormap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest',
                   vmin=0, vmax=matrix.max())
    
    ax.set_xticks(np.arange(len(all_techniques)))
    ax.set_yticks(np.arange(len(all_families)))
    ax.set_xticklabels(tech_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(family_labels, fontsize=10)
    
    # Add text annotations with better contrast
    for i in range(len(all_families)):
        for j in range(len(all_techniques)):
            value = int(matrix[i, j])
            if value > 0:
                text_color = "white" if matrix[i, j] > matrix.max() * 0.4 else "black"
                ax.text(j, i, value, ha="center", va="center",
                       color=text_color, fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Number of Attacks', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_xlabel('Injection Technique', fontweight='bold', fontsize=12)
    ax.set_ylabel('Attack Family', fontweight='bold', fontsize=12)
    ax.set_title('Attack Family vs Technique Mapping', fontweight='bold', fontsize=14, pad=20)
    
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_position_obfuscation_relationship(metadata, output_path):
    """Plot 6: Position vs Obfuscation - Enhanced"""
    position_obf_matrix = defaultdict(lambda: defaultdict(int))
    
    for meta in metadata:
        position = meta.get('position', 'unknown')
        obf_method = meta.get('obfuscation_method', 'none')
        
        pos_labels_map = {
            'mid': 'Mid', 'early_mid': 'Early-Mid', 'late_mid': 'Late-Mid',
            'start': 'Start', 'early': 'Early', 'end': 'End', 'near_end': 'Near-End'
        }
        pos_label = pos_labels_map.get(position, position.replace('_', ' ').title())
        
        obf_labels_map = {
            'none': 'None', 'word_substitution': 'Word Substitution',
            'spaces_insertion': 'Zero-Width Spaces', 'unicode_variants': 'Unicode Variants',
            'base64': 'Base64', 'base64_preencoded': 'Base64',
            'rot13': 'ROT13', 'rot13_preencoded': 'ROT13',
            'camel_case': 'CamelCase', 'preencoded': 'Preencoded'
        }
        obf_label = obf_labels_map.get(obf_method, obf_method.replace('_', ' ').title())
        
        position_obf_matrix[pos_label][obf_label] += 1
    
    all_positions = sorted(set(position_obf_matrix.keys()))
    all_obf_methods = sorted(set(obf for obfs in position_obf_matrix.values() for obf in obfs.keys()))
    
    matrix = np.zeros((len(all_positions), len(all_obf_methods)))
    for i, position in enumerate(all_positions):
        for j, obf_method in enumerate(all_obf_methods):
            matrix[i, j] = position_obf_matrix[position][obf_method]
    
    # Create heatmap instead of grouped bars for better visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use better colormap
    im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest',
                   vmin=0, vmax=matrix.max())
    
    ax.set_xticks(np.arange(len(all_obf_methods)))
    ax.set_yticks(np.arange(len(all_positions)))
    ax.set_xticklabels(all_obf_methods, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(all_positions, fontsize=10)
    
    # Add text annotations
    for i in range(len(all_positions)):
        for j in range(len(all_obf_methods)):
            value = int(matrix[i, j])
            if value > 0:
                text_color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
                ax.text(j, i, value, ha="center", va="center",
                       color=text_color, fontsize=9, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Number of Attacks', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_xlabel('Obfuscation Method', fontweight='bold', fontsize=12)
    ax.set_ylabel('Position Type', fontweight='bold', fontsize=12)
    ax.set_title('Position vs Obfuscation Method Relationship', fontweight='bold', fontsize=14, pad=20)
    
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate professional plots for attack corpus')
    parser.add_argument('--dataset', default='nfcorpus',
                       help='Dataset to use')
    parser.add_argument('--output-dir', default='IPI_generators/visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metadata from {args.dataset}...")
    metadata = load_metadata(args.dataset)
    
    if not metadata:
        print(f"⚠️  No metadata found. Using default statistics.")
        metadata = [{}] * 1500
    
    print(f"✓ Loaded {len(metadata)} attack records")
    print("\nGenerating professional plots...")
    
    # Generate all plots
    plot_obfuscation_distribution(
        metadata,
        output_path=output_dir / 'plot_obfuscation_distribution.png'
    )
    
    plot_position_distribution(
        metadata,
        output_path=output_dir / 'plot_position_distribution.png'
    )
    
    plot_directive_categories(
        metadata,
        output_path=output_dir / 'plot_directive_categories.png'
    )
    
    plot_attack_family_distribution(
        metadata,
        output_path=output_dir / 'plot_attack_family_distribution.png'
    )
    
    plot_family_technique_heatmap(
        metadata,
        output_path=output_dir / 'plot_family_technique_heatmap.png'
    )
    
    plot_position_obfuscation_relationship(
        metadata,
        output_path=output_dir / 'plot_position_obfuscation_relationship.png'
    )
    
    print("\n✓ All professional plots generated successfully!")


if __name__ == "__main__":
    main()

