#!/usr/bin/env python3
"""
Generate essential retriever analysis plots with fixed legend positioning
Fixes overlapping legends in scifact and fiqa plots
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# Retriever colors and labels
RETRIEVER_CONFIG = {
    'bm25': {'color': '#1f77b4', 'label': 'BM25', 'marker': 'o'},
    'dense': {'color': '#ff7f0e', 'label': 'Dense', 'marker': 's'},
    'splade': {'color': '#2ca02c', 'label': 'SPLADE++', 'marker': '^'},
    'hybrid_0.2': {'color': '#d62728', 'label': 'Hybrid (α=0.2)', 'marker': 'D'},
    'hybrid_0.5': {'color': '#9467bd', 'label': 'Hybrid (α=0.5)', 'marker': 'v'},
    'hybrid_0.8': {'color': '#8c564b', 'label': 'Hybrid (α=0.8)', 'marker': '<'},
}

def load_retriever_data(corpus_name: str) -> dict:
    """Load retriever comparison data for a corpus"""
    result_file = Path(f'evaluation/comparative_results/{corpus_name}_retriever_comparison.json')
    if not result_file.exists():
        raise FileNotFoundError(f"Results file not found: {result_file}")
    
    with open(result_file, 'r') as f:
        return json.load(f)

def plot_essential_comparison(corpus_name: str, data: dict, output_path: Path):
    """Generate essential retriever analysis plot with fixed legends"""
    
    # Extract data for each retriever
    retrievers = []
    er_data = {}  # retriever -> {k: ER@k}
    er_top1 = {}  # retriever -> ER@1
    
    for retriever_key, retriever_data in data.items():
        if 'stage3_retrieval' not in retriever_data:
            continue
        
        stage3 = retriever_data['stage3_retrieval']
        poison_presence = stage3.get('poison_presence_top_k', {})
        
        if not poison_presence:
            continue
        
        retrievers.append(retriever_key)
        er_data[retriever_key] = {
            int(k): float(v) for k, v in poison_presence.items()
        }
        er_top1[retriever_key] = float(poison_presence.get('1', 0.0))
    
    if not retrievers:
        print(f"Warning: No retriever data found for {corpus_name}")
        return
    
    # Create figure with better spacing for legends - adjust spacing based on corpus
    if corpus_name == 'fiqa':
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.4, 
                         left=0.08, right=0.95, top=0.93, bottom=0.08)
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.4, 
                             left=0.08, right=0.95, top=0.93, bottom=0.09)
    
    # Panel 1: Top-k Vulnerability Trends (Line plot)
    ax1 = fig.add_subplot(gs[0, :2])
    
    k_values = sorted(set(k for er in er_data.values() for k in er.keys()))
    
    for retriever_key in retrievers:
        config = RETRIEVER_CONFIG.get(retriever_key, {
            'color': '#808080', 
            'label': retriever_key.replace('_', ' ').title(),
            'marker': 'o'
        })
        
        er_values = [er_data[retriever_key].get(k, 0.0) for k in k_values]
        ax1.plot(k_values, er_values, 
                color=config['color'], 
                marker=config['marker'],
                label=config['label'],
                linewidth=2.5,
                markersize=8,
                markeredgewidth=1.5)
    
    ax1.set_xlabel('Top-$k$', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Exposure Rate (ER@$k$) [\%]', fontsize=13, fontweight='bold')
    ax1.set_title('Top-$k$ Vulnerability Trends', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 105])
    
    # For fiqa, place legend inside the plot; for others, place below
    if corpus_name == 'fiqa':
        # Reset plot position to normal for fiqa
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width, box.height])
        # Place legend inside plot in upper right corner
        legend1 = ax1.legend(loc='upper right', 
                            fontsize=9.5, frameon=True, fancybox=True, shadow=True,
                            ncol=1, columnspacing=0.8, handlelength=1.5, handletextpad=0.5,
                            framealpha=0.95, borderpad=0.8)
    else:
        # Adjust plot position to leave room at bottom for legend (for other corpora)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.85])
        # Place legend below the plot - horizontal layout with proper spacing
        legend1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                            fontsize=9.5, frameon=True, fancybox=True, shadow=True,
                            ncol=3, columnspacing=2.0, handlelength=1.3, handletextpad=0.5,
                            framealpha=0.95)
    legend1.get_frame().set_facecolor('white')
    
    # Panel 2: Top-1 Vulnerability Comparison (Bar chart)
    ax2 = fig.add_subplot(gs[0, 2])
    
    retriever_labels = []
    er1_values = []
    colors = []
    
    for retriever_key in retrievers:
        config = RETRIEVER_CONFIG.get(retriever_key, {
            'color': '#808080',
            'label': retriever_key.replace('_', ' ').title()
        })
        retriever_labels.append(config['label'])
        er1_values.append(er_top1[retriever_key])
        colors.append(config['color'])
    
    bars = ax2.barh(range(len(retriever_labels)), er1_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(retriever_labels)))
    ax2.set_yticklabels(retriever_labels, fontsize=9.5)
    ax2.set_xlabel('ER@1 [\%]', fontsize=12, fontweight='bold')
    ax2.set_title('Top-1 Vulnerability', fontsize=13, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Calculate xlim with enough space for text labels to avoid overlap
    max_val = max(er1_values) if er1_values else 1.0
    x_max = max_val * 1.25  # Increased from 1.15 to give more space
    ax2.set_xlim([0, x_max])
    
    # Add value labels on bars with better spacing to avoid overlap
    for i, (bar, val) in enumerate(zip(bars, er1_values)):
        width = bar.get_width()
        # Position text inside bar if bar is wide enough, otherwise outside
        if width > max_val * 0.1:  # If bar is wide enough
            text_x = width * 0.95  # Inside the bar
            text_color = 'white'
            ha = 'right'
        else:  # If bar is narrow, place text outside
            text_x = width + max_val * 0.03
            text_color = 'black'
            ha = 'left'
        ax2.text(text_x, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}\%', ha=ha, va='center', fontsize=8.5, 
                fontweight='bold', color=text_color)
    
    # Panel 3: Heatmap of Vulnerability Across k-values
    ax3 = fig.add_subplot(gs[1, :])
    
    # Prepare heatmap data
    heatmap_data = []
    retriever_names_for_heatmap = []
    
    for retriever_key in retrievers:
        config = RETRIEVER_CONFIG.get(retriever_key, {
            'label': retriever_key.replace('_', ' ').title()
        })
        retriever_names_for_heatmap.append(config['label'])
        row_data = [er_data[retriever_key].get(k, 0.0) for k in k_values]
        heatmap_data.append(row_data)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax3.set_xticks(np.arange(len(k_values)))
    ax3.set_xticklabels([f'Top-{k}' for k in k_values], fontsize=10.5)
    ax3.set_yticks(np.arange(len(retriever_names_for_heatmap)))
    ax3.set_yticklabels(retriever_names_for_heatmap, fontsize=10)
    
    # Add text annotations
    for i in range(len(retriever_names_for_heatmap)):
        for j in range(len(k_values)):
            value = heatmap_data[i, j]
            text_color = 'white' if value > 50 else 'black'
            ax3.text(j, i, f'{value:.1f}\%', ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')
    
    ax3.set_title('Vulnerability Heatmap Across $k$-values', fontsize=14, fontweight='bold', pad=15)
    
    # Add colorbar with better positioning
    cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Exposure Rate [\%]', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Main title
    fig.suptitle(f'Essential Retriever Vulnerability Analysis: {corpus_name.upper()}', 
                fontweight='bold', fontsize=16, y=0.98)
    
    # Save with high DPI - ensure legend is included
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    
    print(f"✅ Generated: {output_path}")

def main():
    """Generate plots for all corpora"""
    corpora = ['scifact', 'fiqa', 'nfcorpus']
    output_dir = Path('evaluation/visualizations')
    
    for corpus_name in corpora:
        try:
            data = load_retriever_data(corpus_name)
            output_path = output_dir / f'essential_retriever_analysis_{corpus_name}.png'
            plot_essential_comparison(corpus_name, data, output_path)
        except Exception as e:
            print(f"❌ Error processing {corpus_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

