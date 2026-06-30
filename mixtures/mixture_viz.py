"""
mixture_viz.py

Visualization utilities for analyzing mixture optimization results:
- Property space scatter (Tc, Pc) with pure vs mixture markers
- Composition analysis (which pairs are selected?)
- Efficiency comparison (pure vs mixture performance)
- Edge coverage heatmap
- Optimization trajectory in property space

Usage:
    from mixture_viz import MixtureVisualizer
    
    viz = MixtureVisualizer(results_csv="scbo_results.csv")
    viz.plot_property_space()
    viz.plot_composition_histogram()
    viz.plot_efficiency_comparison()
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class MixtureVisualizer:
    """Visualization suite for mixture optimization results."""
    
    def __init__(self, results_csv: str, sequence_csv: Optional[str] = None):
        """
        Args:
            results_csv: Path to SCBO results CSV
            sequence_csv: Optional sequence file for trajectory plotting
        """
        self.df = pd.read_csv(results_csv)
        self.sequence_df = pd.read_csv(sequence_csv) if sequence_csv else None
        
        # Parse mixture strings to extract components
        self._parse_mixtures()
    
    def _parse_mixtures(self):
        """Extract component info from mixture strings."""
        is_mixture = []
        comp1_list = []
        comp2_list = []
        x1_list = []
        
        for idx, row in self.df.iterrows():
            mixture_str = row.get('mixture', row.get('fluid', ''))
            
            if '[' in mixture_str and '&' in mixture_str:
                # Mixture: "R32[0.70]&R125[0.30]"
                parts = mixture_str.split('&')
                c1_part = parts[0].split('[')
                c2_part = parts[1].split('[')
                
                comp1 = c1_part[0]
                comp2 = c2_part[0]
                x1 = float(c1_part[1].rstrip(']'))
                
                is_mixture.append(True)
                comp1_list.append(comp1)
                comp2_list.append(comp2)
                x1_list.append(x1)
            else:
                # Pure fluid
                is_mixture.append(False)
                comp1_list.append(mixture_str)
                comp2_list.append('')
                x1_list.append(1.0)
        
        self.df['is_mixture'] = is_mixture
        self.df['comp1'] = comp1_list
        self.df['comp2'] = comp2_list
        self.df['x1'] = x1_list
    
    def plot_property_space(self, save_path: Optional[str] = None):
        """
        Scatter plot in (Tc, Pc) space with pure/mixture distinction.
        
        Args:
            save_path: If provided, save figure to this path
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Separate pure and mixture
        df_pure = self.df[~self.df['is_mixture']]
        df_mix = self.df[self.df['is_mixture']]
        
        # Check for property columns
        tc_col = 'Tc_mix[K]' if 'Tc_mix[K]' in self.df.columns else 'Tcrit[K]'
        pc_col = 'Pc_mix[Pa]' if 'Pc_mix[Pa]' in self.df.columns else 'Pcrit[Pa]'
        
        if tc_col not in self.df.columns or pc_col not in self.df.columns:
            print("Warning: Property columns not found. Available columns:", self.df.columns.tolist())
            return
        
        # Plot pure fluids
        if len(df_pure) > 0:
            scatter1 = ax.scatter(
                df_pure[tc_col], df_pure[pc_col] / 1e6,
                c=df_pure['eta_best'], cmap='viridis',
                s=100, marker='o', edgecolors='k', linewidths=1.5,
                label='Pure fluids', vmin=0, vmax=0.1, alpha=0.8
            )
        
        # Plot mixtures
        if len(df_mix) > 0:
            scatter2 = ax.scatter(
                df_mix[tc_col], df_mix[pc_col] / 1e6,
                c=df_mix['eta_best'], cmap='viridis',
                s=150, marker='D', edgecolors='k', linewidths=1.5,
                label='Binary mixtures', vmin=0, vmax=0.1, alpha=0.8
            )
        
        # Colorbar
        cbar = plt.colorbar(scatter1 if len(df_pure) > 0 else scatter2, ax=ax)
        cbar.set_label('Efficiency η', fontsize=12)
        
        ax.set_xlabel('Critical Temperature [K]', fontsize=12)
        ax.set_ylabel('Critical Pressure [MPa]', fontsize=12)
        ax.set_title('Property Space: Pure Fluids vs Binary Mixtures', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_composition_histogram(self, save_path: Optional[str] = None):
        """
        Histogram of composition ratios for mixtures.
        
        Args:
            save_path: If provided, save figure to this path
        """
        df_mix = self.df[self.df['is_mixture']]
        
        if len(df_mix) == 0:
            print("No mixtures found in results.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Composition histogram
        axes[0].hist(df_mix['x1'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='50:50')
        axes[0].axvline(0.05, color='orange', linestyle=':', linewidth=1.5, label='Min constraint')
        axes[0].axvline(0.95, color='orange', linestyle=':', linewidth=1.5)
        axes[0].set_xlabel('Mole Fraction of Component 1 (x₁)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Mixture Compositions', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Efficiency vs composition
        axes[1].scatter(df_mix['x1'], df_mix['eta_best'], c='steelblue', s=80, alpha=0.6, edgecolors='k')
        axes[1].set_xlabel('Mole Fraction of Component 1 (x₁)', fontsize=12)
        axes[1].set_ylabel('Efficiency η', fontsize=12)
        axes[1].set_title('Efficiency vs Composition', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_efficiency_comparison(self, save_path: Optional[str] = None):
        """
        Box plot comparing pure fluid vs mixture efficiency distributions.
        
        Args:
            save_path: If provided, save figure to this path
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter feasible results (eta >= 0)
        df_feas = self.df[self.df['eta_best'] >= 0].copy()
        df_feas['Type'] = df_feas['is_mixture'].map({True: 'Mixture', False: 'Pure'})
        
        if len(df_feas) == 0:
            print("No feasible results found.")
            return
        
        # Box plot
        df_feas.boxplot(column='eta_best', by='Type', ax=axes[0], patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        axes[0].set_xlabel('Fluid Type', fontsize=12)
        axes[0].set_ylabel('Efficiency η', fontsize=12)
        axes[0].set_title('Efficiency Distribution: Pure vs Mixture', fontsize=13, fontweight='bold')
        axes[0].get_figure().suptitle('')  # Remove default title
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Violin plot
        parts = axes[1].violinplot(
            [df_feas[~df_feas['is_mixture']]['eta_best'].values,
             df_feas[df_feas['is_mixture']]['eta_best'].values],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
        )
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(['Pure', 'Mixture'])
        axes[1].set_xlabel('Fluid Type', fontsize=12)
        axes[1].set_ylabel('Efficiency η', fontsize=12)
        axes[1].set_title('Efficiency Distribution (Violin Plot)', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print("\n=== Efficiency Statistics ===")
        print(df_feas.groupby('Type')['eta_best'].describe())
        
        # Best performers
        print("\n=== Top 5 Performers ===")
        top5 = df_feas.nlargest(5, 'eta_best')[['mixture' if 'mixture' in df_feas.columns else 'fluid', 'eta_best', 'Type']]
        print(top5.to_string(index=False))
    
    def plot_component_pair_heatmap(self, save_path: Optional[str] = None):
        """
        Heatmap showing which component pairs were explored.
        
        Args:
            save_path: If provided, save figure to this path
        """
        df_mix = self.df[self.df['is_mixture']].copy()
        
        if len(df_mix) == 0:
            print("No mixtures found.")
            return
        
        # Get all unique components
        all_comps = sorted(set(df_mix['comp1'].tolist() + df_mix['comp2'].tolist()))
        n_comps = len(all_comps)
        
        # Build adjacency matrix (count of evaluations per pair)
        pair_matrix = np.zeros((n_comps, n_comps))
        
        for _, row in df_mix.iterrows():
            c1, c2 = row['comp1'], row['comp2']
            i1, i2 = all_comps.index(c1), all_comps.index(c2)
            pair_matrix[i1, i2] += 1
            pair_matrix[i2, i1] += 1  # symmetric
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(10, n_comps*0.5), max(8, n_comps*0.5)))
        
        im = ax.imshow(pair_matrix, cmap='YlOrRd', interpolation='nearest')
        
        # Ticks and labels
        ax.set_xticks(np.arange(n_comps))
        ax.set_yticks(np.arange(n_comps))
        ax.set_xticklabels(all_comps, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(all_comps, fontsize=9)
        
        # Annotate cells
        for i in range(n_comps):
            for j in range(n_comps):
                if pair_matrix[i, j] > 0:
                    text = ax.text(j, i, int(pair_matrix[i, j]),
                                   ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Component Pair Exploration Heatmap\n(Number of evaluations per pair)', 
                     fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Evaluation Count', fontsize=11)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_optimization_trajectory(self, save_path: Optional[str] = None):
        """
        Show optimization trajectory in property space (if sequence available).
        
        Args:
            save_path: If provided, save figure to this path
        """
        if self.sequence_df is None:
            print("No sequence file provided. Cannot plot trajectory.")
            return
        
        # Merge with results to get properties
        merged = self.sequence_df.merge(
            self.df, 
            left_on='mixture' if 'mixture' in self.sequence_df.columns else 'fluid',
            right_on='mixture' if 'mixture' in self.df.columns else 'fluid',
            how='left'
        )
        
        tc_col = 'Tc_mix[K]' if 'Tc_mix[K]' in merged.columns else 'Tcrit[K]'
        pc_col = 'Pc_mix[Pa]' if 'Pc_mix[Pa]' in merged.columns else 'Pcrit[Pa]'
        
        if tc_col not in merged.columns or pc_col not in merged.columns:
            print("Property columns not found in merged data.")
            return
        
        fig, ax = plt.subplots(figsize=(11, 8))
        
        # Plot trajectory with color gradient
        n_points = len(merged)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        for i in range(len(merged) - 1):
            ax.plot(
                [merged.iloc[i][tc_col], merged.iloc[i+1][tc_col]],
                [merged.iloc[i][pc_col]/1e6, merged.iloc[i+1][pc_col]/1e6],
                'o-', color=colors[i], linewidth=2, markersize=8, alpha=0.7
            )
        
        # Mark start and end
        ax.scatter(merged.iloc[0][tc_col], merged.iloc[0][pc_col]/1e6,
                   s=300, marker='*', c='green', edgecolors='k', linewidths=2,
                   label='Start', zorder=10)
        ax.scatter(merged.iloc[-1][tc_col], merged.iloc[-1][pc_col]/1e6,
                   s=300, marker='*', c='red', edgecolors='k', linewidths=2,
                   label='End', zorder=10)
        
        ax.set_xlabel('Critical Temperature [K]', fontsize=12)
        ax.set_ylabel('Critical Pressure [MPa]', fontsize=12)
        ax.set_title('Optimization Trajectory in Property Space', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_dir: str = "."):
        """
        Generate all visualizations and save to a directory.
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"Generating visualizations in {output_path}...")
        
        self.plot_property_space(save_path=str(output_path / "property_space.png"))
        self.plot_composition_histogram(save_path=str(output_path / "composition_histogram.png"))
        self.plot_efficiency_comparison(save_path=str(output_path / "efficiency_comparison.png"))
        self.plot_component_pair_heatmap(save_path=str(output_path / "component_pair_heatmap.png"))
        
        if self.sequence_df is not None:
            self.plot_optimization_trajectory(save_path=str(output_path / "optimization_trajectory.png"))
        
        print(f"✓ All visualizations saved to {output_path}")


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize mixture optimization results")
    parser.add_argument("--results", required=True, help="Path to scbo_results.csv")
    parser.add_argument("--sequence", help="Optional path to sequence.csv")
    parser.add_argument("--output-dir", default="viz_output", help="Output directory for plots")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    
    args = parser.parse_args()
    
    viz = MixtureVisualizer(results_csv=args.results, sequence_csv=args.sequence)
    
    if args.report:
        viz.generate_report(output_dir=args.output_dir)
    else:
        # Interactive mode - show all plots
        viz.plot_property_space()
        viz.plot_composition_histogram()
        viz.plot_efficiency_comparison()
        viz.plot_component_pair_heatmap()
        if args.sequence:
            viz.plot_optimization_trajectory()
