# consolidated_graph_analysis.py
# =================================================
# Consolidated Graph Construction Parameter Analysis
# =================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from dataprocess import DataLoader
from trainer import EnhancedExperimentManager
from models import EnhancedGCNModel, EnhancedGATModel, SGCModel, APPNPModel
import torch
import warnings
import os
warnings.filterwarnings('ignore')

class ConsolidatedGraphAnalysis:
    def __init__(self, file_path):
        """Initialize the analyzer"""
        self.file_path = file_path
        self.results_data = []
        self.model_results = {}
        
    def generate_graph_parameters(self):
        """Generate parameter combinations for different graph construction methods"""
        graph_params = {
            "Threshold": [
                ("Threshold (0.1)", "build_threshold_graph", {"threshold": 0.1}),
                ("Threshold (0.3)", "build_threshold_graph", {"threshold": 0.3}),
                ("Threshold (0.5)", "build_threshold_graph", {"threshold": 0.5}),
                ("Threshold (0.7)", "build_threshold_graph", {"threshold": 0.7}),
                ("Threshold (0.9)", "build_threshold_graph", {"threshold": 0.9})
            ],
            "KNN": [
                ("KNN (K=1)", "build_knn_graph_with_weights", {"k": 1}),
                ("KNN (K=3)", "build_knn_graph_with_weights", {"k": 3}),
                ("KNN (K=5)", "build_knn_graph_with_weights", {"k": 5}),
                ("KNN (K=7)", "build_knn_graph_with_weights", {"k": 7}),
                ("KNN (K=9)", "build_knn_graph_with_weights", {"k": 9})
            ],
            "Adaptive Threshold": [
                ("Adaptive Threshold (50%)", "build_adaptive_threshold_graph", {"percentile": 50}),
                ("Adaptive Threshold (65%)", "build_adaptive_threshold_graph", {"percentile": 65}),
                ("Adaptive Threshold (75%)", "build_adaptive_threshold_graph", {"percentile": 75}),
                ("Adaptive Threshold (85%)", "build_adaptive_threshold_graph", {"percentile": 85}),
                ("Adaptive Threshold (95%)", "build_adaptive_threshold_graph", {"percentile": 95})
            ],
            "Random Graph": [
                ("Random Graph(10%)", "build_random_graph", {"edge_probability": 0.1}),
                ("Random Graph(30%)", "build_random_graph", {"edge_probability": 0.3}),
                ("Random Graph(50%)", "build_random_graph", {"edge_probability": 0.5}),
                ("Random Graph(70%)", "build_random_graph", {"edge_probability": 0.7}),
                ("Random Graph(90%)", "build_random_graph", {"edge_probability": 0.9})
            ]
        }
        return graph_params
    
    def run_analysis(self):
        """Run complete parameter analysis"""
        # Load data
        data_loader = DataLoader(self.file_path)
        df = data_loader.load_data()
        X_processed, y = data_loader.preprocess_data(df)
        
        # Create experiment manager
        experiment_manager = EnhancedExperimentManager(
            X_processed, y, df,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        # Get graph parameters
        graph_params = self.generate_graph_parameters()
        
        # Model configurations
        model_classes = {
            "EnhancedGCN": EnhancedGCNModel,
            "EnhancedGAT": EnhancedGATModel,
            "APPNP": APPNPModel
        }
        
        model_configs = Config.MODELS
        
        # Store all results
        all_results = []
        
        # Analyze each graph construction method type
        for graph_type, graph_methods in graph_params.items():
            print(f"\nAnalyzing {graph_type} graph construction methods...")
            
            for model_name, model_class in model_classes.items():
                # Check if model is enabled in config
                if model_name not in model_configs:
                    continue
                    
                print(f"  Testing model: {model_name}")
                
                model_config = model_configs[model_name]
                model_params = model_config["params"]
                training_params = model_config["training"]
                
                # Run experiment for each parameter combination
                for graph_method in graph_methods:
                    try:
                        # Run experiment
                        acc, _, _, _, detailed_metrics = experiment_manager.run_experiment(
                            graph_method,
                            model_class,
                            model_params,
                            training_params
                        )
                        
                        result = {
                            "Graph Type": graph_type,
                            "Graph Method": graph_method[0],
                            "Model": model_name,
                            "Accuracy": acc,
                            "F1-Score (Weighted)": detailed_metrics['f1_weighted'],
                            "F1-Score (Macro)": detailed_metrics['f1_macro'],
                            "Precision (Weighted)": detailed_metrics['precision_weighted'],
                            "Recall (Weighted)": detailed_metrics['recall_weighted'],
                            "AUC": detailed_metrics.get('auc', 0.0)
                        }
                        
                        all_results.append(result)
                        print(f"    {graph_method[0]}: F1={detailed_metrics['f1_weighted']:.4f}")
                        
                    except Exception as e:
                        print(f"    {graph_method[0]}: Failed - {str(e)}")
                        result = {
                            "Graph Type": graph_type,
                            "Graph Method": graph_method[0],
                            "Model": model_name,
                            "Accuracy": 0.0,
                            "F1-Score (Weighted)": 0.0,
                            "F1-Score (Macro)": 0.0,
                            "Precision (Weighted)": 0.0,
                            "Recall (Weighted)": 0.0,
                            "AUC": 0.0
                        }
                        all_results.append(result)
        
        # Save all results
        self.results_data = all_results
        return pd.DataFrame(all_results)
    
    def create_comprehensive_comparison_chart(self, metric='F1-Score (Weighted)'):
        """Create a comprehensive comparison chart showing all models and graph methods"""
        df = pd.DataFrame(self.results_data)
        
        # Create a pivot table for heatmap
        pivot_table = df.pivot_table(
            values=metric,
            index='Model',
            columns='Graph Method',
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': metric})
        plt.title(f'Performance Comparison: Models vs Graph Construction Methods ({metric})')
        plt.xlabel('Graph Construction Methods')
        plt.ylabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('comprehensive_model_graph_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pivot_table
    
    def create_model_performance_evolution(self):
        """Create evolution charts showing how model performance changes with parameters"""
        df = pd.DataFrame(self.results_data)
        graph_types = df['Graph Type'].unique()
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        model_names = df['Model'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        model_colors = dict(zip(model_names, colors))
        
        for i, graph_type in enumerate(graph_types):
            if i >= len(axes):
                break
                
            ax = axes[i]
            type_data = df[df['Graph Type'] == graph_type]
            
            if len(type_data) > 0:
                # Extract parameter values
                if graph_type == "Threshold":
                    param_values = [float(method.split('(')[1].split(')')[0]) for method in type_data['Graph Method']]
                    param_name = "Threshold"
                elif graph_type == "KNN":
                    param_values = [int(method.split('(K=')[1].split(')')[0]) for method in type_data['Graph Method']]
                    param_name = "K Value"
                elif graph_type == "Adaptive Threshold":
                    param_values = [int(method.split('(')[1].split('%')[0]) for method in type_data['Graph Method']]
                    param_name = "Percentile"
                elif graph_type == "Random Graph":
                    param_values = [float(method.split('(')[1].split('%')[0])/100 for method in type_data['Graph Method']]
                    param_name = "Edge Probability"
                else:
                    continue
                
                # Add parameter values to dataframe
                type_data = type_data.copy()
                type_data['Parameter'] = param_values
                
                # Plot each model
                for model_name in model_names:
                    model_data = type_data[type_data['Model'] == model_name]
                    if len(model_data) > 0:
                        model_data = model_data.sort_values('Parameter')
                        ax.plot(model_data['Parameter'], model_data['F1-Score (Weighted)'], 
                               marker='o', linewidth=2, markersize=6, 
                               label=model_name, color=model_colors[model_name])
                
                ax.set_xlabel(param_name)
                ax.set_ylabel('F1-Score (Weighted)')
                ax.set_title(f'{graph_type} Parameter Evolution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(graph_types), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('model_performance_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_summary_chart(self, metric='F1-Score (Weighted)'):
        """Create a summary chart showing best performance for each model-graph combination"""
        df = pd.DataFrame(self.results_data)
        
        # Get best performance for each model-graph type combination
        best_performances = df.groupby(['Model', 'Graph Type'])[metric].max().reset_index()
        
        # Pivot for better visualization
        summary_pivot = best_performances.pivot(index='Model', columns='Graph Type', values=metric)
        
        # Create bar chart
        ax = summary_pivot.plot(kind='bar', figsize=(12, 8), width=0.8)
        plt.title(f'Best {metric} for Each Model-Graph Type Combination')
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Graph Types', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
        
        plt.tight_layout()
        plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return summary_pivot
    
    def create_radar_chart_comparison(self):
        """Create radar charts comparing models across different graph types"""
        df = pd.DataFrame(self.results_data)
        
        # Get best performance for each model-graph type combination
        best_performances = df.groupby(['Model', 'Graph Type'])['F1-Score (Weighted)'].max().reset_index()
        
        # Pivot for radar chart
        radar_data = best_performances.pivot(index='Model', columns='Graph Type', values='F1-Score (Weighted)')
        
        # Normalize data for better visualization
        radar_data_normalized = radar_data.copy()
        for col in radar_data_normalized.columns:
            col_max = radar_data_normalized[col].max()
            if col_max > 0:
                radar_data_normalized[col] = radar_data_normalized[col] / col_max
        
        # Create radar chart
        categories = list(radar_data_normalized.columns)
        N = len(categories)
        
        # Calculate angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories, color='grey', size=10)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        # Plot data for each model
        colors = plt.cm.Set1(np.linspace(0, 1, len(radar_data_normalized.index)))
        for i, (model_name, row) in enumerate(radar_data_normalized.iterrows()):
            values = row.values.flatten().tolist()
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        plt.title('Model Performance Across Graph Types (Normalized)', size=16, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig('radar_chart_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return radar_data
    
    def generate_detailed_report(self):
        """Generate a detailed text report of the analysis"""
        df = pd.DataFrame(self.results_data)
        
        print("="*80)
        print("DETAILED ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\nTotal experiments conducted: {len(df)}")
        print(f"Number of models tested: {df['Model'].nunique()}")
        print(f"Number of graph construction methods: {df['Graph Method'].nunique()}")
        
        # Best overall performance
        best_overall = df.loc[df['F1-Score (Weighted)'].idxmax()]
        print(f"\nBest overall performance:")
        print(f"  Model: {best_overall['Model']}")
        print(f"  Graph Method: {best_overall['Graph Method']}")
        print(f"  F1-Score (Weighted): {best_overall['F1-Score (Weighted)']:.4f}")
        
        # Best performance by model
        print(f"\nBest performance by model:")
        model_best = df.groupby('Model')['F1-Score (Weighted)'].idxmax()
        for idx in model_best:
            row = df.loc[idx]
            print(f"  {row['Model']}: {row['F1-Score (Weighted)']:.4f} "
                  f"({row['Graph Method']})")
        
        # Best performance by graph type
        print(f"\nBest performance by graph type:")
        graph_best = df.groupby('Graph Type')['F1-Score (Weighted)'].idxmax()
        for idx in graph_best:
            row = df.loc[idx]
            print(f"  {row['Graph Type']}: {row['F1-Score (Weighted)']:.4f} "
                  f"({row['Model']} with {row['Graph Method']})")
        
        # Performance statistics by model
        print(f"\nPerformance statistics by model:")
        model_stats = df.groupby('Model')['F1-Score (Weighted)'].agg(['mean', 'std', 'min', 'max'])
        for model, stats in model_stats.iterrows():
            print(f"  {model}:")
            print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"    Range: {stats['min']:.4f} - {stats['max']:.4f}")
        
        # Performance statistics by graph type
        print(f"\nPerformance statistics by graph type:")
        graph_stats = df.groupby('Graph Type')['F1-Score (Weighted)'].agg(['mean', 'std', 'min', 'max'])
        for graph_type, stats in graph_stats.iterrows():
            print(f"  {graph_type}:")
            print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"    Range: {stats['min']:.4f} - {stats['max']:.4f}")

def main():
    """Main function"""
    # File path
    file_path = "teacher_info_2025_name-7-3-english.xlsx"  # Please modify according to your actual file path
    
    # Create analyzer
    analyzer = ConsolidatedGraphAnalysis(file_path)
    
    # Run analysis
    print("Starting consolidated graph construction parameter analysis...")
    results_df = analyzer.run_analysis()
    
    # Save results to CSV
    results_df.to_csv('consolidated_graph_analysis_results.csv', index=False)
    print("Results saved to consolidated_graph_analysis_results.csv")
    
    # Create consolidated visualizations
    print("\nCreating comprehensive comparison chart...")
    analyzer.create_comprehensive_comparison_chart()
    
    print("Creating model performance evolution chart...")
    analyzer.create_model_performance_evolution()
    
    print("Creating performance summary chart...")
    analyzer.create_performance_summary_chart()
    
    print("Creating radar chart comparison...")
    analyzer.create_radar_chart_comparison()
    
    print("Generating detailed report...")
    analyzer.generate_detailed_report()
    
    print("\nAnalysis completed! All consolidated charts and reports have been saved.")

if __name__ == "__main__":
    main()