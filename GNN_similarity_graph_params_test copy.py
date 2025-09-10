# model_comparison_experiment.py
# =================================================
# Model Comparison Experiment for GCN, APPNP and GAT
# =================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import Config
from dataprocess import DataLoader
from trainer import EnhancedExperimentManager
from models import EnhancedGCNModel, APPNPModel, EnhancedGATModel

def run_model_comparison_experiments(X_processed, y, df):
    """
    Run experiments with different models and parameters for each graph construction method
    """
    # Create directory for saving results
    os.makedirs('data', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # Define model configurations
    model_configs = {
        "GCN": {
            "class": "EnhancedGCNModel",
            "model_class": EnhancedGCNModel,
            "configs": {
                "GCN-Base": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "GCN-HighDropout": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.8
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "GCN-LowDropout": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.3
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "GCN-SmallLR": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.01,
                        "weight_decay": 5e-4
                    }
                },
                "GCN-LargeLR": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.1,
                        "weight_decay": 5e-4
                    }
                },
                "GCN-Shallow": {
                    "params": {
                        "hidden_dims": [32, 16],
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                }
            }
        },
        "APPNP": {
            "class": "APPNPModel",
            "model_class": APPNPModel,
            "configs": {
                "APPNP-Base": {
                    "params": {
                        "hidden_dim": 32,
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "APPNP-HighDropout": {
                    "params": {
                        "hidden_dim": 32,
                        "dropout": 0.8
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "APPNP-LowDropout": {
                    "params": {
                        "hidden_dim": 32,
                        "dropout": 0.3
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "APPNP-SmallLR": {
                    "params": {
                        "hidden_dim": 32,
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.01,
                        "weight_decay": 5e-4
                    }
                },
                "APPNP-LargeLR": {
                    "params": {
                        "hidden_dim": 32,
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.1,
                        "weight_decay": 5e-4
                    }
                },
                "APPNP-Shallow": {
                    "params": {
                        "hidden_dim": 16,
                        "dropout": 0.5
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                }
            }
        },
        "GAT": {
            "class": "EnhancedGATModel",
            "model_class": EnhancedGATModel,
            "configs": {
                "GAT-Base": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.6
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "GAT-HighDropout": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.8
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "GAT-LowDropout": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.3
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                },
                "GAT-SmallLR": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.6
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.01,
                        "weight_decay": 5e-4
                    }
                },
                "GAT-LargeLR": {
                    "params": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.6
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.1,
                        "weight_decay": 5e-4
                    }
                },
                "GAT-Shallow": {
                    "params": {
                        "hidden_dims": [32, 16],
                        "dropout": 0.6
                    },
                    "training": {
                        "epochs": 200,
                        "lr": 0.05,
                        "weight_decay": 5e-4
                    }
                }
            }
        }
    }
    
    # Store all results
    all_results = []
    
    # For each model type
    for model_name, model_info in model_configs.items():
        print(f"\n=== Testing {model_name} Model ===")
        
        # For each parameter configuration
        for config_name, config in model_info["configs"].items():
            print(f"\n--- Using {config_name} ---")
            
            # For each graph construction method
            for graph_method in Config.GRAPH_METHODS:
                method_name = graph_method[0]
                print(f"  Testing with {method_name}...")
                
                # Create experiment manager
                experiment_manager = EnhancedExperimentManager(
                    X_processed=X_processed,
                    y=y,
                    df=df,
                    handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
                    imbalance_method=Config.HANDLE_IMBALANCE["method"]
                )
                
                try:
                    # Run experiment
                    acc, _, _, _, detailed_metrics = experiment_manager.run_experiment(
                        graph_method,
                        model_info["model_class"],
                        config["params"],
                        config["training"]
                    )
                    
                    # Store results
                    result = {
                        "Model": model_name,
                        "Model Config": config_name,
                        "Graph Method": method_name,
                        "Accuracy": acc,
                        "F1-Score (Weighted)": detailed_metrics['f1_weighted'],
                        "F1-Score (Macro)": detailed_metrics['f1_macro'],
                        "Precision (Weighted)": detailed_metrics['precision_weighted'],
                        "Recall (Weighted)": detailed_metrics['recall_weighted'],
                        "AUC": detailed_metrics.get('auc', 0.0)
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"    {config_name} with {method_name} failed: {e}")
                    all_results.append({
                        "Model": model_name,
                        "Model Config": config_name,
                        "Graph Method": method_name,
                        "Accuracy": 0.0,
                        "F1-Score (Weighted)": 0.0,
                        "F1-Score (Macro)": 0.0,
                        "Precision (Weighted)": 0.0,
                        "Recall (Weighted)": 0.0,
                        "AUC": 0.0
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    return results_df

def visualize_model_comparison(results_df):
    """
    Create visualization comparing models across graph construction methods
    """
    # Create directory for saving images
    os.makedirs('images', exist_ok=True)
    
    # Get unique models
    models = results_df['Model'].unique()
    
    # Create heatmap for each model
    for model in models:
        model_results = results_df[results_df['Model'] == model]
        
        if model_results.empty:
            continue
            
        # Create heatmaps for different metrics
        metrics = ['Accuracy', 'F1-Score (Weighted)', 'AUC']
        metric_names = ['Accuracy', 'F1-Score', 'AUC']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{model} Model Performance Across Graph Construction Methods', fontsize=16)
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            pivot_data = model_results.pivot(index='Graph Method', columns='Model Config', values=metric)
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[idx], cbar=True)
            axes[idx].set_title(f'{metric_name}', fontsize=12)
            axes[idx].set_xlabel('Model Configurations')
            axes[idx].set_ylabel('Graph Construction Methods')
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[idx].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        filename = f"images/{model}_parameter_comparison_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved heatmap for {model} to {filename}")

def compare_best_performances(results_df):
    """
    Compare the best performance of each model across all metrics
    """
    # Find best configuration for each model and metric
    best_performances = []
    
    models = results_df['Model'].unique()
    metrics = ['Accuracy', 'F1-Score (Weighted)', 'AUC']
    
    for model in models:
        model_results = results_df[results_df['Model'] == model]
        for metric in metrics:
            best_idx = model_results[metric].idxmax()
            best_row = model_results.loc[best_idx]
            best_performances.append({
                'Model': model,
                'Metric': metric,
                'Best Value': best_row[metric],
                'Config': best_row['Model Config'],
                'Graph Method': best_row['Graph Method']
            })
    
    best_df = pd.DataFrame(best_performances)
    
    # Create comparison table
    comparison_pivot = best_df.pivot(index='Metric', columns='Model', values='Best Value')
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    sns.heatmap(comparison_pivot, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title('Best Performance Comparison Across Models and Metrics')
    plt.ylabel('Metrics')
    plt.xlabel('Models')
    plt.tight_layout()
    filename = "images/model_performance_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved performance comparison to {filename}")
    
    return best_df

def main():
    """
    Main function to run model comparison experiments
    """
    print("Starting Model Comparison Experiment")
    print("=" * 60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_loader = DataLoader("teacher_info_2025_name-7-3-english.xlsx")
    try:
        df = data_loader.load_data()
    except FileNotFoundError:
        print("Data file not found. Please ensure 'teacher_info_2025_name-7-3-english.xlsx' exists in the current directory.")
        print("Generating sample data for demonstration...")
        # Create sample data for demonstration
        np.random.seed(Config.RANDOM_SEED)
        n_samples = 1000
        
        # Create numerical features
        num_data = {}
        for feature in Config.NUM_FEATURES:
            num_data[feature] = np.random.randn(n_samples)
        
        # Create categorical features
        cat_data = {}
        cat_data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
        cat_data['Professional Title'] = np.random.choice(['Professor', 'Associate Professor', 'Lecturer'], n_samples)
        cat_data['major'] = np.random.choice(['Computer Science', 'Mathematics', 'Physics'], n_samples)
        cat_data['degree'] = np.random.choice(['PhD', 'Master', 'Bachelor'], n_samples)
        
        # Combine all data
        data_dict = {**num_data, **cat_data}
        data_dict['score'] = np.random.randint(0, 3, n_samples)  # 3 classes: 0, 1, 2
        
        df = pd.DataFrame(data_dict)
        print(f"Generated sample data with {len(df)} samples")
    
    # Preprocess data
    X_processed, y = data_loader.preprocess_data(df)
    print(f"Preprocessed data shape: {X_processed.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Run experiments
    try:
        results_df = run_model_comparison_experiments(X_processed, y, df)
        
        # Save results to CSV
        os.makedirs('data', exist_ok=True)
        results_df.to_csv("data/model_comparison_results.csv", index=False)
        print("\nResults saved to 'data/model_comparison_results.csv'")
        
        # Print summary
        print("\nExperimental Results Summary:")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Create visualizations
        print("\nGenerating visualizations...")
        visualize_model_comparison(results_df)
        best_performances = compare_best_performances(results_df)
        
        print("\nBest Performance for Each Model:")
        print("="*60)
        print(best_performances.to_string(index=False))
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()