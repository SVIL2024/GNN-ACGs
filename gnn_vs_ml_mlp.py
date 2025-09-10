# gnn_vs_ml_mlp.py - Fixed version with different graph methods per model
# =================================================
# GCN/GAT/APPNP (Different Graph Methods) vs Traditional ML vs MLP Comparison
# =================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
# Import project modules
from dataprocess import DataLoader
from trainer import EnhancedExperimentManager
from models import GATModel, EnhancedGCNModel, EnhancedGATModel, APPNPModel
from config import Config

# =================================================
# MLP Model Definition
# =================================================
class MLPModel(nn.Module):
    """Multi-Layer Perceptron Model"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x, dim=1)

# =================================================
# Training Functions
# =================================================
def train_mlp_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.001, weight_decay=1e-4):
    """Train MLP model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 20
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_train_tensor)
        loss = F.nll_loss(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = F.nll_loss(val_output, y_val_tensor)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
    
    return model

def evaluate_mlp_model(model, X_test, y_test):
    """Evaluate MLP model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    with torch.no_grad():
        output = model(X_test_tensor)
        pred = output.argmax(dim=1)
        correct = (pred == y_test_tensor).sum().item()
        accuracy = correct / len(y_test)
        
        # Calculate probabilities for detailed metrics
        prob = torch.softmax(output, dim=1)
        
    return accuracy, pred.cpu().numpy(), prob.cpu().numpy()

# =================================================
# Comparison Functions
# =================================================
def compare_traditional_ml_methods(processed_data, handle_imbalance=False, imbalance_method='smote'):
    """Compare traditional machine learning methods using the same data split as GNN"""
    
    data_loader = DataLoader("")
    
    if handle_imbalance:
        print(f"Processing imbalanced data for traditional ML methods, using method: {imbalance_method}")
        X_train_balanced, y_train_balanced = data_loader.handle_imbalanced_data(
            processed_data['X_train'], processed_data['y_train'], 
            method=imbalance_method
        )
        print(f"Training data before and after balancing: {len(processed_data['y_train'])} -> {len(y_train_balanced)}")
        X_train = X_train_balanced
        y_train = y_train_balanced
    else:
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
    
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_SEED),
        "SVM": SVC(random_state=Config.RANDOM_SEED, probability=True),  # Enable probability prediction
        "Logistic Regression": LogisticRegression(random_state=Config.RANDOM_SEED, max_iter=1000)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("Traditional Machine Learning Methods Comparison")
    print("="*60)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set to select best model
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)  # Get prediction probabilities
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        # Calculate AUC
        try:
            if len(np.unique(y_test)) == 2:  # Binary classification
                test_auc = roc_auc_score(y_test, y_test_prob[:, 1])
            else:  # Multi-class
                test_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
        except:
            test_auc = 0.0
        
        results[name] = {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc
        }
        
        print(f"{name}:")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  F1-Score (weighted): {test_f1:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        
        # Print detailed classification report
        print(f"  Classification report:")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        print()
    
    return results

def compare_with_mlp(processed_data, handle_imbalance=False, imbalance_method='smote'):
    """Compare with MLP model"""
    
    data_loader = DataLoader("")
    
    if handle_imbalance:
        print(f"Processing imbalanced data for MLP, using method: {imbalance_method}")
        X_train_balanced, y_train_balanced = data_loader.handle_imbalanced_data(
            processed_data['X_train'], processed_data['y_train'], 
            method=imbalance_method
        )
        print(f"Training data before and after balancing: {len(processed_data['y_train'])} -> {len(y_train_balanced)}")
        X_train = X_train_balanced
        y_train = y_train_balanced
    else:
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
    
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    
    # Get input dimensions
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\nMLP Model Configuration:")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Hidden layers: [64, 32]")
    
    # Create MLP model
    mlp_model = MLPModel(input_dim, [64, 32], num_classes, dropout=0.5)
    
    # Train MLP model
    print("\nTraining MLP model...")
    trained_mlp = train_mlp_model(mlp_model, X_train, y_train, X_val, y_val, 
                                 epochs=200, lr=0.001, weight_decay=1e-4)
    
    # Evaluate MLP model
    print("\nEvaluating MLP model...")
    accuracy, y_pred, y_prob = evaluate_mlp_model(trained_mlp, X_test, y_test)
    
    # Calculate detailed metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    try:
        if len(np.unique(y_test)) == 2:  # Binary classification
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:  # Multi-class
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except:
        auc = 0.0
    
    print(f"\nMLP Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Print detailed classification report
    print(f"\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    mlp_results = {
        'test_accuracy': accuracy,
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall,
        'test_auc': auc
    }
    
    return mlp_results

# Define graph methods for each model
MODEL_GRAPH_METHODS = {
    "EnhancedGCN": ("Threshold Graph(0.1)", "build_threshold_graph", {"threshold": 0.1}),
    "EnhancedGAT": ("Threshold (0.9)", "build_threshold_graph", {"threshold": 0.95}),
    "APPNP": ("Threshold (0.3)", "build_threshold_graph", {"threshold": 0.3})
}

def compare_gnn_models(processed_data, df, handle_imbalance=False, imbalance_method='smote'):
    """Compare GCN, GAT, and APPNP models with different graph methods using the same data split"""
    
    print("\n" + "="*60)
    print("GNN Models with Different Graph Methods")
    print("="*60)
    
    # Model configurations
    model_configs = {
        "EnhancedGCN": {
            "class": EnhancedGCNModel,
            "config": Config.MODELS["EnhancedGCN"]
        },
        "EnhancedGAT": {
            "class": EnhancedGATModel,
            "config": Config.MODELS["EnhancedGAT"]
        },
        "APPNP": {
            "class": APPNPModel,
            "config": Config.MODELS["APPNP"]
        }
    }
    
    gnn_results = {}
    
    # Create experiment manager with the same processed data
    data_loader = DataLoader("")
    X_processed, y = data_loader.preprocess_data(df)
    
    # Use a custom experiment manager with fixed data splits
    class FixedSplitExperimentManager(EnhancedExperimentManager):
        def __init__(self, X_processed, y, df, processed_data, handle_imbalance=False, imbalance_method='smote'):
            super().__init__(X_processed, y, df, handle_imbalance, imbalance_method)
            self.processed_data = processed_data
        
        def build_graph_data(self, adj_matrix, similarity_matrix):
            """Build PyG graph data object with fixed train/val/test split"""
            # Set seed to ensure consistent data splitting
            np.random.seed(Config.RANDOM_SEED)
            
            if self.handle_imbalance:
                from dataprocess import DataLoader
                data_loader = DataLoader("")  # Create instance to use methods
                print(f"Handling imbalanced data with method: {self.imbalance_method}")
                X_balanced, y_balanced = data_loader.handle_imbalanced_data(
                    self.X_processed, self.y, method=self.imbalance_method
                )
                print(f"Data imbalance before and after: {len(self.y)} -> {len(y_balanced)}")
                self.X_processed = X_balanced
                self.y = y_balanced
                
                # Update number of classes in Config
                Config.NUM_CLASSES = len(np.unique(self.y))
                print(f"Updated number of classes: {Config.NUM_CLASSES}")

            edges = np.array(np.nonzero(adj_matrix))
            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_weight = torch.tensor(similarity_matrix[edges[0], edges[1]], dtype=torch.float)
            
            data = Data(
                x=torch.tensor(self.X_processed.astype(np.float32)),
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=torch.tensor(self.y, dtype=torch.long)
            )
            
            # Debug information
            print(f"Data shape: {data.x.shape}")
            print(f"Number of nodes: {data.num_nodes}")
            print(f"Number of edges: {data.edge_index.shape[1]}")
            print(f"Number of classes: {len(torch.unique(data.y))}")
            print(f"Label distribution: {torch.bincount(data.y)}")
            
            # Use the same train/val/test split as traditional ML methods
            num_nodes = data.num_nodes
            
            # Get the indices from processed_data
            train_indices = self.processed_data['train_indices']
            val_indices = self.processed_data['val_indices']
            test_indices = self.processed_data['test_indices']
            
            # Create masks
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            data.train_mask[train_indices] = True
            data.val_mask[val_indices] = True
            data.test_mask[test_indices] = True
            
            print(f"Train samples: {data.train_mask.sum().item()}")
            print(f"Val samples: {data.val_mask.sum().item()}")
            print(f"Test samples: {data.test_mask.sum().item()}")
            
            return data
    
    for model_name, model_info in model_configs.items():
        print(f"\n--- {model_name} with {MODEL_GRAPH_METHODS[model_name][0]} ---")
        
        model_class = model_info["class"]
        model_config = model_info["config"]
        model_params = model_config["params"].copy()
        training_params = model_config["training"]
        
        # Get the specific graph method for this model
        graph_method = MODEL_GRAPH_METHODS[model_name]
        
        try:
            # Create experiment manager with fixed data splits
            experiment_manager = FixedSplitExperimentManager(
                X_processed, y, df, processed_data,
                handle_imbalance=handle_imbalance,
                imbalance_method=imbalance_method
            )
            
            # Run GNN experiment
            acc, method_name, gnn_model, gnn_data, detailed_metrics = experiment_manager.run_experiment(
                graph_method,
                model_class,
                model_params,
                training_params
            )
            
            gnn_results[model_name] = {
                'accuracy': acc,
                'f1_weighted': detailed_metrics['f1_weighted'],
                'f1_macro': detailed_metrics['f1_macro'],
                'precision_weighted': detailed_metrics['precision_weighted'],
                'recall_weighted': detailed_metrics['recall_weighted'],
                'auc': detailed_metrics.get('auc', 0.0)
            }
            
            print(f"\n{model_name} Results with {method_name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1-Score (weighted): {detailed_metrics['f1_weighted']:.4f}")
            print(f"  F1-Score (macro): {detailed_metrics['f1_macro']:.4f}")
            print(f"  Precision (weighted): {detailed_metrics['precision_weighted']:.4f}")
            print(f"  Recall (weighted): {detailed_metrics['recall_weighted']:.4f}")
            print(f"  AUC: {detailed_metrics.get('auc', 0.0):.4f}")
            
        except Exception as e:
            print(f"Error training {model_name} with {graph_method[0]}: {e}")
            import traceback
            traceback.print_exc()
            gnn_results[model_name] = {
                'accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'precision_weighted': 0.0,
                'recall_weighted': 0.0,
                'auc': 0.0
            }
    
    return gnn_results

# =================================================
# Visualization Functions
# =================================================
def plot_comprehensive_comparison(gnn_results, ml_results, mlp_results):
    """Plot comprehensive comparison of all methods"""
    try:
        # Set font
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Prepare data
        gnn_methods = list(gnn_results.keys())
        ml_methods = list(ml_results.keys())
        all_methods = gnn_methods + ml_methods + ['MLP']
        
        accuracies = [gnn_results[model]['accuracy'] for model in gnn_methods] + \
                    [result['test_accuracy'] for result in ml_results.values()] + \
                    [mlp_results['test_accuracy']]
        
        f1_scores = [gnn_results[model]['f1_weighted'] for model in gnn_methods] + \
                   [result['test_f1'] for result in ml_results.values()] + \
                   [mlp_results['test_f1']]
        
        auc_scores = [gnn_results[model]['auc'] for model in gnn_methods] + \
                   [result['test_auc'] for result in ml_results.values()] + \
                   [mlp_results['test_auc']]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot Accuracy
        bars1 = axes[0].bar(all_methods, accuracies, color=plt.cm.Set3(np.linspace(0, 1, len(all_methods))))
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot F1-Score
        bars2 = axes[1].bar(all_methods, f1_scores, color=plt.cm.Set3(np.linspace(0, 1, len(all_methods))))
        axes[1].set_title('F1-Score (Weighted) Comparison')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, f1_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot AUC
        bars3 = axes[2].bar(all_methods, auc_scores, color=plt.cm.Set3(np.linspace(0, 1, len(all_methods))))
        axes[2].set_title('AUC Comparison')
        axes[2].set_ylabel('AUC')
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars3, auc_scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('gnn_ml_mlp_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison chart saved as gnn_ml_mlp_comparison.png")
        
    except Exception as e:
        print(f"Error plotting comparison: {e}")

def print_summary_table(gnn_results, ml_results, mlp_results):
    """Print summary comparison table"""
    print("\n" + "="*100)
    print("SUMMARY COMPARISON TABLE")
    print("="*100)
    
    # Create summary data
    summary_data = []
    
    # GNN results
    for model_name, results in gnn_results.items():
        summary_data.append({
            'Method': model_name,
            'Accuracy': results['accuracy'],
            'F1-Score (Weighted)': results['f1_weighted'],
            'Precision (Weighted)': results['precision_weighted'],
            'Recall (Weighted)': results['recall_weighted'],
            'AUC': results['auc']
        })
    
    # Traditional ML results
    for method_name, results in ml_results.items():
        summary_data.append({
            'Method': method_name,
            'Accuracy': results['test_accuracy'],
            'F1-Score (Weighted)': results['test_f1'],
            'Precision (Weighted)': results['test_precision'],
            'Recall (Weighted)': results['test_recall'],
            'AUC': results['test_auc']
        })
    
    # MLP results
    summary_data.append({
        'Method': 'MLP',
        'Accuracy': mlp_results['test_accuracy'],
        'F1-Score (Weighted)': mlp_results['test_f1'],
        'Precision (Weighted)': mlp_results['test_precision'],
        'Recall (Weighted)': mlp_results['test_recall'],
        'AUC': mlp_results['test_auc']
    })
    
    # Convert to DataFrame for better display
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Find best method for each metric
    print("\n" + "-"*100)
    print("BEST METHODS BY METRIC")
    print("-"*100)
    
    best_accuracy = summary_df.loc[summary_df['Accuracy'].idxmax()]
    best_f1 = summary_df.loc[summary_df['F1-Score (Weighted)'].idxmax()]
    best_auc = summary_df.loc[summary_df['AUC'].idxmax()]
    
    print(f"Highest Accuracy: {best_accuracy['Method']} ({best_accuracy['Accuracy']:.4f})")
    print(f"Highest F1-Score: {best_f1['Method']} ({best_f1['F1-Score (Weighted)']:.4f})")
    print(f"Highest AUC: {best_auc['Method']} ({best_auc['AUC']:.4f})")

# =================================================
# Main Function
# =================================================
def main():
    """Main function to run GNN vs traditional ML vs MLP comparison experiment"""
    try:
        print("="*100)
        print("GNN (GCN/GAT/APPNP with Different Graph Methods) vs Traditional ML vs MLP Methods Comparison")
        print("="*100)
        
        # Configuration
        file_path = "teacher_info_2025_name-7-3-english.xlsx"
        
        # 1. Data loading
        print("\n" + "="*60)
        print("Data Loading Phase")
        print("="*60)
        data_loader = DataLoader(file_path)
        df = data_loader.load_data()
        
        # 2. Data preprocessing for traditional ML methods (maintaining same split)
        processed_data = data_loader.create_simple_classification_dataset_with_split(df)
        
        # 3. Traditional ML methods comparison
        ml_results = compare_traditional_ml_methods(
            processed_data,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        # 4. MLP comparison
        print("\n" + "="*60)
        print("MLP Model Comparison")
        print("="*60)
        mlp_results = compare_with_mlp(
            processed_data,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        # 5. GNN models with different graph methods - using the same data split
        gnn_results = compare_gnn_models(
            processed_data, df,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        # 6. Summary and visualization
        print_summary_table(gnn_results, ml_results, mlp_results)
        
        # 7. Generate visualization charts
        print("\n" + "="*60)
        print("Generating Visualization Charts")
        print("="*60)
        
        plot_comprehensive_comparison(gnn_results, ml_results, mlp_results)
        
        print("\n" + "="*60)
        print("Comparison Experiment Completed")
        print("="*60)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()