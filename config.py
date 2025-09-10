# config.py
# =================================================
# Configuration Module
# =================================================
import numpy as np

class Config:
    # Random seed for reproducibility
    RANDOM_SEED = 99999
    
    # Device configuration
    DEVICE = None  # Will be set automatically in code
    
    # Data processing parameters
    NUM_FEATURES = [
        'Number of national longitudinal projects',
        'Number of provincial longitudinal projects',
        'Number of municipal longitudinal projects',
        'Collaborative Research Project',
        'Number of Software Copyrights',
        'Second Prize in Provincial Competition',
        'First Prize in Provincial Competition',
        'Third Prize in Provincial Competition',
        'First Prize in National Competition',
        'Second Prize in National Competition',
        'Third Prize in National Competition',
        'Number of Provincial Teaching Awards'
    ]
    
    CAT_FEATURES = ['gender', 'Professional Title', 'major', 'degree']
    
    # Dataset split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    
    # Graph building parameters
    GRAPH_METHODS = [
        ("Threshold (0.5)", "build_threshold_graph", {"threshold": 0.5}),
        ("Threshold (0.3)", "build_threshold_graph", {"threshold": 0.3}),
        ("KNN (K=3)", "build_knn_graph_with_weights", {"k": 3}),
        # ("KNN (K=5)", "build_knn_graph_with_weights", {"k": 5}),
        # ("Adaptive Threshold (55%)", "build_adaptive_threshold_graph", {"percentile": 55}),
        ("Adaptive Threshold (75%)", "build_adaptive_threshold_graph", {"percentile": 75}),
        ("Mixed Graph(K=3, 30%)", "build_mixed_graph", {"k": 3, "threshold_percentile": 75}),
        ("Random Graph(50%)", "build_random_graph", {"edge_probability": 0.5}),
        ("isolated Graph", "build_isolated_nodes_graph", {}),
        # ("Threshold+KNN (0.5, K=3)", "build_threshold_knn_graph", {"threshold": 0.5, "k": 3}),
        ("Random+Min Degree", "build_random_graph_with_min_degree", {"edge_probability": 0.1, "min_degree": 1})
    ]
    
    # Model parameters
    MODELS = {
        "EnhancedGCN": {
            "class": "EnhancedGCNModel",
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
        "EnhancedGAT": {
            "class": "EnhancedGATModel",
            "params": {
                "hidden_dims": [64, 32],
                "dropout": 0.1
            },
            "training": {
                "epochs": 200,
                "lr": 0.05,
                "weight_decay": 5e-4
            }
        },
        
        # "SGC": {
        #     "class": "SGCModel",
        #     "params": {
        #         "hidden_dim": 32,
        #         "K": 2
        #     },
        #     "training": {
        #         "epochs": 200,
        #         "lr": 0.05,
        #         "weight_decay": 5e-4
        #     }
        # },
        "APPNP": {
            "class": "APPNPModel",
            "params": {
                "hidden_dim": 32,
                "dropout": 0.3
            },
            "training": {
                "epochs": 200,
                "lr": 0.05,
                "weight_decay": 5e-4
            }
        }
    }
    
    # Training parameters
    TRAINING_PARAMS = {
        "print_every": 20,
        "patience_limit": 20
    }
    
    # Early stopping scheduler
    SCHEDULER_PARAMS = {
        "mode": 'min',
        "factor": 0.5,
        "patience": 10
    }

    # data imbalance
    HANDLE_IMBALANCE = {
        "enabled": True,
        # Options: 'smote'0.896, 'undersample', BorderlineSMOTE:0.896 
        # RandomOverSampler 0.887、SMOTEENN、SMOTETomek：0.877
        "method": "RandomOverSampler",  
        # "ratio": 0.5   
    }