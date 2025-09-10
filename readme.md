# Teacher Evaluation
This repository contains the code and datasets for the paper titled "Academic Collaboration Graphs for Teacher Performance Evaluation: Exploring GNNs and Graph Construction Strategies".

The goal of this project is to model teacher collaboration through Academic Collaboration Graphs (ACG) and use Graph Neural Networks (GNNs) to assess teacher abilities. We explore various graph construction strategies and GNN architectures to better understand relational patterns in academic collaboration data.

## Requirements
- Python 3.9.23
- Required libraries: pandas, numpy, matplotlib, scikit-learn, pytorch etc.

## Installation
git clone https://github.com/SVIL2024/GNN-ACGs.git

## Project Structure
.
├── data/               # Datasets and graph files
├── models/             # GNN and baseline model implementations
├── utils/              # Helper functions (graph processing, evaluation, etc.)
├── results/            # Output logs, evaluation metrics, and figures
├── notebooks/          # Jupyter notebooks for exploratory analysis
├── config/             # Configuration files for experiments
├── README.md           # This file
└── requirements.txt    # Python dependencies

## introuction for the main files
1. Data Processing (dataprocess.py)
- Data loading and preprocessing
- Feature standardization and encoding
- Imbalanced data handling
- Dataset splitting
2. Graph Building (graphbulider.py)
- Threshold graph construction
- KNN graph construction
- Adaptive threshold graph construction
- Hybrid graph construction
- Random graph construction
3. Model Definition (models.py)
- EnhancedGCNModel: Enhanced Graph Convolutional Network
- EnhancedGATModel: Enhanced Graph Attention Network
- APPNPModel: Approximate Personalized Propagation of Neural Predictions model
4. Training and Evaluation (trainer.py)
- Model training and validation
- Detailed performance evaluation
- Experiment management
5. Main Experiments
- GNN vs Traditional Machine Learning Comparison (gnn_vs_ml_mlp.py)
- Graph Construction Parameter Analysis (GNN_diff_graphContruction_params1.py)
- Model Parameter Testing (GNN_similarity_graph_params_test.py)
6. Configuration

## Usage
1. Prepare data file teacher_info_2025_name-7-3-english.xlsx
2. Modify configuration parameters in config.py as needed
3. Run the corresponding experiment scripts:
bash
python gnn_vs_ml_mlp.py          # GNN vs traditional ML comparison
python GNN_diff_graphContruction_params1.py  # Graph construction parameter analysis
python GNN_similarity_graph_params_test.py   # Model parameter testing

## Dataset
To be more specific, the information gathered from an university is systematically saved in a file named teacher_info***.csv, ensuring its easy access and management for further reference and analysis.

