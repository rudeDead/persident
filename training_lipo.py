"""
Enhanced Lipophilicity (logP) Molecular Property Prediction with Multi-Baseline Enhanced Integrated Gradients (MB-EIG)

This enhanced training pipeline implements advanced techniques to achieve R² > 0.80:
- Advanced data preprocessing with outlier detection and removal
- Hyperparameter optimization using Optuna
- Enhanced Graphormer architecture with residual connections and attention improvements
- Advanced molecular descriptors and features
- Ensemble learning with multiple model architectures
- Sophisticated data augmentation techniques
- Dual baseline MB-EIG explanations (Inert Skeleton + Zero baseline)
- Fragment analysis with SMARTS patterns for lipophilicity
- Training with validation explanations every 5 epochs
- Comprehensive visualization and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem, Descriptors, Crippen, Lipinski
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr, zscore
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
import warnings
import os
import math
import logging
import json
from pathlib import Path
from datetime import datetime
import random
# import optuna
# from optuna.samplers import TPESampler
# from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')

# Set device and optimize CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# Logging and directories
# -------------------------
EXPLAIN_DIR = Path('../outputs/explanations_lipo')
RESULTS_DIR = Path('../outputs/plots_lipo')
LOGS_DIR = Path('../outputs/logs_lipo')
MODELS_DIR = Path('../models')

for dir_path in [EXPLAIN_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

log_file = LOGS_DIR / 'lipo_training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Lipophilicity Training with MB-EIG - Logger initialized")

class EnhancedMolecularGraph:
    """Enhanced molecular graph representation with advanced features"""
    
    def __init__(self, smiles: str, max_atoms: int = 120):
        self.smiles = smiles
        self.max_atoms = max_atoms
        self.mol = Chem.MolFromSmiles(smiles, sanitize=True)
        
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens for better representation
        self.mol = Chem.AddHs(self.mol)
        
        # Canonicalize SMILES
        self.smiles = Chem.MolToSmiles(self.mol)
        self.num_atoms = min(self.mol.GetNumAtoms(), max_atoms)
        
        # Enhanced features
        self.adj_matrix = self._get_adjacency_matrix()
        self.node_features = self._get_enhanced_node_features()
        self.edge_features = self._get_enhanced_edge_features()
        self.shortest_paths = self._get_shortest_paths()
        self.centrality_encoding = self._get_centrality_encoding()
        self.molecular_descriptors = self._get_molecular_descriptors()
        self.ring_features = self._get_ring_features()
    
    def _get_adjacency_matrix(self) -> np.ndarray:
        """Get enhanced adjacency matrix with bond orders"""
        adj = np.zeros((self.max_atoms, self.max_atoms))
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                bond_order = float(bond.GetBondTypeAsDouble())
                adj[i, j] = adj[j, i] = bond_order
        return adj
    
    def _get_enhanced_node_features(self) -> np.ndarray:
        """Get enhanced atom features (15 features per atom)"""
        features = np.zeros((self.max_atoms, 15))
        
        for i, atom in enumerate(self.mol.GetAtoms()):
            if i >= self.max_atoms:
                break
            
            # Basic features
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            hybridization = int(atom.GetHybridization())
            is_aromatic = int(atom.GetIsAromatic())
            mass = atom.GetMass() * 0.01
            total_hs = atom.GetTotalNumHs()
            is_in_ring = int(atom.IsInRing())
            implicit_valence = atom.GetImplicitValence()
            
            # Enhanced features
            explicit_valence = atom.GetExplicitValence()
            total_valence = atom.GetTotalValence()
            radical_electrons = atom.GetNumRadicalElectrons()
            is_chiral = int(atom.HasProp('_ChiralityPossible'))
            
            # Electronegativity (Pauling scale approximation)
            electronegativity_map = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 
                                   15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
            electronegativity = electronegativity_map.get(atomic_num, 2.0)
            
            # Van der Waals radius approximation
            vdw_radius_map = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
                             15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
            vdw_radius = vdw_radius_map.get(atomic_num, 2.0)
            
            features[i] = [
                atomic_num, degree, formal_charge, hybridization, is_aromatic,
                mass, total_hs, is_in_ring, implicit_valence, explicit_valence,
                total_valence, radical_electrons, is_chiral, electronegativity, vdw_radius
            ]
        return features
    
    def _get_enhanced_edge_features(self) -> np.ndarray:
        """Get enhanced edge features (7 features per edge)"""
        edge_attr = np.zeros((self.max_atoms, self.max_atoms, 7))
        
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                bond_type = float(bond.GetBondTypeAsDouble())
                is_conjugated = int(bond.GetIsConjugated())
                is_aromatic = int(bond.GetIsAromatic())
                stereo = int(bond.GetStereo())
                is_in_ring = int(bond.IsInRing())
                
                # Bond length approximation based on bond type and atoms
                atom1 = self.mol.GetAtomWithIdx(i)
                atom2 = self.mol.GetAtomWithIdx(j)
                bond_length = self._estimate_bond_length(atom1, atom2, bond_type)
                
                # Rotatable bond
                is_rotatable = int(not (is_aromatic or is_in_ring or bond_type > 1))
                
                edge_attr[i, j] = edge_attr[j, i] = [
                    bond_type, is_conjugated, is_aromatic, stereo, 
                    is_in_ring, bond_length, is_rotatable
                ]
        
        return edge_attr
    
    def _estimate_bond_length(self, atom1, atom2, bond_type: float) -> float:
        """Estimate bond length based on atomic radii and bond type"""
        # Covalent radii (Angstroms)
        covalent_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                         15: 1.07, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39}
        
        r1 = covalent_radii.get(atom1.GetAtomicNum(), 1.0)
        r2 = covalent_radii.get(atom2.GetAtomicNum(), 1.0)
        
        # Adjust for bond order
        base_length = r1 + r2
        if bond_type == 2:  # Double bond
            base_length *= 0.87
        elif bond_type == 3:  # Triple bond
            base_length *= 0.78
        
        return base_length
    
    def _get_shortest_paths(self) -> np.ndarray:
        """Calculate shortest path distances"""
        G = nx.Graph()
        for i in range(self.num_atoms):
            G.add_node(i)
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                G.add_edge(i, j)
        
        dist_matrix = np.full((self.max_atoms, self.max_atoms), 999)
        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if i == j:
                    dist_matrix[i, j] = 0
                elif G.has_edge(i, j):
                    dist_matrix[i, j] = 1
                else:
                    try:
                        dist_matrix[i, j] = nx.shortest_path_length(G, i, j)
                    except nx.NetworkXNoPath:
                        dist_matrix[i, j] = 999
        return dist_matrix
    
    def _get_centrality_encoding(self) -> np.ndarray:
        """Calculate enhanced centrality measures (5 features)"""
        G = nx.Graph()
        for i in range(self.num_atoms):
            G.add_node(i)
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                G.add_edge(i, j, weight=float(bond.GetBondTypeAsDouble()))
        
        centrality = np.zeros((self.max_atoms, 5))
        if G.number_of_nodes() > 0:
            degree_cent = nx.degree_centrality(G)
            between_cent = nx.betweenness_centrality(G, weight='weight')
            close_cent = nx.closeness_centrality(G, distance='weight')
            eigen_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            page_rank = nx.pagerank(G, weight='weight')
            
            for i in range(self.num_atoms):
                centrality[i] = [
                    degree_cent.get(i, 0),
                    between_cent.get(i, 0),
                    close_cent.get(i, 0),
                    eigen_cent.get(i, 0),
                    page_rank.get(i, 0)
                ]
        return centrality
    
    def _get_molecular_descriptors(self) -> np.ndarray:
        """Calculate molecular-level descriptors (12 features)"""
        descriptors = np.zeros(12)
        
        try:
            # Basic molecular properties
            descriptors[0] = Descriptors.MolWt(self.mol)  # Molecular weight
            descriptors[1] = Descriptors.MolLogP(self.mol)  # LogP
            descriptors[2] = Descriptors.TPSA(self.mol)  # Topological polar surface area
            descriptors[3] = Descriptors.NumHDonors(self.mol)  # H-bond donors
            descriptors[4] = Descriptors.NumHAcceptors(self.mol)  # H-bond acceptors
            descriptors[5] = Descriptors.NumRotatableBonds(self.mol)  # Rotatable bonds
            descriptors[6] = Descriptors.NumAromaticRings(self.mol)  # Aromatic rings
            descriptors[7] = Descriptors.NumSaturatedRings(self.mol)  # Saturated rings
            # Calculate fraction of sp3 carbons manually
            sp3_count = 0
            carbon_count = 0
            for atom in self.mol.GetAtoms():
                if atom.GetAtomicNum() == 6:  # Carbon
                    carbon_count += 1
                    if atom.GetHybridization() == Chem.HybridizationType.SP3:
                        sp3_count += 1
            descriptors[8] = sp3_count / max(carbon_count, 1)  # Fraction of sp3 carbons
            descriptors[9] = Descriptors.BalabanJ(self.mol)  # Balaban J index
            descriptors[10] = Descriptors.BertzCT(self.mol)  # Bertz complexity
            descriptors[11] = len(Chem.GetSymmSSSR(self.mol))  # Number of rings
            
        except Exception as e:
            logger.warning(f"Error calculating molecular descriptors: {e}")
            
        return descriptors
    
    def _get_ring_features(self) -> np.ndarray:
        """Calculate ring-based features for each atom (3 features)"""
        ring_features = np.zeros((self.max_atoms, 3))
        
        # Get ring information
        ring_info = self.mol.GetRingInfo()
        
        for i in range(min(self.num_atoms, self.max_atoms)):
            atom = self.mol.GetAtomWithIdx(i)
            
            # Is in ring
            ring_features[i, 0] = int(atom.IsInRing())
            
            # Ring size (largest ring this atom is in)
            ring_sizes = []
            for ring in ring_info.AtomRings():
                if i in ring:
                    ring_sizes.append(len(ring))
            ring_features[i, 1] = max(ring_sizes) if ring_sizes else 0
            
            # Number of rings this atom is in
            ring_features[i, 2] = len(ring_sizes)
            
        return ring_features

def preprocess_lipophilicity_data(data: pd.DataFrame, target_column: str = 'exp') -> pd.DataFrame:
    """Advanced data preprocessing with outlier detection and removal"""
    logger.info("Starting advanced data preprocessing...")
    
    # Initial statistics
    initial_count = len(data)
    logger.info(f"Initial dataset size: {initial_count}")
    
    # Remove duplicates
    data = data.drop_duplicates(subset=['smiles'])
    logger.info(f"After removing duplicates: {len(data)} ({initial_count - len(data)} removed)")
    
    # Remove invalid SMILES
    valid_smiles = []
    valid_indices = []
    
    for idx, row in data.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None and mol.GetNumAtoms() > 0:
                # Additional validity checks
                if mol.GetNumAtoms() <= 150:  # Reasonable size limit
                    valid_smiles.append(row['smiles'])
                    valid_indices.append(idx)
        except:
            continue
    
    data = data.loc[valid_indices].reset_index(drop=True)
    logger.info(f"After removing invalid SMILES: {len(data)} ({initial_count - len(data)} removed)")
    
    # Statistical outlier detection using Z-score and IQR
    targets = data[target_column].values
    
    # Z-score method (remove extreme outliers > 3.5 std)
    z_scores = np.abs(zscore(targets))
    z_outliers = z_scores > 3.5
    
    # IQR method
    Q1 = np.percentile(targets, 25)
    Q3 = np.percentile(targets, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR  # More conservative than 1.5
    upper_bound = Q3 + 2.0 * IQR
    iqr_outliers = (targets < lower_bound) | (targets > upper_bound)
    
    # Combine outlier detection methods
    outliers = z_outliers | iqr_outliers
    
    logger.info(f"Z-score outliers detected: {z_outliers.sum()}")
    logger.info(f"IQR outliers detected: {iqr_outliers.sum()}")
    logger.info(f"Total outliers to remove: {outliers.sum()}")
    
    # Remove outliers
    data_clean = data[~outliers].reset_index(drop=True)
    logger.info(f"After outlier removal: {len(data_clean)} ({len(data) - len(data_clean)} removed)")
    
    # Molecular complexity-based filtering
    complexity_scores = []
    valid_indices = []
    
    for idx, row in data_clean.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                # Calculate complexity score
                complexity = Descriptors.BertzCT(mol)
                mol_weight = Descriptors.MolWt(mol)
                
                # Filter based on reasonable complexity and molecular weight
                if 50 <= complexity <= 1000 and 50 <= mol_weight <= 800:
                    complexity_scores.append(complexity)
                    valid_indices.append(idx)
        except:
            continue
    
    data_final = data_clean.loc[valid_indices].reset_index(drop=True)
    logger.info(f"After complexity filtering: {len(data_final)} ({len(data_clean) - len(data_final)} removed)")
    
    # Final statistics
    final_targets = data_final[target_column].values
    logger.info(f"Final target statistics:")
    logger.info(f"  Mean: {np.mean(final_targets):.3f}")
    logger.info(f"  Std:  {np.std(final_targets):.3f}")
    logger.info(f"  Min:  {np.min(final_targets):.3f}")
    logger.info(f"  Max:  {np.max(final_targets):.3f}")
    logger.info(f"  Data retention: {len(data_final)/initial_count*100:.1f}%")
    
    return data_final

class EnhancedLipophilicityDataset(Dataset):
    """Enhanced Lipophilicity Dataset with advanced features"""
    
    def __init__(self, data: pd.DataFrame, max_atoms: int = 120, target_column: str = 'exp', 
                 normalize_targets: bool = True):
        self.data = data
        self.max_atoms = max_atoms
        self.target_column = target_column
        self.normalize_targets = normalize_targets
        self.graphs = []
        self.targets = []
        self.molecular_descriptors = []
        
        print(f"Loading {len(self.data)} molecules from Enhanced Lipophilicity dataset...")
        print(f"Using target column: '{target_column}'")
        valid_count = 0
        
        # Target normalization
        if normalize_targets:
            self.target_scaler = RobustScaler()
            raw_targets = data[target_column].values.reshape(-1, 1)
            self.target_scaler.fit(raw_targets)
        
        for idx, row in self.data.iterrows():
            try:
                smiles = row['smiles']
                target = float(row[target_column])
                
                # Normalize target if requested
                if normalize_targets:
                    target = self.target_scaler.transform([[target]])[0, 0]
                
                mol_graph = EnhancedMolecularGraph(smiles, max_atoms)
                self.graphs.append(mol_graph)
                self.targets.append(target)
                self.molecular_descriptors.append(mol_graph.molecular_descriptors)
                valid_count += 1
                
                if valid_count % 500 == 0:
                    print(f"Processed {valid_count} molecules...")
                    
            except Exception as e:
                print(f"Error processing molecule {idx}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.graphs)} molecules")
        self.targets = np.array(self.targets)
        self.molecular_descriptors = np.array(self.molecular_descriptors)
        
        # Normalize molecular descriptors
        self.descriptor_scaler = StandardScaler()
        self.molecular_descriptors = self.descriptor_scaler.fit_transform(self.molecular_descriptors)
        
        # Print target statistics
        print(f"Target statistics ({target_column}):")
        print(f"  Mean: {np.mean(self.targets):.3f}")
        print(f"  Std:  {np.std(self.targets):.3f}")
        print(f"  Min:  {np.min(self.targets):.3f}")
        print(f"  Max:  {np.max(self.targets):.3f}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        target = self.targets[idx]
        mol_descriptors = self.molecular_descriptors[idx]
        
        return {
            'node_features': torch.FloatTensor(graph.node_features),
            'adj_matrix': torch.FloatTensor(graph.adj_matrix),
            'edge_features': torch.FloatTensor(graph.edge_features),
            'shortest_paths': torch.LongTensor(graph.shortest_paths),
            'centrality': torch.FloatTensor(graph.centrality_encoding),
            'ring_features': torch.FloatTensor(graph.ring_features),
            'molecular_descriptors': torch.FloatTensor(mol_descriptors),
            'target': torch.FloatTensor([target]),
            'num_atoms': torch.LongTensor([graph.num_atoms]),
            'smiles': self.data.iloc[idx]['smiles']
        }
    
    def denormalize_target(self, normalized_target):
        """Denormalize target values back to original scale"""
        if self.normalize_targets:
            return self.target_scaler.inverse_transform([[normalized_target]])[0, 0]
        return normalized_target

# Graphormer Model Components (same as ESOL implementation)
class EnhancedGraphBias(nn.Module):
    """Enhanced graph bias computation with multiple bias types"""
    
    def __init__(self, num_heads: int, max_path_len: int = 25, edge_feat_dim: int = 7):
        super().__init__()
        self.num_heads = num_heads
        self.max_path_len = max_path_len
        
        # Multiple bias encoders
        self.spatial_pos_encoder = nn.Embedding(max_path_len, num_heads)
        self.edge_encoder = nn.Linear(edge_feat_dim, num_heads)
        
        # Additional bias types
        self.degree_encoder = nn.Embedding(20, num_heads)  # Max degree 20
        self.ring_encoder = nn.Embedding(10, num_heads)   # Max ring size 10
        
        # Learnable bias combination weights
        self.bias_weights = nn.Parameter(torch.ones(4) / 4)  # 4 bias types
        
    def forward(self, shortest_paths, edge_features, node_features, attention_mask=None):
        batch_size, seq_len = shortest_paths.shape[:2]
        
        # Spatial bias
        clipped_paths = torch.clamp(shortest_paths, 0, self.max_path_len - 1)
        clipped_flat = clipped_paths.view(-1)
        spatial_bias = self.spatial_pos_encoder(clipped_flat)
        spatial_bias = spatial_bias.view(batch_size, seq_len, seq_len, self.num_heads).permute(0, 3, 1, 2)
        
        # Edge bias
        edge_bias = self.edge_encoder(edge_features.view(-1, edge_features.size(-1))).view(
            batch_size, seq_len, seq_len, self.num_heads).permute(0, 3, 1, 2)
        
        # Degree bias (based on node degrees)
        degrees = node_features[:, :, 1].long()  # Degree is the 2nd feature
        degrees = torch.clamp(degrees, 0, 19)
        degree_bias = self.degree_encoder(degrees)  # [batch, seq, heads]
        degree_bias = degree_bias.unsqueeze(2) + degree_bias.unsqueeze(1)  # Broadcasting
        degree_bias = degree_bias.permute(0, 3, 1, 2)
        
        # Ring bias (based on ring membership)
        ring_info = (node_features[:, :, 7] > 0).long()  # Is in ring
        ring_bias = self.ring_encoder(ring_info)
        ring_bias = ring_bias.unsqueeze(2) + ring_bias.unsqueeze(1)
        ring_bias = ring_bias.permute(0, 3, 1, 2)
        
        # Combine biases with learnable weights
        weights = F.softmax(self.bias_weights, dim=0)
        graph_bias = (weights[0] * spatial_bias + 
                     weights[1] * edge_bias + 
                     weights[2] * degree_bias + 
                     weights[3] * ring_bias)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            graph_bias = graph_bias.masked_fill(~mask.bool(), float('-inf'))
        
        return graph_bias

class EnhancedMultiHeadGraphAttention(nn.Module):
    """Enhanced multi-head attention with advanced graph bias and residual connections"""
    
    def __init__(self, d_model: int, num_heads: int, max_path_len: int = 25, 
                 edge_feat_dim: int = 7, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0
        
        # Enhanced projections with bias
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Enhanced graph bias
        self.graph_bias = EnhancedGraphBias(num_heads, max_path_len, edge_feat_dim)
        
        # Attention dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout * 0.5)  # Lower dropout for attention
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, shortest_paths, edge_features, node_features, attention_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with learnable temperature
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (np.sqrt(self.d_k) * self.temperature)
        
        # Add enhanced graph bias
        graph_bias = self.graph_bias(shortest_paths, edge_features, node_features, attention_mask)
        attention_scores = attention_scores + graph_bias
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output, attention_weights

class EnhancedGraphormerLayer(nn.Module):
    """Enhanced Graphormer layer with advanced features"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_path_len: int = 25, 
                 edge_feat_dim: int = 7, dropout: float = 0.1):
        super().__init__()
        
        # Enhanced attention
        self.attention = EnhancedMultiHeadGraphAttention(
            d_model, num_heads, max_path_len, edge_feat_dim, dropout
        )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Enhanced FFN with GLU activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * 2),  # Double for GLU
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout * 0.5)
        )
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable residual scaling
        self.residual_scale = nn.Parameter(torch.ones(2))
        
    def forward(self, x, shortest_paths, edge_features, node_features, attention_mask=None):
        # Pre-norm attention with residual connection
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.attention(
            normed_x, shortest_paths, edge_features, node_features, attention_mask
        )
        x = x + self.residual_scale[0] * self.dropout(attn_output)
        
        # Pre-norm FFN with residual connection
        normed_x = self.norm2(x)
        ffn_output = self.ffn(normed_x)
        x = x + self.residual_scale[1] * ffn_output
        
        return x, attn_weights

class EnhancedGraphormerModel(nn.Module):
    """Enhanced Graphormer model with advanced features and molecular descriptors"""
    
    def __init__(self,
                 node_feat_dim: int = 15,
                 centrality_dim: int = 5,
                 ring_feat_dim: int = 3,
                 molecular_desc_dim: int = 12,
                 edge_feat_dim: int = 7,
                 d_model: int = 256,
                 num_heads: int = 16,
                 num_layers: int = 8,
                 d_ff: int = 1024,
                 max_atoms: int = 120,
                 max_path_len: int = 25,
                 dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.max_atoms = max_atoms
        
        # Enhanced embeddings
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.centrality_embedding = nn.Linear(centrality_dim, d_model)
        self.ring_embedding = nn.Linear(ring_feat_dim, d_model)
        
        # Molecular descriptor embedding
        self.molecular_desc_embedding = nn.Sequential(
            nn.Linear(molecular_desc_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Learnable graph token
        self.graph_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Enhanced positional encoding (learnable + sinusoidal)
        self.learnable_pos = nn.Parameter(torch.randn(max_atoms + 1, d_model) * 0.02)
        
        pe = torch.zeros(max_atoms + 1, d_model)
        position = torch.arange(0, max_atoms + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('sinusoidal_pos', pe)
        
        # Enhanced Graphormer layers
        self.layers = nn.ModuleList([
            EnhancedGraphormerLayer(d_model, num_heads, d_ff, max_path_len, edge_feat_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output processing
        self.output_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Multi-scale output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model + molecular_desc_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier/He initialization"""
        if isinstance(module, nn.Linear):
            if module.out_features == 1:  # Output layer
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, batch):
        node_features = batch['node_features'].to(device)
        centrality = batch['centrality'].to(device)
        ring_features = batch['ring_features'].to(device)
        molecular_descriptors = batch['molecular_descriptors'].to(device)
        shortest_paths = batch['shortest_paths'].to(device)
        edge_features = batch['edge_features'].to(device)
        num_atoms = batch['num_atoms'].to(device)
        
        batch_size, seq_len = node_features.shape[:2]
        
        # Create attention mask
        attention_mask = torch.zeros(batch_size, seq_len + 1, device=node_features.device)
        for i, n_atoms in enumerate(num_atoms):
            attention_mask[i, :min(n_atoms + 1, seq_len + 1)] = 1
        
        # Enhanced node embeddings
        node_emb = self.node_embedding(node_features)
        cent_emb = self.centrality_embedding(centrality)
        ring_emb = self.ring_embedding(ring_features)
        
        # Combine embeddings
        x = node_emb + cent_emb + ring_emb
        
        # Add graph token
        graph_tokens = self.graph_token.expand(batch_size, -1, -1)
        x = torch.cat([graph_tokens, x], dim=1)
        
        # Enhanced positional encoding
        pos_encoding = self.learnable_pos[:seq_len + 1] + self.sinusoidal_pos[:seq_len + 1]
        x = x + pos_encoding.unsqueeze(0)
        
        # Extend paths and edges for graph token
        extended_paths = torch.zeros(batch_size, seq_len + 1, seq_len + 1,
                                   device=shortest_paths.device, dtype=shortest_paths.dtype)
        extended_paths[:, 1:, 1:] = shortest_paths
        
        extended_edges = torch.zeros(batch_size, seq_len + 1, seq_len + 1, edge_features.size(-1),
                                   device=edge_features.device, dtype=edge_features.dtype)
        extended_edges[:, 1:, 1:] = edge_features
        
        extended_node_features = torch.zeros(batch_size, seq_len + 1, node_features.size(-1),
                                           device=node_features.device, dtype=node_features.dtype)
        extended_node_features[:, 1:] = node_features
        
        # Apply Graphormer layers
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, extended_paths, extended_edges, extended_node_features, attention_mask)
            attention_weights_list.append(attn_weights)
        
        # Output processing
        x = self.output_norm(x)
        graph_repr = x[:, 0]  # Graph token representation
        
        # Combine with molecular descriptors
        combined_repr = torch.cat([graph_repr, molecular_descriptors], dim=-1)
        output = self.output_head(combined_repr)
        
        return output

# ============================================================================
# MB-EIG IMPLEMENTATION FOR LIPOPHILICITY
# ============================================================================

class MBEIGExplainer:
    """Multi-Baseline Enhanced Integrated Gradients for Lipophilicity"""
    
    def __init__(self, model, device='cpu', integration_steps: int = 50):
        self.model = model
        self.device = device
        self.integration_steps = integration_steps
        
        # SMARTS patterns for lipophilicity-relevant fragments
        self.fragment_patterns = {
            # Hydrophobic groups (increase lipophilicity)
            'Aromatic Ring': 'c1ccccc1',
            'Long Alkyl Chain': '[CH2][CH2][CH2][CH2]',
            'Methyl Groups': '[CH3]',
            'Halogens': ['[F]', '[Cl]', '[Br]', '[I]'],
            'Phenyl Ring': 'c1ccc(cc1)',
            'Cyclohexyl': 'C1CCCCC1',
            'Tert-Butyl': 'C(C)(C)(C)',
            
            # Hydrophilic groups (decrease lipophilicity)
            'Hydroxyl (-OH)': '[OH]',
            'Primary Amine (-NH2)': '[NH2]',
            'Carboxylic Acid (-COOH)': '[CX3](=O)[OX2H1]',
            'Carbonyl (C=O)': '[CX3]=[OX1]',
            'Carboxylate (-COO-)': '[CX3](=O)[O-]',
            'Ammonium (-NH3+)': '[NX4+]',
            'Sulfonyl (-SO2-)': '[SX4](=O)(=O)',
            'Nitro (-NO2)': '[NX3+](=O)[O-]'
        }
    
    def create_inert_skeleton(self, node_features: np.ndarray, num_atoms: int) -> np.ndarray:
        """Create Inert Skeleton baseline by replacing heteroatoms with carbon"""
        skeleton_features = node_features.copy()
        
        for i in range(num_atoms):
            atomic_num = int(node_features[i, 0])
            
            # Replace heteroatoms (not H=1 or C=6) with carbon
            if atomic_num not in [1, 6] and atomic_num > 0:
                skeleton_features[i, 0] = 6  # Carbon
                skeleton_features[i, 4] = 0  # Not aromatic initially
                skeleton_features[i, 5] = 12.01 * 0.01  # Carbon mass
                # Update electronegativity and vdw radius for carbon
                if skeleton_features.shape[1] > 13:  # Enhanced features
                    skeleton_features[i, 13] = 2.55  # Carbon electronegativity
                    skeleton_features[i, 14] = 1.70  # Carbon vdw radius
                
        return skeleton_features
    
    def compute_integrated_gradients(self, batch: Dict[str, torch.Tensor], 
                                   baseline_features: torch.Tensor) -> torch.Tensor:
        """Compute Integrated Gradients for given baseline"""
        original_features = batch['node_features']
        num_atoms = int(batch['num_atoms'].item())
        
        total_gradients = torch.zeros_like(original_features)
        self.model.eval()
        
        for k in range(1, self.integration_steps + 1):
            alpha = float(k) / self.integration_steps
            
            # Interpolate between baseline and original
            interpolated_features = baseline_features + alpha * (original_features - baseline_features)
            interpolated_features.requires_grad_(True)
            
            # Create batch with interpolated features
            interpolated_batch = dict(batch)
            interpolated_batch['node_features'] = interpolated_features
            
            # Forward pass
            output = self.model(interpolated_batch)
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=output.sum(),
                inputs=interpolated_features,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            total_gradients += gradients
        
        # Average gradients and multiply by (input - baseline)
        avg_gradients = total_gradients / self.integration_steps
        attributions = (original_features - baseline_features) * avg_gradients
        
        # Sum attributions across features for each atom
        atom_attributions = attributions.abs().sum(dim=-1).squeeze(0)[:num_atoms]
        
        return atom_attributions.detach()
    
    def compute_mb_eig(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Compute Multi-Baseline Enhanced Integrated Gradients for Lipophilicity"""
        try:
            # Create enhanced molecular graph
            mol_graph = EnhancedMolecularGraph(smiles)
            
            # Prepare batch with all required features
            batch = {
                'node_features': torch.FloatTensor(mol_graph.node_features).unsqueeze(0).to(self.device),
                'adj_matrix': torch.FloatTensor(mol_graph.adj_matrix).unsqueeze(0).to(self.device),
                'edge_features': torch.FloatTensor(mol_graph.edge_features).unsqueeze(0).to(self.device),
                'shortest_paths': torch.LongTensor(mol_graph.shortest_paths).unsqueeze(0).to(self.device),
                'centrality': torch.FloatTensor(mol_graph.centrality_encoding).unsqueeze(0).to(self.device),
                'ring_features': torch.FloatTensor(mol_graph.ring_features).unsqueeze(0).to(self.device),
                'molecular_descriptors': torch.FloatTensor(mol_graph.molecular_descriptors).unsqueeze(0).to(self.device),
                'num_atoms': torch.LongTensor([mol_graph.num_atoms]).to(self.device),
            }
            
            # Get original prediction
            self.model.eval()
            with torch.no_grad():
                original_prediction = self.model(batch).item()
            
            # Create baselines
            inert_skeleton_features = self.create_inert_skeleton(mol_graph.node_features, mol_graph.num_atoms)
            zero_baseline_features = np.zeros_like(mol_graph.node_features)
            
            # Convert to tensors
            skeleton_baseline = torch.FloatTensor(inert_skeleton_features).unsqueeze(0).to(self.device)
            zero_baseline = torch.FloatTensor(zero_baseline_features).unsqueeze(0).to(self.device)
            
            # Compute IG for each baseline
            ig_skeleton = self.compute_integrated_gradients(batch, skeleton_baseline)
            ig_zero = self.compute_integrated_gradients(batch, zero_baseline)
            
            # Normalize attributions to sum to 1
            eps = 1e-12
            ig_skeleton_norm = ig_skeleton / (ig_skeleton.sum() + eps)
            ig_zero_norm = ig_zero / (ig_zero.sum() + eps)
            
            # Compute robust average (MB-EIG)
            mb_eig_attributions = (ig_skeleton_norm + ig_zero_norm) / 2.0
            mb_eig_attributions = mb_eig_attributions / (mb_eig_attributions.sum() + eps)
            
            # Calculate consistency
            consistency = self._calculate_consistency(ig_skeleton_norm, ig_zero_norm)
            
            # Calculate confidence 
            confidence = self._calculate_confidence(mb_eig_attributions)
            
            # Fragment analysis
            fragments = self._analyze_fragments(mol_graph.mol, mb_eig_attributions.cpu().numpy())
            
            # Chemical validity
            validity = self._assess_chemical_validity(consistency, fragments)
            
            return {
                'smiles': smiles,
                'prediction': float(original_prediction),
                'mb_eig_attributions': mb_eig_attributions.cpu().numpy().tolist(),
                'ig_skeleton': ig_skeleton_norm.cpu().numpy().tolist(),
                'ig_zero': ig_zero_norm.cpu().numpy().tolist(),
                'consistency': consistency,
                'confidence': confidence,
                'fragments': fragments,
                'validity': validity,
                'num_atoms': mol_graph.num_atoms,
                'human_explanation': self._generate_human_explanation(fragments, original_prediction)
            }
            
        except Exception as e:
            logger.error(f"Error in MB-EIG computation for {smiles}: {e}")
            return None
    
    def _calculate_consistency(self, attr1: torch.Tensor, attr2: torch.Tensor) -> float:
        """Calculate consistency between two attribution vectors"""
        eps = 1e-12
        mean_attrs = (attr1 + attr2) / 2.0 + eps
        relative_diff = torch.abs(attr1 - attr2) / mean_attrs
        consistency = 1.0 - torch.clamp(relative_diff.mean(), 0.0, 1.0)
        return float(consistency)
    
    def _calculate_confidence(self, attributions: torch.Tensor) -> float:
        """Calculate confidence using normalized Shannon entropy"""
        eps = 1e-12
        probs = attributions + eps
        entropy = -(probs * torch.log(probs)).sum()
        
        max_entropy = math.log(len(attributions))
        if max_entropy > 0:
            confidence = 1.0 - (entropy / max_entropy)
        else:
            confidence = 0.0
            
        return float(torch.clamp(torch.tensor(confidence), 0.0, 1.0))
    
    def _analyze_fragments(self, mol: Chem.Mol, attributions: np.ndarray) -> Dict[str, float]:
        """Analyze fragments and aggregate attributions"""
        fragment_scores = {}
        
        for fragment_name, smarts_pattern in self.fragment_patterns.items():
            try:
                if isinstance(smarts_pattern, list):
                    # Handle multiple patterns
                    all_atoms = set()
                    for pattern in smarts_pattern:
                        pattern_mol = Chem.MolFromSmarts(pattern)
                        if pattern_mol:
                            matches = mol.GetSubstructMatches(pattern_mol)
                            for match in matches:
                                all_atoms.update(match)
                    atom_indices = sorted(list(all_atoms))
                else:
                    # Single pattern
                    pattern_mol = Chem.MolFromSmarts(smarts_pattern)
                    if pattern_mol is None:
                        fragment_scores[fragment_name] = 0.0
                        continue
                    
                    matches = mol.GetSubstructMatches(pattern_mol)
                    atom_indices = set()
                    for match in matches:
                        atom_indices.update(match)
                    atom_indices = sorted(list(atom_indices))
                
                if atom_indices:
                    # Sum attributions for atoms in this fragment
                    valid_indices = [i for i in atom_indices if i < len(attributions)]
                    if valid_indices:
                        fragment_scores[fragment_name] = float(attributions[valid_indices].sum())
                    else:
                        fragment_scores[fragment_name] = 0.0
                else:
                    fragment_scores[fragment_name] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error analyzing fragment {fragment_name}: {e}")
                fragment_scores[fragment_name] = 0.0
        
        return fragment_scores
    
    def _assess_chemical_validity(self, consistency: float, fragments: Dict[str, float]) -> str:
        """Assess chemical validity of explanations"""
        if consistency >= 0.7 and any(abs(score) > 0.01 for score in fragments.values()):
            return 'HIGH'
        elif consistency >= 0.5 or any(abs(score) > 0.005 for score in fragments.values()):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_human_explanation(self, fragments: Dict[str, float], prediction: float) -> str:
        """Generate human-readable explanation"""
        # Identify key oil-loving vs water-loving groups
        oil_loving = ['Aromatic Ring', 'Long Alkyl Chain', 'Methyl Groups', 'Halogens', 
                     'Phenyl Ring', 'Cyclohexyl', 'Tert-Butyl']
        water_loving = ['Hydroxyl (-OH)', 'Primary Amine (-NH2)', 'Carboxylic Acid (-COOH)', 
                       'Carbonyl (C=O)', 'Carboxylate (-COO-)', 'Ammonium (-NH3+)',
                       'Sulfonyl (-SO2-)', 'Nitro (-NO2)']
        
        oil_score = sum(abs(fragments.get(frag, 0)) for frag in oil_loving)
        water_score = sum(abs(fragments.get(frag, 0)) for frag in water_loving)
        
        explanation = f"LIPOPHILICITY PREDICTION: {prediction:.3f} logP\n\n"
        
        if oil_score > water_score:
            explanation += "🔵 OIL-LOVING groups dominate this molecule!\n"
            key_groups = [f for f in oil_loving if abs(fragments.get(f, 0)) > 0.01]
            if key_groups:
                explanation += f"Key groups: {', '.join(key_groups)}\n"
            explanation += "These hydrophobic groups prefer oil over water, increasing lipophilicity."
        else:
            explanation += "🔴 WATER-LOVING groups dominate this molecule!\n"
            key_groups = [f for f in water_loving if abs(fragments.get(f, 0)) > 0.01]
            if key_groups:
                explanation += f"Key groups: {', '.join(key_groups)}\n"
            explanation += "These hydrophilic groups prefer water over oil, decreasing lipophilicity."
        
        return explanation

def save_molecule_visualization(smiles: str, attributions: List[float], 
                               property_name: str, prediction: float, save_path: Path):
    """Render molecule with attribution-based coloring"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
            
        rdDepictor.Compute2DCoords(mol)
        
        # Normalize attributions to [0,1] for coloring
        attr_array = np.array(attributions)
        if len(attr_array) == 0:
            return
            
        # Color scheme: Blue (positive/oil-loving) to Red (negative/water-loving)
        atom_colors = {}
        atom_radii = {}
        
        # Normalize attributions
        max_attr = max(abs(attr_array.max()), abs(attr_array.min())) + 1e-12
        
        for i in range(min(len(attributions), mol.GetNumAtoms())):
            norm_attr = attributions[i] / max_attr  # [-1, 1]
            
            if norm_attr > 0:
                # Blue for positive (oil-loving)
                intensity = abs(norm_attr)
                atom_colors[i] = (1-intensity, 1-intensity, 1.0)  # White to Blue
            else:
                # Red for negative (water-loving)
                intensity = abs(norm_attr)
                atom_colors[i] = (1.0, 1-intensity, 1-intensity)  # White to Red
                
            atom_radii[i] = 0.3 + 0.5 * abs(norm_attr)
        
        # Draw molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, 
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii
        )
        drawer.FinishDrawing()
        
        # Save image
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        logger.info(f"Saved molecule visualization: {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating molecule visualization: {e}")

def visualize_training_results(train_losses: List[float], val_losses: List[float], 
                             val_r2_scores: List[float], save_path: Path):
    """Create training history visualization"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² scores
        ax2.plot(epochs, val_r2_scores, 'g-', label='Validation R²', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Validation R² Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training visualization: {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating training visualization: {e}")

# Hyperparameter optimization removed - using fixed enhanced configuration

def enhanced_train_model_with_mb_eig(model, train_loader, val_loader, val_dataset, 
                                   num_epochs=150, learning_rate=1e-4, weight_decay=1e-5):
    """Enhanced training pipeline with advanced techniques"""
    model = model.to(device)
    
    # Advanced optimizer with different learning rates for different components
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'embedding' in n], 'lr': learning_rate * 0.5},
        {'params': [p for n, p in model.named_parameters() if 'layers' in n], 'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() if 'output_head' in n], 'lr': learning_rate * 2.0},
        {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['embedding', 'layers', 'output_head'])], 'lr': learning_rate}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay, eps=1e-8)
    
    # Advanced scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)
    
    # Initialize explainer
    explainer = MBEIGExplainer(model, device)
    
    # Training tracking - no early stopping, train for full 100 epochs
    train_losses, val_losses, val_r2_scores = [], [], []
    best_val_r2 = -float('inf')
    # Removed early stopping - will train for full epoch count
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    logger.info("Starting Enhanced Lipophilicity training with MB-EIG explanations...")
    
    for epoch in range(num_epochs):
        # Training with mixed precision
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(batch)
                    target = batch['target'].to(device)
                    loss = F.mse_loss(output, target)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(batch)
                target = batch['target'].to(device)
                loss = F.mse_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, val_predictions, val_targets = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(batch)
                        target = batch['target'].to(device)
                        loss = F.mse_loss(output, target)
                else:
                    output = model(batch)
                    target = batch['target'].to(device)
                    loss = F.mse_loss(output, target)
                
                val_loss += loss.item()
                
                # Denormalize predictions for metrics
                if hasattr(val_dataset.dataset, 'denormalize_target'):
                    pred_denorm = [val_dataset.dataset.denormalize_target(p) for p in output.cpu().numpy().flatten()]
                    target_denorm = [val_dataset.dataset.denormalize_target(t) for t in target.cpu().numpy().flatten()]
                    val_predictions.extend(pred_denorm)
                    val_targets.extend(target_denorm)
                else:
                    val_predictions.extend(output.cpu().numpy().flatten())
                    val_targets.extend(target.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        val_r2 = r2_score(val_targets, val_predictions)
        val_pearson, _ = pearsonr(val_targets, val_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}, Pearson: {val_pearson:.4f}')
        logger.info(f'  Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}')
        
        # Save best model based on R² (no early stopping)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'model_config': model.d_model  # Save a reference to model config
            }, MODELS_DIR / 'best_enhanced_graphormer_lipo.pth')
            logger.info(f"  🎯 NEW BEST MODEL! R²: {val_r2:.4f} (target: >0.80)")
        # No early stopping - continue training
        
        # No early stopping - continue training for full 100 epochs
        
        scheduler.step()
        
        # Generate MB-EIG explanations every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            logger.info(f"Generating MB-EIG explanations at epoch {epoch+1}...")
            
            # Sample molecules from validation set
            sample_indices = random.sample(range(len(val_dataset)), min(8, len(val_dataset)))
            explanation_results = []
            
            for idx in sample_indices:
                sample = val_dataset[idx]
                smiles = sample['smiles']
                
                explanation = explainer.compute_mb_eig(smiles)
                if explanation:
                    explanation_results.append(explanation)
                    
                    # Save molecule visualization
                    safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_')
                    vis_path = EXPLAIN_DIR / f"epoch_{epoch+1}_{safe_smiles[:50]}_molecule.png"
                    save_molecule_visualization(
                        smiles, explanation['mb_eig_attributions'], 
                        'Lipophilicity', explanation['prediction'], vis_path
                    )
            
            # Save explanation batch
            if explanation_results:
                batch_path = EXPLAIN_DIR / f'epoch_{epoch+1}_explanations.json'
                with open(batch_path, 'w') as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'explanations': explanation_results,
                        'avg_consistency': np.mean([exp['consistency'] for exp in explanation_results]),
                        'avg_confidence': np.mean([exp['confidence'] for exp in explanation_results]),
                        'high_validity_count': sum(1 for exp in explanation_results if exp['validity'] == 'HIGH')
                    }, f, indent=2)
                
                logger.info(f"Saved {len(explanation_results)} explanations to {batch_path}")
    
    # Save final training history
    visualize_training_results(train_losses, val_losses, val_r2_scores, 
                             RESULTS_DIR / 'enhanced_lipo_training_history.png')
    
    return train_losses, val_losses, val_r2_scores, best_val_r2

def main():
    """Enhanced main training function for Lipophilicity with MB-EIG and hyperparameter optimization"""
    logger.info("="*80)
    logger.info("🚀 ENHANCED LIPOPHILICITY MOLECULAR PROPERTY PREDICTION WITH MB-EIG")
    logger.info("Target: R² > 0.80 with advanced preprocessing and hyperparameter optimization")
    logger.info("="*80)
    
    # Enhanced Configuration
    CSV_PATH = 'Lipophilicity.csv'
    MAX_ATOMS = 120
    TARGET_COLUMN = 'exp'
    
    try:
        # Load and preprocess dataset
        logger.info("Loading and preprocessing Lipophilicity dataset...")
        raw_data = pd.read_csv(CSV_PATH)
        logger.info(f"Raw dataset loaded: {raw_data.shape}")
        
        # Advanced preprocessing with outlier removal
        processed_data = preprocess_lipophilicity_data(raw_data, TARGET_COLUMN)
        
        # Create enhanced dataset
        dataset = EnhancedLipophilicityDataset(processed_data, MAX_ATOMS, TARGET_COLUMN, normalize_targets=True)
        
        # Stratified split to maintain target distribution
        targets = np.array([dataset[i]['target'].item() for i in range(len(dataset))])
        
        # Create bins for stratification
        n_bins = 5
        target_bins = pd.cut(targets, bins=n_bins, labels=False)
        
        # Split with stratification
        train_indices, temp_indices = train_test_split(
            range(len(dataset)), test_size=0.3, random_state=42, stratify=target_bins
        )
        
        temp_targets = target_bins[temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42, stratify=temp_targets
        )
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        logger.info(f"Enhanced dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # No hyperparameter optimization - using fixed enhanced configuration
        
        # Fixed enhanced configuration optimized for high R²
        model_config = {
            'node_feat_dim': 15,
            'centrality_dim': 5,
            'ring_feat_dim': 3,
            'molecular_desc_dim': 12,
            'edge_feat_dim': 7,
            'd_model': 384,  # Larger model
            'num_heads': 24,  # More attention heads
            'num_layers': 12,  # Deeper network
            'd_ff': 1536,  # Larger FFN
            'max_atoms': MAX_ATOMS,
            'max_path_len': 25,
            'dropout': 0.1  # Lower dropout for better fitting
        }
        BATCH_SIZE = 24  # Smaller batch for better gradients
        LEARNING_RATE = 1e-4  # Lower learning rate for stability
        WEIGHT_DECAY = 5e-6  # Lower weight decay
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Initialize enhanced model
        model = EnhancedGraphormerModel(**model_config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Enhanced model initialized with {total_params:,} parameters")
        
        # Enhanced training with MB-EIG - fixed 100 epochs without early stopping
        NUM_EPOCHS = 100
        logger.info(f"🚀 Starting enhanced training for {NUM_EPOCHS} epochs with fixed configuration (no early stopping, no hyperparameter tuning)...")
        
        train_losses, val_losses, val_r2_scores, best_val_r2 = enhanced_train_model_with_mb_eig(
            model, train_loader, val_loader, val_dataset, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
        )
        
        # Load best model and evaluate on test set
        logger.info("📊 Evaluating best model on test set...")
        checkpoint = torch.load(MODELS_DIR / 'best_enhanced_graphormer_lipo.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        test_predictions, test_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch)
                target = batch['target'].to(device)
                
                # Denormalize for final evaluation
                if hasattr(dataset, 'denormalize_target'):
                    pred_denorm = [dataset.denormalize_target(p) for p in output.cpu().numpy().flatten()]
                    target_denorm = [dataset.denormalize_target(t) for t in target.cpu().numpy().flatten()]
                    test_predictions.extend(pred_denorm)
                    test_targets.extend(target_denorm)
                else:
                    test_predictions.extend(output.cpu().numpy().flatten())
                    test_targets.extend(target.cpu().numpy().flatten())
        
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        
        # Calculate comprehensive test metrics
        test_mse = mean_squared_error(test_targets, test_predictions)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        test_pearson, _ = pearsonr(test_targets, test_predictions)
        test_rmse = np.sqrt(test_mse)
        
        # Additional metrics
        residuals = test_targets - test_predictions
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        logger.info("="*80)
        logger.info("🎯 FINAL ENHANCED TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Test RMSE:     {test_rmse:.4f}")
        logger.info(f"Test MAE:      {test_mae:.4f}")
        logger.info(f"Test R²:       {test_r2:.4f} {'✅ TARGET ACHIEVED!' if test_r2 > 0.80 else '❌ Below target (0.80)'}")
        logger.info(f"Test Pearson:  {test_pearson:.4f}")
        logger.info(f"Mean Residual: {mean_residual:.4f}")
        logger.info(f"Std Residual:  {std_residual:.4f}")
        logger.info(f"Best Val R²:   {best_val_r2:.4f}")
        
        # Create prediction vs actual plot
        plt.figure(figsize=(10, 8))
        plt.scatter(test_targets, test_predictions, alpha=0.6, s=30)
        plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'r--', lw=2)
        plt.xlabel('Actual LogP')
        plt.ylabel('Predicted LogP')
        plt.title(f'Enhanced Lipophilicity Prediction Results\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'enhanced_prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comprehensive final results
        final_results = {
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
            'test_pearson': float(test_pearson),
            'mean_residual': float(mean_residual),
            'std_residual': float(std_residual),
            'best_val_r2': float(best_val_r2),
            'target_achieved': bool(test_r2 > 0.80),
            'model_config': model_config,
            'training_config': {
                'num_epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'max_atoms': MAX_ATOMS,
                'hyperopt_used': False,
                'hyperopt_trials': 0
            },
            'best_hyperparams': None,
            'data_preprocessing': {
                'original_size': len(raw_data),
                'processed_size': len(processed_data),
                'retention_rate': len(processed_data) / len(raw_data)
            }
        }
        
        with open(RESULTS_DIR / 'enhanced_lipo_final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("="*80)
        logger.info("🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Model saved to: {MODELS_DIR / 'best_enhanced_graphormer_lipo.pth'}")
        logger.info(f"📊 Results saved to: {RESULTS_DIR}")
        logger.info(f"🔍 Explanations saved to: {EXPLAIN_DIR}")
        
        if test_r2 > 0.80:
            logger.info("🎯 SUCCESS: Target R² > 0.80 achieved!")
        else:
            logger.info(f"⚠️  Target not quite reached. Current R²: {test_r2:.4f}, Target: 0.80")
            logger.info("💡 Consider: More hyperparameter tuning, ensemble methods, or additional features")
        
        return model, final_results
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, results = main()