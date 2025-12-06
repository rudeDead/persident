"""
ESOL Molecular Property Prediction with Multi-Baseline Enhanced Integrated Gradients (MB-EIG)

This training pipeline implements the complete MB-EIG system for water solubility prediction.
Features:
- Graphormer architecture for molecular representation
- Dual baseline MB-EIG explanations (Inert Skeleton + Zero baseline)
- Fragment analysis with SMARTS patterns
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
from rdkit.Chem import rdMolDescriptors, rdchem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
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
EXPLAIN_DIR = Path('../outputs/explanations')
RESULTS_DIR = Path('../outputs/plots')
LOGS_DIR = Path('../outputs/logs')
MODELS_DIR = Path('../models')

for dir_path in [EXPLAIN_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

log_file = LOGS_DIR / 'esol_training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("ESOL Training with MB-EIG - Logger initialized")

class MolecularGraph:
    """Convert SMILES to molecular graph representation"""
    
    def __init__(self, smiles: str, max_atoms: int = 100):
        self.smiles = smiles
        self.max_atoms = max_atoms
        self.mol = Chem.MolFromSmiles(smiles, sanitize=True)
        
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Canonicalize SMILES
        self.smiles = Chem.MolToSmiles(self.mol)
        self.num_atoms = min(self.mol.GetNumAtoms(), max_atoms)
        self.adj_matrix = self._get_adjacency_matrix()
        self.node_features = self._get_node_features()
        self.edge_features = self._get_edge_features()
        self.shortest_paths = self._get_shortest_paths()
        self.centrality_encoding = self._get_centrality_encoding()
    
    def _get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix"""
        adj = np.zeros((self.max_atoms, self.max_atoms))
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                adj[i, j] = adj[j, i] = 1
        return adj
    
    def _get_node_features(self) -> np.ndarray:
        """Get atom features"""
        features = np.zeros((self.max_atoms, 9))  # 9 atom features
        
        for i, atom in enumerate(self.mol.GetAtoms()):
            if i >= self.max_atoms:
                break
            features[i] = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass() * 0.01,
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetImplicitValence()
            ]
        return features
    
    def _get_edge_features(self) -> np.ndarray:
        """Get edge features"""
        edge_attr = np.zeros((self.max_atoms, self.max_atoms, 4))
        
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                bond_type = int(bond.GetBondType())
                is_conjugated = int(bond.GetIsConjugated())
                is_aromatic = int(bond.GetIsAromatic())
                stereo = int(bond.GetStereo())
                edge_attr[i, j] = edge_attr[j, i] = [bond_type, is_conjugated, is_aromatic, stereo]
        
        return edge_attr
    
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
        """Calculate centrality measures"""
        G = nx.Graph()
        for i in range(self.num_atoms):
            G.add_node(i)
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.max_atoms and j < self.max_atoms:
                G.add_edge(i, j)
        
        centrality = np.zeros((self.max_atoms, 3))
        if G.number_of_nodes() > 0:
            degree_cent = nx.degree_centrality(G)
            between_cent = nx.betweenness_centrality(G, weight=None)
            close_cent = nx.closeness_centrality(G)
            for i in range(self.num_atoms):
                centrality[i] = [
                    degree_cent.get(i, 0),
                    between_cent.get(i, 0),
                    close_cent.get(i, 0)
                ]
        return centrality

class EsolDataset(Dataset):
    """ESOL Dataset"""
    
    def __init__(self, data: pd.DataFrame, max_atoms: int = 100, target_column: str = 'measured log solubility in mols per litre'):
        self.data = data
        self.max_atoms = max_atoms
        self.target_column = target_column
        self.graphs = []
        self.targets = []
        
        print(f"Loading {len(self.data)} molecules from ESOL dataset...")
        print(f"Using target column: '{target_column}'")
        valid_count = 0
        
        for idx, row in self.data.iterrows():
            try:
                smiles = row['smiles']
                target = float(row[target_column])
                
                mol_graph = MolecularGraph(smiles, max_atoms)
                self.graphs.append(mol_graph)
                self.targets.append(target)
                valid_count += 1
                
                if valid_count % 100 == 0:
                    print(f"Processed {valid_count} molecules...")
                    
            except Exception as e:
                print(f"Error processing molecule {idx}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.graphs)} molecules")
        self.targets = np.array(self.targets)
        
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
        
        return {
            'node_features': torch.FloatTensor(graph.node_features),
            'adj_matrix': torch.FloatTensor(graph.adj_matrix),
            'edge_features': torch.FloatTensor(graph.edge_features),
            'shortest_paths': torch.LongTensor(graph.shortest_paths),
            'centrality': torch.FloatTensor(graph.centrality_encoding),
            'target': torch.FloatTensor([target]),
            'num_atoms': torch.LongTensor([graph.num_atoms]),
            'smiles': self.data.iloc[idx]['smiles']
        }

# Graphormer Model Components (same as existing implementation)
class GraphBias(nn.Module):
    """Graph bias computation (shortest path + edge bias)"""
    
    def __init__(self, num_heads: int, max_path_len: int = 20):
        super().__init__()
        self.num_heads = num_heads
        self.max_path_len = max_path_len
        self.spatial_pos_encoder = nn.Embedding(max_path_len, num_heads)
        self.edge_encoder = nn.Linear(4, num_heads)
    
    def forward(self, shortest_paths, edge_features, attention_mask=None):
        batch_size, seq_len = shortest_paths.shape[:2]
        
        # Clip path lengths
        clipped_paths = torch.clamp(shortest_paths, 0, self.max_path_len - 1)
        # Flatten for embedding
        clipped_flat = clipped_paths.view(-1)  # [batch*seq*seq]
        spatial_bias = self.spatial_pos_encoder(clipped_flat)  # [batch*seq*seq, heads]
        spatial_bias = spatial_bias.view(batch_size, seq_len, seq_len, self.num_heads).permute(0, 3, 1, 2)
        
        # Edge encoding
        edge_bias = self.edge_encoder(edge_features.view(-1, 4)).view(
            batch_size, seq_len, seq_len, self.num_heads).permute(0, 3, 1, 2)
        
        # Combine biases
        graph_bias = spatial_bias + edge_bias
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            graph_bias = graph_bias.masked_fill(~mask.bool(), float('-inf'))
        
        return graph_bias

class MultiHeadGraphAttention(nn.Module):
    """Multi-head attention with graph bias"""
    
    def __init__(self, d_model: int, num_heads: int, max_path_len: int = 20, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.graph_bias = GraphBias(num_heads, max_path_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, shortest_paths, edge_features, attention_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        graph_bias = self.graph_bias(shortest_paths, edge_features, attention_mask)
        attention_scores = attention_scores + graph_bias
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output

class GraphormerLayer(nn.Module):
    """Single Graphormer layer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_path_len: int = 20, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadGraphAttention(d_model, num_heads, max_path_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, shortest_paths, edge_features, attention_mask=None):
        attn_output = self.attention(x, shortest_paths, edge_features, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class GraphormerModel(nn.Module):
    """Complete Graphormer model"""
    
    def __init__(self,
                 node_feat_dim: int = 9,
                 centrality_dim: int = 3,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 512,
                 max_atoms: int = 100,
                 max_path_len: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_atoms = max_atoms
        
        self.node_embedding = nn.Linear(node_feat_dim, d_model)
        self.centrality_embedding = nn.Linear(centrality_dim, d_model)
        self.graph_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Sinusoidal positional encoding
        pe = torch.zeros(max_atoms + 1, d_model)
        position = torch.arange(0, max_atoms + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_embedding = nn.Parameter(pe, requires_grad=False)
        
        self.layers = nn.ModuleList([
            GraphormerLayer(d_model, num_heads, d_ff, max_path_len, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, batch):
        node_features = batch['node_features']
        centrality = batch['centrality']
        shortest_paths = batch['shortest_paths']
        edge_features = batch['edge_features']
        num_atoms = batch['num_atoms']
        
        batch_size, seq_len = node_features.shape[:2]
        
        attention_mask = torch.zeros(batch_size, seq_len + 1, device=node_features.device)
        for i, n_atoms in enumerate(num_atoms):
            attention_mask[i, :n_atoms + 1] = 1
        
        node_features = node_features.to(device)
        node_emb = self.node_embedding(node_features)
        cent_emb = self.centrality_embedding(centrality)
        x = node_emb + cent_emb
        
        graph_tokens = self.graph_token.expand(batch_size, -1, -1)
        x = torch.cat([graph_tokens, x], dim=1)
        
        x = x + self.pos_embedding[:seq_len + 1].unsqueeze(0)
        
        extended_paths = torch.zeros(batch_size, seq_len + 1, seq_len + 1,
                                   device=shortest_paths.device, dtype=shortest_paths.dtype)
        extended_paths[:, 1:, 1:] = shortest_paths
        
        extended_edges = torch.zeros(batch_size, seq_len + 1, seq_len + 1, 4,
                                   device=edge_features.device, dtype=edge_features.dtype)
        extended_edges[:, 1:, 1:] = edge_features
        
        for layer in self.layers:
            x = layer(x, extended_paths, extended_edges, attention_mask)
        
        x = self.output_norm(x)
        graph_repr = x[:, 0]
        output = self.output_head(graph_repr)
        
        return output

# ============================================================================
# MB-EIG IMPLEMENTATION (COMPLETE SYSTEM INTEGRATED)
# ============================================================================

class MBEIGExplainer:
    """Multi-Baseline Enhanced Integrated Gradients for ESOL"""
    
    def __init__(self, model, device='cpu', integration_steps: int = 50):
        self.model = model
        self.device = device
        self.integration_steps = integration_steps
        
        # SMARTS patterns for solubility-relevant fragments
        self.fragment_patterns = {
            # Water-loving groups (increase solubility)
            'Hydroxyl (-OH)': '[OH]',
            'Primary Amine (-NH2)': '[NH2]',
            'Carboxylic Acid (-COOH)': '[CX3](=O)[OX2H1]',
            'Carbonyl (C=O)': '[CX3]=[OX1]',
            'Carboxylate (-COO-)': '[CX3](=O)[O-]',
            'Ammonium (-NH3+)': '[NX4+]',
            
            # Water-hating groups (decrease solubility)
            'Aromatic Ring': 'c1ccccc1',
            'Long Alkyl Chain': '[CH2][CH2][CH2][CH2]',
            'Methyl Groups': '[CH3]',
            'Halogens': ['[F]', '[Cl]', '[Br]', '[I]']
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
        """Compute Multi-Baseline Enhanced Integrated Gradients for ESOL"""
        try:
            # Create molecular graph
            mol_graph = MolecularGraph(smiles)
            
            # Prepare batch
            batch = {
                'node_features': torch.FloatTensor(mol_graph.node_features).unsqueeze(0).to(self.device),
                'adj_matrix': torch.FloatTensor(mol_graph.adj_matrix).unsqueeze(0).to(self.device),
                'edge_features': torch.FloatTensor(mol_graph.edge_features).unsqueeze(0).to(self.device),
                'shortest_paths': torch.LongTensor(mol_graph.shortest_paths).unsqueeze(0).to(self.device),
                'centrality': torch.FloatTensor(mol_graph.centrality_encoding).unsqueeze(0).to(self.device),
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
        # Identify key water-loving vs water-hating groups
        water_loving = ['Hydroxyl (-OH)', 'Primary Amine (-NH2)', 'Carboxylic Acid (-COOH)', 
                       'Carbonyl (C=O)', 'Carboxylate (-COO-)', 'Ammonium (-NH3+)']
        water_hating = ['Aromatic Ring', 'Long Alkyl Chain', 'Methyl Groups', 'Halogens']
        
        loving_score = sum(abs(fragments.get(frag, 0)) for frag in water_loving)
        hating_score = sum(abs(fragments.get(frag, 0)) for frag in water_hating)
        
        explanation = f"SOLUBILITY PREDICTION: {prediction:.3f} log(mol/L)\n\n"
        
        if loving_score > hating_score:
            explanation += "🔵 WATER-LOVING groups dominate this molecule!\n"
            key_groups = [f for f in water_loving if abs(fragments.get(f, 0)) > 0.01]
            if key_groups:
                explanation += f"Key groups: {', '.join(key_groups)}\n"
            explanation += "These polar groups act as 'water hooks', increasing solubility."
        else:
            explanation += "🔴 WATER-HATING groups dominate this molecule!\n"
            key_groups = [f for f in water_hating if abs(fragments.get(f, 0)) > 0.01]
            if key_groups:
                explanation += f"Key groups: {', '.join(key_groups)}\n"
            explanation += "These hydrophobic groups resist water, decreasing solubility."
        
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
            
        # Color scheme: Blue (positive/water-loving) to Red (negative/water-hating)
        atom_colors = {}
        atom_radii = {}
        
        # Normalize attributions
        max_attr = max(abs(attr_array.max()), abs(attr_array.min())) + 1e-12
        
        for i in range(min(len(attributions), mol.GetNumAtoms())):
            norm_attr = attributions[i] / max_attr  # [-1, 1]
            
            if norm_attr > 0:
                # Blue for positive (water-loving)
                intensity = abs(norm_attr)
                atom_colors[i] = (1-intensity, 1-intensity, 1.0)  # White to Blue
            else:
                # Red for negative (water-hating)
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

def train_model_with_mb_eig(model, train_loader, val_loader, val_dataset, 
                           num_epochs=100, learning_rate=1e-4):
    """Training pipeline with MB-EIG explanations every 5 epochs"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize explainer
    explainer = MBEIGExplainer(model, device)
    
    train_losses, val_losses, val_r2_scores = [], [], []
    best_val_loss = float('inf')
    
    logger.info("Starting ESOL training with MB-EIG explanations...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            target = batch['target']
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
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
                output = model(batch)
                target = batch['target']
                loss = F.mse_loss(output, target)
                val_loss += loss.item()
                val_predictions.extend(output.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        val_r2 = r2_score(val_targets, val_predictions)
        val_pearson, _ = pearsonr(val_targets, val_predictions)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}, Pearson: {val_pearson:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, MODELS_DIR / 'best_graphormer_esol.pth')
            logger.info(f"  New best model saved (val_loss: {val_loss:.4f})")
        
        scheduler.step()
        
        # Generate MB-EIG explanations every 5 epochs
        if epoch % 5 == 0:
            logger.info(f"Generating MB-EIG explanations at epoch {epoch+1}...")
            
            # Sample 10 random molecules from validation set
            sample_indices = random.sample(range(len(val_dataset)), min(10, len(val_dataset)))
            explanation_results = []
            
            for idx in sample_indices:
                sample = val_dataset[idx]
                smiles = sample['smiles']
                
                explanation = explainer.compute_mb_eig(smiles)
                if explanation:
                    explanation_results.append(explanation)
                    
                    # Save molecule visualization
                    vis_path = EXPLAIN_DIR / f"epoch_{epoch+1}_{smiles.replace('/', '_')}_molecule.png"
                    save_molecule_visualization(
                        smiles, explanation['mb_eig_attributions'], 
                        'ESOL Solubility', explanation['prediction'], vis_path
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
                             RESULTS_DIR / 'esol_training_history.png')
    
    return train_losses, val_losses, val_r2_scores

def main():
    """Main training function for ESOL with MB-EIG"""
    logger.info("="*80)
    logger.info("ESOL MOLECULAR PROPERTY PREDICTION WITH MB-EIG")
    logger.info("="*80)
    
    # Configuration
    CSV_PATH = 'delaney-processed.csv'
    MAX_ATOMS = 100
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    TARGET_COLUMN = 'measured log solubility in mols per litre'
    
    try:
        # Load dataset
        logger.info("Loading ESOL dataset...")
        data = pd.read_csv(CSV_PATH)
        logger.info(f"Dataset loaded: {data.shape}")
        
        dataset = EsolDataset(data, MAX_ATOMS, TARGET_COLUMN)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, temp_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size + test_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_dataset, test_dataset = torch.utils.data.random_split(
            temp_dataset, [val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Initialize model
        model_config = {
            'node_feat_dim': 9,
            'centrality_dim': 3,
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 512,
            'max_atoms': MAX_ATOMS,
            'max_path_len': 20,
            'dropout': 0.1
        }
        
        model = GraphormerModel(**model_config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters")
        
        # Train with MB-EIG
        train_losses, val_losses, val_r2_scores = train_model_with_mb_eig(
            model, train_loader, val_loader, val_dataset.dataset, NUM_EPOCHS, LEARNING_RATE
        )
        
        # Load best model and evaluate on test set
        logger.info("Evaluating best model on test set...")
        checkpoint = torch.load(MODELS_DIR / 'best_graphormer_esol.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        test_predictions, test_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
                output = model(batch)
                test_predictions.extend(output.cpu().numpy().flatten())
                test_targets.extend(batch['target'].cpu().numpy().flatten())
        
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        
        # Calculate test metrics
        test_mse = mean_squared_error(test_targets, test_predictions)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        test_pearson, _ = pearsonr(test_targets, test_predictions)
        
        logger.info("="*80)
        logger.info("FINAL TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Test RMSE: {np.sqrt(test_mse):.4f}")
        logger.info(f"Test MAE:  {test_mae:.4f}")
        logger.info(f"Test R²:   {test_r2:.4f}")
        logger.info(f"Test Pearson: {test_pearson:.4f}")
        
        # Save final results
        final_results = {
            'test_rmse': float(np.sqrt(test_mse)),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
            'test_pearson': float(test_pearson),
            'model_config': model_config,
            'training_config': {
                'num_epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'max_atoms': MAX_ATOMS
            }
        }
        
        with open(RESULTS_DIR / 'esol_final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {MODELS_DIR / 'best_graphormer_esol.pth'}")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        logger.info(f"Explanations saved to: {EXPLAIN_DIR}")
        
        return model, final_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, results = main()