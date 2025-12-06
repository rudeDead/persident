"""
ESOL Single Molecule Prediction with MB-EIG Explanations

This script provides complete explanations for individual molecules using the trained ESOL model.
Usage: python prediction.py --smiles "CCC(C)CO"
       python prediction.py --smiles "CCO" --output custom_output_dir

Output includes:
- Prediction value and confidence
- MB-EIG attributions with dual baseline analysis
- Fragment analysis with chemical interpretation  
- Molecule visualization with attribution coloring
- Human-readable explanation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
import warnings
import os
import math
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------
# Setup logging and directories  
# -------------------------
MODELS_DIR = Path('../models')
OUTPUT_DIR = Path('../xai')

def setup_logging(output_dir: Path):
    """Setup logging for prediction script"""
    log_file = output_dir / 'prediction.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# COPY ALL MODEL COMPONENTS FROM TRAINING.PY (Same architecture)
# ============================================================================

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
        clipped_flat = clipped_paths.view(-1)
        spatial_bias = self.spatial_pos_encoder(clipped_flat)
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
# MB-EIG EXPLAINER (Same as training.py)
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
                'canonical_smiles': mol_graph.smiles,
                'property': 'water_solubility',
                'unit': 'log(mol/L)',
                'predicted_value': float(original_prediction),
                'mb_eig_attributions': mb_eig_attributions.cpu().numpy().tolist(),
                'baseline_attributions': {
                    'ig_skeleton': ig_skeleton_norm.cpu().numpy().tolist(),
                    'ig_zero': ig_zero_norm.cpu().numpy().tolist()
                },
                'explanation_quality': {
                    'consistency': consistency,
                    'confidence': confidence,
                    'validity': validity
                },
                'fragments': self._format_fragments_for_output(fragments, mol_graph.mol, mb_eig_attributions.cpu().numpy()),
                'human_explanation': self._generate_human_explanation(fragments, original_prediction),
                'timestamp': datetime.now().isoformat(),
                'num_atoms': mol_graph.num_atoms
            }
            
        except Exception as e:
            return {'error': f"MB-EIG computation failed: {str(e)}"}
    
    # --------------------------------------------------------------------
    # --- START OF UPDATED BLOCK ---
    # --------------------------------------------------------------------
    
    def _calculate_consistency(self, attr1: torch.Tensor, attr2: torch.Tensor) -> float:
        """
        Calculate consistency between two attribution vectors
        using Cosine Similarity.
        
        This replaces the previous flawed formula which did not
        correctly handle the sparse (zero-filled) skeleton vector.
        """
        eps = 1e-12 # Epsilon for numerical stability

        # Flatten tensors to 1D vectors for cosine similarity
        attr1 = attr1.flatten()
        attr2 = attr2.flatten()

        # Calculate cosine similarity.
        # This measures the angle between the two attribution vectors.
        # A value of 1.0 means they are identical.
        # A value of 0.0 means they are totally different (orthogonal).
        # F.cosine_similarity is imported as torch.nn.functional as F
        cos_sim = F.cosine_similarity(attr1, attr2, dim=0, eps=eps)
        
        # The result of cosine similarity is -1 to 1.
        # Since attributions are (normalized) positive, the result will be 0 to 1.
        # We clamp it just to be safe.
        consistency = torch.clamp(cos_sim, 0.0, 1.0)
        
        # Return as a standard float
        # .item() is needed to convert a 0-dim tensor to a Python float
        return float(consistency.item())
    
    # --------------------------------------------------------------------
    # --- END OF UPDATED BLOCK ---
    # --------------------------------------------------------------------
    
    def _calculate_confidence(self, attributions: torch.Tensor) -> float:
        """Calculate confidence using normalized Shannon entropy"""
        eps = 1e-12
        probs = attributions + eps
        entropy = -(probs * torch.log(probs)).sum()
        
        # Add epsilon to max_entropy to avoid division by zero for single-atom molecules
        max_entropy = math.log(len(attributions)) + eps
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
                fragment_scores[fragment_name] = 0.0
        
        return fragment_scores
    
    def _format_fragments_for_output(self, fragment_scores: Dict[str, float], 
                                     mol: Chem.Mol, attributions: np.ndarray) -> Dict[str, Any]:
        """Format fragment analysis for JSON output"""
        formatted_fragments = {}
        
        for fragment_name, score in fragment_scores.items():
            if abs(score) > 1e-6:  # Only include non-zero fragments
                # Find atom indices for this fragment
                smarts_pattern = self.fragment_patterns[fragment_name]
                atom_indices = []
                
                try:
                    if isinstance(smarts_pattern, list):
                        all_atoms = set()
                        for pattern in smarts_pattern:
                            pattern_mol = Chem.MolFromSmarts(pattern)
                            if pattern_mol:
                                matches = mol.GetSubstructMatches(pattern_mol)
                                for match in matches:
                                    all_atoms.update(match)
                        atom_indices = sorted(list(all_atoms))
                    else:
                        pattern_mol = Chem.MolFromSmarts(smarts_pattern)
                        if pattern_mol:
                            matches = mol.GetSubstructMatches(pattern_mol)
                            atom_indices = set()
                            for match in matches:
                                atom_indices.update(match)
                            atom_indices = sorted(list(atom_indices))
                except:
                    atom_indices = []
                
                # Generate interpretation
                if score > 0:
                    interpretation = "Increases water solubility (water-loving)"
                else:
                    interpretation = "Decreases water solubility (water-hating)"
                
                formatted_fragments[fragment_name] = {
                    "atoms": atom_indices,
                    "attribution": float(score),
                    "interpretation": interpretation
                }
        
        return formatted_fragments
    
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
        
        explanation = "To see why this molecule dissolves in water, we compared it to its 'oily twin' (carbon-only version).\n\n"
        
        if loving_score > hating_score:
            explanation += "🔵 The BLUE parts (water-loving groups) dominate this molecule!\n"
            key_groups = [f for f in water_loving if abs(fragments.get(f, 0)) > 0.01]
            if key_groups:
                explanation += f"Key groups: {', '.join(key_groups)}\n"
            explanation += "These polar groups act as 'water hooks', pulling the molecule into solution.\n\n"
            explanation += f"Prediction: {prediction:.3f} log(mol/L) - INCREASED water solubility"
        else:
            explanation += "🔴 The RED parts (water-hating groups) dominate this molecule!\n"
            key_groups = [f for f in water_hating if abs(fragments.get(f, 0)) > 0.01]
            if key_groups:
                explanation += f"Key groups: {', '.join(key_groups)}\n"
            explanation += "These hydrophobic groups resist water, keeping the molecule out of solution.\n\n"
            explanation += f"Prediction: {prediction:.3f} log(mol/L) - DECREASED water solubility"
        
        return explanation

def save_molecule_visualization(smiles: str, attributions: List[float], 
                                prediction: float, output_dir: Path, logger):
    """Create and save molecule visualization with attributions"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Cannot create molecule from SMILES: {smiles}")
            return None
            
        rdDepictor.Compute2DCoords(mol)
        
        # Normalize attributions for coloring
        attr_array = np.array(attributions)
        if len(attr_array) == 0:
            return None
            
        # Color scheme: Blue (water-loving) to Red (water-hating)
        atom_colors = {}
        atom_radii = {}
        
        # Normalize to [-1, 1] range
        max_attr = max(abs(attr_array.max()), abs(attr_array.min())) + 1e-12
        
        for i in range(min(len(attributions), mol.GetNumAtoms())):
            norm_attr = attributions[i] / max_attr
            
            if norm_attr > 0:
                # Blue for positive (water-loving)
                intensity = abs(norm_attr)
                atom_colors[i] = (1-intensity, 1-intensity, 1.0)
            else:
                # Red for negative (water-hating)
                intensity = abs(norm_attr)
                atom_colors[i] = (1.0, 1-intensity, 1-intensity)
                
            atom_radii[i] = 0.3 + 0.7 * abs(norm_attr)
        
        # Create visualization
        safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_')
        img_path = output_dir / f"{safe_smiles}_attribution_map.png"
        
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
        drawer.drawOptions().addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, 
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii
        )
        drawer.FinishDrawing()
        
        # Save image
        with open(img_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        logger.info(f"Molecule visualization saved: {img_path}")
        return str(img_path)
        
    except Exception as e:
        logger.error(f"Error creating molecule visualization: {e}")
        return None

def create_attribution_plots(explanation_data: Dict[str, Any], output_dir: Path, logger):
    """Create attribution analysis plots"""
    try:
        smiles = explanation_data['smiles']
        safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # Extract data
        mb_eig = explanation_data['mb_eig_attributions']
        ig_skeleton = explanation_data['baseline_attributions']['ig_skeleton']
        ig_zero = explanation_data['baseline_attributions']['ig_zero']
        fragments = explanation_data['fragments']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"MB-EIG Attribution Analysis\nSMILES: {smiles}", fontsize=14)
        
        # Plot 1: Baseline comparison
        atom_indices = range(1, len(mb_eig) + 1)
        width = 0.25
        
        axes[0, 0].bar([x - width for x in atom_indices], ig_skeleton, width, 
                       label='IG Skeleton', color='green', alpha=0.7)
        axes[0, 0].bar(atom_indices, ig_zero, width, 
                       label='IG Zero', color='orange', alpha=0.7)
        axes[0, 0].bar([x + width for x in atom_indices], mb_eig, width, 
                       label='MB-EIG (Average)', color='blue', alpha=0.7)
        
        axes[0, 0].set_xlabel('Atom Index')
        axes[0, 0].set_ylabel('Attribution Score')
        axes[0, 0].set_title('Dual Baseline Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MB-EIG attributions only
        colors = ['blue' if x > 0 else 'red' for x in mb_eig]
        axes[0, 1].bar(atom_indices, mb_eig, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Atom Index')
        axes[0, 1].set_ylabel('MB-EIG Attribution')
        axes[0, 1].set_title('MB-EIG Attributions (Blue=Water-loving, Red=Water-hating)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Fragment contributions
        if fragments:
            frag_names = list(fragments.keys())
            frag_scores = [fragments[name]['attribution'] for name in frag_names]
            
            colors = ['blue' if x > 0 else 'red' for x in frag_scores]
            bars = axes[1, 0].bar(range(len(frag_names)), frag_scores, color=colors, alpha=0.7)
            axes[1, 0].set_xlabel('Fragment Type')
            axes[1, 0].set_ylabel('Fragment Attribution')
            axes[1, 0].set_title('Fragment-Level Analysis')
            axes[1, 0].set_xticks(range(len(frag_names)))
            axes[1, 0].set_xticklabels(frag_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, frag_scores):
                height = bar.get_height()
                axes[1, 0].annotate(f'{score:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Quality metrics
        quality = explanation_data['explanation_quality']
        metrics = ['Consistency', 'Confidence']
        values = [quality['consistency'], quality['confidence']]
        
        bars = axes[1, 1].bar(metrics, values, color=['green', 'purple'], alpha=0.7)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title(f"Explanation Quality (Validity: {quality['validity']})")
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].annotate(f'{val:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"{safe_smiles}_attribution_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attribution plots saved: {plot_path}")
        return str(plot_path)
        
    except Exception as e:
        logger.error(f"Error creating attribution plots: {e}")
        return None

def load_trained_model(model_path: Path, device) -> GraphormerModel:
    """Load the trained Graphormer model"""
    # Model configuration (must match training)
    model_config = {
        'node_feat_dim': 9,
        'centrality_dim': 3,
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 512,
        'max_atoms': 100,
        'max_path_len': 20,
        'dropout': 0.1
    }
    
    model = GraphormerModel(**model_config)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def save_text_explanation(explanation_data: Dict[str, Any], output_dir: Path, logger):
    """Save human-readable text explanation"""
    try:
        smiles = explanation_data['smiles']
        safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        text_path = output_dir / f"{safe_smiles}_explanation.txt"
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("MOLECULAR PROPERTY PREDICTION EXPLANATION\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Input SMILES: {explanation_data['smiles']}\n")
            f.write(f"Canonical SMILES: {explanation_data['canonical_smiles']}\n")
            f.write(f"Property: {explanation_data['property'].replace('_', ' ').title()}\n")
            f.write(f"Predicted Value: {explanation_data['predicted_value']:.3f} {explanation_data['unit']}\n\n")
            
            f.write("EXPLANATION:\n")
            f.write("-" * 20 + "\n")
            f.write(explanation_data['human_explanation'] + "\n\n")
            
            f.write("TECHNICAL DETAILS:\n")
            f.write("-" * 20 + "\n")
            quality = explanation_data['explanation_quality']
            f.write(f"Consistency Score: {quality['consistency']:.3f} (agreement between baselines)\n")
            f.write(f"Confidence Score: {quality['confidence']:.3f} (attribution certainty)\n")
            f.write(f"Chemical Validity: {quality['validity']}\n\n")
            
            f.write("KEY FRAGMENTS:\n")
            f.write("-" * 20 + "\n")
            fragments = explanation_data['fragments']
            if fragments:
                for frag_name, frag_data in fragments.items():
                    f.write(f"• {frag_name}: {frag_data['attribution']:.3f}\n")
                    f.write(f"  {frag_data['interpretation']}\n")
                    f.write(f"  Atoms: {frag_data['atoms']}\n\n")
            else:
                f.write("No significant fragments detected.\n\n")
            
            f.write(f"Analysis completed: {explanation_data['timestamp']}\n")
            f.write("Generated by MB-EIG Explainer for ESOL\n")
        
        logger.info(f"Text explanation saved: {text_path}")
        return str(text_path)
        
    except Exception as e:
        logger.error(f"Error saving text explanation: {e}")
        return None

def predict_single_molecule(smiles: str, output_dir: Path = None, model_path: Path = None):
    """Complete prediction and explanation pipeline for a single molecule"""
    
    # Setup output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("ESOL SINGLE MOLECULE PREDICTION WITH MB-EIG")
    logger.info("=" * 80)
    logger.info(f"Input SMILES: {smiles}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load trained model
        if model_path is None:
            model_path = MODELS_DIR / 'best_graphormer_esol.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        model = load_trained_model(model_path, device)
        logger.info("Model loaded successfully")
        
        # Initialize explainer
        explainer = MBEIGExplainer(model, device, integration_steps=50)
        logger.info("MB-EIG explainer initialized")
        
        # Compute explanation
        logger.info("Computing MB-EIG explanation...")
        explanation = explainer.compute_mb_eig(smiles)
        
        if explanation is None or 'error' in explanation:
            error_msg = explanation.get('error', 'Unknown error') if explanation else 'Explanation returned None'
            logger.error(f"Explanation failed: {error_msg}")
            return None
        
        logger.info("MB-EIG explanation computed successfully")
        logger.info(f"Prediction: {explanation['predicted_value']:.3f} {explanation['unit']}")
        logger.info(f"Consistency: {explanation['explanation_quality']['consistency']:.3f}")
        logger.info(f"Confidence: {explanation['explanation_quality']['confidence']:.3f}")
        logger.info(f"Validity: {explanation['explanation_quality']['validity']}")
        
        # Save results
        safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # 1. Save JSON explanation
        json_path = output_dir / f"{safe_smiles}_prediction.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(explanation, f, indent=2)
        logger.info(f"JSON explanation saved: {json_path}")
        
        # 2. Save molecule visualization
        img_path = save_molecule_visualization(
            smiles, explanation['mb_eig_attributions'], 
            explanation['predicted_value'], output_dir, logger
        )
        
        # 3. Save attribution plots
        plot_path = create_attribution_plots(explanation, output_dir, logger)
        
        # 4. Save text explanation
        text_path = save_text_explanation(explanation, output_dir, logger)
        
        logger.info("=" * 80)
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Generated files:")
        logger.info(f"  📄 JSON: {json_path}")
        if img_path:
            logger.info(f"  🖼️  Molecule: {img_path}")
        if plot_path:
            logger.info(f"  📊 Plots: {plot_path}")
        if text_path:
            logger.info(f"  📝 Text: {text_path}")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="ESOL Single Molecule Prediction with MB-EIG")
    parser.add_argument('--smiles', type=str, required=True,
                        help='SMILES string of the molecule to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: ../outputs/explanations)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (default: ../models/best_graphormer_esol.pth)')
    
    args = parser.parse_args()
    
    # Convert paths
    output_dir = Path(args.output) if args.output else None
    model_path = Path(args.model) if args.model else None
    
    # Run prediction
    result = predict_single_molecule(args.smiles, output_dir, model_path)
    
    if result:
        print("\n" + "=" * 80)
        print("PREDICTION SUMMARY")
        print("=" * 80)
        print(f"SMILES: {result['smiles']}")
        print(f"Predicted Solubility: {result['predicted_value']:.3f} {result['unit']}")
        print(f"Explanation Quality: {result['explanation_quality']['validity']}")
        print("\nHUMAN EXPLANATION:")
        print("-" * 40)
        print(result['human_explanation'])
        print("=" * 80)
    else:
        print("❌ Prediction failed. Check the log for details.")

if __name__ == "__main__":
    main()