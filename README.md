---
title: MB-EIG Molecular Property Prediction
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Explainable AI for Molecular Property Prediction

This application provides **Explainable AI for Molecular Property Prediction** using:
- **Graphormer**: Transformer-based Graph Neural Network
- **MB-EIG**: Multi-Baseline Enhanced Integrated Gradients
- **Three Properties**: ESOL (Solubility), Lipophilicity (logP), FreeSolv (Hydration Energy)

## Features

- 🔬 Predict molecular properties from SMILES strings
- 🎨 Visual attribution maps showing atom importance
- 📊 Dual baseline comparison (Zero vs Skeleton)
- 🧪 Fragment-level analysis
- 💬 Human-readable explanations

## Usage

1. Select a property type (ESOL, Lipophilicity, or FreeSolv)
2. Enter a SMILES string (e.g., `CCO` for ethanol)
3. Click "Predict" to see results
4. Download visualizations and analysis

## Example SMILES

- `CCO` - Ethanol
- `c1ccccc1` - Benzene
- `CC(=O)O` - Acetic Acid
- `c1ccc(cc1)O` - Phenol

## Technology

- Flask web framework
- PyTorch for deep learning
- RDKit for molecular visualization
- Graphormer architecture
- MB-EIG explainability method

## Author

TANMAY | RIT Rajaramnagar | CSE (AI & ML)
