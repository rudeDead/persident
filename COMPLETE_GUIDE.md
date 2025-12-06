# Complete Web Application Guide

## 🎯 Overview

This web application provides **Explainable AI for Molecular Property Prediction** using:
- **Graphormer**: Transformer-based Graph Neural Network
- **MB-EIG**: Multi-Baseline Enhanced Integrated Gradients
- **Three Properties**: ESOL (Solubility), Lipophilicity (logP), FreeSolv (Hydration Energy)

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Test Visualization System
```bash
python test_visualization.py
```

### Step 3: Start the Application
```bash
python app.py
```

Then open: **http://localhost:5000**

## 📊 What You'll See

When you make a prediction, the results are displayed in this order:

### 1. **Prediction Value & Quality** ⭐
```
Predicted Value: -2.345 log(mol/L)
Consistency: 87% ✅
Confidence: 92% ✅
Validity: HIGH ✅
```

### 2. **Baseline Attributions** (3 Charts Horizontally) 📊

#### Chart 1: IG Zero Baseline
- Compares molecule to empty space
- Shows raw importance of all atoms

#### Chart 2: IG Skeleton Baseline
- Compares molecule to carbon-only version
- Shows importance of heteroatoms (O, N, etc.)

#### Chart 3: MB-EIG (Final)
- Robust average of both baselines
- Most reliable attribution values

### 3. **Molecule Attribution Map** 🎨
- PNG image showing the molecule
- Atoms colored by importance
- Darker blue = more important
- Atom indices labeled

### 4. **Detailed Analysis Plot** 📈
- 4-panel comprehensive visualization:
  - Baseline comparison
  - MB-EIG attributions
  - Fragment analysis
  - Quality metrics

### 5. **Fragment Analysis** 🧪
```
• Hydroxyl (-OH): 0.43
  Increases water solubility (water-loving)
  Atoms: [2]

• Aromatic Ring: 0.28
  Decreases water solubility (water-hating)
  Atoms: [3, 4, 5, 6, 7, 8]
```

### 6. **Human Explanation** 💬
```
To see why this molecule dissolves in water, we compared 
it to its 'oily twin' (carbon-only version).

🔵 The BLUE parts (water-loving groups) dominate this molecule!
Key groups: Hydroxyl (-OH), Primary Amine (-NH2)
These polar groups act as 'water hooks', pulling the 
molecule into solution.

Prediction: -2.345 log(mol/L) - INCREASED water solubility
```

## 🎨 Visual Examples

### Example 1: Ethanol (CCO)
```
Input: CCO
Property: ESOL (Water Solubility)

Results:
✓ Prediction: -0.234 log(mol/L) (highly soluble)
✓ Key Group: Hydroxyl (-OH) on oxygen atom
✓ Explanation: "Water-loving groups dominate"
✓ Images: Attribution map + analysis plot
```

### Example 2: Benzene (c1ccccc1)
```
Input: c1ccccc1
Property: Lipophilicity

Results:
✓ Prediction: 2.13 logP (lipophilic)
✓ Key Group: Aromatic Ring
✓ Explanation: "Oil-loving groups dominate"
✓ Images: Attribution map + analysis plot
```

## 📁 File Organization

### Input Files (You Provide)
```
models/
├── best_graphormer_esol.pth      ← ESOL model
├── best_graphormer_lipo.pth      ← Lipo model
└── best_graphormer_freesolv.pth  ← FreeSolv model
```

### Output Files (Auto-Generated)
```
outputs/
├── CCO_attribution_map.png           ← Colored molecule
├── CCO_attribution_analysis.png      ← 4-panel plot
├── CCO_prediction.json               ← Full results
└── CCO_explanation.txt               ← Text summary
```

### Application Files
```
.
├── app.py                    ← Flask backend
├── prediction_Esol.py       ← ESOL prediction
├── prediction_lipo.py       ← Lipo prediction
├── prediction_Free.py       ← FreeSolv prediction
├── templates/               ← HTML pages
│   ├── base.html
│   ├── index.html
│   ├── esol.html
│   ├── lipo.html
│   ├── freesolv.html
│   └── about.html
└── static/                  ← CSS & JavaScript
    ├── css/style.css
    └── js/prediction.js
```

## 🔧 Technical Details

### Backend (Flask)
- **Port**: 5000 (default)
- **Models**: Loaded on startup
- **Prediction**: `/api/predict` endpoint
- **Images**: `/outputs/<filename>` endpoint
- **Format**: JSON responses

### Frontend (JavaScript)
- **Framework**: Vanilla JavaScript
- **Charts**: HTML5 Canvas
- **Images**: Dynamic loading
- **Responsive**: Mobile-friendly

### Visualization (RDKit + Matplotlib)
- **Molecule Images**: RDKit MolDraw2D
- **Analysis Plots**: Matplotlib
- **Format**: PNG (300 DPI)
- **Storage**: `outputs/` folder

## 🎯 Use Cases

### 1. Drug Discovery
- Predict solubility of drug candidates
- Understand why molecules are soluble/insoluble
- Optimize molecular structures

### 2. Chemical Education
- Learn about molecular properties
- Understand structure-property relationships
- Visualize chemical concepts

### 3. Research
- Generate explanations for publications
- Compare different molecules
- Validate predictions

## 📊 Understanding the Results

### Consistency Score (0-100%)
- **>70%**: High agreement between baselines ✅
- **50-70%**: Moderate agreement ⚠️
- **<50%**: Low agreement ❌

### Confidence Score (0-100%)
- **>80%**: High confidence ✅
- **60-80%**: Moderate confidence ⚠️
- **<60%**: Low confidence ❌

### Validity Rating
- **HIGH**: Reliable explanation ✅
- **MEDIUM**: Acceptable explanation ⚠️
- **LOW**: Uncertain explanation ❌

## 🐛 Troubleshooting

### Problem: Models not loading
**Solution**: 
```bash
# Check model files exist
ls models/

# Should show:
# best_graphormer_esol.pth
# best_graphormer_lipo.pth
# best_graphormer_freesolv.pth
```

### Problem: Images not displaying
**Solution**:
```bash
# Check outputs folder exists
mkdir outputs

# Check Flask is serving images
curl http://localhost:5000/outputs/test.png
```

### Problem: Charts not rendering
**Solution**:
- Open browser console (F12)
- Check for JavaScript errors
- Verify canvas elements exist

### Problem: Prediction fails
**Solution**:
- Verify SMILES is valid: `CCO` ✅ not `C-C-O` ❌
- Check molecule has <100 atoms
- Review Flask console for errors

## 🧪 Testing

### Test 1: Visualization System
```bash
python test_visualization.py
```
Expected output:
```
✅ Molecule visualization
✅ Matplotlib plot
✅ Attribution coloring
```

### Test 2: Application Setup
```bash
python test_app.py
```
Expected output:
```
✅ Imports
✅ Prediction modules
✅ Model files
✅ Frontend files
✅ Flask app
```

### Test 3: Live Prediction
1. Start app: `python app.py`
2. Open: http://localhost:5000
3. Try: `CCO` on ESOL page
4. Verify: All 6 sections display

## 📚 Example SMILES

### For ESOL (Water Solubility)
```
CCO              - Ethanol (highly soluble)
c1ccccc1         - Benzene (poorly soluble)
CC(=O)O          - Acetic Acid (highly soluble)
CCCCCCCC         - Octane (insoluble)
c1ccc(cc1)O      - Phenol (moderately soluble)
```

### For Lipophilicity
```
c1ccccc1         - Benzene (lipophilic)
CCO              - Ethanol (hydrophilic)
CCCCCCCC         - Octane (very lipophilic)
c1ccc(cc1)Cl     - Chlorobenzene (lipophilic)
CC(C)(C)c1ccccc1 - Tert-butylbenzene (very lipophilic)
```

### For FreeSolv (Hydration Energy)
```
CCO              - Ethanol (favorable)
CC(C)O           - Isopropanol (favorable)
c1ccccc1         - Benzene (unfavorable)
CC(=O)O          - Acetic Acid (favorable)
CCN              - Ethylamine (favorable)
```

## 🎓 Educational Value

### What Students Learn
1. **Explainable AI**: How to interpret ML predictions
2. **Dual Baselines**: Why multiple baselines improve robustness
3. **Chemical Intuition**: Structure-property relationships
4. **Visualization**: How to communicate AI results

### What Researchers Get
1. **Reliable Explanations**: MB-EIG provides robust attributions
2. **Chemical Insights**: Fragment analysis connects to chemistry
3. **Publication-Ready**: High-quality visualizations
4. **Reproducible**: Complete code and documentation

## 🚀 Advanced Usage

### Custom Integration Steps
1. Import prediction module
2. Load trained model
3. Call `compute_mb_eig(smiles)`
4. Process results

### Example Code
```python
from prediction_Esol import MBEIGExplainer, GraphormerModel
import torch

# Load model
model = GraphormerModel()
checkpoint = torch.load('models/best_graphormer_esol.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create explainer
explainer = MBEIGExplainer(model, device='cpu')

# Get explanation
result = explainer.compute_mb_eig('CCO')
print(result['predicted_value'])
print(result['human_explanation'])
```

## 📞 Support

### Check These First
1. ✅ All dependencies installed?
2. ✅ Model files in `models/` folder?
3. ✅ Python 3.8 or higher?
4. ✅ Port 5000 available?

### Common Issues
- **Import errors**: Run `pip install -r requirements.txt`
- **Model errors**: Verify `.pth` files are valid PyTorch models
- **Port errors**: Change port in `app.py` or kill process on 5000

## 🎉 Success Checklist

Before using the application, verify:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model files in `models/` folder
- [ ] Visualization test passes (`python test_visualization.py`)
- [ ] App test passes (`python test_app.py`)
- [ ] Flask starts without errors (`python app.py`)
- [ ] Can access http://localhost:5000
- [ ] Can make a prediction with `CCO`
- [ ] All 6 result sections display
- [ ] Images load correctly

## 📖 Additional Resources

- **README.md**: Project overview
- **WEB_APP_GUIDE.md**: Detailed usage guide
- **FINAL_IMPLEMENTATION.md**: Technical implementation details
- **CHANGES_SUMMARY.md**: Development notes
- **QUICK_REFERENCE.md**: Quick command reference

---

**Status**: ✅ Production Ready
**Version**: 1.0
**Last Updated**: November 2025
**Author**: TANMAY | RIT Rajaramnagar | CSE (AI & ML)

**Ready to use!** 🚀
