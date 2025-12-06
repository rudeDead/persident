"""
Quick test to verify visualization generation works
"""

import sys
from pathlib import Path

# Test imports
try:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    print("✅ All visualization libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test molecule visualization
def test_molecule_viz():
    print("\n🧪 Testing molecule visualization...")
    try:
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(mol)
        
        # Create simple visualization
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Save test image
        outputs_dir = Path('outputs')
        outputs_dir.mkdir(exist_ok=True)
        
        test_path = outputs_dir / 'test_molecule.png'
        with open(test_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        print(f"✅ Molecule visualization saved: {test_path}")
        return True
    except Exception as e:
        print(f"❌ Molecule visualization failed: {e}")
        return False

# Test matplotlib plot
def test_matplotlib_plot():
    print("\n📊 Testing matplotlib plot...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create sample data
        x = [1, 2, 3, 4, 5]
        y = [0.2, 0.4, 0.3, 0.5, 0.1]
        
        ax.bar(x, y, color='blue', alpha=0.7)
        ax.set_xlabel('Atom Index')
        ax.set_ylabel('Attribution')
        ax.set_title('Test Attribution Plot')
        ax.grid(True, alpha=0.3)
        
        # Save test plot
        outputs_dir = Path('outputs')
        test_path = outputs_dir / 'test_plot.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Matplotlib plot saved: {test_path}")
        return True
    except Exception as e:
        print(f"❌ Matplotlib plot failed: {e}")
        return False

# Test attribution coloring
def test_attribution_coloring():
    print("\n🎨 Testing attribution coloring...")
    try:
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(mol)
        
        # Sample attributions
        attributions = [0.3, 0.5, 0.2]
        
        # Create colors
        atom_colors = {}
        atom_radii = {}
        
        max_attr = max(attributions)
        for i, attr in enumerate(attributions):
            intensity = attr / max_attr
            atom_colors[i] = (1-intensity, 1-intensity, 1.0)  # Blue gradient
            atom_radii[i] = 0.3 + 0.5 * intensity
        
        # Draw with colors
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        drawer.drawOptions().addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii
        )
        drawer.FinishDrawing()
        
        # Save colored molecule
        outputs_dir = Path('outputs')
        test_path = outputs_dir / 'test_colored_molecule.png'
        with open(test_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        print(f"✅ Colored molecule saved: {test_path}")
        return True
    except Exception as e:
        print(f"❌ Attribution coloring failed: {e}")
        return False

def main():
    print("="*60)
    print("🧪 VISUALIZATION SYSTEM TEST")
    print("="*60)
    
    results = {
        'Molecule Visualization': test_molecule_viz(),
        'Matplotlib Plot': test_matplotlib_plot(),
        'Attribution Coloring': test_attribution_coloring()
    }
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nVisualization system is working correctly.")
        print("Check the 'outputs/' folder for test images.")
        print("\nYou can now run the web application:")
        print("   python app.py")
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("\nPlease check the error messages above.")
    
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
