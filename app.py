"""
Flask Web Application for Molecular Property Prediction
Explainable AI for Drug Discovery
Optimized for Hugging Face Spaces Deployment
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path
import traceback
import numpy as np
from datetime import datetime
import zipfile

# Import prediction modules
sys.path.append(os.path.dirname(__file__))
from prediction_Esol import MolecularGraph, GraphormerModel, MBEIGExplainer, device
import prediction_lipo
import prediction_Free
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

from huggingface_hub import hf_hub_download

HF_REPO = "Tanmay0483/mbeig"  # model repo ID

# Model paths (download from Hugging Face model repo)
print("Downloading model files from Hugging Face...")
ESOL_MODEL = hf_hub_download(repo_id="Tanmay0483/mbeig", filename="best_graphormer_esol.pth")
LIPO_MODEL = hf_hub_download(repo_id="Tanmay0483/mbeig", filename="best_graphormer_lipo.pth")
FREE_MODEL = hf_hub_download(repo_id="Tanmay0483/mbeig", filename="best_graphormer_freesolv.pth")
print("✓ Model files downloaded")

# Load models on startup
models = {}
models_loaded = False

def load_models():
    """Load all three models"""
    global models_loaded
    
    if models_loaded:
        print("Models already loaded, skipping...")
        return
    
    try:
        print("\n" + "="*60)
        print("LOADING MODELS")
        print("="*60)
        
        # ESOL Model
        print("Loading ESOL model...")
        esol_model = GraphormerModel().to(device)
        checkpoint = torch.load(ESOL_MODEL, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            esol_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            esol_model.load_state_dict(checkpoint)
        esol_model.eval()
        models['esol'] = {'model': esol_model, 'explainer': MBEIGExplainer(esol_model, device)}
        print("✓ ESOL model loaded")
        
        # Lipo Model
        print("Loading Lipophilicity model...")
        lipo_model = prediction_lipo.GraphormerModel().to(device)
        checkpoint = torch.load(LIPO_MODEL, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            lipo_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            lipo_model.load_state_dict(checkpoint)
        lipo_model.eval()
        models['lipo'] = {'model': lipo_model, 'explainer': prediction_lipo.MBEIGExplainer(lipo_model, device)}
        print("✓ Lipophilicity model loaded")
        
        # FreeSolv Model
        print("Loading FreeSolv model...")
        free_model = prediction_Free.GraphormerModel().to(device)
        checkpoint = torch.load(FREE_MODEL, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            free_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            free_model.load_state_dict(checkpoint)
        free_model.eval()
        models['free'] = {'model': free_model, 'explainer': prediction_Free.MBEIGExplainer(free_model, device)}
        print("✓ FreeSolv model loaded")
        
        models_loaded = True
        print(f"\n✓ All models loaded successfully. Available: {list(models.keys())}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n✗ Error loading models: {e}")
        traceback.print_exc()
        print("\nPlease ensure model files are accessible from Hugging Face.")
        print("="*60 + "\n")
        raise

# Pre-load models on startup
print("Initializing application and loading models...")
load_models()
print("✓ Application ready to serve requests\n")

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/predict/<property_type>')
def predict_page(property_type):
    """Property-specific prediction pages"""
    if property_type not in ['esol', 'lipo', 'freesolv']:
        return "Invalid property type", 404
    return render_template(f'{property_type}.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """Serve generated output files (images, etc.)"""
    outputs_dir = Path('outputs')
    return send_from_directory(outputs_dir, filename)

@app.route('/api/download/<file_type>/<filename>')
def download_file(file_type, filename):
    """Download individual files"""
    outputs_dir = Path('outputs')
    file_path = outputs_dir / filename
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True, download_name=filename)

@app.route('/api/download-all/<smiles_hash>')
def download_all_files(smiles_hash):
    """Download all files as a ZIP archive"""
    outputs_dir = Path('outputs')
    
    matching_files = list(outputs_dir.glob(f"{smiles_hash}*"))
    
    if not matching_files:
        return jsonify({'error': 'No files found'}), 404
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in matching_files:
            zip_file.write(file_path, file_path.name)
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{smiles_hash}_results.zip'
    )

@app.route('/api/health')
def health():
    """Health check endpoint for Hugging Face Spaces"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'available_models': list(models.keys()),
        'device': str(device)
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        smiles = data.get('smiles', '').strip()
        property_type = data.get('property_type', 'esol')
        
        print(f"\n=== Prediction Request ===")
        print(f"SMILES: {smiles}")
        print(f"Property Type: {property_type}")
        
        model_key = 'free' if property_type == 'freesolv' else property_type
        
        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        if model_key not in models:
            return jsonify({
                'error': f'Invalid property type: {property_type}',
                'available': list(models.keys())
            }), 400
        
        explainer = models[model_key]['explainer']
        
        print("Computing MB-EIG explanation...")
        result = explainer.compute_mb_eig(smiles)
        
        if result is None or 'error' in result:
            error_msg = result.get('error', 'Unknown error') if result else 'Prediction failed'
            print(f"Error in prediction: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        if 'predicted_value' in result:
            prediction_value = result['predicted_value']
        elif 'prediction' in result:
            prediction_value = result['prediction']
        else:
            return jsonify({'error': 'No prediction value in result'}), 500
            
        print(f"Prediction successful: {prediction_value:.3f}")
        
        baseline_attributions = result.get('baseline_attributions', {})
        if not baseline_attributions:
            baseline_attributions = {
                'ig_skeleton': result.get('ig_skeleton', []),
                'ig_zero': result.get('ig_zero', [])
            }
        
        print("Generating visualizations...")
        image_paths, file_paths = generate_and_save_visualizations(result, smiles, property_type)
        
        response = {
            'smiles': result.get('smiles', smiles),
            'canonical_smiles': result.get('canonical_smiles', result.get('smiles', smiles)),
            'property_type': property_type,
            'predicted_value': prediction_value,
            'unit': get_unit_for_property(property_type),
            'mb_eig_attributions': result.get('mb_eig_attributions', []),
            'baseline_attributions': baseline_attributions,
            'explanation_quality': result.get('explanation_quality', {
                'consistency': result.get('consistency', 0),
                'confidence': result.get('confidence', 0),
                'validity': result.get('validity', 'UNKNOWN')
            }),
            'fragments': result.get('fragments', {}),
            'human_explanation': result.get('human_explanation', 'No explanation available'),
            'num_atoms': result.get('num_atoms', 0),
            'images': image_paths,
            'download_files': file_paths,
            'timestamp': result.get('timestamp', datetime.now().isoformat())
        }
        
        print("Request completed successfully\n")
        return jsonify(response)
    
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"\n✗ {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

def get_unit_for_property(property_type):
    """Get the unit for each property type"""
    units = {
        'esol': 'log(mol/L)',
        'lipo': 'logP',
        'freesolv': 'kcal/mol'
    }
    return units.get(property_type, '')

def generate_and_save_visualizations(result, smiles, property_type):
    """Generate and save visualization images and downloadable files"""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_')
    safe_smiles = safe_smiles[:100]
    
    image_paths = {}
    file_paths = {}
    
    try:
        # 1. Generate molecule attribution map
        mol = Chem.MolFromSmiles(smiles)
        if mol and 'mb_eig_attributions' in result:
            rdDepictor.Compute2DCoords(mol)
            
            attributions = result['mb_eig_attributions']
            attr_array = np.array(attributions)
            
            if len(attr_array) > 0:
                atom_colors = {}
                atom_radii = {}
                
                max_attr = max(abs(attr_array.max()), abs(attr_array.min())) + 1e-12
                
                for i in range(min(len(attributions), mol.GetNumAtoms())):
                    norm_attr = attributions[i] / max_attr
                    intensity = abs(norm_attr)
                    
                    atom_colors[i] = (1-intensity, 1-intensity, 1.0)
                    atom_radii[i] = 0.3 + 0.5 * intensity
                
                drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
                drawer.drawOptions().addAtomIndices = True
                rdMolDraw2D.PrepareAndDrawMolecule(
                    drawer, mol,
                    highlightAtoms=list(atom_colors.keys()),
                    highlightAtomColors=atom_colors,
                    highlightAtomRadii=atom_radii
                )
                drawer.FinishDrawing()
                
                img_path = outputs_dir / f"{safe_smiles}_attribution_map.png"
                with open(img_path, 'wb') as f:
                    f.write(drawer.GetDrawingText())
                
                image_paths['attribution_map'] = f'/outputs/{safe_smiles}_attribution_map.png'
                file_paths['attribution_map'] = str(img_path)
                print(f"✓ Saved attribution map")
        
        # 2. Generate attribution analysis plots
        if 'mb_eig_attributions' in result:
            mb_eig = result['mb_eig_attributions']
            baseline_attr = result.get('baseline_attributions', {})
            ig_skeleton = baseline_attr.get('ig_skeleton', result.get('ig_skeleton', []))
            ig_zero = baseline_attr.get('ig_zero', result.get('ig_zero', []))
            fragments = result.get('fragments', {})
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"MB-EIG Attribution Analysis\nSMILES: {smiles}", fontsize=14)
            
            if ig_skeleton and ig_zero and mb_eig:
                atom_indices = range(1, len(mb_eig) + 1)
                width = 0.25
                
                axes[0, 0].bar([x - width for x in atom_indices], ig_skeleton, width,
                              label='IG Skeleton', color='green', alpha=0.7)
                axes[0, 0].bar(atom_indices, ig_zero, width,
                              label='IG Zero', color='orange', alpha=0.7)
                axes[0, 0].bar([x + width for x in atom_indices], mb_eig, width,
                              label='MB-EIG', color='blue', alpha=0.7)
                
                axes[0, 0].set_xlabel('Atom Index')
                axes[0, 0].set_ylabel('Attribution Score')
                axes[0, 0].set_title('Dual Baseline Comparison')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            if mb_eig:
                colors = ['#4CAF50' for _ in mb_eig]
                axes[0, 1].bar(atom_indices, mb_eig, color=colors, alpha=0.7)
                axes[0, 1].set_xlabel('Atom Index')
                axes[0, 1].set_ylabel('MB-EIG Attribution')
                axes[0, 1].set_title('MB-EIG Attributions')
                axes[0, 1].grid(True, alpha=0.3)
            
            if fragments:
                frag_names = list(fragments.keys())
                frag_scores = [fragments[name]['attribution'] for name in frag_names]
                
                colors = ['#4CAF50' if x > 0 else '#F44336' for x in frag_scores]
                bars = axes[1, 0].bar(range(len(frag_names)), frag_scores, color=colors, alpha=0.7)
                axes[1, 0].set_xlabel('Fragment Type')
                axes[1, 0].set_ylabel('Fragment Attribution')
                axes[1, 0].set_title('Fragment-Level Analysis')
                axes[1, 0].set_xticks(range(len(frag_names)))
                axes[1, 0].set_xticklabels(frag_names, rotation=45, ha='right')
                
                for bar, score in zip(bars, frag_scores):
                    height = bar.get_height()
                    axes[1, 0].annotate(f'{score:.3f}',
                                       xy=(bar.get_x() + bar.get_width() / 2, height),
                                       xytext=(0, 3),
                                       textcoords="offset points",
                                       ha='center', va='bottom', fontsize=8)
            
            quality = result.get('explanation_quality', {})
            metrics = ['Consistency', 'Confidence']
            values = [quality.get('consistency', 0), quality.get('confidence', 0)]
            
            bars = axes[1, 1].bar(metrics, values, color=['green', 'purple'], alpha=0.7)
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title(f"Explanation Quality (Validity: {quality.get('validity', 'N/A')})")
            axes[1, 1].set_ylim(0, 1)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].annotate(f'{val:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom')
            
            plt.tight_layout()
            
            plot_path = outputs_dir / f"{safe_smiles}_attribution_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            image_paths['analysis_plot'] = f'/outputs/{safe_smiles}_attribution_analysis.png'
            file_paths['analysis_plot'] = str(plot_path)
            print(f"✓ Saved analysis plot")
        
        # 3. Save JSON file
        json_path = outputs_dir / f"{safe_smiles}_prediction.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        file_paths['json'] = str(json_path)
        
        # 4. Save text explanation
        text_path = outputs_dir / f"{safe_smiles}_explanation.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("MOLECULAR PROPERTY PREDICTION EXPLANATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input SMILES: {smiles}\n")
            f.write(f"Canonical SMILES: {result.get('canonical_smiles', smiles)}\n")
            f.write(f"Property: {property_type.upper()}\n")
            f.write(f"Predicted Value: {result.get('predicted_value', 0):.3f} {get_unit_for_property(property_type)}\n\n")
            f.write("EXPLANATION:\n")
            f.write("-" * 30 + "\n")
            f.write(result.get('human_explanation', 'No explanation available') + "\n\n")
            f.write("TECHNICAL DETAILS:\n")
            f.write("-" * 30 + "\n")
            quality = result.get('explanation_quality', {})
            f.write(f"Consistency: {quality.get('consistency', 0):.3f}\n")
            f.write(f"Confidence: {quality.get('confidence', 0):.3f}\n")
            f.write(f"Validity: {quality.get('validity', 'N/A')}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_paths['text'] = str(text_path)
    
    except Exception as e:
        print(f"⚠️  Error generating visualizations: {e}")
        traceback.print_exc()
    
    return image_paths, file_paths

if __name__ == '__main__':
    # For Hugging Face Spaces deployment
    port = int(os.environ.get('PORT', 5000))  # HF Spaces uses port 7860
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
