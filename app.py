"""
Flask Web Application for Molecular Property Prediction
Optimized for Render Deployment
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import sys
import json
from io import BytesIO
from pathlib import Path
import traceback
import numpy as np
from datetime import datetime
import zipfile
import threading

# Import prediction modules
sys.path.append(os.path.dirname(__file__))
from prediction_Esol import MolecularGraph, GraphormerModel, MBEIGExplainer, device
import prediction_lipo
import prediction_Free
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

from huggingface_hub import hf_hub_download

HF_REPO = "Tanmay0483/mbeig"

# Global variables
models = {}
models_loaded = False
loading_lock = threading.Lock()
loading_error = None

def download_models():
    """Download model files from Hugging Face (with error handling)"""
    try:
        print("Downloading model files from Hugging Face...")
        esol = hf_hub_download(repo_id=HF_REPO, filename="best_graphormer_esol.pth")
        lipo = hf_hub_download(repo_id=HF_REPO, filename="best_graphormer_lipo.pth")
        free = hf_hub_download(repo_id=HF_REPO, filename="best_graphormer_freesolv.pth")
        print("✓ Model files downloaded")
        return esol, lipo, free
    except Exception as e:
        print(f"✗ Error downloading models: {e}")
        raise

# Download models at module level
try:
    ESOL_MODEL, LIPO_MODEL, FREE_MODEL = download_models()
except Exception as e:
    print(f"Failed to download models: {e}")
    ESOL_MODEL = LIPO_MODEL = FREE_MODEL = None

def load_models():
    """Load all three models with memory optimization"""
    global models_loaded, loading_error
    
    with loading_lock:
        if models_loaded:
            return True
        
        if loading_error:
            return False
        
        try:
            print("\n" + "="*60)
            print("LOADING MODELS")
            print("="*60)
            
            if not all([ESOL_MODEL, LIPO_MODEL, FREE_MODEL]):
                raise Exception("Model files not downloaded")
            
            # Set PyTorch to use minimal threads to reduce memory
            torch.set_num_threads(1)
            
            # ESOL Model
            print("Loading ESOL model...")
            esol_model = GraphormerModel().to(device)
            checkpoint = torch.load(ESOL_MODEL, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                esol_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                esol_model.load_state_dict(checkpoint)
            esol_model.eval()
            models['esol'] = {'model': esol_model, 'explainer': MBEIGExplainer(esol_model, device)}
            del checkpoint
            print("✓ ESOL model loaded")
            
            # Lipo Model
            print("Loading Lipophilicity model...")
            lipo_model = prediction_lipo.GraphormerModel().to(device)
            checkpoint = torch.load(LIPO_MODEL, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                lipo_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                lipo_model.load_state_dict(checkpoint)
            lipo_model.eval()
            models['lipo'] = {'model': lipo_model, 'explainer': prediction_lipo.MBEIGExplainer(lipo_model, device)}
            del checkpoint
            print("✓ Lipophilicity model loaded")
            
            # FreeSolv Model
            print("Loading FreeSolv model...")
            free_model = prediction_Free.GraphormerModel().to(device)
            checkpoint = torch.load(FREE_MODEL, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                free_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                free_model.load_state_dict(checkpoint)
            free_model.eval()
            models['free'] = {'model': free_model, 'explainer': prediction_Free.MBEIGExplainer(free_model, device)}
            del checkpoint
            print("✓ FreeSolv model loaded")
            
            models_loaded = True
            print(f"\n✓ All models loaded. Available: {list(models.keys())}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            loading_error = str(e)
            print(f"\n✗ Error loading models: {e}")
            traceback.print_exc()
            return False

# Start loading models in background thread
def background_load():
    print("Starting background model loading...")
    load_models()

loading_thread = threading.Thread(target=background_load, daemon=True)
loading_thread.start()

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
    outputs_dir.mkdir(exist_ok=True)
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
    """Health check endpoint - always returns 200 for Render"""
    status = {
        'status': 'healthy' if models_loaded else 'loading',
        'models_loaded': models_loaded,
        'available_models': list(models.keys()) if models_loaded else [],
        'device': str(device),
        'loading_error': loading_error
    }
    # Always return 200 so Render doesn't kill the service
    return jsonify(status), 200

@app.route('/api/status')
def status():
    """Detailed status endpoint"""
    return jsonify({
        'models_loaded': models_loaded,
        'available_models': list(models.keys()),
        'loading_error': loading_error,
        'thread_alive': loading_thread.is_alive()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    # Check if models are loaded
    if not models_loaded:
        if loading_error:
            return jsonify({
                'error': f'Models failed to load: {loading_error}',
                'status': 'error'
            }), 503
        return jsonify({
            'error': 'Models are still loading. Please wait a moment and try again.',
            'status': 'loading'
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        smiles = data.get('smiles', '').strip()
        property_type = data.get('property_type', 'esol')
        
        model_key = 'free' if property_type == 'freesolv' else property_type
        
        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        if model_key not in models:
            return jsonify({
                'error': f'Invalid property type: {property_type}',
                'available': list(models.keys())
            }), 400
        
        explainer = models[model_key]['explainer']
        
        print(f"Computing prediction for: {smiles}")
        result = explainer.compute_mb_eig(smiles)
        
        if result is None or 'error' in result:
            error_msg = result.get('error', 'Unknown error') if result else 'Prediction failed'
            return jsonify({'error': error_msg}), 400
        
        if 'predicted_value' in result:
            prediction_value = result['predicted_value']
        elif 'prediction' in result:
            prediction_value = result['prediction']
        else:
            return jsonify({'error': 'No prediction value in result'}), 500
        
        baseline_attributions = result.get('baseline_attributions', {})
        if not baseline_attributions:
            baseline_attributions = {
                'ig_skeleton': result.get('ig_skeleton', []),
                'ig_zero': result.get('ig_zero', [])
            }
        
        # Generate visualizations
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
        
        return jsonify(response)
    
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"✗ {error_msg}")
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
    
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_')
    safe_smiles = safe_smiles[:100]
    
    image_paths = {}
    file_paths = {}
    
    try:
        # Generate molecule attribution map
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
        
        # Generate plots (simplified to reduce memory)
        if 'mb_eig_attributions' in result:
            mb_eig = result['mb_eig_attributions']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"MB-EIG Analysis - {smiles[:50]}", fontsize=12)
            
            atom_indices = range(1, len(mb_eig) + 1)
            
            # Plot 1: MB-EIG attributions
            axes[0].bar(atom_indices, mb_eig, color='#4CAF50', alpha=0.7)
            axes[0].set_xlabel('Atom Index')
            axes[0].set_ylabel('Attribution Score')
            axes[0].set_title('MB-EIG Attributions')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Quality metrics
            quality = result.get('explanation_quality', {})
            metrics = ['Consistency', 'Confidence']
            values = [quality.get('consistency', 0), quality.get('confidence', 0)]
            axes[1].bar(metrics, values, color=['green', 'purple'], alpha=0.7)
            axes[1].set_ylabel('Score')
            axes[1].set_title('Quality Metrics')
            axes[1].set_ylim(0, 1)
            
            plt.tight_layout()
            plot_path = outputs_dir / f"{safe_smiles}_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')  # Reduced DPI for memory
            plt.close()
            
            image_paths['analysis_plot'] = f'/outputs/{safe_smiles}_analysis.png'
            file_paths['analysis_plot'] = str(plot_path)
        
        # Save JSON and text files
        json_path = outputs_dir / f"{safe_smiles}_prediction.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        file_paths['json'] = str(json_path)
        
        text_path = outputs_dir / f"{safe_smiles}_explanation.txt"
        with open(text_path, 'w') as f:
            f.write(f"MOLECULAR PROPERTY PREDICTION\n")
            f.write(f"SMILES: {smiles}\n")
            f.write(f"Property: {property_type.upper()}\n")
            f.write(f"Value: {result.get('predicted_value', 0):.3f} {get_unit_for_property(property_type)}\n")
        file_paths['text'] = str(text_path)
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    return image_paths, file_paths

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
