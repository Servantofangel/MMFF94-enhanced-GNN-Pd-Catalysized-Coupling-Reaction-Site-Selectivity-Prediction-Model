#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED MMFF-GNN INFERENCE WITH VISUALIZATION
Generates comprehensive graphs and molecular visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDraw2DCairo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🎨 ENHANCED MMFF-GNN INFERENCE WITH VISUALIZATION")
print("="*60)
print("🎯 Optimized threshold: 0.75")
print("📊 Expected: Precision=45.3%, Recall=50.7%, F1=47.8%")
print("="*60)

class ProductionMMFFGNN(nn.Module):
    """Optimized MMFF-GNN for production inference"""
    
    def __init__(self, input_dim=10, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.charge_processor = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 16))
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.charge_integration = nn.Linear(hidden_dim // 4 + 16, hidden_dim // 4)
        self.classifier = nn.Linear(hidden_dim // 4, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        mmff_charges = x[:, 8].unsqueeze(1)
        processed_charges = self.charge_processor(mmff_charges)
        
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = torch.cat([x, processed_charges], dim=1)
        x = F.relu(self.charge_integration(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.classifier(x))

def calculate_mmff_charges(mol):
    """Calculate MMFF94 partial charges for a molecule"""
    try:
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_h)
        
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
        if mmff_props is None:
            return None
            
        charges = []
        for i in range(mol_h.GetNumAtoms()):
            charge = mmff_props.GetMMFFPartialCharge(i)
            charges.append(charge)
        
        mol = Chem.RemoveHs(mol_h)
        return charges
        
    except Exception as e:
        print(f"⚠️ MMFF charge calculation failed: {e}")
        return None

def smiles_to_graph(smiles):
    """Convert SMILES to graph with MMFF charges"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Calculate MMFF charges
        mmff_charges = calculate_mmff_charges(mol)
        if mmff_charges is None:
            return None
        
        # Get atom features
        atom_features = []
        for i, atom in enumerate(mol.GetAtoms()):
            features = [
                atom.GetAtomicNum(),                    # 0: Atomic number
                atom.GetDegree(),                       # 1: Degree
                atom.GetFormalCharge(),                 # 2: Formal charge
                atom.GetTotalValence(),                 # 3: Total valence
                int(atom.GetIsAromatic()),              # 4: Aromaticity
                atom.GetTotalNumHs(),                   # 5: H count
                int(atom.IsInRing()),                   # 6: Ring membership
                atom.GetHybridization().real,           # 7: Hybridization
                mmff_charges[i],                        # 8: MMFF partial charge ⭐
                atom.GetMass() / 100.0,                 # 9: Atomic mass
            ]
            atom_features.append(features)
        
        # Get edge indices (bonds)
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected graph
        
        return {
            'x': torch.tensor(atom_features, dtype=torch.float),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            'smiles': smiles,
            'mol': mol
        }
        
    except Exception as e:
        print(f"⚠️ Error processing {smiles}: {e}")
        return None

def load_production_model(device):
    """Load the optimized production model"""
    model = ProductionMMFFGNN(input_dim=10, hidden_dim=128).to(device)
    try:
        model.load_state_dict(torch.load('mmff_enhanced_final.pth', map_location=device))
        model.eval()  # Set to evaluation mode
        print("✅ Production model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return None

def predict_reaction_sites(model, graph_data, device, threshold=0.75):
    """Predict reaction sites for a single molecule"""
    with torch.no_grad():
        x = graph_data['x'].to(device)
        edge_index = graph_data['edge_index'].to(device)
        
        predictions = model(x, edge_index).cpu().numpy().flatten()
        reaction_sites = (predictions > threshold).astype(int)
        
        return predictions, reaction_sites

def create_comprehensive_visualization(smiles, predictions, reaction_sites, mol, threshold=0.75):
    """Create comprehensive visualization with molecular structure and predictions"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Prediction Probability Plot
    ax1 = fig.add_subplot(gs[0, 0])
    atoms = range(len(predictions))
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Create bar plot with colors based on prediction
    colors = []
    for i, pred in enumerate(predictions):
        if reaction_sites[i] == 1:
            colors.append('red')  # Predicted reaction site
        elif pred > 0.6:
            colors.append('orange')  # High confidence but below threshold
        elif pred > 0.4:
            colors.append('yellow')  # Medium confidence
        else:
            colors.append('lightblue')  # Low confidence
    
    bars = ax1.bar(atoms, predictions, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(y=threshold, color='purple', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    # Add confidence scores on top of bars
    for i, (bar, pred) in enumerate(zip(bars, predictions)):
        if pred > 0.3:  # Only label significant predictions
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{pred:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Atom Index')
    ax1.set_ylabel('Prediction Probability')
    ax1.set_title(f'Reaction Site Predictions for: {smiles}', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Add atom symbols below bars
    ax1.set_xticks(atoms)
    ax1.set_xticklabels([f'{sym}\n({i})' for i, sym in enumerate(atom_symbols)], fontsize=9)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='Predicted Reaction Site'),
        plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.8, label='High Confidence (>0.6)'),
        plt.Rectangle((0,0),1,1, facecolor='yellow', alpha=0.8, label='Medium Confidence (0.4-0.6)'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.8, label='Low Confidence (<0.4)'),
        plt.Line2D([0], [0], color='purple', linestyle='--', label=f'Threshold ({threshold})')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. Molecular Structure with Highlighted Atoms
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create highlighted molecule drawing
    highlight_atoms = [i for i, site in enumerate(reaction_sites) if site == 1]
    highlight_colors = {i: (1.0, 0.0, 0.0) for i in highlight_atoms}  # Red for reaction sites
    
    # Generate molecule image
    try:
        img = Draw.MolToImage(mol, size=(400, 300), highlightAtoms=highlight_atoms, 
                             highlightAtomColors=highlight_colors)
        ax2.imshow(img)
        ax2.set_title('Molecular Structure with Predicted Reaction Sites', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add annotation for highlighted atoms
        if highlight_atoms:
            annotation_text = "Predicted Reaction Sites:\n"
            for atom_idx in highlight_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                confidence = predictions[atom_idx]
                annotation_text += f"Atom {atom_idx} ({atom.GetSymbol()}): {confidence:.3f}\n"
            
            ax2.text(1.02, 0.5, annotation_text, transform=ax2.transAxes, va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    except Exception as e:
        ax2.text(0.5, 0.5, f"Could not generate molecular image:\n{e}", 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Molecular Structure', fontsize=14, fontweight='bold')
        ax2.axis('off')
    
    # 3. Confidence Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if len(predictions) > 1:
        ax3.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=threshold, color='purple', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax3.set_xlabel('Prediction Confidence')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Prediction Confidences', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Not enough data for distribution', ha='center', va='center')
        ax3.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    
    # 4. Summary Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate summary statistics
    num_atoms = len(predictions)
    num_reaction_sites = np.sum(reaction_sites)
    avg_confidence = np.mean(predictions)
    max_confidence = np.max(predictions) if len(predictions) > 0 else 0
    min_confidence = np.min(predictions) if len(predictions) > 0 else 0
    
    summary_text = (
        f"📊 PREDICTION SUMMARY\n"
        f"─────────────────────\n"
        f"• SMILES: {smiles}\n"
        f"• Total atoms: {num_atoms}\n"
        f"• Predicted reaction sites: {num_reaction_sites}\n"
        f"• Reaction site percentage: {num_reaction_sites/num_atoms*100:.1f}%\n"
        f"• Average confidence: {avg_confidence:.3f}\n"
        f"• Maximum confidence: {max_confidence:.3f}\n"
        f"• Minimum confidence: {min_confidence:.3f}\n"
        f"• Threshold: {threshold}\n\n"
        f"🎯 PREDICTED SITES:\n"
    )
    
    if num_reaction_sites > 0:
        for atom_idx in np.where(reaction_sites == 1)[0]:
            atom = mol.GetAtomWithIdx(int(atom_idx))
            confidence = predictions[atom_idx]
            summary_text += f"   Atom {atom_idx} ({atom.GetSymbol()}): {confidence:.3f}\n"
    else:
        summary_text += "   No reaction sites predicted above threshold\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'reaction_prediction_{smiles[:10]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save individual molecular image with highlights
    try:
        drawer = MolDraw2DCairo(400, 300)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        with open(f'molecule_{smiles[:10]}.png', 'wb') as f:
            f.write(drawer.GetDrawingText())
    except Exception as e:
        print(f"⚠️ Could not save molecular image: {e}")

def batch_predict_and_visualize(smiles_list, model, device, threshold=0.75):
    """Batch prediction with visualization for each molecule"""
    results = []
    
    for smiles in tqdm(smiles_list, desc="Predicting and visualizing"):
        graph_data = smiles_to_graph(smiles)
        if graph_data is None:
            results.append({'smiles': smiles, 'error': 'Invalid SMILES or charge calculation failed'})
            continue
        
        try:
            predictions, reaction_sites = predict_reaction_sites(model, graph_data, device, threshold)
            mol = graph_data['mol']
            
            # Create comprehensive visualization
            create_comprehensive_visualization(smiles, predictions, reaction_sites, mol, threshold)
            
            results.append({
                'smiles': smiles,
                'predictions': predictions,
                'reaction_sites': reaction_sites,
                'atom_symbols': [atom.GetSymbol() for atom in mol.GetAtoms()],
                'num_atoms': len(predictions),
                'num_reaction_sites': np.sum(reaction_sites),
                'confidence_scores': predictions[reaction_sites == 1] if np.any(reaction_sites == 1) else [],
                'visualization_file': f'reaction_prediction_{smiles[:10]}.png'
            })
            
        except Exception as e:
            results.append({'smiles': smiles, 'error': str(e)})
    
    return results

def interactive_prediction_with_visualization(model, device):
    """Interactive mode with visualization"""
    print("\n🔍 INTERACTIVE MODE WITH VISUALIZATION")
    print("Enter SMILES strings (type 'quit' to exit)")
    
    while True:
        smiles = input("\nEnter SMILES: ").strip()
        if smiles.lower() in ['quit', 'exit', 'q']:
            break
        
        graph_data = smiles_to_graph(smiles)
        if graph_data is None:
            print("❌ Invalid SMILES or charge calculation failed")
            continue
        
        try:
            predictions, reaction_sites = predict_reaction_sites(model, graph_data, device, 0.75)
            mol = graph_data['mol']
            
            # Create visualization
            create_comprehensive_visualization(smiles, predictions, reaction_sites, mol)
            
            print(f"\n✅ Visualization created: reaction_prediction_{smiles[:10]}.png")
            print(f"🔬 Prediction for: {smiles}")
            print(f"   Number of atoms: {len(predictions)}")
            print(f"   Predicted reaction sites: {np.sum(reaction_sites)}")
            
            if np.any(reaction_sites == 1):
                print(f"   Reaction site indices: {np.where(reaction_sites == 1)[0]}")
                for atom_idx in np.where(reaction_sites == 1)[0]:
                    atom = mol.GetAtomWithIdx(int(atom_idx))
                    confidence = predictions[atom_idx]
                    print(f"     Atom {atom_idx} ({atom.GetSymbol()}): {confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Using device: {device}")
    
    # Load production model
    model = load_production_model(device)
    if model is None:
        return
    
    print("\n🎯 Enhanced MMFF-GNN Ready with Visualization!")
    print("   Threshold: 0.75 (optimized)")
    
    # Check if input file exists or use interactive mode
    input_file = 'input_smiles.txt'
    if os.path.exists(input_file):
        print(f"\n📁 Reading SMILES from {input_file}")
        with open(input_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"   Found {len(smiles_list)} SMILES strings")
        results = batch_predict_and_visualize(smiles_list, model, device, 0.75)
        
        # Save results
        output_data = []
        for result in results:
            if 'error' in result:
                output_data.append({
                    'smiles': result['smiles'],
                    'status': 'error',
                    'error_message': result['error']
                })
            else:
                output_data.append({
                    'smiles': result['smiles'],
                    'status': 'success',
                    'num_atoms': result['num_atoms'],
                    'num_reaction_sites': result['num_reaction_sites'],
                    'reaction_site_indices': ';'.join(map(str, np.where(result['reaction_sites'] == 1)[0])),
                    'confidence_scores': ';'.join(map(lambda x: f"{x:.3f}", result['confidence_scores'])),
                    'visualization_file': result['visualization_file']
                })
        
        df = pd.DataFrame(output_data)
        df.to_csv('reaction_predictions_with_visualization.csv', index=False)
        print("💾 Predictions and visualizations saved!")
        
    else:
        print(f"\n📝 Create '{input_file}' with SMILES strings for batch processing")
        print("   Or use interactive mode below:")
        interactive_prediction_with_visualization(model, device)
    
    print(f"\n✅ Enhanced inference complete!")
    print("📊 Visualizations saved as 'reaction_prediction_*.png'")
    print("💾 Results saved to 'reaction_predictions_with_visualization.csv'")

if __name__ == "__main__":
    main()