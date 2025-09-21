import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MMFFEnhancedGNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        # Specialized processing for MMFF charges
        self.charge_processor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16)
        )
        
        # Main graph convolution layers - FIXED INPUT DIMENSION
        self.conv1 = GCNConv(input_dim, hidden_dim)  # Use original 10 features
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Additional layer to process charge-enhanced features
        self.charge_integration = nn.Linear(hidden_dim // 4 + 16, hidden_dim // 4)
        
        self.classifier = nn.Linear(hidden_dim // 4, 1)
        self.dropout = dropout
        
        print(f"🔧 MMFF-Enhanced Architecture:")
        print(f"   Input: {input_dim} features")
        print(f"   Charge processing: 1 → 32 → 16")
        print(f"   Main layers: {input_dim} → {hidden_dim} → {hidden_dim//2} → {hidden_dim//4}")
        print(f"   Charge integration: {hidden_dim//4 + 16} → {hidden_dim//4}")
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Extract and process MMFF charges (feature index 8)
        mmff_charges = x[:, 8].unsqueeze(1)
        processed_charges = self.charge_processor(mmff_charges)
        
        # Graph convolution on original features
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Integrate processed charges
        x = torch.cat([x, processed_charges], dim=1)
        x = F.relu(self.charge_integration(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return torch.sigmoid(self.classifier(x))

class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=50.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        weights = torch.where(targets == 1, self.pos_weight, 1.0)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss * weights
        return focal_loss.mean()

def load_graphs(data_dir='enhanced_graphs', max_graphs=50000):
    graph_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    graphs = []
    for f in tqdm(graph_files[:max_graphs], desc="Loading MMFF graphs"):
        try:
            graph = torch.load(f'{data_dir}/{f}', weights_only=False)
            if hasattr(graph, 'x') and graph.x.shape[1] == 10:
                graphs.append(graph)
        except:
            continue
    return graphs

def train_mmff_enhanced():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Using device: {device}")
    
    graphs = load_graphs('enhanced_graphs', 50000)
    print(f"✅ Loaded {len(graphs)} MMFF-enhanced graphs")
    
    split_idx = int(0.9 * len(graphs))
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    
    model = MMFFEnhancedGNN(input_dim=10, hidden_dim=128, dropout=0.3).to(device)
    criterion = EnhancedFocalLoss(alpha=0.25, gamma=2.0, pos_weight=50.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses, val_losses, val_aucs = [], [], []
    best_auc = 0
    epochs = 30
    
    print("\n🔥 Training MMFF-Enhanced GNN...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y.view(-1, 1))
                val_loss += loss.item()
                val_preds.extend(out.cpu().numpy().flatten())
                val_labels.extend(batch.y.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
            val_aucs.append(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), 'mmff_enhanced_best.pth')
            
            print(f'Epoch {epoch+1:2d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | AUC: {val_auc:.4f}')
        except:
            print(f'Epoch {epoch+1:2d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}')
    
    torch.save(model.state_dict(), 'mmff_enhanced_final.pth')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MMFF-Enhanced Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation AUC')
    plt.title('Validation AUC Progress')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mmff_enhanced_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ MMFF-Enhanced training complete!")
    print(f"🎯 Best validation AUC: {best_auc:.4f}")
    print(f"💾 Model saved as 'mmff_enhanced_final.pth'")

if __name__ == "__main__":
    train_mmff_enhanced()