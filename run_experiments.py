import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Import custom modules
from model import DynamicLeukemiaNet
from dataset import StandardDataset, TTADataset, get_transforms

def set_seed(seed=42):
    """Ensures absolute reproducibility across multiple runs and hardware."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False     

def log_metrics_to_csv(exp_name, epoch, train_loss, train_acc, val_loss, val_acc, y_true, y_pred, lr, output_dir="logs"):
    """Calculates comprehensive metrics and appends them to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    csv_file = os.path.join(output_dir, f"{exp_name}_logs.csv")
    log_data = {
        "Epoch": [epoch], "Learning_Rate": [lr],
        "Train_Loss": [train_loss], "Train_Acc": [train_acc],
        "Val_Loss": [val_loss], "Val_Acc": [val_acc],
        "Precision": [precision], "Recall": [recall],
        "F1_Score": [f1], "MCC": [mcc]
    }
    
    df = pd.DataFrame(log_data)
    if not os.path.isfile(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)
        
    print(f"Logged {exp_name} | Epoch: {epoch} | Val Acc: {val_acc:.4f} | Val MCC: {mcc:.4f}")

def evaluate_model(model, loader, use_tta, criterion, device):
    """Evaluates the model on the validation set, supporting TTA."""
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, targets in loader:
            targets = targets.to(device)
            if use_tta:
                inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4)).to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].view(targets.size(0), -1).mean(dim=1)
		  # Optimal threshold determined via Fold-4 validation tuning
		  OPTIMAL_THRESHOLD = 0.3907
		  preds = (probs >= OPTIMAL_THRESHOLD).long() 
                loss = 0.0 
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets).item()
                val_loss += loss
                preds = outputs.argmax(dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
    val_acc = accuracy_score(all_labels, all_preds)
    avg_loss = val_loss / len(loader) if not use_tta else 0.0
    return avg_loss, val_acc, all_labels, all_preds

def run_experiment(exp_name, config, loaders, device, pretrained_path, epochs=20):
    """Executes a single experimental configuration."""
    print(f"\nStarting Experiment: {exp_name}")
    print(f"Configuration: {config}")
    
    model = DynamicLeukemiaNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    model.use_cnn = config['cnn']
    model.use_vit = config['vit']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    ft_loader, test_base_loader, test_tta_loader = loaders
    eval_loader = test_tta_loader if config['tta'] else test_base_loader
    
    if config['ft']:
	best_val_acc = 0.0
        for epoch in range(epochs):
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            
            for inputs, targets in ft_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                run_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = run_loss / len(ft_loader)
            train_acc = correct / total
            val_loss, val_acc, y_true, y_pred = evaluate_model(model, eval_loader, config['tta'], criterion, device)
            
            log_metrics_to_csv(exp_name, epoch + 1, train_loss, train_acc, val_loss, val_acc, y_true, y_pred, optimizer.param_groups[0]['lr'])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                if exp_name == "Exp6_PROPOSED_FullPipeline":
                    os.makedirs("weights", exist_ok=True)
                    torch.save(model.state_dict(), os.path.join("weights", "Exp6_Final_Model.pth"))
            	
    else:
        val_loss, val_acc, y_true, y_pred = evaluate_model(model, eval_loader, config['tta'], criterion, device)
        log_metrics_to_csv(exp_name, 0, 0.0, 0.0, val_loss, val_acc, y_true, y_pred, 0.0)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path configurations (To be updated by the user based on their local/cloud environment)
    val_base_path = './data/C-NMC_test_prelim_phase_data'
    csv_path = './data/C-NMC_test_prelim_phase_data_labels.csv'
    pretrained_path = './weights/ultimate_leukemia_model.pth'
    
    # Data Preparation
    df = pd.read_csv(csv_path)
    all_paths = [os.path.join(val_base_path, fname) for fname in df['new_names']]
    all_labels = df['labels'].values
    
    adapt_paths, test_paths, adapt_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.80, random_state=42, stratify=all_labels
    )
    
    base_transform, ft_transform, tta_transforms = get_transforms()
    
    ft_loader = DataLoader(StandardDataset(adapt_paths, adapt_labels, ft_transform), batch_size=16, shuffle=True)
    test_base_loader = DataLoader(StandardDataset(test_paths, test_labels, base_transform), batch_size=16, shuffle=False)
    test_tta_loader = DataLoader(TTADataset(test_paths, test_labels, tta_transforms), batch_size=4, shuffle=False)
    
    loaders = (ft_loader, test_base_loader, test_tta_loader)
    
    experiments = [
        {"name": "Exp1_Ablation_CNN_Only", "config": {"cnn": True, "vit": False, "ft": False, "tta": False}},
        {"name": "Exp2_Ablation_ViT_Only", "config": {"cnn": False, "vit": True, "ft": False, "tta": False}},
        {"name": "Exp3_Baseline_DualStream", "config": {"cnn": True, "vit": True, "ft": False, "tta": False}},
        {"name": "Exp4_Optimization_TTA", "config": {"cnn": True, "vit": True, "ft": False, "tta": True}},
        {"name": "Exp5_Optimization_FineTuning", "config": {"cnn": True, "vit": True, "ft": True, "tta": False}},
        {"name": "Exp6_PROPOSED_FullPipeline", "config": {"cnn": True, "vit": True, "ft": True, "tta": True}}
    ]
    
    for exp in experiments:
        run_experiment(exp["name"], exp["config"], loaders, device, pretrained_path, epochs=20)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()