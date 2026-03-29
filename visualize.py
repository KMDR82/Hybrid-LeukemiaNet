import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

# Global Plotting Configuration for Publication Standards
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'serif', 
    'font.serif': ['Times New Roman']
})


def get_zero_shot_metrics(csv_path):
    """
    Extracts Epoch 0 validation metrics from the experimental log file.
    Returns: [Val_Acc, Precision, Recall, F1_Score, MCC]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing required log file for visualization: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Extract metrics from the first row (Epoch 0 / Zero-Shot Baseline)
    metrics = df.iloc[0][['Val_Acc', 'Precision', 'Recall', 'F1_Score', 'MCC']].tolist()
    
    return metrics

def plot_radar_chart(
    path_cnn='Exp1_Ablation_CNN_Only_logs.csv',
    path_vit='Exp2_Ablation_ViT_Only_logs.csv',
    path_dual='Exp3_Baseline_DualStream_logs.csv',
    save_path='Figure02.png'
):
    """
    Plots a radar chart comparing the zero-shot generalization footprint
    of different architectures (Exp1, Exp2, Exp3) via dynamic log parsing.
    """
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
    N = len(categories)


    values_cnn = get_zero_shot_metrics(path_cnn)
    values_vit = get_zero_shot_metrics(path_vit)
    values_dual = get_zero_shot_metrics(path_dual)

    # Append the first value to the end to close the circular graph
    values_cnn += values_cnn[:1]
    values_vit += values_vit[:1]
    values_dual += values_dual[:1]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='black', size=13, fontweight='bold')
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 0.9)

    # Colorblind-safe palette (Academic standard)
    color_cnn = '#fc8d62'
    color_vit = '#8da0cb'
    color_dual = '#66c2a5'

    ax.plot(angles, values_cnn, linewidth=2, linestyle='dashed', color=color_cnn, label='CNN Only (Exp1)')
    ax.fill(angles, values_cnn, color=color_cnn, alpha=0.15)

    ax.plot(angles, values_vit, linewidth=2, linestyle='dotted', color=color_vit, label='ViT Only (Exp2)')
    ax.fill(angles, values_vit, color=color_vit, alpha=0.15)

    ax.plot(angles, values_dual, linewidth=3, linestyle='solid', color=color_dual, label='Dual-Stream (Exp3)')
    ax.fill(angles, values_dual, color=color_dual, alpha=0.35)

    ax.spines['polar'].set_color('#dddddd')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(csv_path='Exp6_PROPOSED_FullPipeline_logs.csv', save_path='Figure03.png'):
    """
    Plots the training and validation accuracy and loss curves from log files.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Log file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    epochs = df['Epoch']
    train_acc = df['Train_Acc']
    val_acc = df['Val_Acc']
    train_loss = df['Train_Loss']
    val_loss = df['Val_Loss']

    color_train = '#1f78b4'
    color_val = '#e31a1c'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy Subplot
    ax1.plot(epochs, train_acc, marker='o', markersize=4, linestyle='-', linewidth=2, color=color_train, label='Training Accuracy')
    ax1.plot(epochs, val_acc, marker='s', markersize=4, linestyle='--', linewidth=2, color=color_val, label='Validation Accuracy')
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_xticks(epochs[::2])
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower right', frameon=True, edgecolor='black')

    # Loss Subplot
    ax2.plot(epochs, train_loss, marker='o', markersize=4, linestyle='-', linewidth=2, color=color_train, label='Training Loss')
    ax2.plot(epochs, val_loss, marker='s', markersize=4, linestyle='--', linewidth=2, color=color_val, label='Validation Loss')
    ax2.set_xlabel('Epochs', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_xticks(epochs[::2])
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right', frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='Figure04.png'):
    """
    Plots the confusion matrix for the final predictions.
    """
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
    class_names = ['Healthy (HEM)', 'Leukemia (ALL)']

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"}, linewidths=1, linecolor='black')
    
    plt.ylabel('Ground Truth', fontsize=12, fontweight='bold', labelpad=15)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_qualitative_grid(model, test_paths, test_labels, tta_transforms, device, save_path='Figure05.png'):
    """
    Generates a 4x4 grid of qualitative visual results showing true vs predicted labels.
    """
    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=300)
    qual_class_names = ['Healthy', 'Leukemia']
    
    with torch.no_grad():
        for i in range(16):
            if i >= len(test_paths):
                break
                
            ax = axes[i // 4, i % 4]
            ax.set_xticks([])
            ax.set_yticks([])
            
            img_path = test_paths[i]
            true_label = test_labels[i]
            
            orig_img = Image.open(img_path).convert('RGB')
            ax.imshow(orig_img.resize((224, 224)))
            
            tta_tensors = torch.stack([t(orig_img) for t in tta_transforms]).unsqueeze(0)
            inputs = tta_tensors.view(-1, 3, 224, 224).to(device)
            outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].mean()
            pred = int(probs.item() >= 0.35)
            
            true_str = qual_class_names[true_label]
            pred_str = qual_class_names[pred]
            color = 'darkgreen' if true_label == pred else 'darkred'
            
            ax.set_title(f"True: {true_str}\nPred: {pred_str}", color=color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Generate standalone figures using existing CSV data
    print("Generating Figure 02 (Radar Chart)...")
    plot_radar_chart()
    
    print("Generating Figure 03 (Training Curves)...")
    plot_training_curves()
    
    print("Figures 02 and 03 generated successfully. Note: Figures 04 and 05 require model inferences to be passed to their respective functions.")