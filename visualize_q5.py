import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser(description='Visualize Q5 Data')
parser.add_argument('--pt_file', type=str, default='analysis_results/q5_extracted_samples_and_features.pt', help='Path to the .pt file')
parser.add_argument('--output_dir', type=str, default='analysis_results/q5_plots', help='Output directory for plots')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def main():
    if not os.path.exists(args.pt_file):
        print(f"Error: Could not find {args.pt_file}. Please check the path.")
        return

    print(f"Loading data from {args.pt_file}...")
    data = torch.load(args.pt_file, map_location=torch.device('cpu'))
    
    misclassified = data['misclassified']
    correct = data['correct']
    
    print(f"Loaded {len(misclassified)} misclassified and {len(correct)} correct samples.")
    
    # 1. PCA of Penultimate Features
    print("Generating PCA scatter plot...")
    # Extract features
    feat_mis = [sample['penul_feat'].numpy().flatten() for sample in misclassified]
    feat_cor = [sample['penul_feat'].numpy().flatten() for sample in correct]
    
    X = np.vstack(feat_cor + feat_mis)
    y = np.array([0]*len(feat_cor) + [1]*len(feat_mis)) # 0 for correct, 1 for misclassified
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='green', marker='o', label='Correctly Classified', alpha=0.7, s=100)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', marker='x', label='Misclassified', alpha=0.9, s=100)
    plt.title('PCA of Penultimate Features (Correct vs Misclassified)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    pca_path = os.path.join(args.output_dir, 'pca_penultimate_features.png')
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot to {pca_path}")
    
    # 2. Conv1 Activation Heatmap Comparison
    print("Generating Conv1 Activation Heatmap...")
    # We will pick the first misclassified sample and first correct sample
    sample_cor = correct[0]
    sample_mis = misclassified[0]
    
    # Calculate mean activation across channels
    act_cor = sample_cor['conv1_feat'].squeeze(0).mean(dim=0).numpy()
    act_mis = sample_mis['conv1_feat'].squeeze(0).mean(dim=0).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Early Network Attention (Conv1)', fontsize=16)
    
    # Original Images
    img_cor = sample_cor['image'].numpy().transpose(1, 2, 0)
    img_mis = sample_mis['image'].numpy().transpose(1, 2, 0)
    
    axes[0, 0].imshow(img_cor)
    axes[0, 0].set_title(f"Correctly Classified (Label: {sample_cor['true_label']})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_mis)
    axes[0, 1].set_title(f"Misclassified (True: {sample_mis['true_label']} | Pred: {sample_mis['pred_label']})")
    axes[0, 1].axis('off')
    
    # Activation Maps
    im1 = axes[1, 0].imshow(act_cor, cmap='viridis')
    axes[1, 0].set_title("Mean Conv1 Activation")
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(act_mis, cmap='viridis')
    axes[1, 1].set_title("Mean Conv1 Activation")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    conv1_path = os.path.join(args.output_dir, 'conv1_activation_comparison.png')
    plt.savefig(conv1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Conv1 Activation plot to {conv1_path}")
    
    print("\nVisualization complete! You can use these plots for your presentation.")

if __name__ == '__main__':
    main()
