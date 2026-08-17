import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import datasets
import matplotlib.pyplot as plt

from models import resnext_cifar
from collections import OrderedDict

def get_model(model_name, num_real_classes, num_v_classes, dataset='cifar10'):
    if model_name == 'resnext-29_2x64d':
        return resnext_cifar.ResNeXt29_2x64d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    elif model_name == 'resnext-29_2x32d':
        return resnext_cifar.ResNeXt29_2x32d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    elif model_name == 'resnext-20_2x32d':
        return resnext_cifar.ResNeXt20_2x32d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    elif model_name == 'resnext-20_1x16d':
        return resnext_cifar.ResNeXt20_1x16d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    else:
        raise ValueError('un-supported model: {0}'.format(model_name))

def filter_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
parser = argparse.ArgumentParser(description='Analyze Virtual Smoothing Models (Q4 & Q5)')
parser.add_argument('--model_name', default='resnext-29_2x64d', help='Model name')
parser.add_argument('--dataset', default='cifar10', help='Dataset (e.g., cifar10)')
parser.add_argument('--model_file', required=True, help='File path of the trained model')
parser.add_argument('--v_classes', default=10, type=int, help='The number of virtual classes')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--output_dir', default='./analysis_output', help='Directory to save plots and data')
parser.add_argument('--num_samples', type=int, default=20, help='Number of misclassified samples to collect for Q5')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == 'cifar10':
    NUM_REAL_CLASSES = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
else:
    NUM_REAL_CLASSES = 10
    class_names = [str(i) for i in range(10)]

def main():
    # 1. Load Data
    print("Loading data...")
    if args.dataset == 'cifar10':
        transform_test = T.Compose([T.ToTensor()])
        testset = datasets.CIFAR10(root='../../datasets/cifar10', train=False, download=True, transform=transform_test)
    else:
        # Fallback to general loader (SVHN, CIFAR100, etc. would need specific transforms as in eval_clean.py)
        transform_test = T.Compose([T.ToTensor()])
        testset = datasets.CIFAR10(root='../../datasets/cifar10', train=False, download=True, transform=transform_test)
        
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # 2. Load Model
    print(f"Loading model {args.model_name} from {args.model_file}...")
    model = get_model(args.model_name, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes, dataset=args.dataset)
    cpt = filter_state_dict(torch.load(args.model_file, map_location=device))
    model.load_state_dict(cpt)
    model = model.to(device)
    model.eval()

    # Determine base module in case of DataParallel
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Setup hooks for Q5 feature extraction
    features_dict = {'conv1': [], 'penultimate': []}
    
    def hook_conv1(module, inp, out):
        features_dict['conv1'] = out.detach().cpu()
    def hook_penultimate(module, inp, out):
        features_dict['penultimate'] = inp[0].detach().cpu().view(inp[0].size(0), -1)

    # Register hooks (Specific to ResNeXt architecture defined in models/resnext_cifar.py)
    h1 = actual_model.conv1.register_forward_hook(hook_conv1)
    h2 = actual_model.linear.register_forward_hook(hook_penultimate)

    # Tracking variables
    all_targets = []
    all_probs = []
    
    misclassified_samples = []
    correct_samples = []

    print("Evaluating model and extracting features...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x_dev = batch_x.to(device)
            batch_y_dev = batch_y.to(device)
            
            # Forward pass triggers hooks
            logits = model(batch_x_dev)
            
            all_targets.append(batch_y)
            all_probs.append(logits.detach().cpu())
            
            # Normal predictions (slice virtual classes)
            _, normal_preds = torch.max(logits[:, :NUM_REAL_CLASSES], dim=1)
            
            # Collect Q5 Samples
            for i in range(len(batch_y)):
                is_correct = (normal_preds[i].item() == batch_y[i].item())
                
                # Check if we still need more samples
                if not is_correct and len(misclassified_samples) < args.num_samples:
                    misclassified_samples.append({
                        'image': batch_x[i].cpu(),
                        'true_label': batch_y[i].item(),
                        'pred_label': normal_preds[i].item(),
                        'conv1_feat': features_dict['conv1'][i],
                        'penul_feat': features_dict['penultimate'][i]
                    })
                elif is_correct and len(correct_samples) < args.num_samples:
                    correct_samples.append({
                        'image': batch_x[i].cpu(),
                        'true_label': batch_y[i].item(),
                        'pred_label': normal_preds[i].item(),
                        'conv1_feat': features_dict['conv1'][i],
                        'penul_feat': features_dict['penultimate'][i]
                    })

    # Combine all logic
    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    # -------------------------------------------------------------
    # Q4: Incorrect Use of Virtual Classes at Test Time
    # -------------------------------------------------------------
    print("Processing Q4 metrics...")
    # Normal Evaluation (Real only)
    normal_preds = torch.argmax(all_probs[:, :NUM_REAL_CLASSES], dim=1)
    normal_acc = (normal_preds == all_targets).float().mean().item() * 100

    # Incorrect Evaluation (Real + Virtual)
    incorrect_preds = torch.argmax(all_probs, dim=1)
    incorrect_acc = (incorrect_preds == all_targets).float().mean().item() * 100

    print(f"Normal Accuracy (Real Classes Only): {normal_acc:.2f}%")
    print(f"Incorrect Accuracy (Including Virtual Classes): {incorrect_acc:.2f}%")

    # Plot Q4 Bar Chart
    labels = ['Correct (Real Only)', 'Incorrect (Real + Virtual)']
    accuracies = [normal_acc, incorrect_acc]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accuracies, color=['#4CAF50', '#F44336'])
    plt.ylabel('Accuracy (%)')
    plt.title('Q4: Effect of Incorrectly Including Virtual Classes at Test Time')
    plt.ylim(0, 100)
    
    # Add text on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    
    q4_plot_path = os.path.join(args.output_dir, 'q4_accuracy_drop.png')
    plt.savefig(q4_plot_path)
    plt.close()
    print(f"Saved Q4 plot to {q4_plot_path}")

    # -------------------------------------------------------------
    # Q5: Error Analysis
    # -------------------------------------------------------------
    print("Processing Q5 extraction and plots...")
    
    # Save raw data to a .pt file
    q5_data = {
        'misclassified': misclassified_samples,
        'correct': correct_samples
    }
    q5_data_path = os.path.join(args.output_dir, 'q5_extracted_samples_and_features.pt')
    torch.save(q5_data, q5_data_path)
    print(f"Saved raw Q5 features and images to {q5_data_path}")

    # Plot the 20 misclassified images in a grid
    num_cols = 5
    num_rows = int(np.ceil(args.num_samples / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    fig.suptitle('Q5: 20 Misclassified Test Samples', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(misclassified_samples):
            sample = misclassified_samples[i]
            # Convert CHW [0,1] tensor to HWC numpy array
            img = sample['image'].numpy().transpose(1, 2, 0)
            
            true_cls = class_names[sample['true_label']] if args.dataset == 'cifar10' else str(sample['true_label'])
            pred_cls = class_names[sample['pred_label']] if args.dataset == 'cifar10' else str(sample['pred_label'])
            
            ax.imshow(img)
            ax.set_title(f"True: {true_cls}\nPred: {pred_cls}", color='red', fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Adjust for suptitle
    q5_grid_path = os.path.join(args.output_dir, 'q5_misclassified_grid.png')
    plt.savefig(q5_grid_path)
    plt.close()
    print(f"Saved Q5 misclassified grid to {q5_grid_path}")
    
    print("Analysis Complete!")

if __name__ == '__main__':
    main()
