# Refactoring Templates for Remaining Notebooks

This file provides step-by-step templates to apply the same refactoring patterns to `imagenet_models.ipynb`, `3x3_models.ipynb`, and `imagenet_models2.ipynb`.

## Template 1: Configuration Cell (Add to all notebooks)

```python
# =============================================================================
# CELL: EXPERIMENT CONFIGURATION
# =============================================================================

from ml_utils import set_seed, get_system_info, ExperimentConfig

# Random seed for reproducibility
SEED = 42
set_seed(SEED)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# System information
system_info = get_system_info()
print("System Information:")
for key, value in system_info.items():
    print(f"  {key}: {value}")

# Experiment Configuration
config = ExperimentConfig({
    # Reproducibility
    "seed": SEED,
    "device": str(device),
    
    # Dataset
    "dataset": "tiny-imagenet-200",
    "img_size": 64,
    "num_classes": 200,
    "train_val_split": 0.9,
    
    # Data loading
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
    
    # Training
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,
    
    # Paths
    "save_dir": "./saved_models",
    "checkpoint_dir": "./saved_models/checkpoints",
})

os.makedirs(config["save_dir"], exist_ok=True)
os.makedirs(config["checkpoint_dir"], exist_ok=True)

# Save configuration
config.save(os.path.join(config["save_dir"], "config.json"))
print(f"\n✓ Configuration loaded and saved")
```

## Template 2: Dataset Cell (Replace existing data loading)

```python
# =============================================================================
# CELL: DATASET LOADING & PREPROCESSING
# =============================================================================

from ml_utils import get_system_info
import kagglehub

# Download dataset
print("Downloading Tiny ImageNet-200...")
dataset_path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
train_path = os.path.join(dataset_path, "tiny-imagenet-200", "train")

# Define transforms with clear augmentation strategy
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(
        config["img_size"], 
        scale=(0.6, 1.0),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

transform_val = transforms.Compose([
    transforms.Resize(config["img_size"]),
    transforms.CenterCrop(config["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Create datasets
full_dataset = datasets.ImageFolder(train_path)
n_total = len(full_dataset)
n_train = int(config["train_val_split"] * n_total)
n_val = n_total - n_train

# Deterministic split
train_idx, val_idx = random_split(
    range(n_total),
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

train_dataset = datasets.ImageFolder(train_path, transform=transform_train)
val_dataset = datasets.ImageFolder(train_path, transform=transform_val)

train_dataset = torch.utils.data.Subset(train_dataset, train_idx.indices)
val_dataset = torch.utils.data.Subset(val_dataset, val_idx.indices)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"],
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"],
    persistent_workers=True
)

print(f"\n✓ Dataset loaded:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")
print(f"  Total classes: {len(full_dataset.classes)}")
```

## Template 3: Model Definition (With documentation)

```python
# =============================================================================
# CELL: MODEL ARCHITECTURE - [YourModelName]
# =============================================================================

class YourModelName(nn.Module):
    """
    [Brief description of model purpose and key features]
    
    Architecture Overview:
    - [Layer 1 description]
    - [Layer 2 description]
    - [Design choices and why they matter]
    
    Args:
        num_classes: Number of output classes (default 200)
    
    Example:
        >>> model = YourModelName(num_classes=200)
        >>> x = torch.randn(8, 3, 64, 64)
        >>> out = model(x)
        >>> print(out.shape)  # (8, 200)
    """
    
    def __init__(self, num_classes=200):
        super().__init__()
        
        # [Component 1]
        self.layer1 = nn.Sequential(
            # ...
        )
        
        # [Component 2]
        self.layer2 = nn.Sequential(
            # ...
        )
        
        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return self.classifier(x)

# Create model and print info
model = YourModelName(num_classes=config["num_classes"]).to(device)

print("\n✓ Model created")
print(f"Summary: {model_summary(model, device)}")
```

## Template 4: Training Cell (Using ml_utils)

```python
# =============================================================================
# CELL: TRAINING
# =============================================================================

from ml_utils import train_epoch, evaluate, save_model

# Setup optimizer and scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
optimizer = optim.AdamW(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"]
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"]
)

print("Starting training...")
print(f"Configuration: {dict(config.config)}")

best_accuracy = 0.0
history = {"train_acc": [], "val_acc": [], "train_loss": []}

for epoch in range(config["epochs"]):
    # Training
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, device, use_amp=True
    )
    
    # Validation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # LR scheduling
    scheduler.step()
    
    # Track history
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(train_loss)
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        save_model(model, os.path.join(config["save_dir"], "best_model.pth"))
        print(f"  → New best model saved (Acc: {val_acc:.2f}%)")
    
    print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
          f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

print(f"\n✓ Training complete. Best accuracy: {best_accuracy:.2f}%")
```

## Template 5: Evaluation & Results Cell

```python
# =============================================================================
# CELL: EVALUATION & RESULTS ANALYSIS
# =============================================================================

from ml_utils import count_parameters, create_results_summary, model_summary

# Final evaluation
print("="*70)
print("FINAL EVALUATION")
print("="*70)

model.load_state_dict(torch.load(os.path.join(config["save_dir"], "best_model.pth"), map_location=device))
final_loss, final_acc = evaluate(model, val_loader, criterion, device)

# Model complexity
model_info = model_summary(model, device)

# File sizes
model_path = os.path.join(config["save_dir"], "best_model.pth")
model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

print(f"\nValidation Accuracy: {final_acc:.2f}%")
print(f"Validation Loss: {final_loss:.4f}")
print(f"\nModel Complexity:")
for key, value in model_info.items():
    print(f"  {key}: {value}")
print(f"\nModel Size: {model_size_mb:.2f} MB")

# Compile results
results = {
    "accuracy": float(final_acc),
    "loss": float(final_loss),
    "parameters": model_info["total_parameters"],
    "size_mb": float(model_size_mb),
    "history": history,
}

# Save summary
create_results_summary(
    model, 
    results, 
    config.to_dict(), 
    system_info,
    os.path.join(config["save_dir"], "experiment_summary.json")
)

print(f"\n✓ Results saved to {config['save_dir']}/experiment_summary.json")
```

## Template 6: Model Comparison (For multi-model notebooks)

```python
# =============================================================================
# CELL: MODEL COMPARISON
# =============================================================================

from ml_utils import evaluate, count_parameters, model_summary

models_to_compare = {
    "Model1": Model1(num_classes=200),
    "Model2": Model2(num_classes=200),
    "Model3": Model3(num_classes=200),
}

results = {}

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

for model_name, model in models_to_compare.items():
    model_path = os.path.join(config["save_dir"], f"{model_name.lower()}_best.pth")
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # Get complexity
    params = count_parameters(model)
    model_size = os.path.getsize(model_path) / (1024 ** 2)
    
    # Store results
    results[model_name] = {
        "accuracy": float(val_acc),
        "loss": float(val_loss),
        "parameters": int(params),
        "size_mb": float(model_size),
        "params_per_acc": float(params / val_acc) if val_acc > 0 else 0,
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:     {val_acc:.2f}%")
    print(f"  Loss:         {val_loss:.4f}")
    print(f"  Parameters:   {params/1e6:.2f}M")
    print(f"  Size:         {model_size:.2f} MB")
    print(f"  Efficiency:   {val_acc / (params/1e6):.2f} acc/M-param")

# Ranking
print("\n" + "="*80)
print("RANKING (by Accuracy)")
print("="*80)

sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{i:2d}. {name:20s} | Acc: {metrics['accuracy']:6.2f}% | "
          f"Params: {metrics['parameters']/1e6:6.2f}M | Size: {metrics['size_mb']:6.2f} MB")
```

## Integration Checklist

For each notebook, follow this checklist:

- [ ] Add `from ml_utils import *` to top imports
- [ ] Add Configuration cell early in notebook
- [ ] Replace dataset loading with Template 2
- [ ] Add docstrings to all model classes following Template 3
- [ ] Replace train_model() function with Template 4
- [ ] Add comprehensive evaluation cell using Template 5
- [ ] Save final experiment summary
- [ ] Test notebook runs without errors
- [ ] Verify reproducibility (running twice gives same results)

---

## Specific Notes Per Notebook

### For `imagenet_models.ipynb`:
- **Challenge**: Multiple models (FastTinyCNN, StrongCNN, ImprovedTinyCNN, AlexNet, AlexNet3x3)
- **Solution**: Create separate training cells for each model, consolidate comparison at end
- **Key addition**: Add early stopping to avoid over-training weak models

### For `3x3_models.ipynb`:
- **Challenge**: Focus on comparing kernel sizes (2×2, 3×3, 5×5, 7×7)
- **Solution**: Document kernel size choices and receptive field calculations
- **Key addition**: FLOPs comparison table for kernel size efficiency

### For `imagenet_models2.ipynb`:
- **Challenge**: ConvNeXtSETiny is complex with SE blocks
- **Solution**: Break into modular blocks with clear documentation
- **Key addition**: Ablation study tracking (with/without SE, different depths)

---

## Testing Your Refactoring

After refactoring a notebook, verify:

```bash
# 1. Check imports work
python3 -c "from ml_utils import *; print('✓ Imports OK')"

# 2. Run a test cell
# (In Jupyter) Run first configuration cell

# 3. Verify reproducibility
# Run twice, check that final accuracy is identical
```

---

## Common Pitfalls to Avoid

❌ **Don't**: Hardcode hyperparameters in training loops  
✅ **Do**: Use `config["param_name"]` from ExperimentConfig

❌ **Don't**: Mix data loading with model definition  
✅ **Do**: Separate into distinct sections with clear headers

❌ **Don't**: Duplicate `train_model()` in multiple notebooks  
✅ **Do**: Import and use `train_model()` from ml_utils

❌ **Don't**: Skip saving experiment metadata  
✅ **Do**: Always call `create_results_summary()` after training

---

## Performance Tips

To speed up refactoring:
1. Start with one small notebook (e.g., imagine_models2.ipynb)
2. Apply all templates systematically
3. Test reproducibility thoroughly
4. Use the working notebook as template for others

---

For questions or issues, refer to:
- `REFACTORING_GUIDE.md` - Full overview
- `ml_utils.py` - Function documentation and docstrings
- Your refactored `TinyHybridNet_QAT.ipynb` - Working example
