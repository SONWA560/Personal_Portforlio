# train_waste_classifier.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import time
from PIL import Image, ImageFile
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Custom Dataset Class with Robust Loading
class RobustImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        samples = []
        for target_class in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(class_dir):
                continue
                
            for root_dir, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(root_dir, fname)
                        try:
                            with Image.open(path) as img:
                                img.verify()
                            samples.append((path, self.class_to_idx[target_class]))
                        except Exception as e:
                            print(f"Skipping corrupt file: {path}")
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, target
        except Exception as e:
            print(f"Error loading image (skipping): {path}")
            # Return a zero tensor and dummy target
            return torch.zeros(3, 224, 224), -1

# 2. Custom DataLoader with Safe Collation
def safe_collate(batch):
    batch = [item for item in batch if item[1] != -1]  # Filter out failed samples
    if len(batch) == 0:
        return torch.Tensor(), torch.LongTensor()
    return torch.utils.data.dataloader.default_collate(batch)

# 3. Main Training Function
def main():
    # Configuration
    data_dir = '/Users/sonwabise/Documents/Anaconda/Python/venv/Multi Class classification/split_dataset'
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transforms
    image_size = 224
    batch_size = 16  # Reduced for stability

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {
        x: RobustImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in ['train', 'val']
    }

    # Create dataloaders
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=x == 'train',
            num_workers=0,
            collate_fn=safe_collate
        )
        for x in ['train', 'val']
    }

    # Verify we have data
    for phase in ['train', 'val']:
        if len(image_datasets[phase]) == 0:
            raise ValueError(f"No valid images found in {phase} directory")

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Model setup
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
        best_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print('-' * 30)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples = 0

                for inputs, labels in dataloaders[phase]:
                    if inputs.nelement() == 0:  # Skip empty batches
                        continue

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    batch_size = inputs.size(0)
                    running_loss += loss.item() * batch_size
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += batch_size

                if phase == 'train':
                    scheduler.step()

                if total_samples > 0:
                    epoch_loss = running_loss / total_samples
                    epoch_acc = running_corrects.float() / total_samples
                    print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), 'best_model.pth')

        return model

    # Run training
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)
    
    # Save final model
    torch.save(model.state_dict(), "waste_classifier_final.pth")
    print("Training complete and model saved!")

if __name__ == '__main__':
    main()