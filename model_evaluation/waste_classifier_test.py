# test_model_metrics.py
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import os

# Configuration
TEST_DIR = "/Users/sonwabise/Documents/Anaconda/Python/venv/Multi Class classification/split_dataset/test"
MODEL_PATH = "waste_classifier_final.pth"
CLASS_NAMES = ['cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic']
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32

def load_model(model_path, num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def create_test_loader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset.classes

def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return all_preds, all_labels, all_probs, accuracy

def save_classification_report(y_true, y_pred, classes, accuracy):
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Add overall accuracy
    report_df.loc['accuracy', 'support'] = len(y_true)
    report_df.loc['accuracy', 'precision'] = accuracy
    report_df.loc['accuracy', 'recall'] = accuracy
    report_df.loc['accuracy', 'f1-score'] = accuracy
    
    # Save to CSV
    report_df.to_csv('classification_report.csv', float_format='%.4f')
    print("Classification report saved to classification_report.csv")
    
    return report_df

def save_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv('confusion_matrix.csv')
    print("Confusion matrix saved to confusion_matrix.csv")
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix visualization saved to confusion_matrix.png")

def save_predictions(y_true, y_pred, probs, classes):
    results = pd.DataFrame({
        'True_Label': [classes[l] for l in y_true],
        'Predicted_Label': [classes[p] for p in y_pred],
        'Correct': [l == p for l, p in zip(y_true, y_pred)],
        'Confidence': [probs[i][p] for i, p in enumerate(y_pred)]
    })
    
    # Add probabilities for each class
    for i, class_name in enumerate(classes):
        results[f'Prob_{class_name}'] = [prob[i] for prob in probs]
    
    results.to_csv('detailed_predictions.csv', index=False, float_format='%.4f')
    print("Detailed predictions saved to detailed_predictions.csv")

def plot_sample_predictions(model, test_loader, class_names):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    outputs = model(images.to(DEVICE))
    _, preds = torch.max(outputs, 1)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    
    plt.figure(figsize=(12, 8))
    for idx in range(min(8, len(images))):
        plt.subplot(2, 4, idx+1)
        img = images[idx].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}\nConf: {probs[idx][preds[idx]]:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()
    print("Sample predictions saved to sample_predictions.png")

if __name__ == '__main__':
    print("Loading model...")
    model = load_model(MODEL_PATH, len(CLASS_NAMES))
    
    print("Creating test loader...")
    test_loader, detected_classes = create_test_loader()
    
    if detected_classes != CLASS_NAMES:
        print(f"Warning: Detected classes {detected_classes} don't match expected classes {CLASS_NAMES}")
    
    print("Evaluating model...")
    preds, labels, probs, accuracy = evaluate_model(model, test_loader)
    
    print("\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.2%}")
    
    # Save all metrics and results
    report_df = save_classification_report(labels, preds, CLASS_NAMES, accuracy)
    save_confusion_matrix(labels, preds, CLASS_NAMES)
    save_predictions(labels, preds, probs, CLASS_NAMES)
    plot_sample_predictions(model, test_loader, CLASS_NAMES)
    
    print("\nClassification Report:")
    print(report_df)
    
    print("\nEvaluation complete! All results saved to CSV files.")