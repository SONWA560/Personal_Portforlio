# Personal_Portforlio
Smart Waste Sorting Vision System
Project Overview:
As part of my IFS 315 Emerging Trends in Information Systems module, I developed a machine learning-based vision system for smart waste classification. The objective was to build an image classification model capable of automatically sorting waste into appropriate categories, contributing to more efficient recycling processes and waste management systems.

Problem Statement:
The system needed to accurately classify images of waste items into one of seven categories: paper, glass, cardboard, metal, plastic, e-waste, and medical. This project simulated a real-world computer vision challenge, requiring a model that could handle varied image angles, lighting conditions, and partially obscured objects.

Methodology

Dataset
I made use of a publicly available waste classification dataset consisting of approximately 17,000 images evenly distributed across the seven categories.

Data Pre-processing
To prepare the data for training, I resized all images to 244 x 244 pixels, normalised pixel values using the ImageNet dataset’s mean and standard deviation, and split the data into 70% training, 15% validation, and 15% test sets.

Model Development
The model was developed using PyTorch and the MobileNetV2 architecture, selected for its lightweight, efficient structure — an ideal choice for my MacBook M2, which lacks a CUDA-compatible GPU. To improve accuracy, I applied transfer learning by leveraging pretrained weights from ImageNet, which enhanced feature extraction on the new dataset.

Performance Evaluation
The model’s performance was assessed using accuracy and F1 score. Final results were as follows:
Accuracy: 82.5%
F1 Score: 82.6%

Key Insights:
This project offered valuable hands-on experience in deploying computer vision solutions and navigating the practical challenges of working with AI models. I particularly enjoyed the problem-solving and troubleshooting aspect, from debugging model convergence issues to fine-tuning the training pipeline — which gave me a deeper appreciation for the engineering involved in AI systems.

Tools and Technologies:
-Python
-PyTorch
-MobileNetV2 (Transfer Learning)
-MacBook M2 (Apple Silicon)
-Public waste classification dataset (approximately 17,000 images)
