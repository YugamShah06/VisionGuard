VisionGuard: Automated Eye Disease Detection with Deep Learning
Overview
VisionGuard is an AI-powered platform for automated diagnosis of ophthalmological diseases using retinal fundus images. The system leverages advanced Convolutional Neural Network (CNN) architectures-including Conv2D, ResNet50, and VGG19-in both standard (parameter) and hyperparameter-tuned variants to classify images into four categories: Diabetic Retinopathy, Glaucoma, Cataract, and Normal. The project demonstrates the impact of model choice and hyperparameter tuning on classification performance and provides a reproducible, scalable pipeline for medical image analysis.

Dataset
Source: Kaggle Retinal Fundus Image Dataset

Total Images: 4,217

Classes:

Diabetic Retinopathy: 1,098 images

Glaucoma: 1,007 images

Cataract: 1,038 images

Normal: 1,074 images

Structure: Folder-based, stratified 80-10-10 split for training, validation, and testing.

Results
Best Accuracy:
91.78% (ResNet50 Standard)
Most Challenging Class:
Glaucoma (highlighting class imbalance issues)
Key Finding:
Hyperparameter tuning can both improve or significantly degradeperformance (e.g., ResNet50 Tuned: 41.92% accuracy)
Usage1.
Clone the repository:
git clone https://github.com/YugamShah06/VisionGuard.gitcd VisionGuard2.
Install requirements:
pip install -r requirements.txt3.
Prepare data:
Place the dataset in the specified folder structure as described above.4.
Train a model:
Run the training script for your desired architecture (see
/scripts
directory).5.
Evaluate:
Use the evaluation scripts to generate classification reports and confusion matrices.6.
Web App (optional):
Launch the Flask app for interactive predictions.

Tech Stack
Python 3.x
TensorFlow 2.10, Keras 2.10
NVIDIA T4 GPU (16GB VRAM)
Flask (for web deployment)

