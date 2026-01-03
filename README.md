MNIST Digit Classifier (Neural Network)

GDSC AI/ML Inductions – Intermediate Task 1

📌 Problem Statement

Build a digit classification pipeline using the MNIST dataset that takes an input image of a handwritten digit and outputs the predicted digit (0–9). This is a multi-class classification problem.

📂 Dataset

MNIST contains grayscale images of handwritten digits:
Image size: 28 × 28
Classes: 10 (digits 0 to 9)

Each sample consists of:
image tensor
label (ground-truth digit)

⚙️ Preprocessing

To make the images suitable for training:
Images are converted into tensors (ToTensor)
Pixel values are normalized to improve training stability (Normalize)
Images are flattened from 28×28 -> 784 for the feedforward network
Data is fed in mini-batches using a DataLoader for efficient training.

🧠 Model Architecture

A simple feedforward neural network (MLP) was implemented:
Input Layer: 784 (flattened pixels)
Hidden Layer: 128 neurons + ReLU activation
Output Layer: 10 logits (one per digit class)

ReLU introduces non-linearity, allowing the network to learn complex decision boundaries beyond a purely linear classifier.

🏋️ Training Setup

Loss Function: CrossEntropyLoss
Computes classification loss for multi-class problems (internally applies softmax)
Optimizer: Adam
Performs gradient-based updates with adaptive learning rates for faster convergence

Training is done over multiple epochs using:
forward pass → loss computation → backpropagation → weight updates

📊 Evaluation

Model performance is evaluated on the test split using accuracy.

Test Accuracy Achieved: 0.96 (while test-running the code)

🎯 Demo: Image → Predicted Digit

A sample test image is passed through the trained model to demonstrate the full pipeline:
The digit image is displayed
The predicted digit is shown alongside the actual label
This satisfies the requirement of taking an input digit image and outputting the predicted digit as text.

📌 Conclusion

This project demonstrates an end-to-end MNIST digit classification pipeline using a simple neural network. With correct preprocessing, loss setup, and training, a lightweight MLP achieves strong performance (~96% accuracy) without needing complex architectures.

🛠️ Technologies Used

Python
PyTorch (torch, torchvision)
Matplotlib