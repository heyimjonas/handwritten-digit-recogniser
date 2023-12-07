# Handwritten Digit Recogniser

## Description
This project uses a PyTorch-based Convolutional Neural Network (CNN) trained on the [MNIST dataset](https://www.google.com/search?q=MNIST+dataset) to recognise handwritten digits. It also provides a GUI where you can draw a digit and receive a prediction.

## Features
- CNN trained with PyTorch on MNIST.
- GUI for drawing digits using Tkinter.
- Model checkpoints saved for easy loading and inference.

### Installation
```
pip install -r requirements.txt
```
### Training
```
python training.py
```

### Running GUI

```
python gui.py
```

Draw a digit (0–9) in the canvas and click "Predict" to see the model’s prediction.