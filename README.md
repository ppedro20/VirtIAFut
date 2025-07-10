# VirtIAFut

Objective: explore how Artificial Intelligence/Machine Learning can be effectively used to predict the movement of players and the ball in order to improve the accuracy of element virtualization.

How to run:

```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

1. Detections
2. Predictions
3. Data Visualization

**Video of Detections Tracking**

**Current Results of Predictions**

Real Data - with Linear Interpolation

![Soccer Animation](data/animations/frames60s.gif "Real Data")

Transformer

![Soccer Animation](data/animations/tf+train_frames.gif "TF")

LSTM

![Description](data/animations/lstm88+train_frames.gif "lstm")
