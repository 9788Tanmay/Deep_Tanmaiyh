# needs model loss, accuracy, data, they extend compressible models
from models.cifar10 import VGG
import numpy as np

# Training VGG-16 on CIFAT-10
model = VGG(bits=10)

model.train(100000, False)
model.train(150000, True)
# retrain,
print(model.compress(1))


