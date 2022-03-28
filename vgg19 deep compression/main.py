
from models.cifar10 import VGG
import numpy as np

model = VGG(bits=10)

model.train(100000, False)
model.train(150000, True)
# retrain,
print(model.compress(1))


