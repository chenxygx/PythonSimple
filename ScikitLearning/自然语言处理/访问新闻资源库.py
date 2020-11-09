import numpy as np
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train')

print(twenty_train)