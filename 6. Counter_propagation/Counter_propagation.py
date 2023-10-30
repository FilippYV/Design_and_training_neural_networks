import pandas as pd
import random
import numpy as np
from sklearn.datasets import load_iris

def get_input_data():
    data = load_iris
    print(data)
    return data

if __name__ == '__main__':
    get_input_data()