import pickle
import numpy as np

with open('C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/Test/1_222.pkl', 'rb') as f:
    my_dict = pickle.load(f)


with open('C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/Train/1_126.pkl', 'rb') as f:
    my_dict2 = pickle.load(f)


file_path = r"C:/Users/Jacob/Documents/STG-NF-20250414T020943Z-001/Pickle_files-20250418T193829Z-001/Pickle_files/GT/1_222.npy"
data = np.load(file_path)

print(my_dict2)