## coding: UTF-8
import torch
import random

class  OneHotTransform(object):
    def __init__(self, num_character):
        self.num_character =  num_character
    def __call__(self, tensor):
        one_hot = torch.eye(self.num_character)[tensor.long()]
        return one_hot


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_size, transform):
        self.data_size = data_size
        self.transform = transform
        self.x = self.generate_number()
        self.y = self.generate_number()
        self.z = self.x - self.y
        self.char2id = {str(i) : i for i in range(10)}
        self.char2id.update({" ":10, "-":11, "_":12})

        self.inputs = []
        self.targets = []
        for i in range(self.data_size):
            x_y_ = str(self.x[i].item()) + "-" + str(self.y[i].item())
            z_ = "_" + str(self.z[i].item())
            x_y_ = "{: <7}".format(x_y_)
            z_ = "{: <5s}".format(z_)

            input = [self.char2id[x_y_item] for x_y_item in x_y_]
            target = [self.char2id[z_item] for z_item in z_]

            self.inputs.append(input)
            self.targets.append(target)

        self.inputs = torch.Tensor(self.inputs)
        self.targets = torch.Tensor(self.targets)
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        input = self.transform(self.inputs[idx])
        target = self.transform(self.targets[idx])
        print(idx)
        return input, target
        
    def generate_number(self):
        data = [random.randint(0, 999) for _ in range(self.data_size)]
        return torch.tensor(data)
        

def func_kwargs(**kwargs):
    print('kwargs: ', kwargs)
    print('type: ', type(kwargs))
    if not kwargs:
        print('none')

# DATA_SIZE = 50000
# NUM_CHARACTER = 13

# transform = OneHotTransform(num_character=NUM_CHARACTER)
# dataset = Dataset(DATA_SIZE, transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)



import matplotlib.pyplot as plt

for i, c in enumerate(plt.get_cmap("tab10")):
    print(i)


# import pandas as pd
# import numpy as np

# def func(df):
#     df_= pd.DataFrame(np.random.rand((2, 8)), 
#                    index=[0, 1], columns=['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'])
#     df['E'] = pd.Series([1,1], index=['SIX', 'SEVEN'])
#     return df



# df = pd.DataFrame(np.random.rand(16).reshape((2, 8)), 
#                    index=[0, 1], columns=['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'])

            
# print()


