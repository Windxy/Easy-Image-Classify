import os

root = 'flowers'
train = 0.7
val = 0.1
test= 0.2

if __name__ == '__main__':
    data_list = os.listdir(root)
