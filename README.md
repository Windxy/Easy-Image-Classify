# Easy-Image-Classify
简单可扩展的PyTorch图像分类代码（Simple and extensible image classification code based on  PyTorch）



### 1.环境Enviroment

Python == 3.7.6

torch == 1.2.0

torchvision == 0.4.0



### 2.文件目录结构

待完成（to be completed）



### 3.图像分类数据集搭建Build Custom Image Classification Dataset

使用自定义数据集步骤（This is way to build the custom dataset）

- step1.制作/收集数据，并将相同类别的数据放入**相同名字文件夹**下 Create/Collect data and store the same type of data in folders with the same name
- step2.将上述多个类别的数据文件夹，统一放入一个文件夹下，并为该文件夹取一个合适的名字**A**，最终放入CustomDataset文件夹下 Put the folders of the above categories into one folder, give the folder a proper name (eg. ‘dog’ or 'flower') and finally put it in the CustomDataset folder  
- step3.打开CustomDataset文件夹下的train_val_test_split.py文件，将`root`变量修改为名字**A**，并设置合适的训练、验证和测试比例 Open the train_val_test_split.py in the CustomDataset folder, reset the variable `root ` to name A, and set the appropriate training, validation, and test ratios
- step4.运行train_val_test_split.py Run train_val_test_split.py

```
# 在CustormDataset目录下，你应该设置为一下格式 In the CustormDataset directory, you should set as following file tree format  
# 然后再开始执行step3和step4 Then proceed to step3 and step4
CustomDataset
│  train_val_test_split.py
├─ flowers
│  ├─daisy
│  │      xx.jpg
│  │      xx.jpg
│  │      ...
│  ├─rose
│  │      xx.jpg
│  │      xx.jpg
│  │      ...
│  ├─tulip
│  │      xx.jpg
│  │      xx.jpg
│  │ 	    ... 
│  │  ...      
```



### 4.模型搭建

模型搭建的步骤，在model文件夹下，给出了一个示例模型，搭分类模型的方法很简单，同时，非常推荐你使用我们 [整理好的分类模型（注释版）](https://github.com/Windxy/Classic_Network_PyTorch) In the model folder, an example model is given. The method of constructing the classification model is very simple. Meanwhile, we highly recommend you to use our [Classic Classification Model(Annotated Version)](https://github.com/Windxy/Classic_Network_PyTorch).



### 5.训练和验证

运行train_val.py   run train_val.py 


```bash
python train_val.py
```

训练和验证的步骤如下 The steps for training and validation are as follows
```
- step0.参数配置 parameter configuration
- step1.设置数据集和对应的transform Set the dataset and data Transform  
- step2.加载模型 build model
- step3.加载损失函数 build loss function
- step4.加载学习器 build optimizer
- step5.加载预训练权重 Load the pre-training weights
- step6.开始训练 training
  - step6.1数据加载 load data
  - step6.2前向计算 forward
  - step6.3损失计算 calculate loss
  - step6.4后向梯度计算 gradient
  - step6.5参数更新 parameter update
```



### 6.测试和评估

运行test.py   run test.py 

```bash
python test.py
```



### 7.模型部署

待完成（to be completed）



### 参考Reference

待完成（to be completed）

