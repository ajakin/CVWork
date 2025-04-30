## README

暂时有两个文件和一个数据集：

KvasirDataset.py ：继承Dataset类编写了Kvasir数据集的Dataset类，定义了数据图像加载时所需的操作，为后续Dataloader提供数据加载；

segnet.py：定义了基于Mobile结构的Segnet，暂时是基础模型，后面要模块；

数据集：KvasirSeg数据集，包括images、masks两个子文件夹和一个json文件（https://datasets.simula.no/downloads/kvasir-seg.zip）

新增:
SLIC.py:对skimage.segmentation中的slic算法进行封装，使其维度能跟随batchsize变化，得到超像素图superpixel_map用于超像素池化
SuperPixPool.py:超像素池函数，传入特征图和超像素图进行超像素池化



