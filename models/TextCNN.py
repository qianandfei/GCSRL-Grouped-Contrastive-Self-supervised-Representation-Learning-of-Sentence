'''
@Author: Qian wang
@Date: 2022-6-20
'''
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from utils import adjust_learning_rate, accuracy, save_checkpoint, AverageMeter, train, validate, testing

class ModelConfig(object):
    '''
    模型配置参数
    '''
    # 训练参数
    epochs = 120  # epoch数目，除非early stopping, 先开20个epoch不微调,再开多点epoch微调
    batch_size = 512 # batch_size
    workers = 8  # 多处理器加载数据
    lr = 1e-4  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
    weight_decay = 1e-5 # 权重衰减率
    decay_epoch = 20 # 多少个epoch后执行学习率衰减
    improvement_epoch = 6 # 多少个epoch后执行early stopping
    is_Linux = True # 如果是Linux则设置为True,否则设置为else, 用于判断是否多处理器加载
    print_freq = 100  # 每隔print_freq个iteration打印状态
    checkpoint = None  # 模型断点所在位置, 无则None
    best_model = None # 最优模型所在位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    model_name = 'TextCNN' # 模型名

    kernel_sizes = [1,1,1,6,15,20] # 不同尺寸的kernel#,3,4,5,6数据太小（小余一万）不适合用大的参数#1,1,1,1达到47.78
    dropout = 0.5 # dropout
    kernel_num =640 # kernel数量
    model_dim=kernel_num*len(kernel_sizes)
    embed_dim = 300 # 未使用预训练词向量的默认值
    static = True # 是否使用预训练词向量, static=True, 表示使用预训练词向量
    non_static = True # 是否微调，non_static=True,表示微调
    multichannel = True # 是否多通道


class ModelCNN(nn.Module):
    '''
    TextCNN: CNN-rand, CNN-static, CNN-non-static, CNN-multichannel
    '''
    def __init__(self,  kernel_num, kernel_sizes, model_dim):
        '''
        :param vocab_size: 词表大小
        :param embed_dim: 词向量维度
        :param kernel_num: kernel数目
        :param kernel_sizes: 不同kernel size
        :param class_num: 类别数
        :param pretrain_embed: 预训练词向量
        :param dropout: dropout
        :param static: 是否使用预训练词向量, static=True, 表示使用预训练词向量
        :param non_static: 是否微调，non_static=True,表示不微调
        :param multichannel: 是否多通道
        '''
        super(ModelCNN, self).__init__()
        
        # 初始化为单通道
        channel_num = 1
        embed_dim=300
        dropout=False

    
        # 卷积层, kernel size: (size, embed_dim), output: [(batch_size, kernel_num, h,1)] 
        self.convs = nn.ModuleList([
            nn.Conv2d(channel_num, kernel_num, (size, embed_dim)) 
            for size in kernel_sizes
        ])

    
    def forward(self, x):
        '''
        :params x: (batch_size, max_len)
        :return x: logits
        '''

        x = x.unsqueeze(1)    #(batch_size, 1, max_len, word_vec)#输入满足这个维度
        
        # 卷积    
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(batch_size, kernel_num, h)]
        # 池化
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(batch_size, kernel_num)]
        # flatten
        x = torch.cat(x, 1) # (batch_size, kernel_num * len(kernel_sizes))
        return x 



