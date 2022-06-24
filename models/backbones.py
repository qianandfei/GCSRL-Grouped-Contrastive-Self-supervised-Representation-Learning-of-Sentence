from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from transformers import BertTokenizer,BertModel,BertForMaskedLM
from configs import *
import torch.nn as nn
from models import TextCNN




def textcnn():
    model_opt = TextCNN.ModelConfig()
    model =TextCNN.ModelCNN(
                     kernel_num=model_opt.kernel_num,
                     kernel_sizes=model_opt.kernel_sizes,
                        model_dim=model_opt.model_dim
                             )
    return model

if __name__ == "__main__":
    cnn=textcnn()


