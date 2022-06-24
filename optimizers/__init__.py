from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch
from .lr_scheduler import LR_Scheduler
from transformers.optimization import AdamW

def get_optimizer(name, model, lr, momentum, weight_decay):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,nesterov=True)
    elif   name=='adam':# betas���ڼ����ݶ��Լ��ݶ�ƽ��������ƽ��ֵ��ϵbetas1 һ�׾���ָ��˥����    �ڶ���beta2����ϡ���ݶȣ�NLP����Ӧ��Ϊ�ӽ�1����  ��0.999
        optimizer =torch.optim.Adam(parameters,lr=lr,betas=(0.9,0.999),weight_decay=weight_decay)#,eps=1e-6   espΪ�������ֵ���ȶ��Զ���ӵ���ĸ��������ĸΪ��#,amsgrad=True ������ʷ����v_t����Ϊv_{max}��ÿ�μ��㶼��������v_{max}���������õ�ǰv_t
    elif name=='adamw':
        optimizer  =AdamW(parameters,lr=1e-5)

    elif name == 'lars_simclr': # Careful
        optimizer = LARS_simclr(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(
            torch.optim.SGD(
                parameters,
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            ),
            trust_coefficient=0.001, 
            clip=False
        )
    else:
        raise NotImplementedError
    return optimizer



