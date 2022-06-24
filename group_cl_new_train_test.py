import torch
import numpy as np
from tqdm import tqdm
from configs import get_args, set_deterministic
from models import get_model
from selfdatasets import colaDataset
from models.plus_proj_layer import cos_similarity,cl_cal_loss
from optimizers import get_optimizer, LR_Scheduler
from gensim.models import KeyedVectors as Vectors
import  train_test_eval


def main(device, args, embedding, padding_value, max_len):
    dataset_kwargs = {  # 字典
        'data_name': args.data_name,
        'embedding': embedding,
        'padding_value': padding_value,
        'max_len': max_len,
        'data_path': args.data_path
    }
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    train_loader = torch.utils.data.DataLoader(  # Dataset 可寻址  但DataLoader只能迭代
        # DataLoader:将自定义的dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练(就是一个数据迭代器)
        dataset=colaDataset(split='train', Is_train=args.do_train, **dataset_kwargs),
        shuffle=True,  # 随机打乱数据并分成若干个mini-batch
        **dataloader_kwargs)

    # define model

    model = get_model(args.model, args.backbone).to(device)  # 将模型复制到gpu
    if torch.cuda.device_count() > 1:  # distributedDataparallel可进行多机多gpu训练也可单机多gpu
        model = torch.nn.DataParallel(
            model)  # 进行同一服务器多GPU运算，forward返回多个gpu合并的结果torch.nn.DataParallel(model, device_ids=[0, 1, 2])只使用前三张卡
    if args.model == 'simsiam' and args.proj_layers is not None: model.projector.set_layers(args.proj_layers)
    if args.do_train:

        optimizer = get_optimizer(  # sgd
            args.optimizer, model,
            lr=args.base_lr * args.batch_size / 128,  # args.base_lr * args.batch_size / 128#256
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        lr_scheduler = LR_Scheduler(
            optimizer,
            args.warmup_epochs, args.warmup_lr * args.batch_size / 128,
            args.num_epochs, args.base_lr * args.batch_size / 128, args.final_lr * args.batch_size / 128,
            len(train_loader),
            constant_predictor_lr=True  # see the end of section 4.2 predictor
        )

        loss_meter = AverageMeter(name='Loss')  # 计算并储存当前值和平均值
        plot_logger = PlotLogger(params=['lr', 'loss', 'accuracy'])
        # Start training

        global_progress = tqdm(range(0, args.stop_at_epoch), desc='Training')  # python 进度条管理
        total_acc=[0]
        num_nochange=0
        for epoch in global_progress:  # 在此循环中显示总体进度
            loss_meter.reset()
            model.train()

            # plot_logger.update({'epoch':epoch, 'accuracy':accuracy})
            local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}',
                                  disable=args.hide_progress)  # 每个epoch的进度

            for idx, ((data1, data2), labels, mask) in enumerate(local_progress):
                # 计算自注意力

                # data1=data_selfattention(data1,data1,data1,element_num)#在增强之后进行自注意会不会没有之前好？
                # data2=data_selfattention(data2,data2,data2,element_num)#或者根据不同增强的手段来进行自注意力更合理？

                model.zero_grad()  # 将模型的参数梯度设为零，因为每一个batch时并不需要与其他batch的梯度混合起来累积计算若不设为零则梯度将会在整个训练中累计计算

                p1, z2, p2, z1 = model(data1.to(device, non_blocking=True),
                                       data2.to(device, non_blocking=True), mask.to(device,
                                                                                    non_blocking=True))  # model.forward  # 非阻塞允许多个线程同时进入临界区  返回的是由多个gpu计算的向量ueeze(2), x2_sp.unsqueeze(2).transpose(-2, -1))



                n = int(p1.size(-1) / args.num_group)
                p1_sub = p1.view(n, args.batch_size, int(p1.size(-1) / n))
                z2_sub = z2.view(n, args.batch_size, int(p1.size(-1) / n))
                p2_sub = p2.view(n, args.batch_size, int(p1.size(-1) / n))
                z1_sub = z1.view(n, args.batch_size, int(p1.size(-1) / n))
                loss_tem2 = torch.tensor([])
                for i in range(n):#不加group_cl为42.76 若加attention不加group_cl为46.08
                    loss_tem =  cos_similarity(p1_sub[i, :, :], z2_sub[i, :, :]) / 2 + cos_similarity(p2_sub[i, :, :],z1_sub[i, :,:]) / 2#torch.norm(p1_sub[i, :, :]-z2_sub[i, :, :],p=2,dim=1) / 2 + torch.norm(p2_sub[i, :, :]-z1_sub[i, :,:],p=2,dim=1) / 2#cl_cal_loss(p1_sub[i, :, :], z2_sub[i, :, :])/2+ cl_cal_loss(p2_sub[i, :, :],z1_sub[i, :, :])/2
                    loss_tem2 = torch.cat((loss_tem2.to(device, non_blocking=True), loss_tem.unsqueeze(0)), 0)     #cos_similarity(p1_sub[i, :, :], z2_sub[i, :, :]) / 2 + cos_similarity(p2_sub[i, :, :],z1_sub[i, :,:]) / 2
                    loss_tem = []
                loss =torch.mean(loss_tem2)#loss_no_group#torch.mean(loss_tem2)
                loss.backward()
                optimizer.step()  # 更新一个batch计算后的梯度
                loss_meter.update(loss.item())  # .item将张量中的数值取出来(只能一维张量) 不然就是张量无法进行数值计算
                lr = lr_scheduler.step()  # 更新优化器的学习率 用来指定多少个epoch后更换一次学习率

                data_dict = {'lr': lr, "loss": loss_meter.val}
                local_progress.set_postfix(data_dict)
                plot_logger.update(data_dict)
            if epoch>=0:
                accuracy,num_nochanges=train_test_eval.eval_in_train(args,embedding,padding_value,model,epoch,max_len,total_acc,num_nochange)
                total_acc.append(accuracy)
                num_nochange=num_nochanges
                #break
            else:accuracy=0
            #此时model.module.backbone=resnet50是已经训练过的了
            global_progress.set_postfix({"accuracy": accuracy})  # 设置后缀 另外若在tqdm中使用print()时可能会出现多条进度条此时应将print()改为tqdm.write()


           

if __name__ == "__main__":
    a=[[[1,2],[3,4]],[[5,6],[7,8]]]
    b = [[1, 2], [3, 4],[3,4,5]]
    a=torch.tensor(b)
    b=torch.nonzero(a==3)
    set_deterministic(2)
    args = get_args()
    args.download = True
    embedding = Vectors.load_word2vec_format(args.emb_file, binary=True)
    bias = np.sqrt(3.0 / args.emb_dim)
    padding_value = torch.FloatTensor(args.emb_dim)
    padding_value = torch.nn.init.uniform_(padding_value, -bias, bias)
    padding_value = np.array(padding_value, dtype=np.float32)
    main(device=args.device, args=args, embedding=embedding, padding_value=padding_value, max_len=args.max_len)