
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchsummary import summary

import os
import time
import datetime


from opts import args
from helper import LOG, log_summary, log_stats, AverageMeter, accuracy, save_checkpoint, plot_figs
from models import *
from windows_image_data_loader import windows_image_data_loader

# Init Torch/Cuda
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def validate(val_loader, model, criterion):
    model.eval()
    all_acc = AverageMeter()
    all_loss = AverageMeter()
      
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)     
        
        losses = 0
        
        output = model(input_var)
        with torch.no_grad():
            for ix in range(len(output)):
                loss = criterion(output, target_var)
                all_loss.update(loss.item(), input.size(0))
                losses += loss
                
                prec1 = accuracy(output.data, target)
                print("val prec1: ", prec1)
                all_acc.update(prec1[0].item(), input.size(0))

    accs = float(100-all_acc.avg)
    ls = all_loss.avg
    return accs, ls


def train(train_loader, model, criterion, optimizer, epoch):

    model.train()

    lr = None
    all_acc = AverageMeter()
    all_loss = AverageMeter()

    LOG("==========> train ", logFile)
    # print("num_outputs: ", num_outputs)
    
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
            
        output = model(input_var)      
        # 
        optimizer.zero_grad()

        loss = criterion(output, target_var)

        loss.backward()
        
        optimizer.step()

        all_loss.update(loss.item(), input.size(0))

        # top 1 accuracy
        prec1 = accuracy(output.data, target)
        # print("train prec1: ", prec1)

        all_acc.update(prec1[0].item(), input.size(0))
        
    acc = float(100-all_acc.avg)
    ls = all_loss.avg

    try:
        lr = float(str(optimizer).split("\n")[-5].split(" ")[-1])
    except:
        lr = 100
    
    return acc, ls, lr


def main(**kwargs):
    global args
    lowest_error1 = 100
    
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    program_start_time = time.time()
    instanceName = "Classification_Accuracy"
    folder_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + args.model
    
    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + args.model_name + os.sep + ts_str
    print("path: ",path)
    
    tensorboard_folder = path + os.sep + "Graph"
    os.makedirs(path)
    args.savedir = path

    global logFile
    logFile = path + os.sep + "log.txt"
    args.filename = logFile
    global num_outputs
    
    print(args)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_folder = r"C:\Users\yiper\Desktop\20230904_142329_test\windowsImage" 

    if args.data == "windowsImage":
        fig_title_str = " on Windows_Image_Dataset"
    else:
        LOG("ERROR =============================dataset should be CIFAR10 or CIFAR100", logFile)
        NotImplementedError

    captionStrDict = {
        "fig_title" : fig_title_str,
        "x_label" : "epoch"
    }

    # save input parameters into log file

    LOG("program start time: " + ts_str +"\n", logFile)

    
    if args.model == "Elastic_ResNet18" or args.model == "Elastic_ResNet34" or args.model == "Elastic_ResNet50":
        model = Elastic_ResNet(args, logFile)

    else:
        LOG("--model parameter should be in ResNet", logFile)
        exit()    
    
    num_outputs = model.num_outputs

    LOG("successfully create model: " + args.model, logFile)

    args_str = str(args)
    LOG(args_str, logFile)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True


    summary(model, (3,224,224))


    if args.data == "windowsImage":
        train_loader, test_loader, test_real_loader = windows_image_data_loader(data_folder, args)
    else:
        LOG("ERROR ============================= dataset should be windows Images, logFile")
        NotImplementedError
    
    criterion = nn.CrossEntropyLoss().cuda()

    LOG("==> Full training    \n", logFile)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    # summary(model, (3,224,224))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-4, patience=10)
    
    # implement early stop by own
    EarlyStopping_epoch_count = 0

    epochs_train_accs = []
    epochs_train_losses = []
    epochs_test_accs = []
    epochs_test_losses = []
    epochs_lr = []

    for epoch in range(0, args.epochs):
        
        epoch_str = "==================================== epoch %d ==============================" % epoch
        LOG(epoch_str, logFile)
        # Train for one epoch
        accs, losses, lr = train(train_loader, model, criterion, optimizer, epoch)
        epochs_train_accs.append(accs)
        epochs_train_losses.append(losses)
        epochs_lr.append(lr)
        
        epoch_result = "\n train error: " + str(accs) + ", \n loss: " + str(losses) + ", \n learning rate " + str(lr) 
        LOG(epoch_result, logFile) 
        
        # run on val dataset
        LOG("==> val \n", logFile)
        val_accs, val_losses = validate(test_loader, model, criterion)
        epochs_test_accs.append(val_accs)
        epochs_test_losses.append(val_losses)


        test_result_str = "========> Val: \n  output classifier error: " + str(val_accs) + ", \n val_loss" +str(val_losses)
        LOG(test_result_str, logFile)
        
        total_loss = val_losses
        
        log_stats(path, accs, losses, lr, val_accs, val_losses)

        # Remember best prec@1 and save checkpoint
        is_best = val_accs < lowest_error1 #error not accuracy, but i don't want to change variable names
        
        if is_best:
            
            lowest_error1 = val_accs
            
            save_checkpoint({
                'epoch': epoch,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'best_prec1': lowest_error1,
                'optimizer': optimizer.state_dict(),
            }, args)

        
        scheduler.step(total_loss) # adjust learning rate with test_loss
        
        if epoch == 0:
            prev_epoch_loss = total_loss # use all intemediate classifiers sum loss instead of only one classifier loss
        else:
            if total_loss >= prev_epoch_loss: # means this current epoch doesn't reduce test losses
                EarlyStopping_epoch_count += 1
        if EarlyStopping_epoch_count > 20:
            LOG("No improving test_loss for more than 10 epochs, stop running model", logFile)
            break

    # n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    # FLOPS_result = 'Finished training! FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6)
    # LOG(FLOPS_result, logFile)
    # print(FLOPS_result)

    # test result
    # run on test dataset

    epochs_test_real_accs = []
    epochs_test_real_losses = []

    LOG("==> test \n", logFile)
    test_real_accs, test_real_losses = validate(test_real_loader, model, criterion)
    
    epochs_test_real_accs.append(test_real_accs)
    epochs_test_real_losses.append(test_real_losses)

    test_result_str = "=======> test \n classifier error: " + str(test_real_accs) + ", \n test_loss： " +str(test_real_losses) 
    LOG(test_result_str, logFile)
    
    # log_stats(path, accs, losses, lr, test_real_accs, test_real_losses)

    end_timestamp = datetime.datetime.now()
    end_ts_str = end_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    LOG("program end time: " + end_ts_str +"\n", logFile)

    # here plot figures
    # plot_figs(epochs_train_accs, epochs_train_losses, epochs_test_accs, epochs_test_losses, args, captionStrDict)
    LOG("============Finish============", logFile)

if __name__ == "__main__":

    main()
