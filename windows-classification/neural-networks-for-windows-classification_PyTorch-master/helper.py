from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams["figure.figsize"] = (11,5)

import os
import numpy as np
import scipy
import datetime
from io import StringIO

import sys

import sys
import csv


import torch
from torch.autograd import Variable
from functools import reduce
import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


count_ops = 0
count_params = 0
module_number = 0
modules_flops = []
modules_params = []
to_print = False



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def LOG(message, logFile):
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(logFile, "a") as fp:
        fp.write(msg + "\n")

    print(msg)

def log_summary(model, logFile):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model.summary()

    sys.stdout = old_stdout
    summary = mystdout.getvalue()

    LOG("Model summary:", logFile)

    for line in summary.split("\n"):
        LOG(line, logFile) 


def log_stats(path, epochs_acc_train, epochs_loss_train, epochs_lr, epochs_acc_test, epochs_loss_test):

    with open(path + os.sep + "train_errors.txt", "a") as fp:
        fp.write("%.4f " % epochs_acc_train)
        fp.write("\n")

    with open(path + os.sep + "train_losses.txt", "a") as fp:
        fp.write("%.4f " % epochs_loss_train)
        fp.write("\n")

    with open(path + os.sep + "epochs_lr.txt", "a") as fp:
        fp.write("%.7f " % epochs_lr)
        fp.write("\n")    

    with open(path + os.sep + "val_errors.txt", "a") as fp:
        
        fp.write("%.4f " % epochs_acc_test)
        fp.write("\n")
    
    with open(path + os.sep + "val_losses.txt", "a") as fp:
        fp.write("%.4f " % epochs_loss_test)
        fp.write("\n")

    
    
def plot_figs(epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses, args, captionStrDict):
    
    """
    plot epoch test error after model testing is finished
    """

    all_y_labels = ["train error (%)", "train loss", "test error (%)", "test loss"]
    save_file_names = ["train_error.png","train_loss.png","test_error.png","test_loss.png"]
    fig_titles = [args.model + " Train Classification error"+captionStrDict["fig_title"], args.model + " Train Loss"+captionStrDict["fig_title"], args.model + " Test Classification error"+captionStrDict["fig_title"], args.model + " Test Loss"+captionStrDict["fig_title"]]
    all_stats = [epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses]
    for y_label, file_name, fig_title, data in zip(all_y_labels, save_file_names, fig_titles, all_stats):

        fig, ax0 = plt.subplots(1, sharex=True)
        colormap = plt.cm.tab20

        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(data[0]))])

        last = len(data[0])-1

        for k in range(len(data[0])):
            # Plots
            x = np.arange(len(data)) + 1
            y = np.array(data)[:, k]

            if y_label in ["train loss", "test loss"] and len(data[0]) > 1: # means model generates more than one classifier
                if k == last:
                    c_label = "total sum loss"
                elif k == (last-1):
                    c_label = captionStrDict["elastic_final_layer_label"]
                else:
                    c_label = captionStrDict["elastic_intermediate_layer_label"] + str(k)                

            else:
                if k == last:
                    c_label = captionStrDict["elastic_final_layer_label"]
                else:
                    c_label = captionStrDict["elastic_intermediate_layer_label"] + str(k)

            ax0.plot(x, y, label=c_label)
        
        ax0.set_ylabel(y_label)
        ax0.set_xlabel(captionStrDict["x_label"])
        ax0.set_title(fig_title)

        ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        fig_size = plt.rcParams["figure.figsize"]

        plt.rcParams["figure.figsize"] = fig_size
        plt.tight_layout()

        plt.savefig(args.savedir + os.sep + file_name)
        plt.close("all")  

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])



def save_checkpoint(state, args):
    
    model_dir = os.path.join(args.savedir, 'save_models')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    torch.save(state, best_filename)
    print("=> saved checkpoint '{}'".format(best_filename))

    return

def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):

    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr