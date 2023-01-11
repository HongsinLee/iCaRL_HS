from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import os,pdb,copy
from Externals.autoattack import AutoAttack
from Externals.robustbench.eval import benchmark
from Externals.robustbench.data import load_cifar10, load_imagenet
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import yaml




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(args):
    path = Path(os.path.realpath(__file__))
    path = str(path.parent.absolute())
    root = path + "/config/" + args.config_name
    with open(root) as file:
        config = yaml.safe_load(file)
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def convert(s):
        try:
            return float(s)
        except ValueError:
            
            return float(num) / float(denom)

    config = dotdict(config)
    config.alpha = args.alpha if args.alpha != -1 else config.alpha
    config.beta = args.beta if args.beta != -1 else config.beta
    config.gamma = args.gamma if args.gamma != -1 else config.gamma
    config.eta = args.eta if args.eta != -1 else config.eta
    config.epochs = args.epochs if args.epochs != -1 else config.epochs
    config.lr = args.lr if args.lr != -1 else config.lr

    num, denom = config.eps.split('/')
    config.eps = float(num)/float(denom)
    num1,num2,num3 = config.step_size.split('/')
    config.step_size = (float(num1)/float(num2))/float(num3)

    return config

def get_model(config):

    if config.dataset == 'CIFAR10':
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)

        if config.z_normalization:
            from CIFAR10.models.resnet_z import ResNet18_z
            model = ResNet18_z()
        else :
            from CIFAR10.models.resnet import ResNet18
            model = ResNet18()

    
    elif config.dataset == 'CIFAR100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        if config.z_normalization :
            from CIFAR100.models.resnet_z import ResNet18_z
            model = ResNet18_z()
        else :
            from CIFAR100.models.resnet import ResNet18,ResNet50
            model = ResNet18()
            # model = ResNet50()

    elif config.dataset == 'TinyImageNet':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        if config.z_normalization :
            from TinyImageNet.models.PreactResnet_z import PreActResNet18_z
            model = PreActResNet18_z()
        else :
            from TinyImageNet.models.PreactResnet import PreActResNet18,ResNet50
            model = PreActResNet18()
            # model = ResNet50()

    elif config.dataset == 'ImageNet':
        mean = (0.485, 0.456, 0.406)
        std =  (0.229, 0.224, 0.225)
        model = vit_b_16(pretrained = True)

    elif config.dataset == 'CelebA':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if config.z_normalization :
            if config.model =='ResNet18':
                from CelebA.models.resnet_z import resnet18_z
                model = resnet18_z()
            elif config.model =='PreActResNet18':
                from CelebA.models.PreactResnet_z import PreActResNet18_z
                model = PreActResNet18_z()

        else:
            if config.model =='ResNet18':
                from CelebA.models.resnet import resnet18
                model = resnet18()
            elif config.model =='PreActResNet18':
                from CelebA.models.PreactResnet import PreActResNet18
                model = PreActResNet18()


    model = Converter(model,mean,std)
    if config.load:
        path = Path(os.path.realpath(__file__))
        path = str(path.parent.absolute())
        checkpoint = os.path.join(path, config.checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)

    
    return model

def evaluate_l1_l2(model,loader):
    model.eval()
    clean_acc = 0
    pgd_l1_acc = pgd_l2_acc =  0
    total_samples = 0
    fgsm_attack = torchattacks.FGSM(model,eps = 8/255)
    pgd_attack = torchattacks.PGD(model,eps = 8/255,steps = 20, alpha = 2/255, random_start = True)
    PGD_L2_attack = torchattacks.PGDL2(model, eps=128/255, alpha=0.1, steps=20, random_start=True)

    for i,(x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.to(device), y.to(device)
      
        x_pgd_l1 = PGD_L1_Attack(model,x,y)
        x_pgd_l2 = PGD_L2_attack(x,y)

        z_pgd_l1 = model(x_pgd_l1)
        z_pgd_l2 = model(x_pgd_l2)

        z_pgd_l1_out = z_pgd_l1.argmax(dim = 1)
        z_pgd_l2_out = z_pgd_l2.argmax(dim = 1)
        # x_pgd_l1, z_l1_acc = PGD_L1_Attack(model,x,y)

        # pgd_l1_acc += z_l1_acc
        pgd_l1_acc += (z_pgd_l1_out == y).sum() 
        pgd_l2_acc += (z_pgd_l2_out == y).sum() 


    pgd_l1_acc = pgd_l1_acc/total_samples * 100.0
    pgd_l2_acc = pgd_l2_acc/total_samples * 100.0

    return pgd_l1_acc, pgd_l2_acc


def evaluate(model,loader):
    model.eval()
    clean_acc = 0
    fgsm_acc = aa_acc = pgd_acc = pgd_l1_acc = pgd_l2_acc =  0
    total_samples = 0
    fgsm_attack = torchattacks.FGSM(model,eps = 8/255)
    pgd_attack = torchattacks.PGD(model,eps = 8/255,steps = 20, alpha = 2/255, random_start = True)
    pgdl2_attack = torchattacks.PGDL2(model, eps=128/255, alpha = 0.1, steps=20, random_start=True)

    for i,(x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.to(device), y.to(device)
        
        x_fgsm = fgsm_attack(x,y)
        x_pgd = pgd_attack(x,y)
        # x_pgd_l1, z_l1_acc = PGD_L1_Attack(model,x,y)
        x_pgd_l1 = PGD_L1_Attack(model,x,y)
        x_pgd_l2 = pgdl2_attack(x,y)
  
        z_clean = model(x)
        z_fgsm = model(x_fgsm)
        z_pgd = model(x_pgd)
        z_pgd_l1 = model(x_pgd_l1)
        z_pgd_l2 = model(x_pgd_l2)

  
        z_clean_out = z_clean.argmax(dim = 1)
        z_fgsm_out = z_fgsm.argmax(dim = 1)
        z_pgd_out = z_pgd.argmax(dim = 1)
        z_pgd_l1_out = z_pgd_l1.argmax(dim = 1)
        z_pgd_l2_out = z_pgd_l2.argmax(dim = 1)
   
        clean_acc += (z_clean_out ==y).sum().detach()
        fgsm_acc += (z_fgsm_out == y).sum().detach()
        pgd_acc += (z_pgd_out == y).sum().detach()
        pgd_l1_acc += (z_pgd_l1_out == y).sum().detach()
        pgd_l2_acc += (z_pgd_l2_out == y).sum().detach()
        

    clean_acc = clean_acc/total_samples * 100.0
    fgsm_acc = fgsm_acc/total_samples * 100.0
    pgd_acc = pgd_acc/total_samples * 100.0
    pgd_l1_acc = pgd_l1_acc/total_samples * 100.0
    pgd_l2_acc = pgd_l2_acc/total_samples * 100.0


    return clean_acc, fgsm_acc, pgd_acc, pgd_l1_acc, pgd_l2_acc #pgd_l1_acc, pgd_l2_acc

def evaluate_aa(model, dataset = 'CIFAR10'):
    n_samples = 1000
    if dataset == 'CIFAR10':
        x_test, y_test = load_cifar10(n_examples=n_samples)
    elif dataset == 'CIFAR100':
        x_test, y_test = load_cifar100(n_examples=n_samples)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    model.eval() # edit
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    adversary.apgd.n_iter = 20
    adversary.apgd.topk = 3
    x_adv, y_adv, clean_acc = adversary.run_standard_evaluation(
    x_test, y_test, bs = 128, return_labels = True)
    accuracy = (y_test != y_adv).sum()
    clean_acc = clean_acc * 100
    robust_accuracy = (1 - accuracy/n_samples) * 100.0
    idx = (y_test == y_adv)
    x_test = x_test[idx]
    y_test = y_test[idx]

    if x_test.shape[0] == 0 :
        return clean_acc, 0.0

    
    adversary = AutoAttack(model, norm='Linf', eps=4/255, version='custom', attacks_to_run=['apgd-t'])
    adversary.apgd.n_restarts = 1
    adversary.apgd.n_iter = 20
    adversary.apgd.topk = 3
    x_adv, y_adv, _ = adversary.run_standard_evaluation(
    x_test, y_test, bs = 128, return_labels = True)

    accuracy += (y_test != y_adv).sum()
    robust_accuracy = (1 - accuracy/n_samples) * 100.0

    return clean_acc, robust_accuracy
    
def evaluate_clean(model,loader):
    model.eval()
    clean_acc = 0
    total_samples = 0

    for i,(x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.to(device), y.to(device)
        z_clean = model(x)
        z_clean_out = z_clean.argmax(dim = 1)
        clean_acc += (z_clean_out ==y).sum().detach()

    clean_acc = clean_acc/total_samples  * 100.0

    return clean_acc

def evaluate_final_aa(model, loader, args,  n_samples = 10000):
    adversary = AutoAttack(model, norm='Linf', eps=8/255.0, version='standard')
    l = [x for (x, y) in loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in loader]
    y_test = torch.cat(l, 0)    
    adv_complete, robust_acc = adversary.run_standard_evaluation(x_test[:], y_test[:], bs=args.batch_size)
    return robust_acc * 100

def PGD_L1_Attack(model,x,y,eps = 2000/255 ):
    # # if delta_init is not None:
    # #     delta = delta_init
    # # else:
    # #     delta = torch.zeros_like(xvar)
    # bs = x.shape[0]
    # adversary = AutoAttack(model, norm='L1', eps=eps,)
    # adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    # #adversary.apgd.n_restarts = 1
    # with torch.no_grad():
    #     x_adv,robust_acc = adversary.run_standard_evaluation(x, y, bs=bs)
    # return x_adv, robust_acc  * bs
    from Externals.advertorch.attacks import L1PGDAttack
    adversary = L1PGDAttack( model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                        eps=12.0 , nb_iter=20, eps_iter=0.5, rand_init=True, clip_min=0.0,
                        clip_max=1.0, targeted=False)

    x_adv = adversary.perturb(x, y)
    return x_adv



def evaluate_pgd_l2(model,loader):
    model.eval()
    clean_acc = 0
    fgsm_acc = aa_acc = pgd_acc =  0
    total_samples = 0
    pgd_attack = torchattacks.PGDL2(model, eps=128/255, alpha=0.1, steps=10, random_start=True)

    for i,(x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.cuda(), y.cuda()
        x_pgd = pgd_attack(x,y)
        z_pgd = model(x_pgd)

        z_pgd_out = z_pgd.argmax(dim = 1)


        pgd_acc += (z_pgd_out == y).sum()


    
    pgd_acc = pgd_acc/total_samples * 100.0

    return  pgd_acc


def evaluate_pgd_NN(model, loader, class_mean_set):


    clean_acc_NN = 0
    pgd_acc_NN = 0
    clean_acc = 0
    pgd_acc = 0

    model.eval()
    total_samples = 0 
    class_mean_set=torch.tensor(np.array(class_mean_set)).cuda()

    pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)


    for i, (index,x,y) in enumerate(loader): 
        total_samples += x.shape[0]
        x,y = x.cuda(), y.cuda()
        x_pgd = pgd_attack(x,y)
        z_cleans = F.normalize(model.feature_extractor(x))
        z_advs = F.normalize(model.feature_extractor(x_pgd))
       

        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        # class_mean_set = np.array(class_mean_set)
        clean_result = []
        adv_result = []
        for z, z_adv in zip(z_cleans, z_advs):
            
            dist = z.reshape([1,-1]) - class_mean_set
            dist = torch.linalg.norm(dist, ord=2, axis=1)
            out = torch.argmin(dist)
            clean_result.append(out)

            dist = z_adv.reshape([1,-1]) - class_mean_set
            dist = torch.linalg.norm(dist, ord=2, axis=1)
            out = torch.argmin(dist)
            adv_result.append(out)


        clean_acc_NN += (torch.tensor(clean_result) == y.detach().cpu()).sum()
        pgd_acc_NN += (torch.tensor(adv_result) == y.detach().cpu()).sum()


        z = model(x)
        z_pgd = model(x_pgd)

        z_out = z.argmax(dim = 1)
        z_pgd_out = z_pgd.argmax(dim = 1)

        clean_acc += (z_out == y).sum()
        pgd_acc += (z_pgd_out == y).sum()

    clean_acc = clean_acc/total_samples * 100.0
    pgd_acc = pgd_acc/total_samples * 100.0
    clean_acc_NN = clean_acc_NN/total_samples * 100.0
    pgd_acc_NN = pgd_acc_NN/total_samples * 100.0

    return  clean_acc, pgd_acc, clean_acc_NN, pgd_acc_NN
 

def evaluate_pgd(model,loader):
    model.eval()
    clean_acc = 0
    fgsm_acc = aa_acc = pgd_acc = pgd_l1_acc = pgd_l2_acc =  0
    total_samples = 0
    pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)

    for i,(indexs,x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.cuda(), y.cuda()
        x_pgd = pgd_attack(x,y)

        z_pgd = model(x_pgd)

        z = model(x)

        z_pgd_out = z_pgd.argmax(dim = 1)

        z_out = z.argmax(dim = 1)

        pgd_acc += (z_pgd_out == y).sum()

        clean_acc += (z_out == y).sum()

    print ("total_samples : %d"%total_samples)
    clean_acc = clean_acc/total_samples * 100.0
    pgd_acc = pgd_acc/total_samples * 100.0

    return  clean_acc, pgd_acc
 
def write_csv(config, performances):
    path = Path(os.path.realpath(__file__))
    path = str(path.parent.absolute())
    best_clean_acc, best_aa_acc, last_clean_acc, last_aa_acc = performances
    data = {'alpha':config.alpha, 'beta': config.beta, 'gamma' : config.gamma, 'eta' : config.eta,'max_clean':best_clean_acc , 'max_aa':best_aa_acc, 'last_clean':last_clean_acc, 'last_aa':last_aa_acc}
    data_df = pd.DataFrame(data, index = [0])
    output_path = os.path.join(path, config.dataset) + '/csv_files/'
    os.makedirs(output_path, exist_ok=True) 
    output_path = output_path + config.method + ".csv"
    if os.path.isfile(output_path):
        data_df.to_csv(output_path, encoding='utf-8', index = False, mode = 'a', header = False)
    else :
        data_df.to_csv(output_path, encoding='utf-8', index = False)

