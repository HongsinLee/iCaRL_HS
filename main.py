from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import torch
import neptune.new as neptune
from pathlib import Path
from utils import *
import argparse, os
import methods
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))



def main(args, npt):
    numclass=10
    feature_extractor=resnet18_cbam()
    img_size=32
    batch_size=128
    task_size=10
    memory_size=2000
    epochs= args.epochs
    robust_epochs = args.robust_epochs
    learning_rate = 2.0

    iCaRL_model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate, robust_epochs,args)

    for task_idx in range(10):
        iCaRL_model.beforeTrain() # => prepare training sets
        iCaRL_model.train()
        adv_batch_set = getattr(methods, args.method)(iCaRL_model.model, robust_epochs, iCaRL_model.train_loader, args)
        iCaRL_model.afterTrain(args.exemplar, args.exemplar_ratio, adv_batch_set) # => make examplar sets
        clean_acc, pgd_acc, clean_acc_NN, pgd_acc_NN = evaluate_pgd_NN(iCaRL_model.model, iCaRL_model.test_loader, iCaRL_model.class_mean_set)

        if args.npt ==  1 or args.npt == 2:
            npt["test/clean_acc"].log(clean_acc)
            npt["test/pgd_acc"].log(pgd_acc)
            npt["test/clean_acc_NN"].log(clean_acc_NN)
            npt["test/pgd_acc_NN"].log(pgd_acc_NN)


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="cifar100_RFD.yaml")
    parser.add_argument("--epochs", default=100, type = int)
    parser.add_argument("--robust_epochs", default = 10, type = int)
    parser.add_argument("--robust_steps", default = 10, type = int)
    parser.add_argument("--lr", default=-1, type = float)
    parser.add_argument("--tags", default = " ",type = str)
    parser.add_argument("--npt", default = 0, type = int)
    parser.add_argument("--alpha", default = 0, type = float)
    parser.add_argument("--beta", default = 0, type = float)
    parser.add_argument("--method", default = "robust_finetune", type = str)
    parser.add_argument("--exemplar", default = "original", type = str)
    parser.add_argument("--exemplar_ratio", default = -1, type = float)
    return parser
 
if __name__ == "__main__":
    args = argument_parsing().parse_args()
    path = str(Path(os.path.realpath(__file__)).parent.absolute())

    if args.npt == 1:
        npt = {}
    elif args.npt == 2:
        npt = neptune.init(
        project="hongsin/Robust-iCaRL",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzBlMGI2ZS02NjM2LTRiMTMtYWU5NC00MjU1NzIwYWRlODgifQ==",
        tags = [args.tags, "e_ratio : " + str(args.exemplar_ratio), "r_steps : " + str(args.robust_steps), "r_epochs : " + str(args.robust_epochs), args.method, args.exemplar, "alpha : " + str(args.alpha), "beta : " +  str(args.beta)],
        source_files = [path+'/main.py',path+'/iCaRL.py']
        )
    else :
        npt = {}

    main(args, npt)