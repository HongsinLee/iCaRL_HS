import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import pdb, os, copy
import torchattacks
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class iCaRLmodel:

    def __init__(self,numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate, robust_epochs, config):

        super(iCaRLmodel, self).__init__()
        self.epochs=epochs
        self.robust_epochs = robust_epochs 
        self.learning_rate=learning_rate
        self.model = network(numclass,feature_extractor)
        self.exemplar_set = []
        self.herding_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.config = config
        self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             transforms.ToTensor()
                                            # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                            ])
        self.old_model = None

        self.train_transform = transforms.Compose([#transforms.Resize(img_size),
                                                  transforms.RandomCrop((32,32),padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor()
                                                #   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                                  ])
        
        self.test_transform = transforms.Compose([#transforms.Resize(img_size),
                                                   transforms.ToTensor()
                                                #  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                                 ])
        
        self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                    #transforms.Resize(img_size),
                                                    transforms.ToTensor()
                                                #    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                                   ])
        
        self.train_dataset = iCIFAR100('dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size=memory_size
        self.task_size=task_size

        self.train_loader=None
        self.test_loader=None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes = [self.numclass - self.task_size, self.numclass]
    
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
  
        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader
    
    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model

    def train_adv(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass==self.task_size:
                     print(1)
                     opt = optim.SGD(self.model.parameters(), lr=1.0/5, weight_decay=0.00001)
                else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 5
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass>self.task_size:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 25
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                     opt = optim.SGD(self.model.parameters(), lr=1.0/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                  if self.numclass==self.task_size:
                     opt = optim.SGD(self.model.parameters(), lr=1.0 / 125,weight_decay=0.00001)
                  else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 125
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                  print("change learning rate:%.3f" % (self.learning_rate / 100))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                #output = self.model(images)
                loss_value = self._compute_loss_adv(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            # accuracy = self._test(self.test_loader, 1)
            # print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        return 

    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass==self.task_size:
                     print(1)
                     opt = optim.SGD(self.model.parameters(), lr=1.0/5, weight_decay=0.00001)
                else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 5
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass>self.task_size:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 25
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                     opt = optim.SGD(self.model.parameters(), lr=1.0/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                  if self.numclass==self.task_size:
                     opt = optim.SGD(self.model.parameters(), lr=1.0 / 125,weight_decay=0.00001)
                  else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 125
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                  print("change learning rate:%.3f" % (self.learning_rate / 100))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                #output = self.model(images)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            # accuracy = self._test(self.test_loader, 1)
            # print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        return 

    def _test(self, testloader, mode):
        if mode==0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss_adv(self, indexs, imgs, target):
        pgd_attack = torchattacks.PGD(self.model, eps=8/255, alpha=2/225, steps=10, random_start=True)
        imgs_adv = pgd_attack(imgs,target)
        output = self.model(imgs_adv)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
           
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target

            return F.binary_cross_entropy_with_logits(output, target)



    def _compute_loss(self, indexs, imgs, target):
        output=self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
           
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target

            return F.binary_cross_entropy_with_logits(output, target)


    # change the size of examplar
    def afterTrain(self, exemplar_way, ratio, adv_batch_set):
        self.model.eval()
        m=int(self.memory_size/self.numclass)
        print (self.numclass, "m :", m)
        self._reduce_exemplar_sets(m)
        is_fully_compute_class_mean = 1

        exemplar_method_function = "_construct_exemplar_set_" + exemplar_way

        if exemplar_way == "herding_adv" or exemplar_way == "herding_adv_ratio" or exemplar_way == "random_and_herding" :
            is_fully_compute_class_mean = 0


        for i in range(self.numclass-self.task_size,self.numclass):
            images=self.train_dataset.get_image_class(i)
            images_adv = np.array([]).reshape(0, 3, 32, 32)

            for adv_batch, adv_label_batch in adv_batch_set:
                images_adv = np.concatenate([images_adv, adv_batch[adv_label_batch==i]], axis=0)
            images_adv = np.reshape(np.rint(images_adv * 255).astype(np.uint8), (-1, 32, 32, 3)) 


            print(exemplar_way+': construct class %s examplar:'%(i),end='')
            getattr(iCaRLmodel, exemplar_method_function)(self, images, m, ratio, images_adv)

            """
            if exemplar_way == "random":
                print('random : construct class %s examplar:'%(i),end='')
                self._construct_exemplar_set_random(images,m)
            elif exemplar_way == "reverse_herding":
                print('reverse_herding : construct class %s examplar:'%(i),end='')
                self._construct_exemplar_set_reverse_herding(images,m)
            elif exemplar_way == "random_and_herding":
                print('random_and_herding : construct class %s examplar:'%(i),end='')
                self._construct_exemplar_set_random_and_herding(images,m, ratio)
            elif exemplar_way == "entropy":
                print('entropy : construct class %s examplar:'%(i),end='')
                self._construct_exemplar_set_entropy(images,m)
            elif exemplar_way == "herding_adv":
                print('herding_adv : construct class %s examplar:'%(i),end='')
                self._construct_exemplar_set_herding_adv(images_adv,m)
            elif exemplar_way == "herding_adv_ratio":
                print('herding_adv_ratio : construct class %s examplar:'%(i),end='')
                self._construct_exemplar_set_herding_adv_ratio(images, images_adv,m, ratio)
            else:
                print('original : construct class %s examplar:'%(i),end='')
                self._construct_exemplar_set_original(images,m)
            """

        self.numclass+=self.task_size

        if is_fully_compute_class_mean == 1 :
            self.compute_exemplar_class_mean()
        else : 
            self.compute_exemplar_class_mean_adv()
        self.model.train()

        # delete by seungju
        # KNN_accuracy=self._test(self.test_loader, mode = 0)
        # print("NMS accuracy：" + str(KNN_accuracy.item()))
        
        path = str(Path(os.path.realpath(__file__)).parent.absolute())

        # modified by seungju
        # filename= path + '/model/accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10)
        filename= path + '/model/increment:%d_net.pkl' % (i + 1)
        torch.save(self.model,filename)
        self.old_model=torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()
        


    def _construct_exemplar_set_original(self, images, m, *args):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)

    def _construct_exemplar_set_entropy(self, images, m, *args):
        x = self.Image_transform(images, self.transform).to(device)
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        entropy = np.sum(feature_extractor_output * np.log(feature_extractor_output), axis = 1)
        exemplar = []

        top_m_list = np.argsort(-entropy)[:m]

        for i in top_m_list:
            exemplar.append(images[i])

        self.exemplar_set.append(exemplar)

    def _construct_exemplar_set_reverse_herding(self, images, m, *args):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))

        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmax(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)


    def _construct_exemplar_set_random(self, images, m, *args):
        exemplar = []
        random_indices = np.random.choice(len(images),m, replace = False)
        for index in random_indices:
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)


    def _construct_exemplar_set_random_and_herding(self, images, m, ratio, *args):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        temp_herding = []
        temp_random = []
        now_class_mean = np.zeros((1, 512))
        num_herding = int(m * ratio)
        num_random = m - num_herding

        for i in range(num_herding):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            temp_herding.append(images[index])

        random_indices = np.random.choice(len(images), num_random, replace = False)
        for index in random_indices:
            temp_random.append(images[index])

        for i in range(min(num_herding, num_random)):
            exemplar.append(temp_herding[i])
            exemplar.append(temp_random[i])            
        
        if num_herding >= num_random :
            for i in range(num_random, num_herding):
                exemplar.append(temp_herding[i])
        else :
            for i in range(num_herding, num_random):
                exemplar.append(temp_random[i])
                
        self.herding_set.append(temp_herding)
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)


    def _construct_exemplar_set_herding_adv(self, images, m, ratio, images_adv, *args):
        class_mean, feature_extractor_output = self.compute_class_mean(images_adv, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images_adv[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)

    def _construct_exemplar_set_herding_adv_ratio(self, images, m, ratio, images_adv, *args):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        class_mean_adv, feature_extractor_output_adv = self.compute_class_mean(images_adv, self.transform)
        exemplar = []
        temp_herding = []
        temp_adv = []
        num_adv = int(m * ratio)
        num_herding = m - num_adv
        now_class_mean = np.zeros((1, 512))
        now_class_mean_adv = np.zeros((1, 512))

        for i in range(num_herding):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            temp_herding.append(images[index])

        for i in range(num_adv):
            # shape：batch_size*512
            x = class_mean_adv - (now_class_mean_adv + feature_extractor_output_adv) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean_adv += feature_extractor_output_adv[index]
            temp_adv.append(images_adv[index])

        for i in range(min(num_herding, num_adv)):
            exemplar.append(temp_herding[i])
            exemplar.append(temp_adv[i])            
        
        if num_herding >= num_adv :
            for i in range(num_adv, num_herding):
                exemplar.append(temp_herding[i])
        else :
            for i in range(num_herding, num_adv):
                exemplar.append(temp_adv[i])

        self.herding_set.append(temp_herding)
        self.exemplar_set.append(exemplar)






    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar=self.exemplar_set[index]
            #exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def compute_exemplar_class_mean_adv(self):
        self.class_mean_set = []
        for index in range(len(self.herding_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar=self.herding_set[index]
            #exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)

    def robust_finetuning(self):
        # Todo
        origin_model = copy.deepcopy(self.model)
        origin_model.eval()
        self.model.train()
        lr = 0.001
        optimizer = optim.Adam(self.model.parameters(), lr = lr)
        pgd_attack = torchattacks.PGD(self.model, eps=8/255, alpha=2/225, steps=10, random_start=True)
        criterion_kl = nn.KLDivLoss(size_average=False)
        XENT_loss = nn.CrossEntropyLoss()

        for epoch in range(self.robust_epochs):
            for step, (indexs, x, y) in enumerate(self.train_loader):
                x, y = x.to(device), y.to(device)
                N = x.shape[0]
                x_adv = pgd_attack(x,y)
                #output = self.model(images)
                adv_out = self.model(x_adv)

                feat_clean = origin_model.feature_extractor(x)
                feat_adv = origin_model.feature_extractor(x_adv)


                score = torch.exp(-self.config.beta * ((feat_clean - feat_adv)**2)) 
                score = score.reshape(N,-1)
                teacher_out = origin_model.forward_with_score(feat_clean, score)
     

                kl_loss = criterion_kl(F.log_softmax(adv_out, dim=1), F.softmax(teacher_out.detach(), dim=1))
                loss = XENT_loss(adv_out,y) + self.config.alpha * (1.0 / N) *  kl_loss 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
               
            self.adjust_learning_rate(lr, optimizer, epoch)
        return

        

    def adjust_learning_rate(self, lr, optimizer, epoch):
        criteria = self.robust_epochs // 2
        max_lr = lr
        if epoch + 1 <= criteria :
            lr = max_lr/criteria * (epoch + 1)
        else  :
            # lr = max_lr/criteria * (10 - epoch)
            lr = (1/(2**(epoch + 1 - criteria)))*max_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr