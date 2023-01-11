import torch.nn as nn
import torch

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.mean = torch.tensor((0.5071, 0.4867, 0.4408)).cuda().view(-1, 1, 1)
        self.std = torch.tensor((0.2675, 0.2565, 0.2761)).cuda().view(-1, 1, 1)

    def normalize(self, x):
       return (x - self.mean) / self.std

    def forward(self, x):
        N = x.shape[0]
        x = self.normalize(x) 
        x = self.feature(x) # N,C
        
        x = self.fc(x)
        return x
    
    def forward_with_score(self, feat, score ):
        out = self.fc(feat * score) 
        return out

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,x):
        x = self.normalize(x)
        return self.feature(x)
