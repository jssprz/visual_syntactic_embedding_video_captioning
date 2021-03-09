import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, dropout_p=0.5, have_last_bn=False, 
                 pretrained_model_path=''):
        super(MLP, self).__init__()

        self.in_drop = nn.Dropout(dropout_p)
        
        self.fc1 = nn.Linear(in_size, hidden_sizes[0])
        self.fc1_drop = nn.Dropout(dropout_p)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2_drop = nn.Dropout(dropout_p)
        
        self.fc3 = nn.Linear(hidden_sizes[1], out_size)
        
        self.have_last_bn = have_last_bn
        if have_last_bn:
            self.bn = nn.BatchNorm1d(out_size)
        
        self.__init_layers()

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        h = self.in_drop(x)
        h = self.fc1_drop(torch.relu(self.fc1(h)))
        h = self.fc2_drop(torch.relu(self.fc2(h)))
        h = self.fc3(h)
#         h = torch.relu(self.fc3(h))
#         return torch.sigmoid(self.fc3(h))
        
        if self.have_last_bn:
            h = self.bn(h)
     
        return h
    
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class VisualEncoding(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, visual_norm=True, dropout_p=.5, 
                 have_last_bn=False, pretrained_model_path=''):
        super(VisualEncoding, self).__init__()
        self.visual_norm = visual_norm

        self.visual_mapping = MLP(in_size=in_size, hidden_sizes=hidden_sizes, out_size=out_size, dropout_p=dropout_p, have_last_bn=have_last_bn)
        
        if pretrained_model_path != '':
            checkpoint = torch.load(pretrained_model_path)
            self.load_state_dict(checkpoint['visual_model'])
    
    def forward(self, v_feats):
        features = self.visual_mapping(v_feats)
        
        if self.visual_norm:
            features = l2norm(features)
        
        return features