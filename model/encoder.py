import torch
import torch.nn as nn
from torch.autograd import Variable

from model.visual_syntactic_embedding import VisualEncoding

class SCNEncoder(nn.Module):
    def __init__(self, cnn_feature_size, c3d_feature_size, i3d_feature_size, eco_feature_size, 
                 res_eco_features_size, n_tags, hidden_size, global_tagger_hidden_size, 
                 specific_tagger_hidden_size, vis_syn_embedd_space_size=512, pretrained_vis_syn_embedd_path='',
                 input_dropout_p=0.2, rnn_dropout_p=0.5, n_layers=1, bidirectional=False, rnn_cell='gru', device='gpu'):
        super(SCNEncoder, self).__init__()
        
        self.cnn_feature_size = cnn_feature_size
        self.c3d_feature_size = c3d_feature_size
        self.i3d_feature_size = i3d_feature_size
        self.eco_feature_size = eco_feature_size
        self.n_tags = n_tags
        self.hidden_size = hidden_size
        self.global_tagger_hidden_size = global_tagger_hidden_size
        self.specific_tagger_hidden_size = specific_tagger_hidden_size
        self.device = device
        
        # pretrained_model_path = './video_tagging/models/syntactic/h512_tagsNone/MSR-VTT syntactic-tagger bow_tag_ranking None 1e-05/global_tagger_chkpt_39.pkl'
        # common_space_size = 512    
        self.pos_model = VisualEncoding(in_size=res_eco_features_size, hidden_sizes=[2048, 1024], 
                                        out_size=vis_syn_embedd_space_size, visual_norm=True, dropout_p=.5,
                                        have_last_bn=True, pretrained_model_path=pretrained_vis_syn_embedd_path)
        
    def forward_fn(self, v_feats, cnn_globals, v_globals, s_globals):
        batch_size, seq_len, feats_size = v_feats.size()

        h = Variable(torch.zeros(2*2, batch_size, self.hidden_size)).to(self.device)
        c = Variable(torch.zeros(2*2, batch_size, self.hidden_size)).to(self.device)
        
        pos_embs = self.pos_model(v_globals)
        
        v_globals = torch.cat((v_globals, cnn_globals), dim=1)
        
        return v_feats, (h,c), s_globals, v_globals, pos_embs  #pool
        
    def forward(self, cnn_feats, c3d_feats, cnn_globals, cnn_sem_globals, tags_globals, res_eco_globals):
        batch_size = cnn_feats.size(0)

        # (batch_size x max_frames x feature_size) -> (batch_size*max_frames x feature_size)
        cnn_feats = cnn_feats.view(-1, self.cnn_feature_size)
        c3d_feats = c3d_feats.view(-1, self.c3d_feature_size)

        # (batch_size*max_frames x cnn_feature_size+c3d_feature_size+i3d_feature_size)
        v_concat = torch.cat((cnn_feats, c3d_feats), dim=1)

        # (batch_size*max_frames x cnn_feature_size+c3d_feature_size+i3d_feature_size) -> (batch_size x max_frames x cnn_feature_size+c3d_feature_size+i3d_feature_size)
        v_concat = v_concat.view(batch_size, -1, self.cnn_feature_size + self.c3d_feature_size)

        s_globals = torch.cat((tags_globals, torch.softmax(cnn_sem_globals, dim=1)), dim=1)

        return self.forward_fn(v_concat, cnn_globals, res_eco_globals, s_globals)