import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from utils import get_init_weights
from model.attention import Attention


class VNCLCell(nn.Module):
    def __init__(self, in_seq_length, out_seq_length, n_feats, n_tags, embedding_size, hidden_size, rnn_in_size, rnn_hidden_size, vocab, encoder_num_layers, encoder_bidirectional, 
                 pretrained_embedding=None, rnn_cell='gru', num_layers=1, dropout_p=0.5, beam_size=10, temperature=1.0, train_sample_max=False, test_sample_max=True, beam_search_logic='bfs', have_bn=False, var_dropout='per-gate'):
        super(VNCLCell, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.vocab = vocab
        self.beam_size = beam_size
        self.temperature = temperature
        self.train_sample_max = train_sample_max
        self.test_sample_max = test_sample_max
        self.beam_search_logic = beam_search_logic
        self.dropout_p = dropout_p
        self.var_dropout = var_dropout

        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.num_directions = 1  # beause this decoder is not bidirectional

        # Components
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)    
        else:
            self.embedding = nn.Embedding(self.output_size, embedding_size)
        
        self.Wa_i = get_init_weights((rnn_in_size, rnn_hidden_size))
        self.Wa_f = get_init_weights((rnn_in_size, rnn_hidden_size))
        self.Wa_o = get_init_weights((rnn_in_size, rnn_hidden_size))
        self.Wa_c = get_init_weights((rnn_in_size, rnn_hidden_size))

        self.Wb_i = get_init_weights((n_tags, rnn_hidden_size))
        self.Wb_f = get_init_weights((n_tags, rnn_hidden_size))
        self.Wb_o = get_init_weights((n_tags, rnn_hidden_size))
        self.Wb_c = get_init_weights((n_tags, rnn_hidden_size))
        
        self.Wc_i = get_init_weights((rnn_hidden_size, hidden_size))
        self.Wc_f = get_init_weights((rnn_hidden_size, hidden_size))
        self.Wc_o = get_init_weights((rnn_hidden_size, hidden_size))
        self.Wc_c = get_init_weights((rnn_hidden_size, hidden_size))

        self.Ua_i = get_init_weights((hidden_size, rnn_hidden_size))
        self.Ua_f = get_init_weights((hidden_size, rnn_hidden_size))
        self.Ua_o = get_init_weights((hidden_size, rnn_hidden_size))
        self.Ua_c = get_init_weights((hidden_size, rnn_hidden_size))

        self.Ub_i = get_init_weights((n_tags, rnn_hidden_size))
        self.Ub_f = get_init_weights((n_tags, rnn_hidden_size))
        self.Ub_o = get_init_weights((n_tags, rnn_hidden_size))
        self.Ub_c = get_init_weights((n_tags, rnn_hidden_size))
        
        self.Uc_i = get_init_weights((rnn_hidden_size, hidden_size))
        self.Uc_f = get_init_weights((rnn_hidden_size, hidden_size))
        self.Uc_o = get_init_weights((rnn_hidden_size, hidden_size))
        self.Uc_c = get_init_weights((rnn_hidden_size, hidden_size))

        self.Ca_i = get_init_weights((n_feats, rnn_hidden_size))
        self.Ca_f = get_init_weights((n_feats, rnn_hidden_size))
        self.Ca_o = get_init_weights((n_feats, rnn_hidden_size))
        self.Ca_c = get_init_weights((n_feats, rnn_hidden_size))

        self.Cb_i = get_init_weights((n_tags, rnn_hidden_size))
        self.Cb_f = get_init_weights((n_tags, rnn_hidden_size))
        self.Cb_o = get_init_weights((n_tags, rnn_hidden_size))
        self.Cb_c = get_init_weights((n_tags, rnn_hidden_size))

        self.Cc_i = get_init_weights((rnn_hidden_size, hidden_size))
        self.Cc_f = get_init_weights((rnn_hidden_size, hidden_size))
        self.Cc_o = get_init_weights((rnn_hidden_size, hidden_size))
        self.Cc_c = get_init_weights((rnn_hidden_size, hidden_size))
        
        self.b_i = Parameter(torch.zeros(hidden_size))
        self.b_f = Parameter(torch.zeros(hidden_size))        
        self.b_o = Parameter(torch.zeros(hidden_size))        
        self.b_c = Parameter(torch.zeros(hidden_size))
        
        self.out = nn.Linear(self.hidden_size * self.num_directions, self.output_size)
        
        self.have_bn = have_bn
        if have_bn:
            self.bn = nn.LayerNorm(hidden_size)
        
        self.__init_layers()
        
    @property
    def name(self):
        return 'scn-{}-drop{}-bs{}-{}-{}-{}'.format(self.num_layers, self.dropout_p, self.beam_size, self.beam_search_logic, 
                                                    'train-max' if self.train_sample_max else 'train-dist',
                                                    'test-max' if self.test_sample_max else 'test-dist')

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def __dropout(self, x, keep_prob, mask_for):
        # if not self.training or keep_prob >= 1.:
        #     return x
        
        # if mask_for in self.dropM:
        #     mask = self.dropM[mask_for]
        # else:
        #     # op1
        #     # mask = Binomial(probs=keep_prob).sample(x.size()).to(x.device)  # máscara de acuerdo a keep_prob

        #     # op2
        #     mask = x.new_empty(x.size(), requires_grad=False).bernoulli_(keep_prob)
            
        #     self.dropM[mask_for] = mask
            
        # assert x.device == mask.device, 'mask and x must be in the same device'
        
        # return x.masked_fill(mask==0, 0) * (1.0 / keep_prob)
        return x
        
    def precompute_mats(self, v, s, variational_dropout_p):
        self.dropM = {}
                
        keep_prob = 1 - variational_dropout_p
        if self.var_dropout == 'per-gate':
            # use a distinct mask for each gate
            s_i = self.__dropout(s, keep_prob, 's_i')
            s_f = self.__dropout(s, keep_prob, 's_f')
            s_o = self.__dropout(s, keep_prob, 's_o')
            s_c = self.__dropout(s, keep_prob, 's_c')

            v_i = self.__dropout(v, keep_prob, 'v_i')
            v_f = self.__dropout(v, keep_prob, 'v_f')
            v_o = self.__dropout(v, keep_prob, 'v_o')
            v_c = self.__dropout(v, keep_prob, 'v_c')
        else:
            # use the same mask for all gates
            s_i = self.__dropout(s, keep_prob, 's')
            s_f = self.__dropout(s, keep_prob, 's')
            s_o = self.__dropout(s, keep_prob, 's')
            s_c = self.__dropout(s, keep_prob, 's')

            v_i = self.__dropout(v, keep_prob, 'v')
            v_f = self.__dropout(v, keep_prob, 'v')
            v_o = self.__dropout(v, keep_prob, 'v')
            v_c = self.__dropout(v, keep_prob, 'v')
        
        # (batch_size x rnn_hidden_size)
        self.temp2_i = s_i @ self.Wb_i
        self.temp2_f = s_f @ self.Wb_f
        self.temp2_o = s_o @ self.Wb_o
        self.temp2_c = s_c @ self.Wb_c
        
        # (batch_size x rnn_hidden_size)
        self.temp3_i = v_i @ self.Ca_i
        self.temp3_f = v_f @ self.Ca_f
        self.temp3_o = v_o @ self.Ca_o
        self.temp3_c = v_c @ self.Ca_c
        
        # (batch_size x rnn_hidden_size)
        self.temp4_i = s_i @ self.Cb_i
        self.temp4_f = s_f @ self.Cb_f
        self.temp4_o = s_o @ self.Cb_o
        self.temp4_c = s_c @ self.Cb_c
                                
    def __compute_gate(self, activation, temp1, temp2, temp3, temp4, temp5, temp6, Wc, Cc, Uc, b):
        x = (temp1 * temp2) @ Wc
        v = (temp3 * temp4) @ Cc
        h = (temp5 * temp6) @ Uc
        
        logits = x + v + h + b
        
        if self.have_bn:
            logits = self.bn(logits)
        
        return activation(logits)
    
    def forward(self, s, rnn_h, rnn_c, decoder_input, encoder_hidden, encoder_outputs, variational_dropout_p):
        keep_prob = 1 - variational_dropout_p
        if self.var_dropout == 'per-gate':
            # use a distinct mask for each gate
            s_i = self.__dropout(s, keep_prob, 's_i')
            s_f = self.__dropout(s, keep_prob, 's_f')
            s_o = self.__dropout(s, keep_prob, 's_o')
            s_c = self.__dropout(s, keep_prob, 's_c')

            x_i = self.__dropout(decoder_input, keep_prob, 'x_i')
            x_f = self.__dropout(decoder_input, keep_prob, 'x_f')
            x_o = self.__dropout(decoder_input, keep_prob, 'x_o')
            x_c = self.__dropout(decoder_input, keep_prob, 'x_c')

            h_i = self.__dropout(rnn_h, keep_prob, 'h_i')
            h_f = self.__dropout(rnn_h, keep_prob, 'h_f')
            h_o = self.__dropout(rnn_h, keep_prob, 'h_o')
            h_c = self.__dropout(rnn_h, keep_prob, 'h_c')
        else:
            # use the same mask for all gates
            s_i = self.__dropout(s, keep_prob, 's')
            s_f = self.__dropout(s, keep_prob, 's')
            s_o = self.__dropout(s, keep_prob, 's')
            s_c = self.__dropout(s, keep_prob, 's')

            x_i = self.__dropout(decoder_input, keep_prob, 'x')
            x_f = self.__dropout(decoder_input, keep_prob, 'x')
            x_o = self.__dropout(decoder_input, keep_prob, 'x')
            x_c = self.__dropout(decoder_input, keep_prob, 'x')

            h_i = self.__dropout(rnn_h, keep_prob, 'h')
            h_f = self.__dropout(rnn_h, keep_prob, 'h')
            h_o = self.__dropout(rnn_h, keep_prob, 'h')
            h_c = self.__dropout(rnn_h, keep_prob, 'h')

        # (batch_size x rnn_hidden_size)
        temp1_i = x_i @ self.Wa_i
        temp1_f = x_f @ self.Wa_f
        temp1_o = x_o @ self.Wa_o
        temp1_c = x_c @ self.Wa_c

        # (batch_size x rnn_hidden_size)
        temp5_i = s_i @ self.Ub_i
        temp5_f = s_f @ self.Ub_f
        temp5_o = s_o @ self.Ub_o
        temp5_c = s_c @ self.Ub_c

        # (batch_size x rnn_hidden_size)
        temp6_i = h_i @ self.Ua_i
        temp6_f = h_f @ self.Ua_f
        temp6_o = h_o @ self.Ua_o
        temp6_c = h_c @ self.Ua_c

        # (batch_size x hidden_size)
        i = self.__compute_gate(torch.sigmoid, temp1_i, self.temp2_i, self.temp3_i, self.temp4_i, temp5_i, temp6_i, self.Wc_i, self.Cc_i, self.Uc_i, self.b_i)
        f = self.__compute_gate(torch.sigmoid, temp1_f, self.temp2_f, self.temp3_f, self.temp4_f, temp5_f, temp6_f, self.Wc_f, self.Cc_f, self.Uc_f, self.b_f)
        o = self.__compute_gate(torch.sigmoid, temp1_o, self.temp2_o, self.temp3_o, self.temp4_o, temp5_o, temp6_o, self.Wc_o, self.Cc_o, self.Uc_o, self.b_o)
        c = self.__compute_gate(torch.tanh, temp1_c, self.temp2_c, self.temp3_c, self.temp4_c, temp5_c, temp6_c, self.Wc_c, self.Cc_c, self.Uc_c, self.b_c)

        # (batch_size x hidden_size)
        rnn_c = f * rnn_c + i * c
        rnn_h = o * torch.tanh(rnn_c)
        
        return rnn_h, rnn_c 


class SemSynANDecoder(nn.Module):
    def __init__(self, in_seq_length, out_seq_length, n_feats, n_tags, n_pos_emb, 
                 embedding_size, hidden_size, rnn_in_size, rnn_hidden_size, vocab, device, 
                 encoder_num_layers, encoder_bidirectional, pretrained_embedding=None, 
                 rnn_cell='gru', num_layers=1, dropout_p=0.5, beam_size=10, temperature=1.0, 
                 train_sample_max=False, test_sample_max=True, beam_search_logic='bfs',
                 dataset_name='MSVD'):
        super(SemSynANDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.vocab = vocab
        self.device = device
        self.beam_size = beam_size
        self.temperature = temperature
        self.train_sample_max = train_sample_max
        self.test_sample_max = test_sample_max
        self.beam_search_logic = beam_search_logic

        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.num_directions = 1  # beause this decoder is not bidirectional
        
        # Components
        
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)    
        else:
            self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.embedd_drop = nn.Dropout(dropout_p)
        
        self.v_sem_layer = VNCLCell(in_seq_length, out_seq_length, n_feats, n_tags, embedding_size, hidden_size,
                                      rnn_in_size, rnn_hidden_size, vocab, encoder_num_layers, 
                                      encoder_bidirectional, pretrained_embedding, rnn_cell, num_layers, dropout_p, 
                                      beam_size, temperature, train_sample_max, test_sample_max, beam_search_logic,
                                      have_bn=False)
        
        self.v_syn_layer = VNCLCell(in_seq_length, out_seq_length, n_feats, n_pos_emb, embedding_size, hidden_size,
                                      rnn_in_size, rnn_hidden_size, vocab, encoder_num_layers, 
                                      encoder_bidirectional, pretrained_embedding, rnn_cell, num_layers, dropout_p, 
                                      beam_size, temperature, train_sample_max, test_sample_max, beam_search_logic,
                                      have_bn=False)
        
        self.se_sy_layer = VNCLCell(in_seq_length, out_seq_length, n_tags, n_pos_emb, embedding_size, hidden_size,
                                      rnn_in_size, rnn_hidden_size, vocab, encoder_num_layers, 
                                      encoder_bidirectional, pretrained_embedding, rnn_cell, num_layers, dropout_p, 
                                      beam_size, temperature, train_sample_max, test_sample_max, beam_search_logic,
                                      have_bn=False)
        
        self.merge1 = nn.Linear(self.hidden_size + 6144, self.hidden_size)
        self.merge2 = nn.Linear(self.hidden_size + 6144, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        self.dataset_name = dataset_name
        if dataset_name == 'MSVD':
            self.v_sem_attn = Attention(self.in_seq_length, self.embedding_size, self.hidden_size, self.num_layers, self.num_directions, mode='soft')
            self.v_syn_attn = Attention(self.in_seq_length, self.embedding_size, self.hidden_size, self.num_layers, self.num_directions, mode='soft')
            self.se_sy_attn = Attention(self.in_seq_length, self.embedding_size, self.hidden_size, self.num_layers, self.num_directions, mode='soft')
        elif dataset_name == 'MSR-VTT':
            self.v_attn = Attention(self.in_seq_length, self.embedding_size, self.hidden_size*3, self.num_layers, self.num_directions, mode='soft')
            self.s_attn = Attention(self.in_seq_length, self.embedding_size, self.hidden_size*3, self.num_layers, self.num_directions, mode='soft')

        self.__init_layers()
        
    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
        
    def __adaptive_merge(self, rnn_h, v_attn, v_sem_h, v_syn_h, sem_syn_h):
        h = torch.cat((rnn_h, v_attn), dim=1)
        beta1 = torch.sigmoid(self.merge1(h))
        beta2 = torch.sigmoid(self.merge2(h))
        aa1 = beta1 * v_sem_h + (1 - beta1) * v_syn_h
        return beta2 * aa1 + (1 - beta2) * sem_syn_h
    
    def forward_fn(self, v_pool, s_pool, pos_emb, enc_hidden, v_feats, captions, teacher_forcing_ratio=0.5):
        # Determine whether it is an inferred mode based on whether it is passed into caption
        infer = captions is None
        
        batch_size = v_pool.size(0)

        # (batch_size x embedding_size)
        decoder_input = Variable(torch.Tensor(batch_size, self.embedding_size).fill_(0)).to(self.device)
        
        if type(enc_hidden) is tuple:
            # (encoder_n_layers * encoder_num_directions x batch_size x hidden_size) -> (encoder_n_layers x encoder_num_directions x batch_size x hidden_size) 
            rnn_h = enc_hidden[0].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.hidden_size)
            rnn_c = enc_hidden[1].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.hidden_size)
        
        v_sem_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        v_sem_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        
        v_syn_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        v_syn_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        
        se_sy_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        se_sy_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        
        rnn_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        outputs = []
        
        self.v_sem_layer.precompute_mats(v_pool, s_pool, variational_dropout_p=0.1)
        self.v_syn_layer.precompute_mats(v_pool, pos_emb, variational_dropout_p=0.1)
        self.se_sy_layer.precompute_mats(s_pool, pos_emb, variational_dropout_p=0.1)
        
        words = []
        if infer:
            for step in range(self.out_seq_length):               
                v_sem_h, v_sem_c = self.v_sem_layer(s_pool, v_sem_h, v_sem_c, decoder_input, enc_hidden, v_feats, variational_dropout_p=0.1)
                v_syn_h, v_syn_c = self.v_syn_layer(pos_emb, v_syn_h, v_syn_c, decoder_input, enc_hidden, v_feats, variational_dropout_p=0.1)
                se_sy_h, se_sy_c = self.se_sy_layer(pos_emb, se_sy_h, se_sy_c, decoder_input, enc_hidden, v_feats, variational_dropout_p=0.1)

                if self.dataset_name == 'MSVD':
                    v_attn1 = self.v_sem_attn(v_feats, v_sem_h)
                    v_attn2 = self.v_syn_attn(v_feats, v_syn_h)
                    v_attn3 = self.se_sy_attn(v_feats, se_sy_h)
                    v_attn = (v_attn1 + v_attn2 + v_attn3) / 3
                elif self.dataset_name == 'MSR-VTT':
                    h = torch.cat((v_sem_h,v_syn_h,se_sy_h),dim=1)
                    v_attn = self.v_attn(v_feats, h)
                
                rnn_h = self.__adaptive_merge(rnn_h, v_attn, v_sem_h, v_syn_h, se_sy_h)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                # compute word probs
                if self.test_sample_max:
                    # sample max probailities
                    # (batch_size x 1), (batch_size x 1)
                    word_id = word_logits.max(dim=1)[1]
                else:
                    # sample from distribution
                    # (batch_size x 1)
                    word_id = torch.multinomial(torch.softmax(word_logits, dim=1), 1)
                
                # (batch_size x 1) -> (batch_size x embedding_size)
                decoder_input = self.embedding(word_id).squeeze(1)
                
                outputs.append(word_logits)
                words.append(word_id)
                    
            return torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), torch.cat([w.unsqueeze(1) for w in words], dim=1).contiguous()
        else:
            for seq_pos in range(self.out_seq_length):
                v_sem_h, v_sem_c = self.v_sem_layer(s_pool, v_sem_h, v_sem_c, decoder_input, enc_hidden, v_feats, variational_dropout_p=0.1)
                v_syn_h, v_syn_c = self.v_syn_layer(pos_emb, v_syn_h, v_syn_c, decoder_input, enc_hidden, v_feats, variational_dropout_p=0.1)
                se_sy_h, se_sy_c = self.se_sy_layer(pos_emb, se_sy_h, se_sy_c, decoder_input, enc_hidden, v_feats, variational_dropout_p=0.1)

                if self.dataset_name == 'MSVD':
                    v_attn1 = self.v_sem_attn(v_feats, v_sem_h)
                    v_attn2 = self.v_syn_attn(v_feats, v_syn_h)
                    v_attn3 = self.se_sy_attn(v_feats, se_sy_h)
                    v_attn = (v_attn1 + v_attn2 + v_attn3) / 3
                elif self.dataset_name == 'MSR-VTT':
                    h = torch.cat((v_sem_h,v_syn_h,se_sy_h),dim=1)
                    v_attn = self.v_attn(v_feats, h)

                rnn_h = self.__adaptive_merge(rnn_h, v_attn, v_sem_h, v_syn_h, se_sy_h)
                
                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio or seq_pos == 0 else False
                if use_teacher_forcing:
                    word_id = captions[:, seq_pos]  # use the correct words, (batch_size x 1)
                elif self.train_sample_max:
                    # select the words ids with the max probability,
                    # (batch_size x 1)
                    word_id = word_logits.max(1)[1]
                else:
                    # sample words from probability distribution
                    # (batch_size x 1)
                    word_id = torch.multinomial(torch.softmax(word_logits, dim=1), 1)

                # (batch_size x 1) -> (batch_size x embedding_size)
                decoder_input = self.embedding(word_id).squeeze(1)
                decoder_input = self.embedd_drop(decoder_input)
                
                outputs.append(word_logits)
                words.append(word_id)
            
            # (batch_size x out_seq_length x output_size), none
            return torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), torch.cat([w.unsqueeze(1) for w in words], dim=1).contiguous()
        
    def forward(self, videos_encodes, captions, teacher_forcing_ratio=0.5):
        return self.forward_fn(v_pool=videos_encodes[3], 
                               s_pool=videos_encodes[2],
                               pos_emb=videos_encodes[4],
                               enc_hidden=videos_encodes[1], 
                               v_feats=videos_encodes[0], 
                               captions=captions, 
                               teacher_forcing_ratio=teacher_forcing_ratio)

    def sample(self, videos_encodes):
        return self.forward(videos_encodes, None, teacher_forcing_ratio=0.0)
