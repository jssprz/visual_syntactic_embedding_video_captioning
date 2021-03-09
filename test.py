import os
import argparse
import pickle

from utils import decode_from_tokens
from vocabulary import Vocabulary
from configuration_file import ConfigurationFile
from model.encoder import SCNEncoder
from model.decoder import SemSynANDecoder

import h5py
import torch
import numpy as np


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate captions por test samples')
  parser.add_argument('-chckpt', '--checkpoint_path', type=str, default='pretrain/chckpt.pt',
                      help='Set the path to pre-trained model (default is pretrain/chckpt.pt).')
  parser.add_argument('-data', '--dataset_folder', type=str, default='data/MSVD',
                      help='Set the path to dataset folder (default is data/MSVD).')
  parser.add_argument('-out', '--output_folder', type=str, default='results/MSVD',
                      help='Set the path to output folder (default is results/MSVD).')

  args = parser.parse_args()

  # load vocabulary
  with open(os.path.join(args.dataset_folder, 'corpus.pkl'), "rb") as f:
      corpus = pickle.load(f)
      idx2word_dict = corpus[4]
  vocab = Vocabulary.from_idx2word_dict(idx2word_dict, False)
  print('Size of vocabulary: {}'.format(len(vocab)))

  # Pretrained Embedding
  pretrained_embedding = torch.Tensor(corpus[5])

  cnn_feature_size = 2048
  c3d_feature_size = 4096
  i3d_feature_size = 400
  eco_feature_size = 1536
  res_eco_features_size = 3584
  cnn_global_size = 512
  projected_size = 512
  hidden_size = 1024  # Number of hidden layer units of the cyclic network
  mid_size = 128  # The middle of the boundary detection layer represents the dimension

  n_tags = 300
  global_tagger_hidden_size = 1024
  specific_tagger_hidden_size = 128
  hidden_size = 1024
  embedding_size = 300  #1024
  rnn_in_size = 300  #1024
  rnn_hidden_size = 1024

  config = ConfigurationFile(os.path.join(args.dataset_folder, 'config.ini'), 'sem-syn-cn-max')

  # Models
  encoder = SCNEncoder(cnn_feature_size=cnn_feature_size,
                        c3d_feature_size=c3d_feature_size,
                        i3d_feature_size=i3d_feature_size,
                        eco_feature_size=eco_feature_size,
                        res_eco_features_size=res_eco_features_size,
                        n_tags=n_tags,
                        hidden_size=hidden_size,
                        global_tagger_hidden_size=global_tagger_hidden_size,
                        specific_tagger_hidden_size=specific_tagger_hidden_size,
                        n_layers=config.encoder_num_layers,
                        input_dropout_p=config.encoder_dropout_p,
                        rnn_dropout_p=config.encoder_dropout_p,
                        bidirectional=config.encoder_bidirectional,
                        rnn_cell=config.encoder_rnn_cell,
                        device='cpu')

  decoder = SemSynANDecoder(in_seq_length=config.max_frames, 
                            out_seq_length=config.max_words,
                            n_feats=res_eco_features_size + cnn_global_size,
#                            n_feats=cnn_feature_size+c3d_feature_size,
                            n_tags=n_tags + 400, #+ 174,
                            n_pos_emb=512,
                            embedding_size=embedding_size,
                            pretrained_embedding=pretrained_embedding,
                            hidden_size=hidden_size, 
                            rnn_in_size=rnn_in_size, 
                            rnn_hidden_size=rnn_hidden_size,
                            vocab=vocab,
                            device='cpu',
                            rnn_cell=config.decoder_rnn_cell,
                            encoder_num_layers=config.encoder_num_layers,
                            encoder_bidirectional=config.encoder_bidirectional,
                            num_layers=config.decoder_num_layers,
                            dropout_p=config.decoder_dropout_p,
                            beam_size=config.decoder_beam_size,
                            temperature=config.decoder_temperature, 
                            train_sample_max=config.decoder_train_sample_max,
                            test_sample_max=config.decoder_test_sample_max,
                            beam_search_logic=config.decoder_beam_search_logic,
                            dataset_name=config.dataset_name)

  # Checkpoint
  checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

  # 1. filter out unnecessary keys for encoder
  chckpt_dict = {k: v for k, v in checkpoint['encoder'].items() if k not in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']}
  encoder_dict = encoder.state_dict()
  encoder_dict.update(chckpt_dict)

  encoder.load_state_dict(encoder_dict)
  decoder.load_state_dict(checkpoint['decoder'])

  #load test set features
  test_vidxs = sorted(list(set(corpus[2][1])))

  with h5py.File(os.path.join(args.dataset_folder, config.features_path), 'r') as feats_file:
      print('loading visual feats...')
      dataset = feats_file[config.dataset_name]
      cnn_feats = torch.from_numpy(dataset['cnn_features'][test_vidxs]).float()
      c3d_feats = torch.from_numpy(dataset['c3d_features'][test_vidxs]).float()
      cnn_globals = torch.zeros(cnn_feats.size(0), 512)  # torch.from_numpy(dataset['cnn_globals'][test_vidxs]).float()
      cnn_sem_globals = torch.from_numpy(dataset['cnn_sem_globals'][test_vidxs]).float()
      f_counts = dataset['count_features'][test_vidxs]
      print('visual feats loaded')

  res_eco_globals = torch.from_numpy(np.load(os.path.join(args.dataset_folder, 'resnext_eco.npy'))[test_vidxs])
  tags_globals = torch.from_numpy(np.load(os.path.join(args.dataset_folder, 'tag_feats.npy'))[test_vidxs])

  encoder.eval()
  decoder.eval()

  with torch.no_grad():
      video_encoded = encoder(cnn_feats, c3d_feats, cnn_globals, cnn_sem_globals, tags_globals, res_eco_globals)
      logits, tokens = decoder(video_encoded, None, teacher_forcing_ratio=0)

      scores = logits.max(dim=2)[0].mean(dim=1)

      confidences, sentences = [], []
      for score, seq in zip(scores, tokens):
          s = decode_from_tokens(seq, vocab)
          print(score, s)
          sentences.append(s)
          confidences.append(score)

  if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

  with open(os.path.join(args.output_folder, 'predictions.txt'), 'w') as fo:
    for vidx, sentence in zip(test_vidxs, sentences):
      fo.write(f'{vidx}\t{sentence}\n')