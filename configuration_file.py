#!/usr/bin/env python
from configparser import ConfigParser

__author__ = 'jssprz'
__version__ = '0.0.1'
__email__ = 'jperezmartin90@gmail.com'
__maintainer__ = 'jssprz'
__status__ = 'Development'


class ConfigurationFile:
    def __init__(self, config_path, section_name):
        self.__config = ConfigParser()
        self.__config.read(config_path)

        try:
            section = self.__config[section_name]
        except Exception:
            raise ValueError(" {} is not a valid section".format(section_name))

        self.__dataset_name = section['dataset_name']
        self.__data_dir = section['data_dir'] if 'data_dir' in section else None
        self.__corpus_pkl_path = section['corpus_pkl_path'] if 'corpus_pkl_path' in section else None
        self.__features_path = section['features_path'] if 'features_path' in section else None
        if 'train_range' in section and 'val_range' in section and 'test_range' in section:
            self.__train_range = section['train_range']
            self.__val_range = section['val_range']
            self.__test_range = section['test_range']
        if 'num_epochs' in section:
            self.__num_epochs = int(section['num_epochs'])
        self.__batch_size = int(section['batch_size'])
        self.__learning_rate = float(section['learning_rate'])
        self.__lr_decay_factor = float(section['lr_decay_factor'])
        self.__optimizer_name = section['optimizer_name']
        self.__criterion_name = section['criterion_name']
        self.__criterion_reduction = section['criterion_reduction']
        self.__criterion_param = float(section['criterion_param'])
        self.__convergence_speed_factor = int(section['convergence_speed_factor'])
        if 'train_caption_pkl_path' in section and 'val_caption_pkl_path' in section and 'test_caption_pkl_path' in section:
            self.__train_caption_pkl_path = section['train_caption_pkl_path']
            self.__val_caption_pkl_path = section['val_caption_pkl_path']
            self.__test_caption_pkl_path = section['test_caption_pkl_path']

        self.__encoder_rnn_cell = section['encoder_rnn_cell']
        self.__encoder_num_layers = int(section['encoder_num_layers'])
        self.__encoder_dropout_p = float(section['encoder_dropout_p'])
        self.__encoder_bidirectional = section.getboolean('encoder_bidirectional')
        self.__encoder_vis_syn_embedd_space_size = int(section['vis_syn_embedd_space_size'])

        self.__decoder_rnn_cell = section['decoder_rnn_cell']
        self.__decoder_attn = section.getboolean('decoder_attn')
        self.__decoder_num_layers = int(section['decoder_num_layers'])
        self.__decoder_dropout_p = float(section['decoder_dropout_p'])
        self.__decoder_bidirectional = section.getboolean('decoder_bidirectional')
        self.__decoder_teacher_forcing_ratio = float(section['decoder_teacher_forcing_ratio'])
        self.__decoder_beam_size = int(section['decoder_beam_size'])
        self.__decoder_train_sample_max = section.getboolean('decoder_train_sample_max')
        self.__decoder_test_sample_max = section.getboolean('decoder_test_sample_max')
        self.__decoder_temperature = float(section['decoder_temperature'])
        self.__decoder_beam_search_logic = section['decoder_beam_search_logic']

        if 'max_words' in section:
            self.__max_words = int(section['max_words'])
        if 'max_frames' in section:
            self.__max_frames = int(section['max_frames'])

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def corpus_pkl_path(self):
        return self.__corpus_pkl_path

    @property
    def max_words(self):
        return self.__max_words

    @property
    def max_frames(self):
        return self.__max_frames

    @property
    def features_path(self):
        return self.__features_path

    @property
    def train_range(self):
        return self.__train_range

    @property
    def val_range(self):
        return self.__val_range

    @property
    def test_range(self):
        return self.__test_range

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def num_epochs(self):
        return self.__num_epochs

    @property
    def learning_rate(self):
        return self.__learning_rate
    
    @property
    def lr_decay_factor(self):
        return self.__lr_decay_factor

    @property
    def optimizer_name(self):
        return self.__optimizer_name

    @property
    def criterion_name(self):
        return self.__criterion_name
    
    @property
    def criterion_reduction(self):
        return self.__criterion_reduction
    
    @property
    def criterion_param(self):
        return self.__criterion_param
    
    @property
    def convergence_speed_factor(self):
        return self.__convergence_speed_factor

    @property
    def train_caption_pkl_path(self):
        return self.__train_caption_pkl_path

    @property
    def val_caption_pkl_path(self):
        return self.__val_caption_pkl_path

    @property
    def test_caption_pkl_path(self):
        return self.__test_caption_pkl_path

    @property
    def encoder_rnn_cell(self):
        return self.__encoder_rnn_cell

    @property
    def encoder_num_layers(self):
        return self.__encoder_num_layers

    @property
    def encoder_bidirectional(self):
        return self.__encoder_bidirectional

    @property
    def encoder_dropout_p(self):
        return self.__encoder_dropout_p

    @property
    def encoder_vis_syn_embedd_space_size(self):
        return self.__encoder_vis_syn_embedd_space_size

    @property
    def decoder_rnn_cell(self):
        return self.__decoder_rnn_cell

    @property
    def decoder_attn(self):
        return self.__decoder_attn

    @property
    def decoder_num_layers(self):
        return self.__decoder_num_layers

    @property
    def decoder_bidirectional(self):
        return self.__decoder_bidirectional

    @property
    def decoder_dropout_p(self):
        return self.__decoder_dropout_p

    @property
    def decoder_teacher_forcing_ratio(self):
        return self.__decoder_teacher_forcing_ratio

    @property
    def decoder_beam_size(self):
        return self.__decoder_beam_size

    @property
    def decoder_train_sample_max(self):
        return self.__decoder_train_sample_max
    
    @property
    def decoder_test_sample_max(self):
        return self.__decoder_test_sample_max

    @property
    def decoder_temperature(self):
        return self.__decoder_temperature

    @property
    def decoder_beam_search_logic(self):
        return self.__decoder_beam_search_logic
    
    def __str__(self):
        attrs = vars(self)
        return '\n '.join("%s: %s" % item for item in attrs.items())
