import sys
sys.path.append('../code/') 
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import BaseMetricLossFunction

import mocap_functions
from functools import partial
import numpy as np
import pandas as pd
import neo
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.interpolate import interp1d
import spike_train_functions
import elephant
import quantities as pq
# import h5py
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F
from joblib import Parallel, delayed
import multiprocessing
import Neural_Decoding
import pickle
import seaborn as sns
from hnn_core.utils import smooth_waveform
from scipy.signal import savgol_filter
# import haste_pytorch as haste
#sns.set()
#sns.set_style("white")

num_cores = multiprocessing.cpu_count()
scaler = StandardScaler()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0")
#device = torch.device('cpu')

torch.backends.cudnn.benchmark = True

def load_mocap_df(data_path, kinematic_suffix=None):
    kinematic_df = pd.read_pickle(f'{data_path}kinematic_df{kinematic_suffix}_preevent.pkl')
    neural_df = pd.read_pickle(data_path + 'neural_df_preevent.pkl')

    # read python dict back from the file
    metadata_file = open(data_path + 'metadata_preevent.pkl', 'rb')
    metadata = pickle.load(metadata_file)
    metadata_file.close()

    return kinematic_df, neural_df, metadata

# Prepare dataframes for movement decoding
def get_marker_decode_dataframes(noise_fold=0):
    cam_idx = 1
    # kinematic_df, neural_df, metadata = mocap_functions.load_mocap_df('../data/SPK20220308/task_data/', kinematic_suffix=f'_cam{cam_idx}')
    kinematic_df, neural_df, metadata = load_mocap_df('../data/SPK20220308/task_data/', kinematic_suffix=f'_cam{cam_idx}')

    num_trials = len(kinematic_df['trial'].unique())

    pos_filter = [1,2,3,4]
    layout_filter = [1,2,3,4]
    pos_remove_filter = [f'position_{pos_idx}' for pos_idx in []]
    layout_remove_filter = [f'layout_{layout_idx}' for layout_idx in []]

    # pos_filter = [1, 4]
    # layout_filter = [1,2]
    # pos_remove_filter = [f'position_{pos_idx}' for pos_idx in [2,3]]
    # layout_remove_filter = [f'layout_{layout_idx}' for layout_idx in [3,4]]


    neural_df = neural_df[np.in1d(neural_df['position'], pos_filter)].reset_index(drop=True)
    neural_df = neural_df[np.in1d(neural_df['layout'], layout_filter)].reset_index(drop=True)
    neural_df = neural_df[~np.in1d(neural_df['unit'], pos_remove_filter)].reset_index(drop=True)
    neural_df = neural_df[~np.in1d(neural_df['unit'], layout_remove_filter)].reset_index(drop=True)

    kinematic_df = kinematic_df[np.in1d(kinematic_df['position'], pos_filter)].reset_index(drop=True)
    kinematic_df = kinematic_df[np.in1d(kinematic_df['layout'], layout_filter)].reset_index(drop=True)
    kinematic_df = kinematic_df[~np.in1d(kinematic_df['name'], pos_remove_filter)].reset_index(drop=True)
    kinematic_df = kinematic_df[~np.in1d(kinematic_df['name'], layout_remove_filter)].reset_index(drop=True)

    # Subselect specific marker
    # marker_list = ['ulnarDistal', 'carpal', 'thumbProx', 'ringProx','pinkyProx'] # cam4
    # marker_list = ['ringProx', 'pinkyProx', 'middleProx'] # cam4
    marker_list = ['ringProx'] # cam4
    # marker_list = ['indexProx', 'carpal', 'ringProx'] # cam1

    mask_list = [kinematic_df['name'].str.contains(pat=pat) for pat in marker_list]
    wrist_df = kinematic_df[np.logical_or.reduce(mask_list)]

    # Remove trials with velocity outliers
    velocity_outlier_thresh = 6
    velocity_std = np.concatenate(wrist_df['posData'].map(np.diff).values).std()
    velocity_outlier_mask = wrist_df['posData'].map(np.diff).apply(
        lambda x: np.any(np.abs(x - np.mean(x)) > velocity_outlier_thresh * velocity_std))
    velocity_outlier_trials = wrist_df[velocity_outlier_mask]['trial'].unique()

    wrist_df = wrist_df[wrist_df['trial'].apply(lambda x: x not in velocity_outlier_trials)]
    neural_df = neural_df[neural_df['trial'].apply(lambda x: x not in velocity_outlier_trials)]

    # Remove trials with length outliers
    length_outlier_thresh = 3
    trial_lengths = wrist_df['posData'].map(len).values
    length_outlier_mask = np.abs(trial_lengths - np.mean(trial_lengths)) > length_outlier_thresh * np.std(trial_lengths)
    length_outlier_trials = wrist_df[length_outlier_mask]['trial'].unique()

    wrist_df = wrist_df[wrist_df['trial'].apply(lambda x: x not in length_outlier_trials)]
    neural_df = neural_df[neural_df['trial'].apply(lambda x: x not in length_outlier_trials)]

    assert np.array_equal(neural_df['trial'].unique(), wrist_df['trial'].unique())

    trial_ids = neural_df['trial'].unique()
    num_trials_filtered = len(trial_ids)

    # Give each layout/position combination a unique label
    trial_type_dict = dict()
    label_idx = 0
    for layout_idx in range(1,5):
        for position_idx in range(1,5):
            trial_type_dict[(layout_idx, position_idx)] = label_idx
            label_idx = label_idx + 1

    # Populate list with labels to match each trials
    trial_labels = list()
    for trial_id in trial_ids:
        layout_idx = neural_df[neural_df['trial'] == trial_id]['layout'].values[0]
        position_idx = neural_df[neural_df['trial'] == trial_id]['position'].values[0]
        trial_labels.append(trial_type_dict[(layout_idx, position_idx)])
    trial_labels = np.array(trial_labels)

    

    #Generate cv_dict for regular train/test/validate split (no rolling window)
    # cv_split = StratifiedShuffleSplit(n_splits=5, test_size=.25, random_state=3)
    # val_split = StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=3)

    cv_split = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=3)
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=3)

    cv_dict = {}
    for fold, (train_val_idx, test_idx) in enumerate(cv_split.split(trial_ids, trial_labels)):
        val_trial_labels = trial_labels[train_val_idx]
        for t_idx, v_idx in val_split.split(train_val_idx, val_trial_labels): #No looping, just used to split train/validation sets
            cv_dict[fold] = {'train_idx':trial_ids[train_val_idx[t_idx]], 
                            'test_idx':trial_ids[test_idx], 
                            'validation_idx':trial_ids[train_val_idx[v_idx]]} 

    neural_df, wrist_df = add_noise(neural_df, wrist_df, cv_dict, noise_fold, num_trials)

    # Smooth everything after adding noise
    smooth_func = partial(savgol_filter, window_length=31, polyorder=3)
    neural_df['rates'] = neural_df['rates'].map(smooth_func)
    wrist_df['posData'] = wrist_df['posData'].map(smooth_func)

    nolayout_neural_mask = ~(neural_df['unit'].str.contains(pat='layout'))
    noposition_neural_mask = ~(neural_df['unit'].str.contains(pat='position'))
    notask_neural_df = neural_df[np.logical_and.reduce([nolayout_neural_mask, noposition_neural_mask])]
    task_neural_df = neural_df.copy()

    data_dict = {'wrist_df': wrist_df, 'task_neural_df': task_neural_df, 'notask_neural_df': notask_neural_df,
                 'metadata': metadata, 'cv_dict': cv_dict, 'noise_fold': noise_fold}
    return data_dict

#RNN architecture for decoding kinematics
class model_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device, bidirectional=False,
                 cat_features=None):
        super(model_lstm, self).__init__()

        #multiplier based on bidirectional parameter
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # Defining some parameters
        self.hidden_dim = hidden_dim       
        self.n_layers = n_layers * num_directions
        self.device = device
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.cat_features = cat_features
        self.input_size = input_size

        if self.cat_features is not None:
            self.num_cat_features = np.sum(self.cat_features).astype(int)
            self.hidden_fc = nn.Linear(self.num_cat_features, self.hidden_dim)

            self.input_size = self.input_size - self.num_cat_features
            # self.input_size = self.input_size

            
        else:
            self.fc = nn.Linear(self.hidden_dim * num_directions, output_size)

        self.fc = nn.Linear((self.hidden_dim* num_directions), output_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional) 
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden, cell = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        if self.cat_features is not None:
            # cat_hidden = self.hidden_fc(torch.tanh(x[:, -1, self.cat_features]))
            # hidden = hidden + cat_hidden
            # cell = cell + cat_hidden

            out_lstm, (hidden, cell) = self.lstm(x[:, :, ~self.cat_features], (hidden, cell))

        else:
            out_lstm, (hidden, cell) = self.lstm(x, (hidden, cell))

        out_final = out_lstm.contiguous()
        out_final = self.fc(out_final)
        return out_final, hidden, cell
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data.to(self.device)

        #LSTM initialization
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        cell = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device) + 1

        return hidden, cell

#RNN architecture for decoding kinematics
class model_lstm_single(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device, bidirectional=False,
                 cat_features=None):
        super(model_lstm_single, self).__init__()

        #multiplier based on bidirectional parameter
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # Defining some parameters
        self.hidden_dim = hidden_dim       
        self.n_layers = 1
        self.device = device
        self.dropout_func = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        self.cat_features = cat_features
        self.input_size = input_size

        if self.cat_features is not None:
            self.num_cat_features = np.sum(self.cat_features).astype(int)

            self.input_size = self.input_size - self.num_cat_features
            # self.input_size = self.input_size

            
        else:
            self.fc = nn.Linear(self.hidden_dim * num_directions, output_size)

        n_layers = 1
        self.fc = nn.Linear((self.hidden_dim* num_directions), output_size)
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_dim, n_layers, batch_first=True, dropout=0, bidirectional=bidirectional)
        self.lstm2  = nn.LSTM(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True, dropout=0, bidirectional=bidirectional)

        # self.forget_bias(self.lstm1)
        # self.forget_bias(self.lstm2)
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden0, cell0 = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        if self.cat_features is not None:
            out1, (hidden1, cell1) = self.lstm1(x[:, :, ~self.cat_features], (hidden0, cell0))

        else:
            out1, (hidden1, cell1) = self.lstm1(x, (hidden0, cell0))

        out2, (hidden2, cell2) = self.lstm2(self.dropout_func(out1), (hidden1, cell1))

        out_final = out2.contiguous()
        out_final = self.fc(out_final)
        return out_final, (out1, out2), (cell1, cell2)
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data.to(self.device)

        #LSTM initialization
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        cell = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)

        return hidden, cell

    def forget_bias(self, layer):
        for names in layer._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(layer, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

def add_noise(neural_df, wrist_df, cv_dict, fold, num_trials):
    rng = np.random.default_rng(111)

    # kinematic_noise = 20
    kinematic_noise = 10
    neural_noise = 1
    # neural_noise = 5

    noise_rounds = 10

    neural_df_list = [neural_df]
    wrist_df_list = [wrist_df]
    for round in (1, noise_rounds + 1):
        # Neural data
        neural_temp_df = neural_df[np.in1d(neural_df['trial'], cv_dict[fold]['train_idx'])].copy()
        
        nolayout_mask = ~(neural_temp_df['unit'].str.contains(pat='layout'))
        noposition_mask = ~(neural_temp_df['unit'].str.contains(pat='position'))

        unit_mask = np.logical_and(nolayout_mask, noposition_mask)

        noise_data = neural_temp_df[unit_mask]['rates'].apply(
            lambda x: x + rng.normal(loc=0, scale=neural_noise, size=len(x)))
        neural_temp_df['rates'][unit_mask] = noise_data
        neural_temp_df['trial'] += (round * num_trials)
        neural_df_list.append(neural_temp_df)

        # Kinematic data
        wrist_temp_df = wrist_df[np.in1d(wrist_df['trial'], cv_dict[fold]['train_idx'])].copy()

        nolayout_mask = ~(wrist_temp_df['name'].str.contains(pat='layout'))
        noposition_mask = ~(wrist_temp_df['name'].str.contains(pat='position'))

        unit_mask = np.logical_and(nolayout_mask, noposition_mask)

        noise_data = wrist_temp_df[unit_mask]['posData'].apply(
            lambda x: x + rng.normal(loc=0, scale=kinematic_noise, size=len(x)))
        wrist_temp_df['posData'][unit_mask] = noise_data
        wrist_temp_df['trial'] += (round * num_trials)
        wrist_df_list.append(wrist_temp_df)

        new_trials = np.concatenate([cv_dict[fold]['train_idx'], wrist_temp_df['trial'].unique()])
        cv_dict[fold]['train_idx'] = new_trials

    neural_df = pd.concat(neural_df_list).reset_index(drop=True)
    wrist_df = pd.concat(wrist_df_list).reset_index(drop=True)

    return neural_df, wrist_df

        

# Dataset class to handle mocap dataframes from SEE project
class SEE_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, cv_dict, fold, partition, kinematic_df, neural_df, offset, window_size, data_step_size, device,
                 kinematic_type='posData', scale_neural=True, scale_kinematics=True, flip_outputs=False,
                 exclude_neural=None, exclude_kinematic=None, neural_scaler=None, kinematic_scaler=None,
                 label_col=None):
        #'Initialization'
        self.cv_dict = cv_dict
        self.fold = fold
        self.flip_outputs = flip_outputs
        self.partition = partition
        self.trial_idx = cv_dict[fold][partition]
        self.num_trials = len(self.trial_idx) 
        self.offset = offset
        self.window_size = window_size
        self.data_step_size = data_step_size
        self.label_col = label_col
        self.device = device
        self.posData_list, self.neuralData_list = self.process_dfs(kinematic_df, neural_df)
        if neural_scaler is None:
            neural_scaler = StandardScaler()
            if exclude_neural is not None:
                neural_scaler.fit(np.vstack(self.neuralData_list)[:, ~exclude_neural])
            else:
                neural_scaler.fit(np.vstack(self.neuralData_list))
        self.neural_scaler = neural_scaler

        
        if kinematic_scaler is None:
            kinematic_scaler = StandardScaler()
            if exclude_kinematic is not None:
                kinematic_scaler.fit(np.vstack(self.posData_list)[:, ~exclude_kinematic])
            else:
                kinematic_scaler.fit(np.vstack(self.posData_list))
        self.kinematic_scaler = kinematic_scaler

        # Extract labels for trial
        if self.label_col is not None:
            self.make_labels = True
            self.trial_labels = self.get_labels(kinematic_df, neural_df)
            self.batch_labels = list()
            self.batch_trials = list()
        else:
            self.make_labels = False

        # Boolean array of 1's for features to not be scaled
        if scale_kinematics:
            self.posData_list = self.transform_data(self.posData_list, self.kinematic_scaler, exclude_kinematic)
        
        if scale_neural:
            self.neuralData_list = self.transform_data(self.neuralData_list, self.neural_scaler, exclude_neural)

        self.split_offset = np.round((self.offset/self.data_step_size) / 2).astype(int)

        self.X_tensor, self.y_tensor = self.load_splits()
        self.num_samples = np.sum(self.X_tensor.size(0))

        if self.make_labels:
            assert len(self.batch_labels) == self.num_samples
            assert len(self.batch_trials) == self.num_samples
            self.batch_labels = torch.tensor(self.batch_labels)
            self.batch_trials = torch.tensor(self.batch_trials)

    def __len__(self):
        #'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, slice_index):
        return self.X_tensor[slice_index,:,:], self.y_tensor[slice_index,:,:], self.batch_labels[slice_index]

    #**add functionality to separate eye, object, and body markers
    def process_dfs(self, kinematic_df, neural_df):
        posData_list, neuralData_list = [], []
        for trial in self.trial_idx:
            posData_array = np.stack(kinematic_df['posData'][kinematic_df['trial'] == trial].values).transpose() 
            neuralData_array = np.stack(neural_df['rates'][neural_df['trial'] == trial].values).transpose() 

            posData_list.append(posData_array)
            neuralData_list.append(neuralData_array)

        return posData_list, neuralData_list
    
    # Use label col to pair each trial with specific category
    def get_labels(self, kinematic_df, neural_df):
        labels = list()
        for trial in self.trial_idx:
            kinematic_labels = kinematic_df[self.label_col][kinematic_df['trial'] == trial].values
            neural_labels = neural_df[self.label_col][neural_df['trial'] == trial].values

            assert np.all(kinematic_labels == kinematic_labels[0])
            assert np.all(neural_labels == neural_labels[0])
            assert kinematic_labels[0] == neural_labels[0]
            labels.append(neural_labels[0])
        return labels

    def format_splits(self, data_list, make_labels=False):
        unfolded_data_list = list()
        for trial_idx in range(self.num_trials):
            if self.window_size == 1:
                padded_trial = torch.from_numpy(data_list[trial_idx])
            else:
                # padded_trial = torch.nn.functional.pad(
                #     torch.from_numpy(data_list[trial_idx].T), pad=(self.window_size, 0)).transpose(0, 1)
                padded_trial = torch.from_numpy(data_list[trial_idx])
            
            unfolded_trial = padded_trial.unfold(0, self.window_size, self.data_step_size).transpose(1, 2)
            unfolded_data_list.append(unfolded_trial)
            
            if make_labels:
                self.batch_labels.extend(np.repeat(self.trial_labels[trial_idx], unfolded_trial.size(0)))
                self.batch_trials.extend(np.repeat(trial_idx, unfolded_trial.size(0)))
        
        data_tensor = torch.concat(unfolded_data_list, axis=0)

        return data_tensor
    
    def load_splits(self):
        if not self.flip_outputs:
            X_tensor = self.format_splits(self.posData_list, make_labels=self.make_labels)
            y_tensor = self.format_splits(self.neuralData_list)
        else:
            y_tensor = self.format_splits(self.posData_list, make_labels=self.make_labels)
            X_tensor = self.format_splits(self.neuralData_list)

        # X_tensor, y_tensor = X_tensor[:,:-self.split_offset:self.data_step_size,:], y_tensor[:,self.split_offset::self.data_step_size,:]
        X_tensor, y_tensor = X_tensor[:,self.split_offset::self.data_step_size,:], y_tensor[:,:-self.split_offset:self.data_step_size,:]

        assert X_tensor.shape[0] == y_tensor.shape[0]
        return X_tensor, y_tensor

    #Zero mean and unit std
    def transform_data(self, data_list, scaler, exclude_processing):
        #Iterate over trials and apply normalization
     
        scaled_data_list = []
        for data_trial in data_list:
            if exclude_processing is None:
                scaled_data_trial = scaler.transform(data_trial)
            else:
                scaled_data_trial = np.zeros(data_trial.shape)
                scaled_data_trial[:, exclude_processing] = data_trial[:, exclude_processing]
                processed_data = scaler.transform(data_trial[:, ~exclude_processing])
                scaled_data_trial[:, ~exclude_processing] = processed_data
            scaled_data_list.append(scaled_data_trial)

        return scaled_data_list


# Prepare dataloaders for batch training
def make_generators(pred_df, neural_df, neural_offset, cv_dict, metadata,
                    exclude_neural=None, exclude_kinematics=None, window_size=1, 
                    flip_outputs=False, fold=0, batch_size=1000, device='cpu', label_col=None):
    sampling_rate = 100
    kernel_offset = int(metadata['kernel_halfwidth'] * sampling_rate)  #Convolution kernel centered at zero, add to neural offset
    # offset = neural_offset + kernel_offset
    offset = neural_offset 
    data_step_size = 1 

    # Set up PyTorch Dataloaders
    
    # Parameters
    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_cores, 'pin_memory':False}
    train_eval_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_cores, 'pin_memory':False}
    validation_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_cores, 'pin_memory':False}
    test_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_cores, 'pin_memory':False}

    scale_neural = True
    scale_kinematics = True
    flip_outputs=flip_outputs

    # Generators
    training_set = SEE_Dataset(cv_dict, fold, 'train_idx', pred_df, neural_df, offset, window_size, 
                               data_step_size, device, 'posData', scale_neural=scale_neural,
                               scale_kinematics=scale_kinematics, flip_outputs=flip_outputs,
                               exclude_neural=exclude_neural, exclude_kinematic=exclude_kinematics, label_col=label_col)
    training_neural_scaler = training_set.neural_scaler
    training_kinematic_scaler = training_set.kinematic_scaler

    training_generator = torch.utils.data.DataLoader(training_set, **train_params)
    training_eval_generator = torch.utils.data.DataLoader(training_set, **train_eval_params)

    validation_set = SEE_Dataset(cv_dict, fold, 'validation_idx', pred_df, neural_df, offset, window_size, 
                                 data_step_size, device, 'posData', scale_neural=scale_neural,
                                 scale_kinematics=scale_kinematics, flip_outputs=flip_outputs,
                                 exclude_neural=exclude_neural, exclude_kinematic=exclude_kinematics,
                                 neural_scaler=training_neural_scaler, kinematic_scaler=training_kinematic_scaler, label_col=label_col)
    validation_generator = torch.utils.data.DataLoader(validation_set, **validation_params)

    testing_set = SEE_Dataset(cv_dict, fold, 'test_idx', pred_df, neural_df, offset, window_size, 
                              data_step_size, device, 'posData', scale_neural=scale_neural,
                              scale_kinematics=scale_kinematics, flip_outputs=flip_outputs,
                              exclude_neural=exclude_neural, exclude_kinematic=exclude_kinematics,
                              neural_scaler=training_neural_scaler, kinematic_scaler=training_kinematic_scaler, label_col=label_col)
    testing_generator = torch.utils.data.DataLoader(testing_set, **test_params)

    data_arrays = (training_set, validation_set, testing_set)
    generators = (training_generator, training_eval_generator, validation_generator, testing_generator)

    return data_arrays, generators

#Helper function to pytorch train networks for decoding
def train_validate_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, device, print_freq=10, early_stop=20):
    train_loss_array = []
    validation_loss_array = []
    # Loop over epochs
    min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
    for epoch in range(max_epochs):
        #___Train model___
        model.train()
        train_batch_loss = []
        validation_batch_loss = []
        for batch_x, batch_y, labels in training_generator:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            labels = labels.float().to(device)

            # output, hidden = model(batch_x)
            # train_loss = criterion(output[:,-10:-1,:], batch_y[:,-10:-1,:], hidden, labels)

            output, hidden, cell = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y[:,-1,:], hidden, cell, labels)
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
        
        train_loss_array.append(train_batch_loss)

        #___Evaluate Model___
        with torch.no_grad():
            model.eval()
            #Generate train set predictions
            for batch_x, batch_y, labels in validation_generator:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                labels = labels.float().to(device)

                # output, hidden = model(batch_x)
                # validation_loss = criterion(output[:,-10:-1,:], batch_y[:,-10:-1,:], hidden, labels)

                output, hidden, cell = model(batch_x)
                validation_loss = criterion(output[:,-1,:], batch_y[:,-1,:], hidden, cell, labels)

                validation_batch_loss.append(validation_loss.item())

        validation_loss_array.append(validation_batch_loss)

        #Compute average loss on batch
        train_epoch_loss = np.mean(train_batch_loss)
        train_epoch_std = np.std(train_batch_loss)
        validation_epoch_loss = np.mean(validation_batch_loss)
        validation_epoch_std = np.std(validation_batch_loss)

       #Check if validation loss reaches minimum 
        if validation_epoch_loss < min_validation_loss:
            print('*',end='')
            min_validation_loss = np.copy(validation_epoch_loss)
            min_validation_std = np.copy(validation_epoch_std)
            min_validation_counter = 0
            min_validation_epoch = np.copy(epoch+1)

            min_train_loss = np.copy(train_epoch_loss)
            min_train_std = np.copy(train_epoch_std)
            
        else:
            print('.',end='')
            min_validation_counter += 1

        #Print Loss Scores
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: {:.4f}  ... Validation Loss: {:.4f}'.format(train_epoch_loss,validation_epoch_loss))
        
        #Early stop if no validation improvement over set number of epochs
        if min_validation_counter > early_stop:
            print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
            break

    loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
    'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
    'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'max_epochs':max_epochs}
    return loss_dict


#Helper function to evaluate decoding performance on a trained model
def evaluate_model(model, generator, device):
    #Run model through test set
    with torch.no_grad():
        model.eval()
        #Generate predictions
        y_pred_tensor = torch.zeros(len(generator.dataset),  generator.dataset[0][1].shape[1])
        batch_idx = 0
        for batch_x, batch_y, labels in generator:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            labels = labels.float().to(device)

            # output, _ = model(batch_x)
            output, _, _ = model(batch_x)
            y_pred_tensor[batch_idx:(batch_idx+output.size(0)),:] = output[:,-1,:]
            batch_idx += output.size(0)

    y_pred = y_pred_tensor.detach().cpu().numpy()
    return y_pred
    
# Joint loss function: contrastive_loss + MSE
def contrast_mse(y_pred, y_true, hidden, cell, labels, weight=0.1):
    hidden = torch.concat([hidden[0], hidden[1]], dim=2) # Hidden states returned separately for each layer
    hidden = hidden[:,-10:-1,:]
    
    # cell = torch.concat(cell, dim=0) # Hidden states returned separately for each layer

    # hidden = hidden.transpose(0,1)

    hidden = hidden.flatten(start_dim=1, end_dim=2)

    # cell = cell.transpose(0,1)
    # cell = cell.flatten(start_dim=1, end_dim=2)

    # loss_func = losses.SupConLoss(temperature=0.1, distance=LpDistance(power=2))
    loss_func = losses.SupConLoss(temperature=0.1)

    hidden_loss = loss_func(hidden, labels)
    # cell_loss = loss_func(cell, labels)
    # output_loss = loss_func(output.flatten(start_dim=1, end_dim=2), labels)


    mse_loss_func = nn.MSELoss()
    mse_loss = mse_loss_func(y_pred, y_true)

    # loss = mse_loss + (weight * (hidden_loss + cell_loss))
    loss = mse_loss + (weight * hidden_loss)
    # loss = mse_loss + (weight * cell_loss)
    # loss = mse_loss + (weight * output_loss)


    return loss

# Dummy wrapper for MSE (last 3 parameters unused)
def mse(y_pred, y_true, hidden, cell, labels, weight=1):
    mse_loss_func = nn.MSELoss()
    loss = mse_loss_func(y_pred, y_true)
    return loss


def run_wiener(pred_df, neural_df, neural_offset, cv_dict, metadata, task_info=True, window_size=10, num_cat=0, label_col=None):
    # window_size = 1 # doesn't matter for kalman filter
    # neural_offset = 2
    if task_info:
        exclude_processing = np.zeros(len(neural_df['unit'].unique()))
        exclude_processing[-num_cat:] = np.ones(num_cat)
        exclude_processing = exclude_processing.astype(bool)
    else:
        exclude_processing = None

    data_arrays, generators = make_generators(
    pred_df, neural_df, neural_offset, cv_dict, metadata, 
    exclude_neural=exclude_processing, window_size=window_size,
    flip_outputs=True, label_col=label_col)

    # Unpack tuple into variables
    training_set, validation_set, testing_set = data_arrays
    training_generator, training_eval_generator, validation_generator, testing_generator = generators

    X_train_data = training_set[:][0][:,-1,:].detach().cpu().numpy()
    y_train_data = training_set[:][1][:,-1,:].detach().cpu().numpy()

    X_test_data = testing_set[:][0][:,-1,:].detach().cpu().numpy()
    y_test_data = testing_set[:][1][:,-1,:].detach().cpu().numpy()

    #Fit and run wiener filter
    model_wr = Neural_Decoding.decoders.WienerFilterDecoder() 
    model_wr.fit(X_train_data,y_train_data)

    wr_train_pred = model_wr.predict(X_train_data)
    wr_test_pred = model_wr.predict(X_test_data)

    #Compute decoding performance
    wr_train_corr = mocap_functions.matrix_corr(wr_train_pred,y_train_data)
    wr_test_corr = mocap_functions.matrix_corr(wr_test_pred,y_test_data)

    res_dict = {'train_pred': wr_train_pred, 'test_pred': wr_test_pred,
                'train_corr': wr_train_corr, 'test_corr': wr_test_corr}

    return model_wr, res_dict

def run_rnn(pred_df, neural_df, neural_offset, cv_dict, metadata, task_info=True,
            window_size=10, num_cat=0, label_col=None):
    if task_info:
        exclude_processing = np.zeros(len(neural_df['unit'].unique()))
        exclude_processing[-num_cat:] = np.ones(num_cat)
        exclude_processing = exclude_processing.astype(bool)
        criterion = contrast_mse

    else:
        exclude_processing = None
        criterion = mse

    data_arrays, generators = make_generators(
    pred_df, neural_df, neural_offset, cv_dict, metadata, exclude_neural=exclude_processing,
    window_size=window_size, flip_outputs=True, batch_size=1000, label_col=label_col)

    # Unpack tuple into variables
    training_set, validation_set, testing_set = data_arrays
    training_generator, training_eval_generator, validation_generator, testing_generator = generators

    X_train_data = training_set[:][0][:,-1,:].detach().cpu().numpy()
    y_train_data = training_set[:][1][:,-1,:].detach().cpu().numpy()

    X_test_data = testing_set[:][0][:,-1,:].detach().cpu().numpy()
    y_test_data = testing_set[:][1][:,-1,:].detach().cpu().numpy()

    #Define hyperparameters
    lr = 1e-4
    weight_decay = 1e-4
    hidden_dim = 100
    # dropout = 0.0
    dropout = 0.5
    n_layers = 2
    max_epochs = 1000
    input_size = X_train_data.shape[1] 
    output_size = y_train_data.shape[1] 

    # model_rnn = model_lstm(input_size, output_size, hidden_dim, n_layers, dropout, device, cat_features=exclude_processing).to(device)
    model_rnn = model_lstm_single(input_size, output_size, hidden_dim, n_layers, dropout, device, cat_features=exclude_processing).to(device)

    # Define Loss, Optimizerints h
    optimizer = torch.optim.Adam(model_rnn.parameters(), lr=lr, weight_decay=weight_decay)

    #Train model
    loss_dict = train_validate_model(model_rnn, optimizer, criterion, max_epochs, training_generator, validation_generator, device, 10, 5)

    #Evaluate trained model
    rnn_train_pred = evaluate_model(model_rnn, training_eval_generator, device)
    rnn_test_pred = evaluate_model(model_rnn, testing_generator, device)

    rnn_train_corr = mocap_functions.matrix_corr(rnn_train_pred, y_train_data)
    rnn_test_corr = mocap_functions.matrix_corr(rnn_test_pred, y_test_data)

    res_dict = {'loss_dict': loss_dict,
                'train_pred': rnn_train_pred, 'test_pred': rnn_test_pred,
                'train_corr': rnn_train_corr, 'test_corr': rnn_test_corr}

    return model_rnn, res_dict

