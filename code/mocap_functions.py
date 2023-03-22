from scipy.signal import argrelextrema
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
import neo.core as neo
import matplotlib.pyplot as plt
import elephant 
import quantities as pq
import spike_train_functions
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
import torch
from torch import nn
import torch.nn.functional as F
import multiprocessing
from joblib import Parallel, delayed
import pickle
import imageio
num_cores = multiprocessing.cpu_count()

#Simple feedforward ANN for decoding kinematics
class model_ann(nn.Module):
    def __init__(self, input_size, output_size, layer_size):
        super(model_ann, self).__init__()
        self.input_size,  self.layer_size, self.output_size = input_size, layer_size, output_size

        #List layer sizes
        self.layer_hidden = np.concatenate([[input_size], layer_size, [output_size]])
        
        #Compile layers into lists
        self.layer_list = nn.ModuleList(
            [nn.Linear(in_features=self.layer_hidden[idx], out_features=self.layer_hidden[idx+1]) for idx in range(len(self.layer_hidden)-1)] )        
 
    def forward(self, x):
        #Encoding step
        for idx in range(len(self.layer_list)):
            x = F.tanh(self.layer_list[idx](x))

        return x


#LSTM/GRU architecture for decoding
class model_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device, bidirectional=False):
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

        #Defining the layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout)   

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*num_directions, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous()
        out = self.fc(out)
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        weight = next(self.parameters()).data.to(self.device)

        # LSTM cell initialization
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
    

        return hidden


#GRU architecture for decoding kinematics
class model_gru(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device, bidirectional=False,
                 cat_features=None):
        super(model_gru, self).__init__()

        #multiplier based on bidirectional parameter
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # Defining some parameters
        if cat_features is not None:
            self.hidden_dim = hidden_dim + np.sum(cat_features)
        else:
            self.hidden_dim = hidden_dim

        self.n_layers = n_layers * num_directions
        self.device = device
        self.dropout = dropout
        self.bidirectional = bidirectional

        #Defining the layers
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)   

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*num_directions, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.gru(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous()
        out = torch.hstack()
        out = self.fc(out)
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        weight = next(self.parameters()).data.to(self.device)

        #GRU initialization
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)

        return hidden


#Helper function to pytorch train networks for decoding
def train_model(model, optimizer, criterion, max_epochs, training_generator, device, print_freq=10):
    train_loss_array = []
    model.train()
    # Loop over epochs
    for epoch in range(max_epochs):
        train_batch_loss = []
        for batch_x, batch_y in training_generator:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
        print('*',end='')
        train_loss_array.append(train_batch_loss)
        #Print Loss
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: ' + str(np.mean(train_batch_loss)))
    return train_loss_array

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
        for batch_x, batch_y in training_generator:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
        
        train_loss_array.append(train_batch_loss)

        #___Evaluate Model___
        with torch.no_grad():
            model.eval()
            #Generate train set predictions
            for batch_x, batch_y in validation_generator:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                validation_loss = criterion(output[:,-1,:], batch_y[:,-1,:])

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


#Helper function to pytorch train networks for decoding
def train_validate_test_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, testing_generator,device, print_freq=10, early_stop=20):
    train_loss_array = []
    validation_loss_array = []
    test_loss_array = []
    # Loop over epochs
    min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
    for epoch in range(max_epochs):
        #___Train model___
        model.train()
        train_batch_loss = []
        validation_batch_loss = []
        test_batch_loss = []
        for batch_x, batch_y in training_generator:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            train_loss = criterion(output[:,-1,:], batch_y[:,-1,:])
            # train_loss = criterion(output[:,:,:], batch_y[:,:,:])
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
        
        train_loss_array.append(train_batch_loss)

        #___Evaluate Model___
        with torch.no_grad():
            model.eval()
            #Generate validation set predictions
            for batch_x, batch_y in validation_generator:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                validation_loss = criterion(output[:,-1,:], batch_y[:,-1,:])

                validation_batch_loss.append(validation_loss.item())

            validation_loss_array.append(validation_batch_loss)

            #Generate test set predictions
            for batch_x, batch_y in testing_generator:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x)
                test_loss = criterion(output[:,-1,:], batch_y[:,-1,:])

                test_batch_loss.append(test_loss.item())

            test_loss_array.append(test_batch_loss)

        #Compute average loss on batch
        train_epoch_loss = np.mean(train_batch_loss)
        train_epoch_std = np.std(train_batch_loss)
        validation_epoch_loss = np.mean(validation_batch_loss)
        validation_epoch_std = np.std(validation_batch_loss)
        test_epoch_loss = np.mean(test_batch_loss)
        test_epoch_std = np.std(test_batch_loss)

       #Check if validation loss reaches minimum 
        if validation_epoch_loss < min_validation_loss:
            print('*',end='')
            min_validation_loss = np.copy(validation_epoch_loss)
            min_validation_std = np.copy(validation_epoch_std)
            min_validation_counter = 0
            min_validation_epoch = np.copy(epoch+1)

            min_train_loss = np.copy(train_epoch_loss)
            min_train_std = np.copy(train_epoch_std)
            min_test_loss = np.copy(test_epoch_loss)
            min_test_std = np.copy(test_epoch_std)


        else:
            print('.',end='')
            min_validation_counter += 1

        #Print Loss Scores
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: {:.2f}  ... Validation Loss: {:.2f} ... Test Loss: {:.2f}'.format(train_epoch_loss, validation_epoch_loss, test_epoch_loss))
        
        #Early stop if no validation improvement over set number of epochs
        if min_validation_counter > early_stop:
            print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
            break

    loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
    'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
    'min_test_loss':min_test_loss, 'min_test_std':min_test_std,
    'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'test_loss_array':test_loss_array, 'max_epochs':max_epochs}
    return loss_dict


#Dataset class to handle mocap dataframes from SEE project
class SEE_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, cv_dict, fold, partition, kinematic_df, neural_df, offset, window_size, data_step_size, device,
                 kinematic_type='posData', scale_neural=True, scale_kinematics=True, flip_outputs=False,
                 exclude_neural=None, exclude_kinematic=None, neural_scaler=None, kinematic_scaler=None):
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

        # Boolean array of 1's for features to not be scaled
        if scale_kinematics:
            self.posData_list = self.transform_data(self.posData_list, self.kinematic_scaler, exclude_kinematic)
        
        if scale_neural:
            self.neuralData_list = self.transform_data(self.neuralData_list, self.neural_scaler, exclude_neural)

        self.kinematic_type = kinematic_type
        self.split_offset = np.round((self.offset/self.data_step_size) / 2).astype(int)

        self.X_tensor, self.y_tensor = self.load_splits()
        self.num_samples = np.sum(self.X_tensor.size(0))

    def __len__(self):
        #'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, slice_index):
        return self.X_tensor[slice_index,:,:], self.y_tensor[slice_index,:,:]

    #**add functionality to separate eye, object, and body markers
    def process_dfs(self, kinematic_df, neural_df):
        posData_list, neuralData_list = [], []
        for trial in self.trial_idx:
            posData_array = np.stack(kinematic_df['posData'][kinematic_df['trial'] == trial].values).transpose() 
            neuralData_array = np.stack(neural_df['rates'][neural_df['trial'] == trial].values).squeeze().transpose() 

            posData_list.append(posData_array)
            neuralData_list.append(neuralData_array)

        return posData_list, neuralData_list

    def format_splits(self, data_list):
        unfolded_data_list = list()
        for trial_idx in range(self.num_trials):
            if self.window_size == 1:
                padded_trial = torch.from_numpy(data_list[trial_idx])
            else:
                padded_trial = torch.nn.functional.pad(
                    torch.from_numpy(data_list[trial_idx].T), pad=(self.window_size, 0)).transpose(0, 1)
            
            unfolded_trial = padded_trial.unfold(0, self.window_size, self.data_step_size).transpose(1, 2)
            unfolded_data_list.append(unfolded_trial)
        
        data_tensor = torch.concat(unfolded_data_list, axis=0)

        return data_tensor
    
    def load_splits(self):
        if not self.flip_outputs:
            X_tensor = self.format_splits(self.posData_list)
            y_tensor = self.format_splits(self.neuralData_list)
        else:
            y_tensor = self.format_splits(self.posData_list)
            X_tensor = self.format_splits(self.neuralData_list)

        X_tensor, y_tensor = X_tensor[:,:-self.split_offset:self.data_step_size,:], y_tensor[:,self.split_offset::self.data_step_size,:]
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

# Utility function to load dataframes of preprocessed kinematic/neural data
def load_mocap_df(data_path, kinematic_suffix=None):
    kinematic_df = pd.read_pickle(f'{data_path}kinematic_df{kinematic_suffix}.pkl')
    neural_df = pd.read_pickle(data_path + 'neural_df.pkl')

    # read python dict back from the file
    metadata_file = open(data_path + 'metadata.pkl', 'rb')
    metadata = pickle.load(metadata_file)
    metadata_file.close()

    return kinematic_df, neural_df, metadata


#Vectorized correlation coefficient of two matrices on specified dimension
def matrix_corr(x, y, axis=0):
    num_tpts, _ = np.shape(x)
    mean_x, mean_y = np.tile(np.mean(x, axis=axis), [num_tpts,1]), np.tile(np.mean(y, axis=axis), [num_tpts,1])
    corr = np.sum(np.multiply((x-mean_x), (y-mean_y)), axis=axis) / np.sqrt(np.multiply( np.sum((x-mean_x)**2, axis=axis), np.sum((y-mean_y)**2, axis=axis) ))
    return corr

#Helper function to evaluate decoding performance on a trained model
def evaluate_model(model, generator, device):
    #Run model through test set
    with torch.no_grad():
        model.eval()
        #Generate predictions
        y_pred_tensor = torch.zeros(len(generator.dataset),  generator.dataset[0][1].shape[1])
        batch_idx = 0
        for batch_x, batch_y in generator:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x)
            y_pred_tensor[batch_idx:(batch_idx+output.size(0)),:] = output[:,-1,:]
            batch_idx += output.size(0)

    y_pred = y_pred_tensor.detach().cpu().numpy()
    return y_pred

def make_generators(pred_df, neural_df, neural_offset, cv_dict, metadata,
                    exclude_neural=None, exclude_kinematics=None, window_size=1, 
                    flip_outputs=False, fold=0, batch_size=10000, device='cpu',):
    sampling_rate = 100
    kernel_offset = int(metadata['kernel_halfwidth'] * sampling_rate)  #Convolution kernel centered at zero, add to neural offset
    offset = neural_offset + kernel_offset
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
                               exclude_neural=exclude_neural, exclude_kinematic=exclude_kinematics)
    training_neural_scaler = training_set.neural_scaler
    training_kinematic_scaler = training_set.kinematic_scaler

    training_generator = torch.utils.data.DataLoader(training_set, **train_params)
    training_eval_generator = torch.utils.data.DataLoader(training_set, **train_eval_params)

    validation_set = SEE_Dataset(cv_dict, fold, 'validation_idx', pred_df, neural_df, offset, window_size, 
                                 data_step_size, device, 'posData', scale_neural=scale_neural,
                                 scale_kinematics=scale_kinematics, flip_outputs=flip_outputs,
                                 exclude_neural=exclude_neural, exclude_kinematic=exclude_kinematics,
                                 neural_scaler=training_neural_scaler, kinematic_scaler=training_kinematic_scaler)
    validation_generator = torch.utils.data.DataLoader(validation_set, **validation_params)

    testing_set = SEE_Dataset(cv_dict, fold, 'test_idx', pred_df, neural_df, offset, window_size, 
                              data_step_size, device, 'posData', scale_neural=scale_neural,
                              scale_kinematics=scale_kinematics, flip_outputs=flip_outputs,
                              exclude_neural=exclude_neural, exclude_kinematic=exclude_kinematics,
                              neural_scaler=training_neural_scaler, kinematic_scaler=training_kinematic_scaler)
    testing_generator = torch.utils.data.DataLoader(testing_set, **test_params)

    data_arrays = (training_set, validation_set, testing_set)
    generators = (training_generator, training_eval_generator, validation_generator, testing_generator)

    return data_arrays, generators

def make_movie(image_folder, output_name, images, fps=30, quality=10):
    """Turn folder of images into movie.
    Parameters
    ----------
    image_folder : str
        Path where images are stored.
    output_name : str
        Name of video file.
    images : list of str
        File names of images stored in `image_folder`.
        Order of images determines sequence of frames.
    fps : int | None
        Frames per second. Default: 30.
    quality : int | None
        Quality of output video. Default: 10.
    """
    writer = imageio.get_writer(output_name, fps=fps, quality=quality)
    for filename in images:
        image_path = str(image_folder) + str(filename)
        writer.append_data(imageio.imread(image_path))
    writer.close()

#GRU architecture for decoding kinematics
class model_gru_categorical(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device, bidirectional=False,
                 cat_features=None):
        super(model_gru, self).__init__()

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
            # self.fc1 = nn.Linear((self.hidden_dim* num_directions) + self.num_cat_features, self.hidden_dim)

            
        else:
            self.fc = nn.Linear(self.hidden_dim * num_directions, output_size)
        #     self.fc1 = nn.Linear((self.hidden_dim* num_directions), self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, output_size)

        self.fc = nn.Linear((self.hidden_dim* num_directions), output_size)
        self.gru = nn.GRU(self.input_size, self.hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional) 

      

        #Defining the layers
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        if self.cat_features is not None:
            cat_hidden = self.hidden_fc(torch.tanh(x[:, -1, self.cat_features]))
            hidden = hidden + cat_hidden
            out, hidden = self.gru(x[:, :, ~self.cat_features], hidden)
            out = out.contiguous()
            # out = torch.cat([out, x[:, :, self.cat_features]], 2)

        else:
            out, hidden = self.gru(x, hidden)
            out = out.contiguous()
        
        out = self.fc(out)
        # out = self.fc1(torch.tanh(out))
        # out = self.fc2(out)
             
        # if self.cat_features is not None:
        # #     out = torch.cat([out, x[:, :, self.cat_features]], 2)
        # out = self.fc1(torch.tanh(out))
        # out = self.fc2(out)


        return out
    
    def init_hidden(self, batch_size):
        # weight = next(self.parameters()).data
        # a = weight.new(self.n_layers, batch_size, 1).normal_(-1,1)
        # b = weight.new(self.n_layers, batch_size, self.hidden_dim-1).zero_()  
        # return torch.autograd.Variable(torch.cat([a,b], 2))

        weight = next(self.parameters()).data.to(self.device)

        #GRU initialization
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)

        return hidden