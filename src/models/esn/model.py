import pandas as pd
import torch
import torch.sparse
import torch.nn as nn
import numpy as np
import scipy
from dataset import index_agreement_torch, rmse_torch


# Echo state base class
class ESN(nn.Module):

    def __init__(self, **kwargs):
        super(ESN, self).__init__()
        self.load_dict(**kwargs)
        self.training_samples = 0

        self.reservoir_state = torch.zeros(self.reservoir_size, dtype=self.torch_type, device=self.device)

        if self.with_bias:
            self.size = self.reservoir_size + 1
        else:
            self.size = self.reservoir_size

        # ESN matrix generations

        self.W = torch.tensor(
            self.generate_reservoir_matrix(self.reservoir_size, self.connectivity, self.spectral_radius),
            dtype=self.torch_type, device=self.device).to_sparse()
        self.W_bias = torch.tensor(
            self.generate_norm_sparce_matrix(self.reservoir_size, 1, self.connectivity, self.bias_scaling),
            dtype=self.torch_type, device=self.device)
        self.W_input = torch.tensor(
            self.generate_norm_sparce_matrix(self.reservoir_size, self.input_dim, self.connectivity,
                                             self.input_scaling), dtype=self.torch_type, device=self.device)
        self.W_out = torch.squeeze(torch.zeros((self.output_dim, self.size), dtype=self.torch_type, device=self.device))

        # Ridge regression parameters
        # X = neuron states
        # Y = desired output
        # W_out = (XX^T - λI)^-1 XY

        self.XX_sum = torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device)  # Sum (XX^T)
        self.XY_sum = torch.zeros((self.size, self.output_dim), dtype=self.torch_type, device=self.device)  # Sum (XY)

    def load_dict(self, **kwargs):
        self.kwargs = kwargs
        self.input_dim = kwargs['input_dim']
        self.output_dim = kwargs['output_dim']
        self.reservoir_size = kwargs['reservoir_size']
        self.spectral_radius = kwargs['spectral_radius']
        self.bias_scaling = kwargs['bias_scaling']
        self.input_scaling = kwargs['input_scaling']
        self.ridge_parameter = kwargs['ridge_parameter']
        self.connectivity = kwargs['connectivity']
        if 'torch_type' in kwargs.keys():
            self.torch_type = kwargs['torch_type']
        else:
            self.torch_type = torch.float32
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        if 'activation' in kwargs.keys():
            self.activation = kwargs['activation']
        else:
            self.activation = torch.tanh
        if 'loss' in kwargs.keys():
            self.loss = kwargs['loss']
        else:
            self.loss = index_agreement_torch
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        if 'sparse' in kwargs.keys():
            self.sparse = kwargs['sparse']
        else:
            self.sparse = True
        if 'with_bias' in kwargs.keys():
            self.with_bias = kwargs['with_bias']
        else:
            self.with_bias = True

    def generate_reservoir_matrix(self, size, density, spectral_radius):
        distribution = scipy.stats.norm().rvs
        matrix = scipy.sparse.random(size, size, density, data_rvs=distribution).toarray()
        largest_eigenvalue = np.max(np.abs(np.linalg.eigvals(matrix)))
        matrix = matrix * spectral_radius / largest_eigenvalue
        return matrix

    def generate_norm_sparce_matrix(self, rows, columns, density, scaling):
        distribution = scipy.stats.norm().rvs
        matrix = scipy.sparse.random(rows, columns, density, data_rvs=distribution).toarray()
        matrix = matrix * scaling
        return np.squeeze(matrix)

    def save(self, name):
        torch.save((self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples,
                    self.kwargs), name)

    def load(self, name):
        self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples, kwargs = torch.load(
            name, map_location=self.device)
        self.load_dict(**kwargs)

    def forward(self, x):

        self.reservoir_state = self.activation(
            torch.mul(input=self.W_input, other=x.cuda()) + self.W_bias + torch.matmul(self.W, self.reservoir_state))
        output = torch.matmul(input=self.W_out, other=torch.unsqueeze(self.reservoir_state, 1))

        return torch.squeeze(output)

    def load_matrices(self, w, w_in, w_bias, w_out):

        wv = torch.load(w, weights_only=True)
        w_inv = torch.load(w_in, weights_only=True)
        w_bv = torch.load(w_bias, weights_only=True)
        w_ov = torch.load(w_out, weights_only=True)
        if w_inv.shape[0] == w_bv.shape[0] == wv.shape[0] == wv.shape[1]:
            if w_ov.shape[1] == wv.shape[1]:
                self.with_bias = False
            elif w_ov.shape[1] == wv.shape[1] + 1:
                self.with_bias = True
            else:
                return
            self.input_dim = min([min(w_inv.shape),1])
            self.reservoir_size = wv.shape[0]
            self.size = w_ov.shape[1]
            self.W = torch.tensor(wv.clone().detach(), device=self.device, dtype=self.torch_type)
            self.W_bias = torch.tensor(w_bv.clone().detach(), device=self.device, dtype=self.torch_type)
            self.W_input = torch.squeeze(
                torch.tensor(w_inv.clone().detach(), device=self.device, dtype=self.torch_type))
            self.W_out = torch.tensor(w_ov.clone().detach(), device=self.device, dtype=self.torch_type)
        else:
            print("Inconsistent matrix sizes while trying to load tensors from files")
            return
        print('ok')

    def save_weights(self, prefix):
        torch.save(self.W_input, prefix + '_w_in')
        torch.save(self.W, prefix + '_w')
        torch.save(self.W_bias, prefix + '_w_bias')
        torch.save(self.W_out, prefix + '_w_out')

    def load_weights(self, prefix):
        self.W_input = torch.load(prefix + '_w_in', weights_only=True)
        self.W = torch.load(prefix + '_w', weights_only=True)
        self.W_bias = torch.load(prefix + '_w_bias', weights_only=True)
        self.W_out = torch.load(prefix + '_w_out', weights_only=True)

    def save_parquet(self, path, predictions):
        for idp, p in enumerate(predictions):
            if p is not None:
                ts, pp = p # ts = time series, pp = prediction pairs
                pred = [np.array(c[:,0].cpu()) for c in pp]
            else:
                pred =list()
            df = pd.DataFrame(data=pred)
            df = df.transpose()
            df.to_parquet(path + '/' + str(idp) + '.parquet')

    def predict(self, x, n):
        pass

    def predict_batches(self, batches, future_steps, calc_loss=True):
        pass

    def train_batch(self, x, y):
        pass

    def reset_internal_state(self):
        self.reservoir_state.fill_(0.0)

    def train_epoch(self, dataloader):
        pass

    def get_ridge_sums(self):
        return self.XX_sum, self.XY_sum, self.training_samples

    def set_ridge_sums(self, XX_sum, XY_sum, n_samples):
        pass

    def train_finalize(self):
        pass

# Leaky-Integrator ESN
class LiESN(ESN):
    def __init__(self, **kwargs):
        super(LiESN, self).__init__(**kwargs)
        self.leak_rate = torch.tensor(kwargs['leak_rate'], dtype=self.torch_type, device=self.device)

    def load_dict(self, **kwargs):
        super().load_dict(**kwargs)
        self.leak_rate = torch.tensor(kwargs['leak_rate'], dtype=self.torch_type, device=self.device)

    def forward(self, x, *args, **kwargs):
        self.reservoir_state = torch.mul(
            self.activation(
                torch.mul(input=self.W_input, other=x) + self.W_bias + torch.matmul(self.W, self.reservoir_state)),
            self.leak_rate) \
                               + torch.mul(self.reservoir_state, 1 - self.leak_rate)
        output = torch.matmul(input=self.W_out, other=torch.unsqueeze(self.reservoir_state, 1))

        return torch.squeeze(output)


# Single reservoir LiESN with explicit time step
class TCLiESN(LiESN):
    def __init__(self, **kwargs):
        super(TCLiESN, self).__init__(**kwargs)
        if 'time_constant' in kwargs.keys():
            self.tc = torch.tensor(kwargs['time_constant'], dtype=self.torch_type, device=self.device)
        else:
            self.tc = torch.tensor(1.0, dtype=self.torch_type, device=self.device)
        if 'output_map' in kwargs.keys():
            self.out_maps = kwargs['output_map']  # Relation between the outputs and inputs
            self.inp_maps = np.empty(self.input_dim)  # Relation between the inputs and outputs
            self.inp_maps[:] = np.nan
            for idx, value in enumerate(self.out_maps):
                self.inp_maps[value] = idx
        elif 'out_maps' in kwargs.keys():
            self.out_maps = kwargs['out_maps']
            self.inp_maps = range(self.input_dim)
        else:
            self.out_maps = range(self.input_dim)
            self.inp_maps = range(self.input_dim)
        self.training_samples = np.zeros(self.output_dim)
        self.XX_sums = list()
        self.XY_sums = list()
        for idx in range(self.output_dim):
            self.XX_sums.append(torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device))
            self.XY_sums.append(torch.zeros((self.size, 1), dtype=self.torch_type, device=self.device))
        if self.with_bias:
            self.extended_states = torch.cat(
                (torch.tensor([1.0], dtype=self.torch_type, device=self.device), self.reservoir_state))
        else:
            self.extended_states = self.reservoir_state.clone().detach()
        self.ridges = None
        print('Using LiESN with leakage only on previous states')

    def define_is_size(self, size):
        self.size = size
        self.reset()

    def define_heterogeneous_ridge(self, own, others):
        self.ridges = own[1] * torch.ones(own[0] + self.with_bias, device=self.device, dtype=self.torch_type)
        for other in others:
            self.ridges = torch.cat(
                (self.ridges, other[1] * torch.ones(other[0], device=self.device, dtype=self.torch_type)))

    def save(self, name):
        torch.save((self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples,
                    self.inp_maps, self.out_maps, self.tc, self.kwargs), name)

    def load(self, name):
        self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples, self.inp_maps, \
        self.out_maps, self.tc, kwargs = torch.load(name, map_location=self.device)
        self.load_dict(**kwargs)

    # forward_w_extended_states(x, dt, extended_states)
    def forward_w_extended_states(self, x, *args, **kwargs):
        dt = args[0] / self.tc
        extended_states = args[1]

        while dt > 1.0:
            self.reservoir_state = self.activation(torch.mul(self.W_input, x) + self.W_bias + torch.matmul(self.W, self.reservoir_state)) \
                                   + torch.mul(self.reservoir_state, 1 - self.leak_rate)
            dt = dt - 1.0
        self.reservoir_state = torch.mul(self.activation(torch.mul(self.W_input, x) + self.W_bias + torch.matmul(self.W, self.reservoir_state)),
            dt) + torch.mul(self.reservoir_state, 1 - dt * self.leak_rate)
        # self.reservoir_state[-2:] = x.detach().clone()
        if self.with_bias:
            self.extended_states = torch.cat(
                (torch.tensor([1.0], dtype=self.torch_type, device=self.device), self.reservoir_state, extended_states))
        else:
            self.extended_states = torch.cat((self.reservoir_state, extended_states))

        output = torch.matmul(input=self.W_out, other=self.extended_states)

        return output

    def forward(self, x, *args, **kwargs):
        dt = args[0] / self.tc
        if self.input_dim == 1:
            inputs = self.W_input * x
        else:
            inputs = self.W_input @ x
        while dt > 1.0:
            self.reservoir_state = self.activation(inputs + self.W_bias + torch.matmul(self.W, self.reservoir_state)) \
                                   + torch.mul(self.reservoir_state, 1 - self.leak_rate)
            dt = dt - 1.0
        self.reservoir_state = (torch.mul(self.activation(inputs + self.W_bias + torch.matmul(self.W, self.reservoir_state)), dt)
                                + torch.mul(self.reservoir_state, 1 - dt * self.leak_rate))

        if self.with_bias:
            self.extended_states = torch.cat(
                (torch.tensor([1.0], dtype=self.torch_type, device=self.device), self.reservoir_state))
        else:
            self.extended_states = self.reservoir_state.clone().detach()

        output = self.W_out @ self.extended_states

        return output

    def train_epoch(self, dataloader):
        for idx, batch in enumerate(dataloader):
            self.train_batch(batch)
            print("Trained batch " + str(idx))
        print('finished_training')

    def train_batch(self, batch, throwaway=None):
        for element in batch:
            inp, out = element
            dt = torch.tensor(inp[1], dtype=self.torch_type, device=self.device)
            self.forward(inp[0].clone().detach(), dt)
            out_value, predict, time_out = out
            for idx in range(self.output_dim):
                if predict[idx]:
                    self.add_train_point(idx, torch.tensor([out_value[idx]], dtype=self.torch_type, device=self.device))
                if torch.isnan(self.XX_sums[idx]).any() or torch.isnan(self.XY_sums[idx]).any():
                    raise ("ERROR: NAN DETECTED DURING RIDGE REGRESSION")
        self.reset_internal_state()

    def add_train_point(self, idx, y):
        self.XX_sums[idx] = self.XX_sums[idx] + torch.matmul(torch.unsqueeze(self.extended_states, 1),
                                                             torch.unsqueeze(self.extended_states, 1).t()).t()
        self.XY_sums[idx] = self.XY_sums[idx] + torch.matmul(torch.unsqueeze(self.extended_states, 1),
                                                             torch.unsqueeze(y.t(), 1))
        self.training_samples[idx] = self.training_samples[idx] + 1

    def train_finalize(self):
        Wout = list()
        for idx in range(self.output_dim):
            self.XX_sums[idx] = self.XX_sums[idx] / self.training_samples[idx]
            self.XY_sums[idx] = self.XY_sums[idx] / self.training_samples[idx]
            eye = torch.mul(input=torch.eye(self.size, device=self.device), other=self.ridge_parameter)
            if self.ridges is not None:
                eye[range(self.ridges.shape[0]), range(self.ridges.shape[0])] = self.ridges
            try:
                xx_in = (self.XX_sums[idx] + eye).inverse()
            except RuntimeError:
                print('Ridge Regression: matrix has no inverse! Using pseudoinverse instead')
                xx_in = (self.XX_sums[idx] + eye).pinverse()
            Wout.append(torch.matmul(input=xx_in, other=self.XY_sums[idx]).t())
        self.W_out = torch.cat(tuple(Wout))
        self.W_out = self.W_out.clone().detach()

    def reset(self):
        self.W_out = torch.squeeze(
            torch.zeros((self.output_dim, self.size), dtype=self.torch_type, device=self.device))
        self.reset_internal_state()
        self.training_samples = np.zeros(self.output_dim)
        self.XX_sums = list()
        self.XY_sums = list()
        for idx in range(self.output_dim):
            self.XX_sums.append(torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device))
            self.XY_sums.append(torch.zeros((self.size, 1), dtype=self.torch_type, device=self.device))

    @torch.no_grad()
    def predict_batches(self, batches, forecast_horizon, warmup, calc_loss=True, return_rmse = False):
        predictions = list()
        losses = list()
        rmses = list()
        inputs = list()
        for idb, batch in enumerate(batches):
            # print('batch: ' + str(idx))
            inp, prediction, comparison_pairs = self.predict(batch, warmup)
            valid = True
            if calc_loss:
                loss = list()
                rmse = list()
                for idx in range(self.output_dim):
                    if len(comparison_pairs[idx]) > 0:
                        loss.append(self.loss(comparison_pairs[idx][:, 0], comparison_pairs[idx][:, 1]).cpu().numpy())
                        rmse.append(rmse_torch(comparison_pairs[idx][:, 0], comparison_pairs[idx][:, 1]).cpu().numpy())
                    else:
                        valid = False
                        loss.append(
                            torch.tensor(np.nan, device=torch.device('cuda'), dtype=torch.float32).cpu().numpy())
                        rmse.append(
                            torch.tensor(np.nan, device=torch.device('cuda'), dtype=torch.float32).cpu().numpy())
                if valid:
                    losses.append(loss)
                    rmses.append(rmse)
            if valid:
                inputs.append(inp)
                predictions.append([prediction, comparison_pairs])
                print("Predicted batch: " + str(idb))
            else:
                print('Droped batch: ' + str(idb) + ' due to lack of data in the forecasting period')
                nanarray = np.empty(self.output_dim)
                nanarray[:] = np.nan
                inputs.append(None)
                predictions.append(None)
                losses.append(nanarray)
                rmses.append(nanarray)
        if return_rmse :
            return inputs, predictions, losses, rmses
        else:
            return inputs, predictions, losses

    @torch.no_grad()
    def predict(self, batch, warmup=7*24, return_internal_states=False):
        output = list()
        inputs = list()
        internal_states = list()
        for idx in range(self.output_dim):
            internal_states.append(list())
        batch.set_warmup(pd.to_timedelta(warmup, unit='hours'))
        LiESN_prediction = None
        prediction_gt_pairs = list()
        self.reset_internal_state()
        for idx in range(self.output_dim):
            prediction_gt_pairs.append(list())
        broi = torch.zeros(self.input_dim, dtype=self.torch_type, device=self.device) # broi = Best representation of inputs
        for element in batch:
            inp, out = element
            dt = torch.tensor(inp[1], dtype=self.torch_type, device=self.device)
            dt_i = dt.clone().detach()
            #inputs.append((inp[0].copy(), dt_i.cpu().numpy(), inp[2].copy()))
            #broi = np.copy(inp[0])  # broi = Best representation of inputs
            inputs.append((inp[0].clone().detach(), dt_i.cpu().clone().detach(), inp[2]))
            for idx in range(self.input_dim):
                if torch.isnan(inp[0][idx]):
                    if LiESN_prediction is not None:
                        if not np.isnan(self.out_maps[idx]):
                            broi[idx] = LiESN_prediction[self.out_maps[idx]]
                    else:
                        broi[idx] = 0.0
                else:
                    broi[idx] = inp[0][idx]

            while dt_i > self.tc:
                LiESN_prediction = self.forward(broi.clone().detach(),
                                                torch.tensor(1.0, dtype=self.torch_type, device=self.device))
                dt_i = dt_i - self.tc
                for idx in range(len(self.out_maps)):
                    if not np.isnan(self.out_maps[idx]):
                        broi[idx] = LiESN_prediction[self.out_maps[idx]]

            LiESN_prediction = self.forward(broi.clone().detach(), dt_i)
            output.append([LiESN_prediction.detach().clone(), dt.detach().clone(), inp[2]])
            out_value, predict, time = out
            # print(str(predict) + ' - ' + str(inp[2]))
            for idx in range(self.output_dim):
                if predict[idx]:
                    prediction_gt_pairs[idx].append([LiESN_prediction[idx].detach().clone(),
                                                     out_value[idx].clone().detach()])
                if return_internal_states:
                    if any(predict):
                        internal_states[idx].append(self.reservoir_state.clone().detach())

        list_pred = list()
        for pred in prediction_gt_pairs:
            list_pred.append(torch.tensor(pred, dtype=self.torch_type, device=self.device))

        prediction_gt_pairs = tuple(list_pred)
        if return_internal_states:
            return inputs, output, prediction_gt_pairs, internal_states
        else:
            return inputs, output, prediction_gt_pairs

    @staticmethod
    def config_2_dict(**config):
        esn_dict = dict(config)

        esn_dict['spectral_radius'] = config.pop('spectral_radius_1')
        esn_dict['leak_rate'] = config.pop('leak_rate_1')
        esn_dict['reservoir_size'] = config.pop('reservoir_size_1')
        esn_dict['connectivity'] = config.pop('connectivity_1')
        esn_dict['input_scaling'] = config.pop('input_scaling_1')
        esn_dict['bias_scaling'] = config.pop('bias_scaling_1')
        esn_dict['time_constant'] = config.pop('time_constant')
        esn_dict['ridge_parameter'] = config.pop('ridge_parameter_1')
        esn_dict['seed'] = None
        esn_dict['sparse'] = None
        esn_dict['torch_type'] = None
        esn_dict["device"] = None
        esn_dict['input_dim'] = None
        esn_dict['output_dim'] = None
        esn_dict['out_maps'] = None
        esn_dict['input_map'] = None

        return esn_dict
