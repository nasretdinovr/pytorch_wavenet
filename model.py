import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import autograd
from ops import mu_law_encode, time_to_batch, batch_to_time
import Queue
import time
import math


class WaveNetModel(nn.Module):
    
    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 num_classes=2,
                 quantization_channels=2**8,
                 use_biases=True,
                 scalar_input=False,
                 initial_filter_width=32,
                 use_cuda = True):

        super(WaveNetModel, self).__init__()
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.use_cuda = use_cuda
        self.num_classes = num_classes
        
        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        
        self.causal_conv_layer = nn.Conv1d(
             self.quantization_channels,
             self.residual_channels,
             self.filter_width)
        
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dences = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        
        for _ in xrange (len(dilations)):
            self.filters.append(nn.Conv1d(
                 self.residual_channels,
                 self.dilation_channels,
                 self.filter_width))
            self.gates.append(nn.Conv1d(
                 self.residual_channels,
                 self.dilation_channels,
                 self.filter_width))
            self.dences.append(nn.Conv1d(
                 self.dilation_channels,
                 self.residual_channels,
                 1))
            self.skips.append(nn.Conv1d(
                 self.dilation_channels,
                 self.skip_channels,
                 1))

        self.postprocess1 = nn.Conv1d(
        self.skip_channels, self.skip_channels, 1)
        self.postprocess2 = nn.Conv1d(
            self.skip_channels, self.quantization_channels, 1)
        
        self.queue = []
        self.queue_0 = Queue.Queue(1)
        if self.use_cuda:
            self.queue_0.put(autograd.Variable(torch.zeros(self.batch_size, self.quantization_channels).cuda()))
        else:
            self.queue_0.put(autograd.Variable(torch.zeros(self.batch_size, self.quantization_channels)))
            
        for d in dilations:
            q = Queue.Queue(d)
            for i in range(d):
                if self.use_cuda:
                    q.put(autograd.Variable(torch.zeros(self.batch_size, self.residual_channels).cuda()))
                else:
                    q.put(autograd.Variable(torch.zeros(self.batch_size, self.residual_channels)))
            self.queue.append(q)
                          
    @staticmethod        
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field
    
    '''def one_hot_2(self, input):
        len = input.size()
        ret = torch.zeros(len[0], len[1], self.quantization_channels)
        for batch_idx in range(len[0]):
            for i in range(len[1]):
                if self.cuda:
                    ret[batch_idx][i][input.cpu().data.numpy()[batch_idx, i]] = 1
                else:
                    ret[batch_idx][i][input.data.numpy()[batch_idx, i]] = 1
        ret = torch.transpose(ret, 1, 2)
        if self.cuda:
            return autograd.Variable(ret).cuda()
        else:
            return autograd.Variable(ret)'''
    
    def one_hot(self, input):
        size = input.size()
        ret = torch.zeros(size[0]*size[1], self.quantization_channels)
        input = input.view(-1,1)
        ret.scatter_(1, input.data, 1)
        ret = ret.view(size[0], size[1], self.quantization_channels).transpose_(1, 2)
        return autograd.Variable(ret).cuda()
    
    def causal_conv(self, value, layer,  dilation):
        value_length = value.size()[2]
        if dilation > 1:
            value = torch.transpose(value, 1, 2)
            transposed = time_to_batch(value, dilation)
            conv = layer(transposed)
            restored = batch_to_time(conv, dilation)
        else:
            restored = layer(value)
        # Remove excess elements at the end.
        out_width = value_length - (self.filter_width - 1) * dilation
        result = restored[:, :, :out_width]
        return result
    
    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        layer = self.causal_conv_layer
        return self.causal_conv(input_batch, layer, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                              output_width ):
        layer_filter = self.filters[layer_index]
        layer_gate = self.gates[layer_index]
        conv_filter = self.causal_conv(input_batch, layer_filter, dilation)
        conv_gate = self.causal_conv(input_batch, layer_gate, dilation)
        

        out = F.tanh(conv_filter) * F.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        layer_dense = self.dences[layer_index]
        transformed = layer_dense(out)

        # The 1x1 conv to produce the skip outputs
        skip_cut = out.size()[2] - output_width

        out_skip = out[:,:,skip_cut:]
        #print (out_skip)
        layer_skip = self.skips[layer_index]
        skip_contribution = layer_skip(out_skip)

        input_cut = input_batch.size()[2] - transformed.size()[2]
 
        input_batch = input_batch[:,:,input_cut:]

        return skip_contribution, input_batch + transformed
    
    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2

        weights=torch.transpose(weights,0,1)
        past_weights = weights[:, :, 0]
        curr_weights = weights[:, :, 1]
        output = torch.matmul(state_batch, past_weights) + torch.matmul(
             input_batch, curr_weights)
        return output

        
    def predict_proba_incremental(self, waveform):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        
        encoded = one_hot(waveform, self.quantization_channels,self.use_cuda)
        encoded = encoded.contiguous().view(self.quantization_channels,-1).t()
        raw_output = self._create_generator(encoded)
        out = raw_output.view(-1, self.quantization_channels)
        proba = F.softmax(out.type(torch.DoubleTensor)).type(torch.FloatTensor)
        last = proba[-1]
        return last

    
    def _generator_causal_layer(self, input_batch, state_batch):
        weights_filter = self.causal_conv_layer.weight
        output = self._generator_conv(
            input_batch, state_batch, weights_filter)
        return output
    
    
    
    
    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation):
        weights_filter = self.filters[layer_index].weight
        weights_gate = self.gates[layer_index].weight
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        out = F.tanh(output_filter) * F.sigmoid(output_gate)

        weights_dense = self.dences[layer_index].weight       
        weights_dense=torch.transpose(weights_dense,0,1)
        transformed = torch.matmul(out, weights_dense[:, :, 0])
    
        transformed = transformed.t()

        weights_skip = self.skips[layer_index].weight     
        weights_skip=torch.transpose(weights_skip,0,1)
        skip_contribution = torch.matmul(out, weights_skip[:, :, 0])
        skip_contribution = skip_contribution.t()

        return skip_contribution, input_batch + transformed
    
    
    
    def _create_generator(self, input_batch):
        '''Construct an efficient incremental generator.'''
        outputs = []
        current_layer = input_batch
        

        q = self.queue_0
        
        current_state = q.get()
        q.put(current_layer)        
        
        current_layer = self._generator_causal_layer(
                            current_layer, current_state)

        # Add all defined dilation layers.
        for layer_index, dilation in enumerate(self.dilations):

            q = self.queue[layer_index]
            
            current_state = q.get()
            q.put(current_layer)

            output, current_layer = self._generator_dilation_layer(
                current_layer, current_state, layer_index, dilation)
            outputs.append(output)

        # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
        # postprocess the output.
        w1 = self.postprocess1.weight
        w2 = self.postprocess2.weight
        if self.use_biases:
            b1 = self.postprocess1.bias
            b2 = self.postprocess2.bias

        # We skip connections from the outputs of each layer, adding them
        # all up here.
        total = sum(outputs)
        transformed1 = F.relu(total)
        conv1 = torch.matmul(transformed1.t(), w1[:, :, 0].t())
        if self.use_biases:
            conv1 = conv1 + b1
        transformed2 = F.relu(conv1)
        conv2 = torch.matmul(transformed2, w2[:, :, 0].t())
 
        if self.use_biases:
            conv2 = conv2 + b2
        
        return conv2

    
    def _create_network(self, input_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch
        
        # Pre-process the input with a regular convolution
        if self.scalar_input:
            initial_channels = 1
        else:
            initial_channels = self.quantization_channels

        current_layer = self._create_causal_layer(current_layer)

        output_width = input_batch.size()[2] - self.receptive_field + 1

        # Add all defined dilation layers.
        for layer_index, dilation in enumerate(self.dilations):
            output, current_layer = self._create_dilation_layer(
                current_layer, layer_index, dilation, output_width)
            outputs.append(output)

        # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
        # postprocess the output.
        l1 = self.postprocess1
        l2 = self.postprocess2

        # We skip connections from the outputs of each layer, adding them
        # all up here.
        total = sum(outputs)
        transformed1 = F.relu(total)
        conv1 = l1(transformed1)
        transformed2 = F.relu(conv1)
        conv2 = l2(transformed2)
        return conv2
    
    def accurancy(self,
             input_batch,
             targets):
        prediction = self._prediction(input_batch)
        acc_val = (prediction.cpu().data.numpy() == targets.cpu().data.numpy()).mean()
        return acc_val
    
    def _prediction(self,input_batch):
        raw_output = self.forward(input_batch)
        loss = F.log_softmax(raw_output)
        _, index = torch.max(loss,1)
        return index
        
    def forward(self, input_batch):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
            # We mu-law encode and quantize the input audioform.
        encoded_input = mu_law_encode(input_batch,
                                      self.quantization_channels)


        encoded = self.one_hot(encoded_input)
        if self.scalar_input:
            network_input = input_batch.type(torch.FloatTensor).view(self.batch_size, -1, 1)
        else:
            network_input = encoded

        # Cut off the last sample of network input to preserve causality.
        network_input_width = network_input.size()[2] - 1
        network_input = network_input[:, :, :network_input_width]
        
        
        raw_output = self._create_network(network_input)
        # Cut off the samples corresponding to the receptive field
        # for the first predicted sample.
        raw_output = torch.transpose(raw_output, 1, 2).view(-1, self.num_classes)
        return raw_output
    
    def _loss(self,raw_output, targets):
        loss = F.cross_entropy(raw_output, targets)
        return loss