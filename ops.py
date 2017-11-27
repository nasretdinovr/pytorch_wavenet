import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
import math

def one_hot(input, quantization_channels, cuda=True):
    len = input.size()[0]
    ret = torch.zeros(1,len,quantization_channels)
    for i in range(len):
        if cuda:
            ret[0][i][input.cpu().data.numpy()[i]] = 1
        else:
            ret[0][i][input.data.numpy()[i]] = 1
    ret = torch.transpose(ret, 1, 2)
    if cuda:
        return autograd.Variable(ret).cuda()
    else:
        return autograd.Variable(ret)
    
def time_to_batch(value, dilation, name=None): 
    shape = value.size()
    pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
    value = value.unsqueeze(1)
    padded = F.pad(value, (0, 0, 0, pad_elements))
    padded = padded.squeeze(1)
    reshaped = padded.view(-1, dilation, shape[2])
    '''TODO: generalize for batch size  > 1'''
    transposed = torch.transpose(reshaped, 2, 0)
    transposed = torch.transpose(transposed,1, 0)
    return transposed

def batch_to_time(value, dilation, name=None):
    shape = value.size()
    transposed = torch.transpose(value, 1, 2)
    transposed = torch.transpose(transposed, 1, 0)
    reshaped = transposed.contiguous().view((shape[0] // dilation), -1, shape[1])
    return torch.transpose(reshaped, 1, 2)
    
    
def mu_law_encode(audio, quantization_channels, cuda=True):
    '''Quantizes waveform amplitudes.'''
    mu = (quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    if cuda:
        safe_audio_abs = torch.min(torch.abs(audio), autograd.Variable(torch.ones(audio.size())).cuda())
    else:
        safe_audio_abs = torch.min(torch.abs(audio), autograd.Variable(torch.ones(audio.size())))
    magnitude = torch.log1p(safe_audio_abs * mu) / math.log1p(mu)
    signal = torch.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).type(torch.LongTensor)

def mu_law_decode(output, quantization_channels, cuda=True):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (output.float() / mu) - 1
    # Perform inverse of mu-law transformation."
    if cuda:
        magnitude = (1.0 / mu) * (torch.pow((autograd.Variable(torch.ones(signal.size()).float() *(1 + mu))), (torch.abs(signal))) -1)
    else:
        magnitude = (1.0 / mu) * (torch.pow((autograd.Variable(torch.ones(signal.size()).float() *(1 + mu)).cuda()), (torch.abs(signal))) -1)
    return torch.sign(signal) * magnitude

def causal_conv(value, filter_, dilation):
    filter_width = filter_.size()[2]
    value_length = value.size()[2]
    if dilation > 1:
        value = torch.transpose(value, 1, 2)
        transposed = time_to_batch(value, dilation)
        conv = F.conv1d(transposed, filter_, stride=1)
        restored = batch_to_time(conv, dilation)
    else:
        #print ("value")
        #print (value.size())
        #print ("filter")
        #print (filter_.size())
        restored = F.conv1d(value, filter_, stride=1)
    # Remove excess elements at the end.
    out_width = value_length - (filter_width - 1) * dilation
    result = restored[:, :, :out_width]
    return result
