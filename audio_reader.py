import fnmatch
import os
import random
import re
import threading
import Queue

import librosa
import numpy as np
import torch

FILE_PATTERN = r'([0-9]*)/audio([0-9]*)\.wav'

def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate, amount):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    randomized_files = randomize_files(files)
    for it, filename in enumerate(randomized_files):
        if it == amount:
            break
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            folder_num = int(ids[0][0])
            file_num = int(ids[0][1])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        targets_filename = directory+'/targets/{}/audio{}.txt'.format(folder_num,file_num)
        targets = np.loadtxt(targets_filename, delimiter='\n')
        yield audio, targets


def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 sample_rate,
                 receptive_field,
                 load_size=1):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.receptive_field = receptive_field
        self.data_set = Queue.Queue()
        self.target_queue = Queue.Queue()
        self.load_size = load_size


        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def thread_main(self):
        # Go through the dataset multiple times
        
        iterator = load_generic_audio(self.audio_dir, self.sample_rate, self.load_size)
        for audio, targets  in iterator:
            
            audio = audio.reshape(-1, 1)
            
            # Cut samples into pieces of size receptive_field
            size = len(audio)/self.receptive_field 
            rand = np.arange(size)
            np.random.shuffle(rand)
            piece = np.zeros((self.receptive_field+1, 1))

            for i in range(size):
                if rand[i] == 0:
                    piece = audio[self.receptive_field*rand[i]:self.receptive_field*(rand[i]+1)+1, :]
                else:
                    piece = audio[self.receptive_field*rand[i]-1:self.receptive_field*(rand[i]+1), :]
                self.target_queue.put(targets[rand[i]])
                self.data_set.put(piece)
        return self.data_set.qsize()