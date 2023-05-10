#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:10:37 2021

This script is for de novo generation of novel antimicrobial peptides.

The RNN language model was adapted from the pytorch tutorial "NLP from 
scratch: generating names with a character-level RNN" by Sean Robertson.
(https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

@author: Chenkai Li
"""


import os
import argparse
from textwrap import dedent
import time
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import torch.nn as nn


NUM_AA = 21 # 20 standard amino acids + 1 EOS signal
MAX_LEN = 50 # maximum length of sequences
aa_all = 'ACDEFGHIKLMNPQRSTVWY'


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def inputTensor(seq):
    """
    Get the input tensor: one-hot encoding matrix for first to last amino acids
    """
    tensor = torch.zeros(len(seq), 1, NUM_AA)
    for i in range(len(seq)):
        aa = seq[i]
        tensor[i][0][aa_all.find(aa)] = 1
        
    return tensor


def targetTensor(seq):
    """
    Get the target tensor: long tensor for second amino acid to EOS
    """
    aa_ix = [aa_all.find(seq[i]) for i in range(1, len(seq))]
    aa_ix.append(NUM_AA - 1)
    
    return torch.LongTensor(aa_ix)


def randomTrainingSample(seq_lst):
    """
    Get a random training sample
    """
    seq = seq_lst[random.randint(0, len(seq_lst) - 1)]
    input_tensor = inputTensor(seq)
    target_tensor = targetTensor(seq)
    
    return input_tensor, target_tensor


def train(input_tensor, target_tensor, model):
    """
    Train the RNN model
    """
    loss_function = nn.NLLLoss()
    learning_rate = 0.0005
    target_tensor.unsqueeze_(-1)
    hidden = model.initHidden()
    
    model.zero_grad()
    loss = 0
    
    for i in range(input_tensor.size(0)):
        output, hidden = model(input_tensor[i], hidden)
        loss_temp = loss_function(output, target_tensor[i])
        loss = loss + loss_temp
        
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()/input_tensor.size(0)


def sample(model, first_aa):
    """
    Sample a peptide from the model given the N-terminal amino acid 
    """
    with torch.no_grad():
        input = inputTensor(first_aa)
        hidden = model.initHidden()

        output_seq = first_aa # initialize the sequence 
        output_score = 1.0 # geometric mean of probablities

        for i in range(MAX_LEN):
            output, hidden = model(input[0], hidden)
            top_score, top_index = output.topk(1)
            top_index = top_index[0][0]
            # If reaching EOS before or at position 51,
            # scores are calculated based on all generated amino acids + EOS;
            # else, scores are calculated based on all generated amino acids.
            if (i != MAX_LEN - 1) or (top_index == NUM_AA - 1):
                output_score = output_score * float(np.e**top_score[0][0])
            if top_index == NUM_AA - 1:
                break
            else:
                aa = aa_all[top_index]
                output_seq = output_seq + aa
            input = inputTensor(aa)
        
        if len(output_seq) > MAX_LEN:
            output_score = output_score**(1/(MAX_LEN-1))
        else:
            output_score = output_score**(1/len(output_seq))

        return output_seq, output_score


def generate(model):
    """
    Generate peptides from 20 possible N-terminal amino acids
    """
    output_seq_lst = []
    output_score_lst = []
    for aa in aa_all:
        out = sample(model, aa)
        output_seq_lst.append(out[0])
        output_score_lst.append(out[1])
        
    return output_seq_lst, output_score_lst


def compute_length(seq_lst):
    """
    Compute the length of a list of sequences
    """
    length_lst = [len(seq) for seq in seq_lst]
    
    return length_lst


def compute_charge(seq_lst):
    """
    Compute the charge of a list of sequences
    """
    charge_lst = []
    for seq in seq_lst:
        temp = seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')
        charge_lst.append(temp)
        
    return charge_lst


def truncate(seq_lst):
    """
    Truncate the sequences if they are longer than 50
    """
    seq_trunc = [] + seq_lst
    complete = []
    for i in range(len(seq_lst)):
        if len(seq_lst[i]) > 50:
            seq_trunc[i] = seq_lst[i][:50]
            complete.append('incomplete')
        else:
            complete.append('complete')
    
    return(seq_trunc, complete)
        
    
def compute_results(model):
    """
    Compute results generated from a model (20 peptides)
    """
    out_instance = generate(model)
    pept, complete = truncate(out_instance[0])
    score = out_instance[1]
    length = compute_length(pept)
    charge = compute_charge(pept)
    
    return pept, score, length, charge, complete
    

def check_positive_integer(inp):
    """
    Check if the input of an argument is a positive number.
    """
    message = '-n/--num_seq should be a positive integer!'
    try:
        inp = int(inp)
        if inp <= 0:
            raise argparse.ArgumentTypeError(message)
    except ValueError:
        raise Exception(message)
        
    return inp


def main():
    parser = argparse.ArgumentParser(description=dedent('''
        AMPd-Up v1.0.0
        --------------------------------------------------------------
        Generate antimicrobial peptide sequences with recurrent neural 
        network.
        Uses can either generate sequences by training new models or 
        from the exiting models.
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-fm', '--from_model', help="Directory of the existing models;\
                        only specify this argument if you want to sample from existing models (optional)", 
                        required=False)
    parser.add_argument('-n', '--num_seq', help="Number of sequences to sample", 
                        type=check_positive_integer, required=True)
    parser.add_argument('-sm', '--save_model', help="Prefix of the models if you want to save them;\
                        only specify this argument if you want to sample by training new models (optional)", 
                       required=False)
    parser.add_argument('-od', '--out_dir', help="Output directory (optional)", 
                        default=os.getcwd(), required=False)
    parser.add_argument('-of', '--out_format', help="Output format, fasta or tsv (tsv by default, optional)", 
                        choices=['fasta', 'tsv'], default='tsv', required=False)
    
    args = parser.parse_args()
    
    pept = []
    pept_id = []
    score = []
    charge = []
    length = []
    completeness = []

    n_model = int(np.ceil(args.num_seq/20)) # number of models required
    
    
    # if sample sequences from existing models
    if args.from_model is not None:
        
        print("Sampling from existing models. %s models required.\n"%n_model)
        
        model_names = []
        model_load = []
        
        # find all models in the specified directory ending with .pt
        for file in os.listdir(args.from_model):
            if file.endswith(".pt"):
                model_names.append(file)
        # load saved models in order
        # model_names = ['RNN_generative_20210709_%d.pt'%(i+1) for i in range(1000)]
                
        # check whether the number of models are enough
        if len(model_names) < n_model:
            n = len(model_names)*20
            warn = "Warning: Number of existing models not enough!" + \
            " Only %d sequences from %d models can be generated.\n"%(n, len(model_names))
            print(warn)
            model_load = [] + model_names
        else:
            model_load = [] + model_names[:n_model]
            
        # load the models and generate sequences
        for i in range(len(model_load)):
            print("Loading model %d from "%(i+1) + args.from_model + "/" + model_load[i] + "...")
            model = torch.load(args.from_model + '/' + model_load[i])
            model.eval()
            pept_temp, score_temp, length_temp, charge_temp, completeness_temp = compute_results(model)
            pept_id_temp = ['DeNo_' + model_load[i][:-3] + '_' + str(k+1).zfill(2) + 
                            '|length=%d'%length_temp[k] + '|charge=%d'%charge_temp[k] + 
                            '|score=%f'%score_temp[k] + '|%s'%completeness_temp[k] for k in range(20)]
            pept = pept + pept_temp
            score = score + score_temp
            length = length + length_temp
            charge = charge + charge_temp
            completeness = completeness + completeness_temp
            pept_id = pept_id + pept_id_temp
        
        
    # if sample sequences by training new models (default)
    else:
        
        print("Sampling by training new models. %s models required.\n"%n_model)
        
        # read training data
        train_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/data/training/APD3_ABP_20190320.fa'
        train_seq = []
        for seq_record in SeqIO.parse(train_dir, 'fasta'):
            train_seq.append(str(seq_record.seq))
        
        # train new models
        for i in range(n_model):
            
            # train the current model with n_itr iterations
            print("Training model %d..."%(i+1))
            model = RNN(NUM_AA, 128, NUM_AA)
            n_itr = 100000
            for itr in range(1, n_itr + 1):
                output, loss = train(*randomTrainingSample(train_seq), model) 

            # sample 20 sequences from the current model
            model.eval()
            pept_temp, score_temp, length_temp, charge_temp, completeness_temp = compute_results(model)
            
            # save the model and write the peptide IDs if -sm/--save_model is specified
            if args.save_model is not None:
                save_dir = args.out_dir + '/' + args.save_model + '_%d_.pt'%(i+1)
                torch.save(model, save_dir)
                print("Model %d saved to %s"%(i+1, save_dir))
                pept_id_temp = ['DeNo_' + args.save_model + '_' + str(k+1).zfill(2) + 
                                '|length=%d'%length_temp[k] + '|charge=%d'%charge_temp[k] + 
                                '|score=%f'%score_temp[k] + '|%s'%completeness_temp[k] for k in range(20)]
            else:
                pept_id_temp = ['DeNo_' + 'M%d'%(k+1) + '_' + str(k+1).zfill(2) + 
                                '|length=%d'%length_temp[k] + '|charge=%d'%charge_temp[k] + 
                                '|score=%f'%score_temp[k] + '|%s'%completeness_temp[k] for k in range(20)]
                
            pept = pept + pept_temp
            score = score + score_temp
            length = length + length_temp
            charge = charge + charge_temp
            completeness = completeness + completeness_temp
            pept_id = pept_id + pept_id_temp

            
    # take n sequences as specified by the user
    pept = [] + pept[:args.num_seq]
    score = [] + score[:args.num_seq]
    length = [] + length[:args.num_seq]
    charge = [] + charge[:args.num_seq]
    completeness = [] + completeness[:args.num_seq]
    pept_id = [] + pept_id[:args.num_seq]

    
    # save results
    out_name = 'AMPd-Up_de_novo_results_' + \
    time.strftime('%Y%m%d%H%M%S', time.localtime()) + '.' +args.out_format
    
    if args.out_format == 'tsv':
        out = pd.DataFrame({'Sequence_ID':[s.split('|length=')[0] for s in pept_id],
                            'Sequence': pept,
                            'Length': length,
                            'Charge': charge,
                            'Score': score,
                            'Completeness': completeness})
        out.to_csv(args.out_dir + '/' + out_name, sep='\t', index=False)
        
    else:
        out = open(args.out_dir + '/' + out_name, 'w')
        for i in range(len(pept)):
            out.write('>' + pept_id[i] + '\n' + pept[i] + '\n')
        out.close()

    print("\n%d sequences have been generated and saved to "%len(pept) + args.out_dir + '/' + out_name)
    
    
if __name__ == "__main__":
    main()
    
