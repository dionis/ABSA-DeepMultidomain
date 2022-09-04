#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import nlp_data_utils as data_utils
from nlp_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split

import math
datasets = [
            './dat/absa/XuSemEval/asc/14/rest',
            './dat/absa/XuSemEval/asc/14/laptop',

            './dat/absa/Bing3Domains/asc/Speaker',
            './dat/absa/Bing3Domains/asc/Router',
            './dat/absa/Bing3Domains/asc/Computer',

            './dat/absa/Bing5Domains/asc/Nokia6610',
            './dat/absa/Bing5Domains/asc/NikonCoolpix4300',
            './dat/absa/Bing5Domains/asc/CreativeLabsNomadJukeboxZenXtra40GB',
            './dat/absa/Bing5Domains/asc/CanonG3',
            './dat/absa/Bing5Domains/asc/ApexAD2600Progressive',

            './dat/absa/Bing9Domains/asc/CanonPowerShotSD500',
            './dat/absa/Bing9Domains/asc/CanonS100',
            './dat/absa/Bing9Domains/asc/DiaperChamp',
            './dat/absa/Bing9Domains/asc/HitachiRouter',
            './dat/absa/Bing9Domains/asc/ipod',
            './dat/absa/Bing9Domains/asc/LinksysRouter',
            './dat/absa/Bing9Domains/asc/MicroMP3',
            './dat/absa/Bing9Domains/asc/Nokia6600',
            './dat/absa/Bing9Domains/asc/Norton',

            './dat/absa/TripAdvisor/asc',

            './dat/absa/FewItemsDomains/asc/CanonG3',
            './dat/absa/FewItemsDomains/asc/Nokia6610',
            './dat/absa/FewItemsTripAdvisor/asc'
            ]


domains = [
     'XuSemEval14_rest',
     'XuSemEval14_laptop',

     'Bing3domains_Speaker',
     'Bing3domains_Router',
     'Bing3domains_Computer',

     'Bing5domains_Nokia6610',
     'Bing5domains_NikonCoolpix4300',
     'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',
     'Bing5domains_CanonG3',
     'Bing5domains_ApexAD2600Progressive',

     'Bing9domains_CanonPowerShotSD500',
     'Bing9domains_CanonS100',
     'Bing9domains_DiaperChamp',
     'Bing9domains_HitachiRouter',
     'Bing9domains_ipod',
     'Bing9domains_LinksysRouter',
     'Bing9domains_MicroMP3',
     'Bing9domains_Nokia6600',
     'Bing9domains_Norton',

     'TripAdvisor',
     'FewItemsDomains_CanonG3',
     'FewItemsDomains_Nokia6610',
     'FewItemsTripAdvisor'
   ]





def get(logger=None,args=None):
    data={}
    taskcla=[]

    # Others
    f_name = 'asc_random'

    if args != None and args.dataloaders != None and type( args.dataloaders) == str and  args.dataloaders != '':
        f_name =  args.dataloaders

    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)
    print('domains: ',domains)

    print('random_sep: ',len(random_sep))
    print('domains: ',len(domains))

    for t in range(args.ntasks):
        dataset = datasets[domains.index(random_sep[t])]
        print('dataset: ',dataset)
        data[t]={}
        if 'Bing' in dataset:
            data[t]['name']=dataset
            if args.baseline=='derpp' or args.baseline=='a-gem': data[t]['ncla']=3
            else: data[t]['ncla']=2
        elif 'XuSemEval' in dataset:
            data[t]['name']=dataset
            data[t]['ncla']=3
        else:
            data[t]['name'] = dataset
            data[t]['ncla'] = 3

        processor = data_utils.AscProcessor(args)
        label_list = processor.get_labels()
        #,cache_dir = "Transformer" + os.path.sep,  local_files_only=True
        tokenizer = ABSATokenizer.from_pretrained( args.bert_model)
        if "TripAdvisor" in dataset:
          train_examples = processor.get_train_examplesEx(dataset)
        else:
          train_examples = processor.get_train_examples(dataset)

        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, "asc")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        #all_example_ids = torch.tensor([f.example_id for f in train_features], dtype=torch.long) #All id in task to extract bat examples
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        #train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids,all_example_ids,all_tasks)
        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids,all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps

        eval_examples = None
        if "TripAdvisor" in dataset:
            #Remember split in dev and test examples
            valid_examples = processor.get_test_examplesEx(dataset)
            assert 0 <= args.valset_ratio < 1
            if args.valset_ratio > 0: #Split text dataset in valid and test subset
                valset_len = int(len(valid_examples) * args.valset_ratio)
                valid_examples, eval_examples= random_split(valid_examples, (len(valid_examples) - valset_len, valset_len))
            else:
                eval_examples = valid_examples

        else:
            valid_examples = processor.get_dev_examples(dataset)

        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "asc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids,valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid']=valid_data


        processor = data_utils.AscProcessor(args)
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)

        if not ("TripAdvisor"  in dataset):
            eval_examples = processor.get_test_examples(dataset)

        eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, "asc")

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids,all_tasks)
        # Run prediction for full data

        data[t]['test']=eval_data
        domainName = os.path.basename(data[t]['name'])
        if domainName == 'asc':
            domainName = os.path.dirname(data[t]['name'])
            domainName = os.path.basename(domainName)
        print(domainName)
        taskcla.append((t,int(data[t]['ncla']),domainName))



    # Others
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n


    return data,taskcla, tokenizer


