import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.twitter_dataset import twitter_dataset
from dataset.multioff_dataset import multioff_dataset
from dataset.prompt_dataset import prompt_dataset
from dataset.hateful_prompt import hateful_prompt
from dataset.harmC_dataset import harmC_dataset
from dataset.harmP_dataset import harmP_dataset


from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset

from itertools import cycle


from dataset.randaugment import RandomAugment

def create_dataset(dataset, config, temps=None, tokenizer=None):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)                  
        return dataset      
               
    elif dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train') 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])       
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])     
        return train_dataset, val_dataset, test_dataset 

    elif dataset=='hateful':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_seen_dataset = ve_dataset(config['val_seen_file'], test_transform, config['image_root'])  
        test_seen_dataset = ve_dataset(config['test_seen_file'], test_transform, config['image_root'])     
        return train_dataset, val_seen_dataset, test_seen_dataset
    
    elif dataset=='harmP':   
        train_dataset = harmP_dataset(config['train_file'], train_transform, config['image_root'])  
        val_seen_dataset = harmP_dataset(config['val_seen_file'], test_transform, config['image_root'])  
        test_seen_dataset = harmP_dataset(config['test_seen_file'], test_transform, config['image_root'])     
        return train_dataset, val_seen_dataset, test_seen_dataset
    
    elif dataset=='harmC':   
        train_dataset = harmC_dataset(config['train_file'], train_transform, config['image_root'])  
        val_seen_dataset = harmC_dataset(config['val_seen_file'], test_transform, config['image_root'])  
        test_seen_dataset = harmC_dataset(config['test_seen_file'], test_transform, config['image_root'])     
        return train_dataset, val_seen_dataset, test_seen_dataset
        
    elif dataset=='twitter':   
        train_dataset = twitter_dataset(config['train_file_t'], train_transform, config['image_root_t'])  
        val_dataset = twitter_dataset(config['val_file_t'], test_transform, config['image_root_t'])  
        test_dataset = twitter_dataset(config['test_file_t'], test_transform, config['image_root_t']) 
        return train_dataset, val_dataset, test_dataset  

    elif dataset=='multioff':   
        train_dataset = multioff_dataset(config['train_file_m'], train_transform, config['image_root_m'])  
        val_dataset = multioff_dataset(config['val_file_m'], test_transform, config['image_root_m'])  
        test_dataset = multioff_dataset(config['test_file_m'], test_transform, config['image_root_m']) 
        return train_dataset, val_dataset, test_dataset  

    elif dataset=='cckt':  
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_seen_dataset = ve_dataset(config['val_seen_file'], test_transform, config['image_root'])  
        test_seen_dataset = ve_dataset(config['test_seen_file'], test_transform, config['image_root']) 
    
        train_dataset_t = twitter_dataset(config['train_file_t'], train_transform, config['image_root_t'])  
        val_dataset = twitter_dataset(config['val_file_t'], test_transform, config['image_root_t'])  
        test_dataset = twitter_dataset(config['test_file_t'], test_transform, config['image_root_t'])                
        return train_dataset, train_dataset_t, val_seen_dataset, val_dataset, test_seen_dataset, test_dataset  
    
    elif dataset=='mmtf':  
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_seen_dataset = ve_dataset(config['val_seen_file'], test_transform, config['image_root'])  
        test_seen_dataset = ve_dataset(config['test_seen_file'], test_transform, config['image_root']) 
    
        train_dataset_t = twitter_dataset(config['train_file_t'], train_transform, config['image_root_t'])  
        val_dataset = twitter_dataset(config['val_file_t'], test_transform, config['image_root_t'])  
        test_dataset = twitter_dataset(config['test_file_t'], test_transform, config['image_root_t'])                
        return train_dataset_t, train_dataset, val_dataset, val_seen_dataset, test_dataset, test_seen_dataset  

    elif dataset=='prompt':   
        train_dataset = prompt_dataset(config['train_file'], train_transform, config['image_root'], temps, tokenizer)  
        val_dataset = prompt_dataset(config['val_file'], test_transform, config['image_root'], temps, tokenizer)  
        test_dataset = prompt_dataset(config['test_file'], test_transform, config['image_root'], temps, tokenizer)                
        return train_dataset, val_dataset, test_dataset 

    elif dataset=='hateful_prompt':   
        train_dataset = hateful_prompt(config['train_file'], train_transform, config['image_root'])  
        val_seen_dataset = hateful_prompt(config['val_seen_file'], test_transform, config['image_root'])  
        val_unseen_dataset = hateful_prompt(config['val_unseen_file'], test_transform, config['image_root'])  
        test_seen_dataset = hateful_prompt(config['test_seen_file'], test_transform, config['image_root'])     
        test_unseen_dataset = hateful_prompt(config['test_unseen_file'], test_transform, config['image_root'])     
        return train_dataset, val_seen_dataset, val_unseen_dataset, test_seen_dataset, test_unseen_dataset




    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')       
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return train_dataset, test_dataset    
    

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     



def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
