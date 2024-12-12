from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_constant_schedule, AutoImageProcessor, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from model import Model
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from PIL import Image
from torchvision import transforms


logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 pixel_values,
                 labels):
        self.pixel_values = pixel_values
        self.labels = labels
        
class TextDataset(Dataset):
    def __init__(self, feature_processor, args, file_type="train"):
        crop_size = (feature_processor.size["shortest_edge"], feature_processor.size["shortest_edge"])
        
        if file_type == "train":
            file_path = args.train_data_file
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),  # RandomResizedCrop
            transforms.RandomHorizontalFlip(),        # RandomHorizontalFlip
            transforms.ToTensor(),                    # Convert image to PyTorch tensor
            transforms.Normalize(mean=feature_processor.image_mean, std=feature_processor.image_std)  # Normalize with ImageNet stats
            ])
        elif file_type == "val":
            file_path = args.eval_data_file
            self.transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=feature_processor.image_mean, std=feature_processor.image_std)  # Normalize with ImageNet stats
            ])
        elif file_type == "test":
            file_path = args.test_data_file
            self.transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=feature_processor.image_mean, std=feature_processor.image_std)  # Normalize with ImageNet stats
            ])
        
        image_label = []
        # Lists to store all opened images and their corresponding labels
        if args.classify_pneumonia_type:
            logger.info("loading images for pneumonia type classification")
            
            all_file_path = [file_path+"/PNEUMONIA"]
            # Walk through the directory and load each image
            for path in all_file_path:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpeg'):
                            # Construct the full path to the image file
                            image_path = os.path.join(root, file)
                            # Open the image
                            image = Image.open(image_path)
                            if image.mode != 'RGB':
                                image = image.convert('RGB')                        
                            if 'bacteria' in file.lower(): # bacteria -> 0
                                label = 0
                            elif 'virus' in file.lower(): # virus -> 1
                                label = 1
                            else: # ignore normal images
                                logger.info("error occur when loading images!")
                                exit()
                            # Store the opened image and its label in the lists
                            image_label.append([image, label])
        else:
            logger.info("loading images for pneumonia detection")
            
            all_file_path = [file_path+"/NORMAL", file_path+"/PNEUMONIA"]
            # Walk through the directory and load each image
            for path in all_file_path:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpeg'):
                            # Construct the full path to the image file
                            image_path = os.path.join(root, file)
                            # Open the image
                            image = Image.open(image_path)
                            if image.mode != 'RGB':
                                image = image.convert('RGB')                        
                            # Determine the label based on the filename
                            if ('bacteria' in file.lower()) or ('virus' in file.lower()):
                                label = 1
                            else:
                                label = 0
                            # Store the opened image and its label in the lists
                            image_label.append([image, label])
        random.shuffle(image_label)
        
        # for testing
        #image_label = image_label[:256]
        
        self.examples = []

        for i in tqdm(range(len(image_label))):
            self.examples.append(convert_examples_to_features(image_label[i], self.transform))
        
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("pixel_values: {}".format(' '.join(map(str, example.pixel_values))))
                logger.info(f"labels: {example.labels}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].pixel_values, torch.tensor(self.examples[i].labels).long()

def convert_examples_to_features(inputs, feature_processor):
    image, label = inputs
    # Preprocess the image
    features = feature_processor(image)
    return InputFeatures(features, label) # features['pixel_values'][0]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, feature_processor, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 10

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = get_constant_schedule(optimizer)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.epochs*0.1, num_training_steps=len(train_dataloader)*args.epochs)


    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(args.device)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 1e6

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (pixel_values, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss = model(pixel_values=pixel_values, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # grad clipping
            ##torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(args, model, feature_processor, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)               
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(args, model, feature_processor, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []
    num_batch = 0
    loss_sum = 0
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (pixel_values, labels) = [x.to(args.device) for x in batch]
            loss = model(pixel_values=pixel_values, labels=labels)
            num_batch += 1
            loss_sum += loss.sum().sum().item()
            #preds = torch.where(probs>0.5, 1, 0)
            #preds = torch.argmax(probs, dim=1)
            #y_trues += labels.tolist()
            #y_preds += preds.tolist()

    eval_loss = loss_sum / num_batch
    model.train()
    #f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    logger.info("***** Eval results *****")
    logger.info(f"Loss: {str(eval_loss)}")
    return eval_loss

def test(args, model, feature_processor, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (pixel_values, labels) = [x.to(args.device) for x in batch]
            probs = model(pixel_values=pixel_values)
            preds = torch.where(probs>0.5, 1, 0)
            preds = torch.argmax(probs, dim=1)
            y_trues += labels.tolist()
            y_preds += preds.tolist()
            
    model.train()
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    recall = recall_score(y_true=y_trues, y_pred=y_preds)
    precision = precision_score(y_true=y_trues, y_pred=y_preds)
    auc = roc_auc_score(y_trues, y_preds, multi_class='ovr', average='weighted')
    
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    # Calculate specificity
    specificity = tn / (tn + fp)
    
    logger.info("***** Test results *****")
    logger.info(f"f1: {str(f1)}")
    logger.info(f"precision: {str(precision)}")
    logger.info(f"recall: {str(recall)}")
    logger.info(f"specificity: {str(specificity)}")
    logger.info(f"Acc: {str(acc)}")
    logger.info(f"AUC: {str(auc)}")


def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--eval_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--test_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--pretrain_language", default="", type=str, required=False,
                        help="python, go, ruby, php, javascript, java, c_cpp")
    ps.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ps.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    ps.add_argument("--encoder_block_size", default=512, type=int,
                        help="")
    ps.add_argument("--max_line_length", default=64, type=int,
                        help="")
    ps.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    ps.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    ps.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    ps.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    ps.add_argument("--feature_processor_name", default="", type=str,
                        help="Optional pretrained feature_processor name or path if not the same as model_name_or_path") 
    ps.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    ps.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    ps.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    ps.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    ps.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    ps.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    ps.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for AdamW.")
    ps.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    ps.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    ps.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    ps.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    ps.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    ps.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    ps.add_argument('--epochs', type=int, default=3,
                        help="training epochs")
    ps.add_argument("--classify_pneumonia_type", action='store_true',
                        help="")
    args = ps.parse_args()

    # Setup CUDA, GPU
    args.n_gpu = 2
    args.device = "cuda"
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    # Set seed
    set_seed(args)
    
    feature_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    vit = AutoModel.from_pretrained(args.model_name_or_path)
        
    model = Model(vit, feature_processor, args)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        checkpoint_prefix = 'checkpoint-best-f1/pretrained_dinov2-large.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        # Load the checkpoint
        checkpoint = torch.load(output_dir, map_location=args.device)
        # Remove classifier weights from the checkpoint
        incompatible_keys = ['classifier.weight', 'classifier.bias']
        for key in incompatible_keys:
            if key in checkpoint:
                del checkpoint[key]
        model.load_state_dict(checkpoint, strict=False)
        model.to(args.device)
           
        train_dataset = TextDataset(feature_processor, args, file_type='train')
        eval_dataset = TextDataset(feature_processor, args, file_type='val')
        train(args, train_dataset, model, feature_processor, eval_dataset)
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(feature_processor, args, file_type='test')
        test(args, model, feature_processor, test_dataset)


if __name__ == "__main__":
    main()
