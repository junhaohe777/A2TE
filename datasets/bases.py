from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
import spacy


nlp = spacy.load('en_core_web_sm')

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    # 获取序列起始和结束的特殊令牌
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
     # 将字幕转换为令牌序列，并在开始和结束位置添加特殊令牌
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
    # 初始化结果张量，长度为指定的文本长度，初始值为0
    result = torch.zeros(text_length, dtype=torch.long)
     # 如果生成的令牌序列超过指定长度
    if len(tokens) > text_length:
         # 如果允许截断，则调整序列长度，确保末尾是结束令牌
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
             # 如果不允许截断，则抛出异常
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    # 将调整后的令牌序列复制到结果张量中
    result[:len(tokens)] = torch.tensor(tokens)
    return result


# def get_attribute_mask(caption: str, tokenizer, text_length=77,max_length = 200, truncate=True):
#     # 什么时候截断也是一个探索的地方。
#     text = tokenizer.encode(caption)
#     text = tokenizer.decode(text)
#     doc = nlp(text)
#     attribute_mask = torch.zeros(max_length)
#     text_words=text.split()

#     idx_words=0
#     char_count=0

#     count=1
#     for chunk in doc.noun_chunks:
#         pos=[]
#         adj_founded=False # just to be sure
#         for tok in chunk:
#             if tok.pos_ == "NOUN":
#                 # noun = tok.text
#                 #manage she's to be split in she is
#                 while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
#                     char_count+=len(text_words[idx_words])+1
#                     idx_words+=1
#                 pos.append(idx_words+1) # +1 to account for [CLS]
#             if tok.pos_ == "ADJ":
#                 adj_founded=True
#                 while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
#                     char_count+=len(text_words[idx_words])+1
#                     idx_words+=1
#                 pos.append(idx_words+1) # +1 to account for [CLS]
                
#         if len(pos)>1 and adj_founded:
#             attribute_mask[pos]=count
#             count+=1

#     bpe_attribute_mask = [attribute_mask[0].item()]
    
#     for idx, word in enumerate(text_words):
#         bpe_pieces = tokenizer.bpe(word).split()
#         if idx + 1 < len(attribute_mask):
#             mask_value = attribute_mask[idx + 1].item()# +1 because attribute_mask[0] is CLS
#         else:
#             mask_value = 0  # 越界就用0填充
#         bpe_attribute_mask.extend([mask_value] * len(bpe_pieces))
    
#     bpe_attribute_mask.append(0)
    
#     if len(bpe_attribute_mask) < text_length:
#         pad_len = text_length - len(bpe_attribute_mask)
#         bpe_attribute_mask.extend([0] * pad_len)
#     else:
#         bpe_attribute_mask = bpe_attribute_mask[:text_length]
#         bpe_attribute_mask[-1] = 0

#     bpe_attribute_mask = torch.tensor(bpe_attribute_mask)


#     return bpe_attribute_mask

def get_attribute_mask(caption: str, tokenizer, text_length=77, truncate=True):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    text_ids = [sot_token] + tokenizer.encode(caption)
    if len(text_ids) > text_length:
         # 如果允许截断，则调整序列长度，确保末尾是结束令牌
        if truncate:
            text_ids = text_ids[:text_length]
            text_ids[-1] = eot_token
        else:
             # 如果不允许截断，则抛出异常
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    
    text = tokenizer.decode(text_ids)
    doc = nlp(text)
    attribute_mask = torch.zeros(text_length)
    
    text_words=text.split()

    idx_words=0
    char_count=0

    count=1
    for chunk in doc.noun_chunks:
        pos=[]
        adj_founded=False # just to be sure
        for tok in chunk:
            if tok.pos_ == "NOUN":
                # noun = tok.text
                #manage she's to be split in she is
                while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
                    char_count+=len(text_words[idx_words])+1
                    idx_words+=1
                pos.append(idx_words+1) # +1 to account for [CLS]
            if tok.pos_ == "ADJ":
                adj_founded=True
                while (char_count<=tok.idx and tok.idx<=char_count+len(text_words[idx_words]))==False:
                    char_count+=len(text_words[idx_words])+1
                    idx_words+=1
                pos.append(idx_words+1) # +1 to account for [CLS]
                
        if len(pos)>1 and adj_founded:
            attribute_mask[pos]=count
            count+=1
    return attribute_mask

def extract_attributes(caption):
    # 定义语法规则
    grammar = r"""
        NP: {<DT>?<JJ>*<NN>}
        {<JJ><NN>}
        {<NN>}
    """
    parser = RegexpParser(grammar)
    # 分词
    words = word_tokenize(caption.lower())
    # 词性标注
    tagged_words = pos_tag(words)
    # 解析语法树
    tree = parser.parse(tagged_words)
    attributes = []
    # 遍历语法树子树，提取名词短语
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            attribute = ' '.join(word for (word, _) in subtree.leaves())
            attributes.append(attribute)
    return attributes

class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        attributes = get_attribute_mask(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'attributes_mask':attributes,
            'images': img,
            # 'images_aug':img_aug,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
        attributes = get_attribute_mask(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            # 'images_aug':img_aug,
            'attributes_mask':attributes,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)