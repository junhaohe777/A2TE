from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from torch.nn import functional as F
from .detaillayer import TexualEmbeddingLayer, VisualEmbeddingLayer

#添加了属性损失
class A2TE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        
        self.base_model_m, _ = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.model_pairs = [[self.base_model, self.base_model_m]]

        # Instantiate the Memory class for local cache
        # self.memory = Memory(self.base_model, feature_dim=self.embed_dim, memory_size=50)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)


        if 'amm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            # 添加 itm_head 层
            self.itm_head = nn.Linear(self.embed_dim, 2)
            nn.init.normal_(self.itm_head.weight.data, std=0.001)
            nn.init.constant_(self.itm_head.bias.data, val=0.0)
        
        if 'detail' in args.loss_names:
            self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.pool_ratio)
            self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.pool_ratio)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.cross_modal_transformer([x])
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def get_image_attention_map(self, image):
        _, attn = self.base_model.encode_image(image)
        return attn
    
    def get_text_attention_map(self, text):
        _, attn = self.base_model.encode_text(text)
        return attn
    
    def compute_attribute_loss(self, attribute_masks, text_feats, image_feats, is_positive=True):
        # print(f"attribute_masks shape: {attribute_masks.shape}")
        # print(f"text_feats shape: {text_feats.shape}")
        # print(f"image_feats shape: {image_feats.shape}")
        #是否使用跨模态交互 positive
        fusion = self.cross_former(text_feats, image_feats, image_feats)
        fusion_norm = F.normalize(fusion)
        loss = 0
        count = 0
        for attribute_mask, text_emb in zip(attribute_masks, fusion_norm):
            max_attribute_value = torch.max(attribute_mask).to(torch.int32)

            # P-itm with attribute
            averaged_attribute = []

            for i in range(1, max_attribute_value + 1):
                mask = attribute_mask == i
                averaged_attribute.append(text_emb[mask].mean(0))

            if len(averaged_attribute) > 0:
                # print(f"Query shape: {torch.stack(averaged_attribute).shape}")
                # print(f"Key/Value shape: {image_feats.shape}")
                vl_avg_output = self.itm_head(torch.stack(averaged_attribute))
                target = torch.ones(vl_avg_output.size(0), dtype=torch.long, device=image_feats.device) if is_positive else torch.zeros(vl_avg_output.size(0), dtype=torch.long, device=image_feats.device)
                loss += F.cross_entropy(vl_avg_output, target)
                    
                count += 1
        return loss, count

    def forward(self, batch, alpha):
        ret = dict()

        images = batch['images']
        # images_aug = batch['images_aug']
        caption_ids = batch['caption_ids']
        masks = batch['attributes_mask']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_feats_norm = F.normalize(i_feats).float()
        t_feats_norm = F.normalize(t_feats).float()
        
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        with torch.no_grad():
            # 计算图像和文本特征的相似度矩阵
            # similarity_matrix = torch.matmul(i_feats_norm, t_feats_norm.T)
            similarity_matrix = torch.matmul(i_feats, t_feats.T)
            weights_i2t = F.softmax(similarity_matrix, dim=1)
            # 创建掩码，将正样本对应的相似度得分置为 0
            idx = torch.arange(i_feats.size(0), device=i_feats.device)
            mask = torch.eq(idx.unsqueeze(1), idx.unsqueeze(0))
            weights_i2t.masked_fill_(mask, 0)
        
        text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
        mask_neg = masks[text_neg_idx]
        
        
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:   
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
        
        if 'detail' in self.current_task:
            i_tse_f = self.visul_emb_layer(image_feats, atten_i)
            t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            ret.update({'detail_loss':objectives.compute_detail_tri(i_tse_f, t_tse_f,batch['pids'], logit_scale)*self.args.detail_loss_weight})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
            
            
        if 'amm' in self.current_task:

            # MLM
            mlm_ids = batch['mlm_ids']

            mlm_feats,_ = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
            
#           #AIM
            loss_attribute = 0
            count = 0
            # 处理正样本
            loss_pos, count_pos = self.compute_attribute_loss(masks, text_feats, image_feats, is_positive=True)

            # 处理负样本
            loss_neg, count_neg = self.compute_attribute_loss(mask_neg, text_feats[text_neg_idx], image_feats, is_positive=False)

            loss_attribute = loss_pos + loss_neg
            count = count_pos + count_neg

            if count > 0:
                ret.update({'attr_loss': (loss_attribute / count) * self.args.attribute_weight})  # 归一化损失
            else:
                ret.update({'attr_loss': 0})             

        return ret

def build_model(args, num_classes=11003):
    model = A2TE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
