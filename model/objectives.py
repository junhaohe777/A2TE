import torch
import torch.nn as nn
import torch.nn.functional as F

  
def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels, temp=1.0, epsilon=1e-8):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

    # input features are normed
def compute_nitc(image_features, text_features, image_features_s, text_features_s, pid, alpha, logit_scale):
    
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    with torch.no_grad():
        sim_i2t_s = logit_scale * image_features_s @ text_features_s.t()
        sim_t2i_s = logit_scale * text_features_s @ image_features_s.t()
        sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * labels
        sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * labels  # soft + hard
        
    sim_i2t_targets = sim_i2t_targets / sim_i2t_targets.sum(1, keepdim=True)
    sim_t2i_targets = sim_t2i_targets / sim_t2i_targets.sum(1, keepdim=True)
    
    sim_i2t = logit_scale * image_features @ text_features.t()
    sim_t2i = logit_scale * text_features @ image_features.t()
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
    loss_ita = (loss_i2t + loss_t2i) / 2
    return loss_ita


def compute_ritc(i_feats_norm, t_feats_norm, pid, logit_scale, eps=1e-2):
    batch_size = i_feats_norm.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
        
    labels_distribute = labels / labels.sum(dim=1)
    
    logits_per_image_1 = logit_scale * i_feats_norm @ t_feats_norm.t()
    logits_per_text_1 = logit_scale * t_feats_norm @ i_feats_norm.t()
    img_log = F.log_softmax(logits_per_image_1, dim=1)
    txt_log = F.log_softmax(logits_per_text_1, dim=1)
    target_log = (labels_distribute + eps).log()
    kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
    kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
    ritc_loss = 0.5 * (kl_img + kl_txt)
    
    return ritc_loss

def compute_citc(i_feats_norm, t_feats_norm, logit_scale, l1=0.25, l2 = 0.25):
    logits_image_per_image = logit_scale * i_feats_norm @ i_feats_norm.t()
    logits_text_per_text = logit_scale * t_feats_norm @ t_feats_norm.t()
    inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
            logit_scale * logit_scale)
    logits_text_per_image = logit_scale * i_feats_norm @ t_feats_norm.t()
    logits_image_per_text = logit_scale * i_feats_norm @ t_feats_norm.t()
    crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
            logit_scale * logit_scale)
    citc_loss = l1 * inmodal_cyclic_loss + l2 * crossmodal_cyclic_loss
    
    return citc_loss


def compute_detail(image_features,text_features, pid, tau = 0.015, margin=0.1):
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()
    
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    
    return loss.sum()

def compute_detail_tri(image_features,text_features, pid, tau = 0.015, margin=0.1):     
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t() 
    
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    
    loss = (cost_1 + cost_2).sum()
    
    return loss