import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

import numpy as np 
from .losses import angle_loss, FocalLoss
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 loss: str = 'cross_entropy',
                 focal_alpha=None, focal_gamma=None
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.loss = loss

        if loss == "cross_entropy" or loss == "cross_entropy_t_b":
            self.loss_f = nn.CrossEntropyLoss(reduction='mean')
            if loss == "cross_entropy_t_b":
                self.temperature_l = nn.Parameter(data=torch.tensor(np.log(5), dtype=torch.float32, requires_grad=True))
                self.bias = nn.Parameter(data=torch.tensor(-5, dtype=torch.float32, requires_grad=True))
        if loss == "angle":
            self.loss_f = angle_loss
        if loss == "sigmoid" or loss == "sigmoid2":
            self.temperature_l = nn.Parameter(data=torch.tensor(np.log(5), dtype=torch.float32, requires_grad=True))
            self.bias = nn.Parameter(data=torch.tensor(-5, dtype=torch.float32, requires_grad=True))
            if loss == 'sigmoid2':
                normlized = False
        if self.loss == 'focal':
            self.loss_f = FocalLoss(alpha= focal_alpha, gamma = focal_gamma, reduction = 'mean')

        #self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.normlized = normlized
        
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)

            if self.loss in ["cross_entropy", "cross_entropy_t_b", "focal"]:
                if self.use_inbatch_neg:
                    if self.loss == "cross_entropy" or self.loss == "focal":
                        scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                    if self.loss == "cross_entropy_t_b":
                        t = torch.exp(self.temperature_l)
                        scores = self.compute_similarity(q_reps, p_reps) * t + self.bias 
                    scores = scores.view(q_reps.size(0), -1)

                    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                    target = target * group_size
                    loss = self.compute_loss(scores, target)
                else:
                    if self.loss == "cross_entropy" or self.loss == "focal":
                        scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G
                    if self.loss == "cross_entropy_t_b":
                        t = torch.exp(self.temperature_l)
                        scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) * t + self.bias 
                    scores = scores.view(q_reps.size(0), -1)
                    target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                    loss = self.compute_loss(scores, target)
            if self.loss == "sigmoid":
                b = q_reps.shape[0]
                t = torch.exp(self.temperature_l)
                scores = self.compute_similarity(q_reps, p_reps) * t + self.bias # B B*G
                scores = scores.view(q_reps.size(0), -1)
                scores_z = -scores
                j = 0
                for i in range(0, scores.shape[1], group_size):
                    scores_z[j][i]*=-1
                    j+=1
                loss = -torch.sum(F.logsigmoid(scores_z)) / b
                scores = self.compute_similarity(q_reps, p_reps)
            if self.loss == "sigmoid2":
                b = q_reps.shape[0]
                t = torch.exp(self.temperature_l)
                scores = self.compute_similarity(q_reps, p_reps) * t + self.bias # B B*G
                scores = scores.view(q_reps.size(0), -1)
                mask = torch.zeros(scores.shape, dtype=torch.bool)
                j = 0
                for i in range(0, scores.shape[1], group_size):
                    mask[j][i]=True
                    j+=1
                pos = torch.stack(scores[mask].chunk(b), dim=0)
                neg = torch.stack(scores[~mask].chunk(b), dim=0)
                neg = -torch.stack(torch.log(torch.sum(torch.exp(neg),dim=1)).chunk(b), dim=0)
                loss = -torch.sum(F.logsigmoid(torch.hstack((pos,neg)))) / b
                scores = self.compute_similarity(q_reps, p_reps)
            if self.loss == 'angle':
                batch_size = q_reps.size(0)
                mask_pos = group_size*torch.arange(batch_size,  dtype=torch.long)
                positive = p_reps[mask_pos]
                positive = torch.stack((torch.flatten(positive.unsqueeze(1).expand(-1, group_size-1, -1)).chunk((group_size-1)*batch_size)),dim=0)
                mask_neg = torch.arange(batch_size*group_size,  dtype=torch.long)
                mask_neg = mask_neg[(mask_neg[:, None] != mask_pos).all(dim=1)]
                negative = p_reps[mask_neg]
                text = q_reps
                text = torch.stack((torch.flatten(text.unsqueeze(1).expand(-1, group_size-1, -1)).chunk((group_size-1)*batch_size)),dim=0)
                #positive, negative = torch.chunk(p_reps, 2, dim=0)
                assert text.shape == positive.shape == negative.shape, f'text.shape={text.shape}, postive.shape={positive.shape}, negative.shape={negative.shape}'
                _, fea_dim = q_reps.shape
                positive_inputs = torch.stack((text, positive), dim=1).reshape(-1, fea_dim)  # zip(text, positive), tensor (bath*2,seq_len) 
                positive_labels = torch.ones_like(positive_inputs[:, :1]).long()
                negative_inputs = torch.stack((text, negative), dim=1).reshape(-1, fea_dim)  # zip(text, negative)
                negative_labels = torch.zeros_like(negative_inputs[:, :1]).long()
                combined_inputs = torch.cat((positive_inputs, negative_inputs), dim=0)
                combined_labels = torch.cat((positive_labels, negative_labels), dim=0)
                loss = self.compute_loss(combined_inputs, combined_labels)
                scores = self.compute_similarity(q_reps, p_reps)


        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.loss_f(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
