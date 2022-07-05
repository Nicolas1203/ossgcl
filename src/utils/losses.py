import torch
import logging as lg

from torch import nn

eps = 1e-7

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SemiSupConLoss(nn.Module):
    """Adapted version of Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', stream_size=10):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.stream_size = stream_size

    def forward(self, features, input_weights, labels=None, wposmask=None):
        """Compute loss for model. If both `labels` and `wposmask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [memory_batch_size + stream_size, n_views, ...].
            labels: ground truth of shape [memory_batch_size].
            wposmask: contrastive weighted positive mask of shape [mbs, mbs], wposmask_{i,j}=1 if sample j
                        has the same class as sample i and is taken from memory.
                        wposmask_{i,i}=alpha if taken from the stream.
                        # TODO cite paper section.
                        Memory samples MUST be first in batch for this code to work as expected
                
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and wposmask is not None:
            raise ValueError('Cannot define both `labels` and `wposmask`')
        elif labels is None and wposmask is None:
            wposmask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif wposmask is None:
            labels = labels.contiguous().view(-1, 1)
            # Complete missing label for strema data. Add arbitrary non-existing labels.
            # This is just a trick to obtain the wposmask matrix
            max_label = labels.max().item() if labels.shape[0] > 0 else 0
            n_missing_labels = batch_size - labels.shape[0]
            stream_labels = torch.tensor([max_label + i + 1 for i in range(n_missing_labels)]).view(-1, 1).to(device)
            labels = torch.cat([labels, stream_labels])
            posmask = torch.eq(labels, labels.T).float().to(device)
            wposmask = torch.eq(labels, labels.T).float().to(device)
            mask_out = None

            # Weight mask matrix
            for j, valj in enumerate(input_weights):
                wposmask[j, :] = valj

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        wposmask = wposmask.repeat(anchor_count, contrast_count)
        posmask = posmask.repeat(anchor_count, contrast_count)

        if mask_out is None: mask_out = torch.ones_like(wposmask).to(device) - torch.eye(wposmask.size(0)).to(device)
        posmask = posmask * mask_out
        wposmask = wposmask * posmask

        # compute log_prob
        exp_logits = torch.exp(logits) * mask_out
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (wposmask * log_prob).sum(1) / posmask.sum(1)
        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss
