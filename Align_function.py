import torch
import torch.nn as nn
import torch.nn.functional as F


class global_align(nn.Module):
    def __init__(self, temperature=0.3):
        super(global_align, self).__init__()
        self.temperature = temperature

    def forward(self, text_features, image_features):
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        similarity_matrix = torch.matmul(image_features, text_features.t())
        positive_logits = torch.diag(similarity_matrix) / self.temperature
        negative_logits = similarity_matrix / self.temperature
        negative_logits_max = torch.max(negative_logits, dim=1, keepdim=True)[0]
        negative_logits = negative_logits - negative_logits_max
        logits_max = torch.max(positive_logits)
        loss = torch.mean(torch.log(torch.exp(positive_logits - logits_max) / torch.sum(torch.exp(negative_logits), dim=1)))
        loss = -loss
        
        return loss
    
        
class local_align(nn.Module):
    def __init__(self, temperature=0.3):
        super(local_align, self).__init__()
        self.temperature = temperature

    def forward(self, similarity_matrix):

        batch_size, n1, n2 = similarity_matrix.shape
        pos_row = similarity_matrix.max(dim=2, keepdim=True).values
        neg_row = similarity_matrix.clone()
        neg_row[pos_row == similarity_matrix] = float('-inf')
        pos_exp_row = torch.exp((pos_row - pos_row.max()) / self.temperature)
        neg_exp_row = torch.exp((neg_row - neg_row.max()) / self.temperature)
        row_sum = torch.sum(neg_exp_row, dim=2)
        row_sum = torch.clamp(row_sum, min=torch.finfo(row_sum.dtype).eps)
        row_loss = -torch.log(pos_exp_row.squeeze() / row_sum).sum() / (batch_size * n1)

        pos_col = similarity_matrix.max(dim=1, keepdim=True).values
        neg_col = similarity_matrix.clone()
        neg_col[pos_col == similarity_matrix] = float('-inf')
        pos_exp_col = torch.exp((pos_col - pos_col.max()) / self.temperature)
        neg_exp_col = torch.exp((neg_col - neg_col.max()) / self.temperature)
        col_sum = torch.sum(neg_exp_col, dim=1)
        col_sum = torch.clamp(col_sum, min=torch.finfo(col_sum.dtype).eps)
        col_loss = -torch.log(pos_exp_col.squeeze() / col_sum).sum() / (batch_size * n2)

        return (row_loss + col_loss) / 2
