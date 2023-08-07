import torch
import random
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn import MultiheadAttention
import numpy as np
from torch.nn.functional import softplus
from functools import partial

random.seed(510)

class Space_Encoder(nn.Module):
    def __init__(self, input_dim = 768, shared_dim = 128):
        super(Space_Encoder, self).__init__()
        self.text_attention = MultiheadAttention(embed_dim = input_dim, num_heads= 8, dropout = 0.5, batch_first= True)
        self.image_attention = MultiheadAttention(embed_dim = input_dim, num_heads= 8, dropout = 0.5, batch_first= True)
        self.text_MLP = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.LayerNorm(shared_dim, eps=1e-5),
            nn.ReLU()
        )
        
        self.Imgae_MLP = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.LayerNorm(shared_dim, eps=1e-5),
            nn.ReLU()
        )
        
    def forward(self, post, image, post_mask = None, image_mask = None):
        post, _ = self.text_attention(post, post, post, post_mask)
        image, _ = self.image_attention(image, image, image, image_mask)
        text_shared = self.text_MLP(post)
        image_shared = self.Imgae_MLP(image)
        return text_shared, image_shared
    
    
class Similarity_space(nn.Module):
    def __init__(self, shared_dim = 128, cal_dim = 64):
        super(Similarity_space, self).__init__()
        self.Encoder = Space_Encoder()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim, eps=1e-5),
            nn.ReLU(),
            nn.Linear(shared_dim, cal_dim),
            nn.LayerNorm(cal_dim, eps=1e-5),
            nn.ReLU()
        )
        
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim, eps=1e-5),
            nn.ReLU(),
            nn.Linear(shared_dim, cal_dim),
            nn.LayerNorm(cal_dim, eps=1e-5),
            nn.ReLU()
        )

    def forward(self, post, image):
        post, image = self.Encoder(post, image)
        post = self.text_aligner(post)
        image = self.image_aligner(image)


        sim_matrix = torch.matmul(post, image.transpose(1, 2))
        sim_matrix = sim_matrix / torch.norm(post, dim=-1, keepdim=True)
        sim_matrix = sim_matrix / torch.norm(image, dim=-1, keepdim=True).transpose(1, 2)

        return post, image, sim_matrix
    

class Normal_mapping(nn.Module):
    def __init__(self, z_dim=2):
        super(Normal_mapping, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )
    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  
        return Independent(Normal(loc=mu, scale=sigma), 1)
    
    
class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.encoding = Space_Encoder()
        self.encoder_text = Normal_mapping()
        self.encoder_image = Normal_mapping()

    def forward(self, text_encoding, image_encoding):
        # text_encoding, image_encoding = self.encoding(text, image)
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        return nn.functional.sigmoid(kl_1_2), nn.functional.sigmoid(kl_2_1)



class PatchLevelKL(nn.Module):
    def __init__(self, input_dim = 64, z_dim = 2):
        super(PatchLevelKL, self).__init__()
        #self.similarity_space = Similarity_space(shared_dim, cal_dim, resim = True)
        self.text_normal_mapping = Normal_mapping(z_dim)
        self.image_normal_mapping = Normal_mapping(z_dim)
        self.dim = input_dim
        self.z_dim = z_dim

    def forward(self, post_align, image_align, sim_matrix):


        kl_patches_post = []
        for i in range(1, post_align.shape[1]):

            max_idx = torch.argmax(sim_matrix[:, i, 1:], dim=1)
            
            row_indices = torch.arange(len(max_idx)).unsqueeze(1)
            column_indices = max_idx.unsqueeze(1)


            image_align_post = image_align[row_indices, column_indices]

            image_align_post = image_align_post.reshape(len(max_idx), image_align.shape[2])

            q_z_post = self.text_normal_mapping(post_align[:, i])
            q_z_image = self.image_normal_mapping(image_align_post)

            z_post = q_z_post.rsample()
            log_prob_post = q_z_post.log_prob(z_post)
            log_prob_image = q_z_image.log_prob(z_post)
            kl = log_prob_post - log_prob_image
            kl = nn.functional.sigmoid(kl)
            kl_patches_post.append(kl)
        kl_patches_post = torch.stack(kl_patches_post, dim=1)
        kl_patches_image = []
        for i in range(1, image_align.shape[1]):
            # Find the index of the maximum cosine similarity in the i-th row
            max_idx = torch.argmax(sim_matrix[:, 1:, i], dim= 1)
            
            row_indices = torch.arange(len(max_idx)).unsqueeze(1)
            column_indices = max_idx.unsqueeze(1)

            post_align_image = post_align[row_indices, column_indices]
            post_align_image = post_align_image.reshape(len(max_idx), image_align.shape[2])
            q_z_image = self.image_normal_mapping(image_align[:, i])
            q_z_post = self.text_normal_mapping(post_align_image)
            z_image = q_z_image.rsample()
            log_prob_image = q_z_image.log_prob(z_image)
            log_prob_post = q_z_post.log_prob(z_image)

            kl = log_prob_image - log_prob_post 
            kl = nn.functional.sigmoid(kl)
            kl_patches_image.append(kl)
        kl_patches_image = torch.stack(kl_patches_image, dim=1)

        return nn.functional.sigmoid(kl_patches_post), nn.functional.sigmoid(kl_patches_image)
     
    
class Attention(nn.Module):

    def __init__(self, attention_dropout=0.5):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, ratio = None, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask.unsqueeze(1), -np.inf)       
        if ratio is not None:
            attention = attention * ratio.unsqueeze(1).unsqueeze(2)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)


        return attention   
    
class Gloabal_MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=128, num_heads=8, dropout=0.5):
        super(Gloabal_MultiHeadAttention, self).__init__()
        
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention =Attention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, ratio = None, attn_mask=None):
        B1, N1, C1 = query.shape
        B2, N2, C2 = key.shape   
        residual = query
        
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        

        
        key = key.reshape(B2, N2, num_heads, dim_per_head)
        value = value.reshape(B2, N2, num_heads, dim_per_head) 
        query = query.reshape(B1, N1, num_heads, dim_per_head)

        
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        scale = (self.dim_per_head)**-0.5
        attention = self.dot_product_attention(query, key, value, 
                                               ratio, scale, attn_mask)

        attention = attention.transpose(1, 2)
        attention = attention.reshape(B1, N1, C1)
        
        output = self.linear_final(attention)

        output = self.dropout(output)

        output = self.layer_norm(residual + output)

        
        return output    
    
    
class class_token_pos_embed(nn.Module):
    def __init__(self, embed_dim):
        super(class_token_pos_embed, self).__init__()

        num_patches = patchembed().num_patches
        
        self.num_tokens = 1  

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+self.num_tokens, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
 

    def forward(self, x):  

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super(MLP, self).__init__()
        
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, inputs):
        
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class encoder_block(nn.Module):
    def __init__(self, dim, num_heads = 8, mlp_ratio=4., drop_ratio=0.5):
        super(encoder_block, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.atten = nn.MultiheadAttention(dim, num_heads, dropout=drop_ratio)
        self.drop = nn.Dropout()
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features)
        
    def forward(self, inputs):
        
        x = self.norm1(inputs)
        x, _ = self.atten(x, x, x)
        x = self.drop(x)
        feat1 = x + inputs  

        x = self.norm2(feat1)
        x = self.mlp(x)
        x = self.drop(x)
        return x + feat1
        

class patchembed(nn.Module):
    def __init__(self, img_size=int(224 * 0.75), patch_size=24, in_c=3, embed_dim=64):
        super(patchembed, self).__init__()
        
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size//patch_size, img_size//patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        

    
    def forward(self, inputs):

        B, C, H, W = inputs.shape
        

        assert H==self.img_size[0] and W==self.img_size[1]
        

        x = self.proj(inputs)
        x = x.flatten(start_dim=2, end_dim=-1)  
        x = x.transpose(1, 2) 
        x = self.norm(x)
        
        return x



class DCTDetectionModel(nn.Module):
    def __init__(self, embedding_dim, dropout = 0.5):
        super(DCTDetectionModel, self).__init__()
        self.dct_stem = patchembed()
        self.positional_encoding = class_token_pos_embed(embed_dim=embedding_dim)
        self.encoder_layer = nn.Sequential(*[encoder_block(dim = embedding_dim) for _ in range(3)]) 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embedding_dim)
        
        
        self.ratio = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, img):
        
        patched_img = self.dct_stem(img)
        patched_img = self.positional_encoding(patched_img)
        attention_output = self.encoder_layer(patched_img)
        attention_output = self.norm(attention_output)
        

        cls_ratio = self.ratio(attention_output[:, 0])
        cls_ratio = (cls_ratio + 1) / 2
        return attention_output, cls_ratio