import torch
import torch.nn as nn
from model_encoder import Space_Encoder, Gloabal_MultiHeadAttention, AmbiguityLearning, PatchLevelKL
from torch.nn import functional as F

class Modal_Fushion(nn.Module):
    def __init__(self, align_dim, model_dim):
        super(Modal_Fushion, self).__init__()
        self.Fushion_layer_1 = Gloabal_MultiHeadAttention(model_dim = align_dim)
        self.Fushion_layer_2 = Gloabal_MultiHeadAttention(model_dim = model_dim)
        self.Fushion_MLP_1 = nn.Sequential(
            nn.Linear(align_dim, model_dim),
            nn.BatchNorm1d(align_dim),
            nn.ReLU()
        )
        self.Fushion_MLP_2 = nn.Sequential(
            nn.Linear(align_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU()
        )
        
        self.Fushion_linear = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU()
        )
        
    def forward(self, post_align, image_align, dct_img, post_modal_ratio = None, img_modal_ratio = None):
        
        if img_modal_ratio is not None:
            Fushion_feature_img = self.Fushion_layer_1(image_align, post_align, post_align, post_modal_ratio)

        fushion_img = self.Fushion_layer_2(Fushion_feature_img * img_modal_ratio.unsqueeze(-1), dct_img, dct_img)
        return self.Fushion_linear(fushion_img[:, 0])
    
    
        
class InformationDetection(nn.Module):
    def __init__(self, DCT_Encoder,align_dim = 64, shared_dim = 128, model_dim = 64):
        super(InformationDetection, self).__init__()
        
        self.uniencoder = Space_Encoder()
        
        self.global_ambiguity = AmbiguityLearning()
        self.patch_ambiguity = PatchLevelKL()
        self.text_uni = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, model_dim),
                nn.BatchNorm1d(model_dim),
                nn.ReLU()
            )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU()
        )
        self.multimodal = Modal_Fushion(align_dim, model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3*model_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 2)
        )
        
        self.dct_encoder = DCT_Encoder
        self.dct_multiple = Modal_Fushion(align_dim, model_dim)
        self.classifier_dct = nn.Sequential(
        nn.Linear(4*model_dim, model_dim),
        nn.BatchNorm1d(model_dim),
        nn.ReLU(),
        nn.Linear(model_dim, 2)
        )
        
    
    
    def forward(self, text_origin, image_origin, text_align, image_align, dct_image, sim_matrix = None):

        dct_feature, cls_ratio = self.dct_encoder(dct_image)
        p2i, i2p = self.global_ambiguity(text_align[:, 0], image_align[:, 0])
        kl_patches_post, kl_patches_image = self.patch_ambiguity(text_align, image_align, sim_matrix)
        post_ratio = F.softmax(torch.cat([p2i.unsqueeze(1), kl_patches_post], dim = 1))
        image_ratio = F.softmax(torch.cat([i2p.unsqueeze(1), kl_patches_image], dim = 1))        
        multiple_feature = self.multimodal(text_align, image_align, dct_feature, post_ratio, image_ratio)
        post_shared, _ = self.uniencoder(text_origin, image_origin)
        uni_post = self.text_uni(post_shared[:, 0])
        feature = torch.cat([multiple_feature * cls_ratio, uni_post, dct_feature[:, 0]], dim =1)
        return self.classifier(feature)
        
        