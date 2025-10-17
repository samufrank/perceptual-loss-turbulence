"""
Loss functions for turbulence-robust feature learning.
Combines spatial, frequency, and contrastive objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TurbulenceRobustLoss(nn.Module):
    """
    Multi-component loss for learning turbulence-invariant features.
    
    Components:
        1. Spatial invariance: L2 distance between clean and turbulent features
        2. Frequency consistency: L2 distance in frequency domain
        3. Contrastive: Features from same scene close, different scenes far
    """
    
    def __init__(self, spatial_weight=1.0, frequency_weight=1.0, 
                 contrastive_weight=0.5, temperature=0.07, margin=1.0):
        """
        Args:
            spatial_weight: Weight for spatial invariance loss (α)
            frequency_weight: Weight for frequency consistency loss (β)
            contrastive_weight: Weight for contrastive loss (γ)
            temperature: Temperature for contrastive loss
            margin: Margin for contrastive loss (push negatives beyond this)
        """
        super().__init__()
        
        self.spatial_weight = spatial_weight
        self.frequency_weight = frequency_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, clean_features, turb_features, batch_clean_filenames=None):
        """
        Compute total loss.
        
        Args:
            clean_features: Features from clean images (B, D)
            turb_features: Features from turbulent images (B, D)
            batch_clean_filenames: List of clean filenames for identifying pairs
            
        Returns:
            dict with total_loss and individual loss components
        """
        losses = {}
        
        # 1. Spatial invariance loss
        if self.spatial_weight > 0:
            spatial_loss = self.spatial_invariance_loss(clean_features, turb_features)
            losses['spatial'] = spatial_loss
        else:
            losses['spatial'] = torch.tensor(0.0, device=clean_features.device)
        
        # 2. Frequency consistency loss
        if self.frequency_weight > 0:
            freq_loss = self.frequency_consistency_loss(clean_features, turb_features)
            losses['frequency'] = freq_loss
        else:
            losses['frequency'] = torch.tensor(0.0, device=clean_features.device)
        
        # 3. Contrastive loss
        if self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(
                clean_features, turb_features, batch_clean_filenames
            )
            losses['contrastive'] = contrastive_loss
        else:
            losses['contrastive'] = torch.tensor(0.0, device=clean_features.device)
        
        # Total weighted loss
        total_loss = (
            self.spatial_weight * losses['spatial'] +
            self.frequency_weight * losses['frequency'] +
            self.contrastive_weight * losses['contrastive']
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def spatial_invariance_loss(self, clean_features, turb_features):
        """
        L2 distance between clean and turbulent features.
        Encourages features to be invariant to turbulence.
        
        Args:
            clean_features: (B, D)
            turb_features: (B, D)
            
        Returns:
            Scalar loss
        """
        return F.mse_loss(clean_features, turb_features)
    
    def frequency_consistency_loss(self, clean_features, turb_features):
        """
        L2 distance in frequency domain.
        Turbulence affects frequency content (blur), so enforcing
        frequency consistency encourages robustness.
        
        Args:
            clean_features: (B, D)
            turb_features: (B, D)
            
        Returns:
            Scalar loss
        """
        # Apply FFT to feature vectors
        clean_fft = torch.fft.rfft(clean_features, dim=1)
        turb_fft = torch.fft.rfft(turb_features, dim=1)
        
        # L2 distance between magnitudes
        clean_mag = torch.abs(clean_fft)
        turb_mag = torch.abs(turb_fft)
        
        return F.mse_loss(clean_mag, turb_mag)
    
    def contrastive_loss(self, clean_features, turb_features, batch_clean_filenames):
        """
        Contrastive loss: features from same scene should be close,
        features from different scenes should be far.
        
        Uses InfoNCE-style loss.
        
        Args:
            clean_features: (B, D)
            turb_features: (B, D)
            batch_clean_filenames: List of clean filenames for identifying pairs
            
        Returns:
            Scalar loss
        """
        if batch_clean_filenames is None:
            # If no filenames provided, assume all are from different scenes
            # Use simple margin-based contrastive loss
            return self.margin_contrastive_loss(clean_features, turb_features)
        
        # InfoNCE-style contrastive loss
        batch_size = clean_features.size(0)
        
        # Normalize features
        clean_norm = F.normalize(clean_features, dim=1)
        turb_norm = F.normalize(turb_features, dim=1)
        
        # Compute similarity matrix (all pairs)
        # similarity[i, j] = similarity between clean[i] and turb[j]
        similarity = torch.matmul(clean_norm, turb_norm.t()) / self.temperature
        
        # Create labels: positive pairs have same filename
        labels = torch.zeros(batch_size, batch_size, device=clean_features.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if batch_clean_filenames[i] == batch_clean_filenames[j]:
                    labels[i, j] = 1.0
        
        # InfoNCE loss: maximize similarity to positives, minimize to negatives
        # For each clean image, compute cross-entropy over all turbulent images
        exp_sim = torch.exp(similarity)
        
        # Sum over positives and negatives
        positive_sum = (exp_sim * labels).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        
        # Avoid division by zero
        loss = -torch.log(positive_sum / (all_sum + 1e-8))
        
        return loss.mean()
    
    def margin_contrastive_loss(self, clean_features, turb_features):
        """
        Simple margin-based contrastive loss.
        Used when batch_clean_filenames not provided.
        
        Assumes:
        - clean[i] and turb[i] are positive pairs
        - All other combinations are negative pairs
        
        Args:
            clean_features: (B, D)
            turb_features: (B, D)
            
        Returns:
            Scalar loss
        """
        batch_size = clean_features.size(0)
        
        # Positive pairs (same index)
        positive_dist = F.pairwise_distance(clean_features, turb_features)
        positive_loss = positive_dist.mean()
        
        # Negative pairs (different indices)
        negative_loss = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    neg_dist = F.pairwise_distance(
                        clean_features[i:i+1], 
                        turb_features[j:j+1]
                    )
                    # Hinge loss: push beyond margin
                    negative_loss += F.relu(self.margin - neg_dist).mean()
                    count += 1
        
        if count > 0:
            negative_loss = negative_loss / count
        
        return positive_loss + negative_loss


class FeatureDistanceMetric:
    """
    Compute feature distance metrics for evaluation.
    Not a loss, but used to evaluate turbulence robustness.
    """
    
    @staticmethod
    def l2_distance(feat1, feat2):
        """
        Euclidean distance between feature vectors.
        
        Args:
            feat1: (B, D) or (D,)
            feat2: (B, D) or (D,)
            
        Returns:
            Scalar or (B,) tensor
        """
        return torch.norm(feat1 - feat2, p=2, dim=-1)
    
    @staticmethod
    def cosine_similarity(feat1, feat2):
        """
        Cosine similarity between feature vectors.
        
        Args:
            feat1: (B, D) or (D,)
            feat2: (B, D) or (D,)
            
        Returns:
            Scalar or (B,) tensor (range: [-1, 1])
        """
        return F.cosine_similarity(feat1, feat2, dim=-1)
    
    @staticmethod
    def mean_feature_distance(model, dataloader, device):
        """
        Compute mean feature distance over entire dataset.
        
        Args:
            model: Encoder model
            dataloader: DataLoader with clean/turbulent pairs
            device: torch device
            
        Returns:
            dict with l2_distance and cosine_similarity
        """
        model.eval()
        
        all_l2_dist = []
        all_cosine_sim = []
        
        with torch.no_grad():
            for batch in dataloader:
                clean = batch['clean'].to(device)
                turb = batch['turbulent'].to(device)
                
                # Extract features
                clean_out = model(clean)
                turb_out = model(turb)
                
                clean_feat = clean_out['bottleneck']
                turb_feat = turb_out['bottleneck']
                
                # Compute metrics
                l2 = FeatureDistanceMetric.l2_distance(clean_feat, turb_feat)
                cosine = FeatureDistanceMetric.cosine_similarity(clean_feat, turb_feat)
                
                all_l2_dist.append(l2)
                all_cosine_sim.append(cosine)
        
        # Concatenate and compute mean
        all_l2_dist = torch.cat(all_l2_dist)
        all_cosine_sim = torch.cat(all_cosine_sim)
        
        return {
            'l2_distance': all_l2_dist.mean().item(),
            'l2_std': all_l2_dist.std().item(),
            'cosine_similarity': all_cosine_sim.mean().item(),
            'cosine_std': all_cosine_sim.std().item()
        }


if __name__ == '__main__':
    # Test loss functions
    print("Testing TurbulenceRobustLoss...")
    
    batch_size = 4
    feature_dim = 512
    
    # Dummy features
    clean_feat = torch.randn(batch_size, feature_dim)
    turb_feat = torch.randn(batch_size, feature_dim)
    filenames = ['img1.jpg', 'img2.jpg', 'img1.jpg', 'img3.jpg']  # img1 appears twice
    
    # Test with all components
    criterion = TurbulenceRobustLoss(
        spatial_weight=1.0,
        frequency_weight=1.0,
        contrastive_weight=0.5
    )
    
    losses = criterion(clean_feat, turb_feat, filenames)
    
    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    # Test individual components
    print("\nTesting individual components:")
    
    # Spatial only
    criterion_spatial = TurbulenceRobustLoss(
        spatial_weight=1.0,
        frequency_weight=0.0,
        contrastive_weight=0.0
    )
    losses_spatial = criterion_spatial(clean_feat, turb_feat)
    print(f"Spatial only: {losses_spatial['total'].item():.4f}")
    
    # Frequency only
    criterion_freq = TurbulenceRobustLoss(
        spatial_weight=0.0,
        frequency_weight=1.0,
        contrastive_weight=0.0
    )
    losses_freq = criterion_freq(clean_feat, turb_feat)
    print(f"Frequency only: {losses_freq['total'].item():.4f}")
    
    # Contrastive only
    criterion_contr = TurbulenceRobustLoss(
        spatial_weight=0.0,
        frequency_weight=0.0,
        contrastive_weight=1.0
    )
    losses_contr = criterion_contr(clean_feat, turb_feat, filenames)
    print(f"Contrastive only: {losses_contr['total'].item():.4f}")
    
    # Test metrics
    print("\nTesting FeatureDistanceMetric...")
    l2 = FeatureDistanceMetric.l2_distance(clean_feat, turb_feat)
    cosine = FeatureDistanceMetric.cosine_similarity(clean_feat, turb_feat)
    
    print(f"L2 distances: {l2}")
    print(f"Cosine similarities: {cosine}")
    print(f"Mean L2: {l2.mean().item():.4f}")
    print(f"Mean cosine: {cosine.mean().item():.4f}")
    
    print("\nLoss tests complete!")

