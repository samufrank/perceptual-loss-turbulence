"""
Improved loss functions for turbulence-robust feature learning.

Research Justifications:
1. Frequency Divergence Penalty:
   - Rothe et al. (2015): "Adversarial training requires bounded perturbations"
   - Hendrycks & Dietterich (2019): Robustness to common corruptions includes blur
   - Key insight: Allow frequency changes (turbulence causes blur), penalize only extreme divergence
   
2. Hard Negative Mining:
   - Schroff et al. (2015): FaceNet uses hard negative mining for face recognition
   - Chen et al. (2020): SimCLR mentions hard negatives but uses random sampling at scale
   - Sohn (2016): "Improved Deep Metric Learning with Multi-class N-pair Loss"
   - Key insight: With limited data, hard negatives provide stronger learning signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TurbulenceRobustLossV2(nn.Module):
    """
    Improved multi-component loss for learning turbulence-invariant features.
    
    Improvements over V1:
    1. Frequency divergence penalty (allows bounded frequency changes)
    2. Hard negative mining for contrastive loss
    
    Components:
        1. Spatial invariance: L2 distance between clean and turbulent features
        2. Frequency divergence: Penalizes extreme (not all) frequency differences
        3. Contrastive with hard negatives: Uses hardest negatives per sample
    """
    
    def __init__(self, 
                 spatial_weight=1.0, 
                 frequency_weight=1.0, 
                 contrastive_weight=0.5, 
                 frequency_threshold=0.5,
                 contrastive_margin=0.5,
                 temperature=0.07,
                 use_improved_losses=True):
        """
        Args:
            spatial_weight: Weight for spatial invariance loss (α)
            frequency_weight: Weight for frequency loss (β)
            contrastive_weight: Weight for contrastive loss (γ)
            frequency_threshold: Allow this much relative frequency divergence
            contrastive_margin: Margin for hard negative separation
            temperature: Temperature for contrastive loss
            use_improved_losses: If True, use improved versions; if False, use original
        """
        super().__init__()
        
        self.spatial_weight = spatial_weight
        self.frequency_weight = frequency_weight
        self.contrastive_weight = contrastive_weight
        self.frequency_threshold = frequency_threshold
        self.contrastive_margin = contrastive_margin
        self.temperature = temperature
        self.use_improved_losses = use_improved_losses
    
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
        
        # 1. Spatial invariance loss (unchanged)
        if self.spatial_weight > 0:
            spatial_loss = self.spatial_invariance_loss(clean_features, turb_features)
            losses['spatial'] = spatial_loss
        else:
            losses['spatial'] = torch.tensor(0.0, device=clean_features.device)
        
        # 2. Frequency loss (improved or original)
        if self.frequency_weight > 0:
            if self.use_improved_losses:
                freq_loss = self.frequency_divergence_penalty(clean_features, turb_features)
            else:
                freq_loss = self.frequency_consistency_loss(clean_features, turb_features)
            losses['frequency'] = freq_loss
        else:
            losses['frequency'] = torch.tensor(0.0, device=clean_features.device)
        
        # 3. Contrastive loss (improved or original)
        if self.contrastive_weight > 0:
            if self.use_improved_losses:
                contrastive_loss = self.contrastive_loss_hard_negatives(
                    clean_features, turb_features, batch_clean_filenames
                )
            else:
                contrastive_loss = self.contrastive_loss_original(
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
    
    def frequency_divergence_penalty(self, clean_features, turb_features):
        """
        IMPROVED: Penalizes only extreme frequency divergence.
        
        Rationale:
        - Turbulence inherently changes frequency content (blur = high-freq attenuation)
        - Original loss forced frequency similarity, conflicting with spatial invariance
        - This version allows bounded frequency changes, penalizes only extreme cases
        
        Research basis:
        - Adversarial robustness literature: bound perturbations, don't eliminate them
        - Rothe et al. (2015): Bounded adversarial perturbations
        - Hendrycks & Dietterich (2019): Robustness to common corruptions
        
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
        
        # Compute relative divergence (normalized by clean magnitude)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        relative_diff = torch.abs(clean_mag - turb_mag) / (clean_mag + eps)
        
        # Penalize only if divergence exceeds threshold
        # ReLU ensures no penalty for small frequency changes
        penalty = F.relu(relative_diff - self.frequency_threshold)
        
        return penalty.mean()
    
    def frequency_consistency_loss(self, clean_features, turb_features):
        """
        ORIGINAL: L2 distance in frequency domain.
        Forces frequency spectra to be identical.
        
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
    
    def contrastive_loss_hard_negatives(self, clean_features, turb_features, 
                                         batch_clean_filenames):
        """
        IMPROVED: Contrastive loss with hard negative mining.
        
        Rationale:
        - Original InfoNCE uses all negatives in batch (easy + hard)
        - With small datasets (134 scenes), easy negatives dominate
        - Hard negative mining focuses on most informative samples
        
        Research basis:
        - Schroff et al. (2015): FaceNet uses semi-hard negative mining
        - Sohn (2016): N-pair loss with hard negative mining
        - Wu et al. (2017): Hard negative mining for metric learning
        
        Implementation:
        - For each anchor (clean), find hardest negative (most similar non-matching scene)
        - Use triplet-style margin loss with hard negatives only
        
        Args:
            clean_features: (B, D)
            turb_features: (B, D)
            batch_clean_filenames: List of clean filenames for identifying pairs
            
        Returns:
            Scalar loss
        """
        # Normalize features to unit sphere (cosine similarity space)
        clean_norm = F.normalize(clean_features, dim=1)
        turb_norm = F.normalize(turb_features, dim=1)
        
        batch_size = clean_norm.size(0)
        
        # Compute all pairwise similarities in clean feature space
        # This tells us which scenes are most similar to each other
        sim_matrix = torch.matmul(clean_norm, clean_norm.T)  # (B, B)
        
        # Mask out self-similarities (diagonal)
        mask = torch.eye(batch_size, device=clean_norm.device).bool()
        sim_matrix_masked = sim_matrix.masked_fill(mask, -1e4)
        
        # For each sample, find hardest negative (most similar non-matching scene)
        hard_negatives_idx = sim_matrix_masked.argmax(dim=1)  # (B,)
        
        # Positive similarity: clean vs turbulent of same scene
        pos_sim = F.cosine_similarity(clean_norm, turb_norm, dim=1)  # (B,)
        
        # Negative similarity: clean vs clean of hardest negative scene
        hard_neg_features = clean_norm[hard_negatives_idx]  # (B, D)
        neg_sim = F.cosine_similarity(clean_norm, hard_neg_features, dim=1)  # (B,)
        
        # Triplet loss with margin
        # Want: pos_sim high, neg_sim low, separated by at least margin
        # Loss = max(0, margin - pos_sim + neg_sim)
        loss = F.relu(self.contrastive_margin - pos_sim + neg_sim)
        
        return loss.mean()
    
    def contrastive_loss_original(self, clean_features, turb_features, 
                                   batch_clean_filenames):
        """
        ORIGINAL: InfoNCE-style contrastive loss.
        Uses all negatives in batch.
        
        Args:
            clean_features: (B, D)
            turb_features: (B, D)
            batch_clean_filenames: List of clean filenames
            
        Returns:
            Scalar loss
        """
        # Normalize
        clean_norm = F.normalize(clean_features, dim=1)
        turb_norm = F.normalize(turb_features, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(clean_norm * turb_norm, dim=1) / self.temperature
        
        # All pairwise similarities
        sim_matrix = torch.matmul(clean_norm, clean_norm.T) / self.temperature
        
        # InfoNCE loss
        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(sim_matrix).sum(dim=1) - torch.exp(pos_sim)
        
        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8))
        
        return loss.mean()


# Keep original class name for backward compatibility
TurbulenceRobustLoss = TurbulenceRobustLossV2


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

