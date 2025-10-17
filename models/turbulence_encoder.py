"""
Turbulence-Robust Feature Encoder
Multi-scale CNN encoder for learning turbulence-invariant features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TurbulenceEncoder(nn.Module):
    """
    CNN encoder for turbulence-invariant feature learning.
    Progressively downsamples with residual connections.
    """
    
    def __init__(self, input_channels=3, base_channels=64, depth=4, 
                 bottleneck_dim=512, use_multiscale=False):
        """
        Args:
            input_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels (doubles each layer)
            depth: Number of downsampling layers
            bottleneck_dim: Dimension of bottleneck feature vector
            use_multiscale: Extract and concatenate features from multiple layers
        """
        super().__init__()
        
        self.depth = depth
        self.use_multiscale = use_multiscale
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder layers (progressive downsampling)
        self.encoder_layers = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            layer = EncoderBlock(in_ch, out_ch, downsample=(i > 0))
            self.encoder_layers.append(layer)
            in_ch = out_ch
        
        # Global pooling + bottleneck
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        final_ch = base_channels * (2 ** (depth - 1))
        self.bottleneck = nn.Linear(final_ch, bottleneck_dim)
        
        # Multi-scale feature dimensions (if used)
        if use_multiscale:
            self.multiscale_dims = [base_channels * (2 ** i) for i in range(depth)]
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            dict with keys:
                - bottleneck: Final feature vector (B, bottleneck_dim)
                - layer_features: List of intermediate features (if use_multiscale)
        """
        features = []
        
        # Initial conv
        x = self.conv_in(x)
        
        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.use_multiscale:
                # Pool to fixed size for concatenation
                pooled = F.adaptive_avg_pool2d(x, (8, 8))
                features.append(pooled)
        
        # Global pooling + bottleneck
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        bottleneck = self.bottleneck(x)
        
        output = {'bottleneck': bottleneck}
        
        if self.use_multiscale:
            output['layer_features'] = features
        
        return output


class EncoderBlock(nn.Module):
    """
    Encoder block with residual connection.
    Optional downsampling via strided convolution.
    """
    
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        identity = self.residual(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out + identity
        return F.relu(out)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_names=['relu3_3'], requires_grad=False, use_bottleneck=False, bottleneck_dim=512):
        super().__init__()
        
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features
        
        self.layer_names = layer_names
        self.use_bottleneck = use_bottleneck
        
        # VGG layer mapping
        self.layer_indices = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29
        }
        
        # Channel counts
        self.layer_channels = {
            'relu1_2': 64,
            'relu2_2': 128,
            'relu3_3': 256,
            'relu4_3': 512,
            'relu5_3': 512
        }
        
        # Extract layers
        max_idx = max(self.layer_indices[name] for name in layer_names)
        self.features = nn.Sequential(*list(vgg.children())[:max_idx + 1])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # Add bottleneck projection if requested
        if use_bottleneck:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            # For single layer, just project that layer's channels
            in_channels = self.layer_channels[layer_names[0]]
            self.projection = nn.Linear(in_channels, bottleneck_dim)
        
        self.eval()
    
    def forward(self, x):
        outputs = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            for name, idx in self.layer_indices.items():
                if i == idx and name in self.layer_names:
                    outputs[name] = x
        
        # If using bottleneck, pool and project
        if self.use_bottleneck:
            # Take the first (and only) layer
            feat = outputs[self.layer_names[0]]
            pooled = self.global_pool(feat).view(feat.size(0), -1)
            bottleneck = self.projection(pooled)
            return {'bottleneck': bottleneck}
        
        return outputs


class VGGMultiLayerExtractor(nn.Module):
    """
    VGG extractor that combines multiple layers with learned weights.
    Tests whether multi-layer fusion alone solves turbulence robustness.
    """
    
    def __init__(self, layer_names=['relu2_2', 'relu3_3', 'relu4_3'],
                 layer_weights=None, bottleneck_dim=512):
        """
        Args:
            layer_names: Layers to extract and combine
            layer_weights: Optional fixed weights for combining layers
            bottleneck_dim: Output feature dimension
        """
        super().__init__()
        
        self.vgg = VGGFeatureExtractor(layer_names)
        self.layer_names = layer_names
        
        if layer_weights is None:
            # Default weights (prioritize relu3_3)
            layer_weights = [0.2, 0.5, 0.3]
        
        self.layer_weights = nn.Parameter(
            torch.tensor(layer_weights, dtype=torch.float32),
            requires_grad=False
        )
        
        # Projection to bottleneck
        # Note: Actual implementation would need to handle different spatial sizes
        # For simplicity, using global average pooling + concatenation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Layer channels for VGG
        self.layer_channels = {
            'relu1_2': 64,
            'relu2_2': 128,
            'relu3_3': 256,
            'relu4_3': 512,
            'relu5_3': 512
        }
        
        total_ch = sum(self.layer_channels[name] for name in layer_names)
        self.projection = nn.Linear(total_ch, bottleneck_dim)
    
    def forward(self, x):
        """
        Extract and combine multi-layer features.
        
        Returns:
            dict with key 'bottleneck'
        """
        # Extract features
        features = self.vgg(x)
        
        # Pool each layer and concatenate
        pooled = []
        for name in self.layer_names:
            feat = features[name]
            pooled_feat = self.global_pool(feat).view(feat.size(0), -1)
            pooled.append(pooled_feat)
        
        # Concatenate
        combined = torch.cat(pooled, dim=1)
        
        # Project to bottleneck
        bottleneck = self.projection(combined)
        
        return {'bottleneck': bottleneck}


def get_model(model_type='custom', **kwargs):
    """
    Factory function for creating models.
    
    Args:
        model_type: 'custom', 'vgg_single', or 'vgg_multi'
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    if model_type == 'custom':
        return TurbulenceEncoder(**kwargs)
    elif model_type == 'vgg_single':
        layer = kwargs.get('layer_name', 'relu3_3')
        return VGGFeatureExtractor(layer_names=[layer])
    elif model_type == 'vgg_multi':
        return VGGMultiLayerExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    print("Testing TurbulenceEncoder...")
    
    # Standard encoder
    model = TurbulenceEncoder(depth=4, bottleneck_dim=512, use_multiscale=False)
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Bottleneck shape: {output['bottleneck'].shape}")
    
    # Multi-scale encoder
    print("\nTesting multi-scale encoder...")
    model_ms = TurbulenceEncoder(depth=4, bottleneck_dim=512, use_multiscale=True)
    output_ms = model_ms(x)
    print(f"Bottleneck shape: {output_ms['bottleneck'].shape}")
    print(f"Number of layer features: {len(output_ms['layer_features'])}")
    for i, feat in enumerate(output_ms['layer_features']):
        print(f"  Layer {i}: {feat.shape}")
    
    # VGG baseline
    print("\nTesting VGG baseline...")
    vgg = VGGFeatureExtractor(layer_names=['relu3_3'], use_bottleneck=True, bottleneck_dim=512)
    output_vgg = vgg(x)
    print(f"VGG bottleneck shape: {output_vgg['bottleneck'].shape}")
        
    # VGG multi-layer
    print("\nTesting VGG multi-layer...")
    vgg_multi = VGGMultiLayerExtractor()
    output_vgg_multi = vgg_multi(x)
    print(f"VGG multi-layer bottleneck: {output_vgg_multi['bottleneck'].shape}")
    
    print("\nModel tests complete!")

