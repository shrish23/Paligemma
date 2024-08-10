from typing import Optional, Tuple
import torch
import torch.nn as nn

# Paligemma comes in different sizes so we need to define the configuration
class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size: int = 768,
            intermediate_size: int = 3072,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            num_channels: int = 3,
            image_size: int = 224,
            patch_size: int = 16,
            layer_norm_eps: float = 1e-6,
            attention_dropout: float = 0.0,
            num_image_tokens: int = None,
            **kwargs
    ):
        
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


# Siglip Embeddings: converts the image into patches and then into embeddings
class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size = self.patch_size,
            stride=self.patch_size,
            padding="valid",# This indicates no padding is added to the input image
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2# number of patches in the image: (image_size/patch_size)^2
        self.num_positions = self.num_patches# number of positional embeddings we need
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)# learned positional embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),# expand the tensor to the desired size i.e. [1, num_patches]
            persistent=False,
        )# register_buffer: This is used to register a buffer that should not to be considered a model parameter

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:

        _,_,height, width = pixel_values.shape# [batch_size, num_channels, height, width]
        #Convolve the 'patch size' kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The ouput of the covolution will have shape [batch_size, embed_dim, num_patches_height, num_patches_width]
        # where num_patches_height = height/patch_size and num_patches_width = width/patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [batch_size, embed_dim, num_patches_height, num_patches_width] -> [batch_size, embed_dim, num_patches]
        # num_patches = num_patches_height * num_patches_width
        embeddings = patch_embeds.flatten(2)# flatten the last two dimensions
        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim]
        embeddings = embeddings.transpose(1, 2)# transpose the last two dimensions
        # Add position embeddings to each patch. Each positional encoding is a vector size of embed_dim
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [batch_size, num_patches, embed_dim]
        return embeddings


# Siglip Multi-Layer Perceptron
class SiglipMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)# vectors are expanded to a larger size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [batch_size, num_patches, intermediate_size]
        # we use GeLU because it is a smooth approximation of the ReLU function and it is used in the original transformer paper
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")# activation function: GELU -> GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        # [batch_size, num_patches, embed_dim]
        return hidden_states


# Siglip Attention: The self-attention mechanism in the transformer model
class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # scale factor: Equivalent to 1/sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)# linear layer for key
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)# linear layer for value
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)# linear layer for query
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)# linear layer for output

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)
        # split each tokens into smaller tokens
        # query_states: [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)# [batch_size, num_heads, num_patches, head_dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)# [batch_size, num_heads, num_patches, head_dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)# [batch_size, num_heads, num_patches, head_dim]
        # compute the attention scores
        # attn_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3))* self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is {attn_weights.size()}")
        
        # Apply the softmask row-wise, attn_weights: [batch_size, num_heads, Num_patches, Num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #Multiply attention weights with value_states, attn_output: [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}")
        
        # [batch_size, num heads, num patches, Head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)# concatenate the heads
        # [batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Siglip Encoder: The encoder is a stack of N layers
class SiglipEncoder(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [batch_size, num_patches, embed_dim]
        hidden_states = inputs_embeds
        for layer in self.layers:
            # hidden_states: [batch_size, num_patches, embed_dim]
            hidden_states = layer(hidden_states)
        # hidden_states: [batch_size, num_patches, embed_dim]
        return hidden_states


# Siglip Encoder: The encoder is a stack of N layers
class SiglipEncoderLayer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)# self attention layer
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)# feedforward layer
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states = hidden_states + residual
        #residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states = hidden_states + residual
        return hidden_states


# Siglip Transformer
class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)# extract patches from image
        self.encoder = SiglipEncoder(config)# encoder
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        #pixel values: [batch_size, num_channels, height, width] -> [batch_size, num_patches, emed_dim]
        hidden_states = self.embeddings(pixel_values=pixel_values)# extract patches from image

        last_hidden_state = self.encoder(inputs_embeds = hidden_states)# encoder: list of layers of transformer including the attention and feedforward layers

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


# The SiglipVision model is a simple vision transformer
# Now we define the model

class SiglipVisionModel(nn.Module):

    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        #pixel values loaded with numpy array
        #[batch_size, num_channels, height, width] -> [batch_size, num_patches, emed_dim]: converts the image into patches and then into embeddings, gives a list of embeddig for each patch
        return self.vision_model(pixel_values=pixel_values)