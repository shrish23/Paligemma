import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modelling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache:

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # THe shape of the key_cache is [Batch_size, num_heads_KV, seq_len, Head_dim]
            return self.key_cache[0].shape[-2]
        
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_size, Num_heads_kv, seq_len, Head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)# concatenate along the seq_len dimension
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

# This is the configuration class for the Gemma model
class GemmaConfig:

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,# number of heads for the query
        num_key_value_heads,# number of heads for the key and value
        head_dim=256,# dimensions each head should have
        max_position_embeddings=8192,# maximum number of positions the model can handle
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

# This is the configuration class for the PaliGemma model
class PaliGemmaConfig:

    def __init__(
            self,
            vision_config = None,
            text_config = None,
            ignore_index=-100,
            image_token_index=256000,# The index of the image token in the vocabulary
            vocab_size=257152,
            projection_dim=2048,
            hidden_size=2048,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)# Config of the text model
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2# how many patches for each image: no. of tokens
        self.vision_config.projection_dim = projection_dim


# This is the class for the multi-modal projector
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)# linear layer to project the image features to the same size as the text embeddings

    def forward(self, image_features: torch.Tensor):
        #[batch size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states
    

#The RMS Normalization layer
class GemmaRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))# number of parameters to learn

    def _norm(self, x):
        # We add eps to the denominator to avoid division by zero. This is a common trick in machine learning
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)# RMS normalization. We are calculating 1/sqrt(mean(x^2)).
    
    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16)*w whilst Gemma is (x*w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

# THe Gemma MLP layer
class GemmaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)# linear layer to project the hidden states to the intermediate size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)# linear layer to project the hidden states to the intermediate size
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)# linear layer to project the intermediate states to the hidden size

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        # y = torch.gelu(y, approximate="tanh") # [batch_size, seq_len, intermediate_size]
        # j = self.up_proj(x) # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        # z = y * j # [batch_size, seq_len, intermediate_size]
        # z = self.down_proj(z) # [batch_size, seq_len, intermediate_size] -> [batch_size, seq_len, hidden_size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
    

# The repeated key and value function
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# The rotary positional embedding
class GemmaRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Caluculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, .., dim//2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy inv_freq tensor for batch in the sequence
        # inv_freq expanded: [Batch_size, Head_dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)# expand the inv_freq tensor to the batch size
        #position_ids_expanded: [Batch_size, 1, Seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):# used for mixed precision training
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_size, Head_dim // 2, Seq_len] @ [Batch_Size, 1, Seq_len] -> [Batch_size, Seq_len, Head_dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [batch_size, seq_len, Head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_size, Seq_len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2]# Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]# Takes the second half of the last dimension
    return torch.cat([-x2, x1], dim=-1)
    
# Apply the rotary positional embedding
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)# Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)# Add the head dimension
    # apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# The Gemma Attention Layer
class GemmaAttention(nn.Module):
    
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim# dimensions each head should have
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, "The hidden size must be divisible by the number of heads"

        # NUmber of heads = 8
        # hidden size = 1024
        # head dimensions = 1024 / 8 = 128
        # Wq: [1024, 8* 128]= [1024, 1024]
        # Wk: [1024, 4* 128]= [1024, 512]
        # Wv: [1024, 4* 128]= [1024, 512]
        # Grouped-query attention: Every two head of the query will share the same key and value
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)# linear layer to project the hidden states to the hidden size
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)# linear layer to project the hidden states to the hidden size
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)# linear layer to project the hidden states to the hidden size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)# linear layer to project the hidden states to the hidden size
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base = self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()# [batch_size, seq_len, hidden_size]
        # [batch_size, seq_len, num_heads_q * head_dim]
        query_states = self.q_proj(hidden_states)
        # [batch_size, seq_len, num_heads_KV * head_dim]
        key_states = self.k_proj(hidden_states)
        # [batch_size, seq_len, num_heads_KV * head_dim]
        value_states = self.v_proj(hidden_states)
        # [batch_size, num_heads_Q, seq_len, Head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads_KV, seq_len, Head_dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads_KV, seq_len, Head_dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [batch_size, seq_len, head_dim], [batch_size, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [batch_size, num_heads_q, seq_len, Head_dim], [Batch_size, Num_heads_KV, seq_len, Head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads in the query to reverse the grouping as we do not have a custom cuda kernel for grouped attention
        # this can be done in a more efficient way if used Flash Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax
        # [batch_size, num_heads_Q, seq_len_Q, Seq_len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # Multiply by the values. [Batch_size, Num_heads_q, Seq_Len_Q, Seq_len_KV] x [Batch_size, Num_heads_KV, Seq_len_KV, Head_dim] -> [Batch_size, Num_heads_Q, Seq_len_Q, Head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should have the shape {(bsz, self.num_heads, q_len, self.head_dim)}, but has shape {attn_output}"
            )
        
        # Make sure the sequence length is the second dimension. [Batch_size, num_heads_Q, Seq_len_Q, Head_dim] -> [Batch_size, Seq_len_Q, num_heads_Q, Head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_size, Seq_len_Q, Num_heads_Q, Head_dim] -> [Batch_size, Seq_len_Q, Num_heads_Q * Head_dim]
        attn_output = attn_output.view(bsz, q_len, -1)

        # Multiply by Wo to get the final output. [Batch_size, Seq_len_Q, Num_heads_Q * Head_dim] -> [Batch_size, Seq_len_Q, hidden_size]
        attn_output = self.o_proj(attn_output)# mixing the heads together

        return attn_output, attn_weights


# The Gemma Decoder Layer
class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        #[batch_size, seq_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)

        # [batch_size, seq_len, hidden_size]
        hidden_states, _, = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # [batch_size, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        # [Batch_size, seq_len, hidden_size]
        residual = hidden_states
        # [Batch_size, seq_len, hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_size, seq_len, hidden_size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_size, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        return hidden_states


# The Gemma model: The embedding layer and the list of transformer layers
class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)# embedding layer for the tokens
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]# list of layers for transformer
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)# RMSNorm layer

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,# applying rotary positional encoding
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        
        #[Batch_size, seq_len, hidden_size]
        hidden_states = inputs_embeds
        #[Batch_size, seq_len, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer# this is to scale the hidden states: magnitude of numbers remain same even if the dimensionality changes

        for decoder_layer in self.layers:
            # Batch_size, seq_len, hidden_size
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)

        #[Batch_size, seq_len, hidden_size]
        return hidden_states

# The GEmma Language model
class GemmaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)# linear layer to project the hidden states to the logits

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):# Tie the weights between the language model and the token embeddings to reduce the number of parameters: parameter sharing
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        
        #input_embeds: [batch_size, sequence_length, hidden_size]
        #outputs: batch_size, sequence_length, hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data


# THis is called Conditional Generation because the model conditions the generation of text based on the input image
class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.vision_tower = SiglipVisionModel(config.vision_config)# image encoder
        self.config = config
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)# linear layer to project the image features to the same size as the text embeddings
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)# Transformer dcoder
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        # Tie the weights between the language model and the token embeddings to reduce the number of parameters
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype,device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: (batch_size, sequence_length, embed_dim)
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # Combine the embeddings of the image tokens, the text and mask out all the padding tokens
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # We create masks to identify the image tokens and the text tokens and padding tokens: these will help us to merge the image features with the text embeddings
        # shape: [batch_size, sequence_length] true for text tokens, false for image tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # shape: [batch_size, sequence_length] true for image tokens, false for text tokens
        image_mask = input_ids == self.config.image_token_index
        # shape: [batch_size, sequence_length] true for padding tokens
        padding_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we will not be able to multiply them with the embeddings in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Add image embeddings: we cannot use the torch.where function because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        # We will use the image mask to index the image tokens and add the image features to the final embeddings
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)# copy the values from scaled_image_features to final_embeddings where image_mask_expanded is true
        # Zero out the paddings
        final_embedding = torch.where(padding_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ### CREATE THE ATTENTION MASK###

        dtype,device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase(that is, we send all the prompt of the user to the kv caches q,k,v)
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1, "The query length must be 1 when generating tokens"
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous
            # This on;y works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [batch_size, Q_len, KV_len] -> [Batch_size, Num_heads_Q, Q_len, KV_len]
        causal_mask = causal_mask.unsqueeze(1)# add the head dimension


        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position of the key and value
            # suppose
            position_ids = attention_mask.cumsum(-1)[:,-1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,# image extracted from paligemma processor
            attention_mask: Optional[torch.Tensor] = None,# provided ny the tokenizer
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The attention mask should be 1 for all elements in the input_ids"

        #1. Extract input features
        #shape: (batch_size, sequence_length, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)# converting all input tokens into embeddings. embeddings by image placeholder tokens are junk and will be replaced

        # 2. Merge text and images
        #[batch size, channels, height, width] -> [batch size, num_patches, embed_dim]
        selected_image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # Project the image features to the same size as the text embeddings
        #[batch size, num_patches, embed_dim] -> [batch size, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_features)

        # Merge the image features with the text embeddings
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        # 3. Forward pass through the transformer decoder

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs