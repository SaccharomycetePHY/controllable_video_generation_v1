from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint

from src.models.unet import UNet3DConditionModel, UNet3DConditionOutput, logger
from einops import rearrange
from src.models.orig_attention import CrossAttention

class HackUNet3DConditionModel(UNet3DConditionModel):
    def __init__(self, *args, **kwargs):
        in_channels = 12
        kwargs["in_channels"] = in_channels
        kwargs["cross_attention_dim"] = 768
        kwargs["use_motion_module"] = True
        kwargs["motion_module_resolutions"] = [1, 2, 4, 8]
        kwargs["unet_use_cross_frame_attention"] = False
        kwargs["unet_use_temporal_attention"] = False

        kwargs["motion_module_type"] = "Vanilla"
        kwargs["motion_module_kwargs"] = {
            "num_attention_heads": 8,
            "num_transformer_block": 1,
            "attention_block_types": ["Temporal_Self", "Temporal_Self"],
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 32,
            "temporal_attention_dim_div": 1,
            "zero_initialize": True
        }

        # 调用父类的初始化
        super().__init__(*args, **kwargs)
        
        # 初始化时序交叉注意力层
        self.init_temporal_crossattention()

    def init_temporal_crossattention(self):
        # 为每个block初始化交叉注意力层
        for i in range(len(self.down_blocks)):
            block = self.down_blocks[i]
            setattr(block, "pose_attn", CrossAttention(
                query_dim=block.resnets[-1].conv2.out_channels,
                cross_attention_dim=512,
                heads=8,
                dim_head=64,
                dropout=0.0,
                bias=False
            ))

        # 中间block的交叉注意力
        setattr(self.mid_block, "pose_attn", CrossAttention(
            query_dim=self.mid_block.resnets[-1].conv2.out_channels,
            cross_attention_dim=512,
            heads=8,
            dim_head=64,
            dropout=0.0,
            bias=False
        ))

        # 上采样block的交叉注意力
        for i in range(len(self.up_blocks)):
            block = self.up_blocks[i]
            setattr(block, "pose_attn", CrossAttention(
                query_dim=block.resnets[-1].conv2.out_channels,
                cross_attention_dim=512,
                heads=8,
                dim_head=64,
                dropout=0.0,
                bias=False
            ))

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        latent_pose: Optional[Tuple[torch.Tensor]] = None,  # new add
        latent_pose_1d: Optional[torch.Tensor] = None,  # new add
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        print(latent_pose_1d.shape)
        latent_pose_1d = latent_pose_1d.reshape(latent_pose_1d.shape[0], 1, latent_pose_1d.shape[1])

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        sample = self.conv_in(sample)

        # add latent_pose
        if latent_pose is not None and isinstance(latent_pose, (tuple, list)) and len(latent_pose) > 0:
            sample = sample + latent_pose[0]
            latent_pose = latent_pose[1:]

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

            if latent_pose_1d is not None:
                # 将sample从[B, C, F, H, W]转为[B, C, H, W]
                is_5d = False
                if len(sample.shape) == 5:
                    sample = rearrange(sample, 'b c f h w -> (b f) c h w')
                    is_5d = True
                
                # 将sample从[B, C, H, W]转为[B, L, C]
                b, c, h, w = sample.shape
                sample = rearrange(sample, 'b c h w -> b (h w) c')
                sample = downsample_block.pose_attn(sample, encoder_hidden_states=latent_pose_1d) + sample
                # 将sample从[B, L, C]转为[B, C, H, W]
                sample = rearrange(sample, 'b (h w) c -> b c h w', h=h, w=w)
                if is_5d:
                    sample = rearrange(sample, '(b f) c h w -> b c f h w', b=1)

            if latent_pose is not None and isinstance(latent_pose, (tuple, list)) and len(latent_pose) > 0:
                sample += latent_pose[0]
                latent_pose = latent_pose[1:]
            down_block_res_samples += res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        if latent_pose_1d is not None:
            # 将sample从[B, C, F, H, W]转为[B, C, H, W]
            is_5d = False
            if len(sample.shape) == 5:
                sample = rearrange(sample, 'b c f h w -> (b f) c h w')
                is_5d = True
            # 将sample从[B, C, H, W]转为[B, L, C]
            b, c, h, w = sample.shape
            sample = rearrange(sample, 'b c h w -> b (h w) c')
            sample = self.mid_block.pose_attn(sample, encoder_hidden_states=latent_pose_1d) + sample
            # 将sample从[B, L, C]转为[B, C, H, W]
            sample = rearrange(sample, 'b (h w) c -> b c h w', h=h, w=w)
            if is_5d:
                sample = rearrange(sample, '(b f) c h w -> b c f h w', b=1)

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size, encoder_hidden_states=encoder_hidden_states,
                )

            if latent_pose_1d is not None:
                # 将sample从[B, C, F, H, W]转为[B, C, H, W]
                is_5d = False
                if len(sample.shape) == 5:
                    sample = rearrange(sample, 'b c f h w -> (b f) c h w')
                    is_5d = True
                # 将sample从[B, C, H, W]转为[B, L, C]
                b, c, h, w = sample.shape
                sample = rearrange(sample, 'b c h w -> b (h w) c')
                sample = upsample_block.pose_attn(sample, encoder_hidden_states=latent_pose_1d) + sample
                # 将sample从[B, L, C]转为[B, C, H, W]
                sample = rearrange(sample, 'b (h w) c -> b c h w', h=h, w=w)
                if is_5d:
                    sample = rearrange(sample, '(b f) c h w -> b c f h w', b=1)
                
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
