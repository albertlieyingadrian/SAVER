import torch
import torch.nn as nn

from .utils import (
    load_vlm_model,
)

from .modules import (
    SemanticRouter,
    MoEFusion,
    VisionTokenProcessor
)


class SAVERModel(nn.Module):
    '''
    Wrapper model that augments a frozen Vision-Language Model (VLM) with
    view-specific soft prompts and cross-attention over visual features.

    The model has two stages:
        - Stage 1: Learn view-specific query prompts that attend to visual features.
          Each sample uses a single target view (scene, object, etc.).
        - Stage 2: Fuse multiple view prompts (across all view types) and replace
          the vision token region with a concatenation of all view prompts
          and visual features, then train a high-level fusion behavior.

    The base VLM remains frozen in both stages. Only view queries and cross-attn
    modules are trainable in Stage 1; in Stage 2, the model is typically used
    in a frozen/inference-like manner (depending on your training setup).
    '''
    def __init__(self, base_model, processor, view_names, stage="stage1"):
        super().__init__()
        self.vlm = base_model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.embedding = self.vlm.get_input_embeddings()
        self.stage = stage
        self.view_names = view_names
        self.image_token_dropout_prob = 0.0  # Placeholder for potential future use

        # 1) Freeze the entire base VLM by default
        self._freeze_vlm_parameters()

        # 2) Initialize view-specific query prompts and cross-attention modules
        self._init_view_prompts_and_cross_attn()

        # 3) Configure which parameters are trainable depending on stage
        self._configure_trainable_parameters(stage)
        
        hidden_dim = self.embedding.embedding_dim

        # --- MoE Router ---
        self.router = SemanticRouter(
            hidden_dim=hidden_dim,
            num_views=len(self.view_names)
        )
        
        self.fusion = MoEFusion()
        
        self.token_processor = VisionTokenProcessor()

    def _freeze_vlm_parameters(self):
        '''
        Freeze all parameters of the underlying VLM so that only the
        added modules (e.g., view prompts, cross-attention) can be trained.
        '''
        for p in self.vlm.parameters():
            p.requires_grad = False

    def _init_view_prompts_and_cross_attn(self, num_query_tokens=5, num_heads=1):
        '''
        Initialize learnable view-specific query prompts and cross-attention modules.

        Args:
            num_query_tokens (int): Number of query tokens per view.
            num_heads (int): Number of attention heads for MultiheadAttention.
        '''
        N = num_query_tokens

        # Learnable query matrix for each view: (N, D)
        self.q_view = nn.ParameterDict({
            view: nn.Parameter(torch.randn(N, self.embedding.embedding_dim))
            for view in self.view_names
        })

        # View-specific cross-attention modules (query: q_view, key/value: vision features)
        self.cross_attn = nn.ModuleDict({
            view: nn.MultiheadAttention(
                embed_dim=self.embedding.embedding_dim,
                num_heads=num_heads,
                batch_first=True
            )
            for view in self.view_names
        })

    def _configure_trainable_parameters(self, stage):
        '''
        Set requires_grad flags according to the training stage.

        Stage 1:
            - Only q_view and cross_attn parameters are trainable.
        Stage 2:
            - By default, all parameters are frozen (you can customize this
              if you want to train an extra fusion module).
        '''
        for name, param in self.named_parameters():
            if stage == "stage1":
                # Train only view queries and cross-attention modules
                param.requires_grad = ("q_view" in name) or ("cross_attn" in name)
            elif stage == "stage2":
                # Freeze everything (can be relaxed if you add extra modules)
                param.requires_grad = (
                    ("router" in name)
                )
                

        # Explicitly ensure q_view is frozen in Stage 2
        for p in self.q_view.parameters():
            if stage == "stage2":
                p.requires_grad = False

    def _forward_stage1(self, **kwargs):
        '''
        Forward pass for Stage 1 (single-view learning).

        For each sample:
            - Extract visual features using the VLM's visual backbone.
            - Select the corresponding view name from `task`.
            - Apply view-specific cross-attention (q_view[view] → vision_feats).
            - Replace the original vision token region with the resulting z.
            - Compute cross-entropy loss using the VLM language head.
        '''
        pixel_values = kwargs["pixel_values"]
        messages = kwargs["messages"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        task = kwargs["task"]

        input_device = pixel_values.device
        B = len(messages)

        # 1) Extract visual features from the VLM
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)

        # 2) Compute view-specific prompts via cross-attention
        z_list = []
        for i in range(B):
            view_name = task[i]
            q = self.q_view[view_name]
            z_i, _ = self.cross_attn[view_name](q, vision_feats, vision_feats)
            z_list.append(z_i)
        z_stack = torch.stack(z_list, dim=0)  # (B, N, D) under current assumptions

        # 3) Replace vision tokens with z for each example
        input_embeds, attn_mask, labels_list = [], [], []
        for i in range(B):
            embed, attn, label = self.token_processor._replace_vision_tokens_stage1(
                self.vlm,
                input_ids[i],
                attention_mask[i],
                z_stack[i],
                input_device
            )
            input_embeds.append(embed)
            attn_mask.append(attn)
            labels_list.append(label)

        # 4) Pad to batch (batch_first=True)
        input_embeds = torch.nn.utils.rnn.pad_sequence(
            input_embeds, batch_first=True
        )
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_mask, batch_first=True
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True
        )

        # 5) Forward through the frozen VLM language head
        outputs = self.vlm(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.loss, outputs

    def _forward_stage2(self, **kwargs):
        '''
        Forward pass for Stage 2 (multi-view fusion).

        For each sample:
            - Extract visual features using the VLM's visual backbone.
            - For every view in VIEW_NAMES, apply cross-attention (q_view[view] → vision_feats).
            - Concatenate all view-specific z's into a single vector z_i.
            - Replace the original vision token region with [z_i, vision_feats].
            - Forward through the VLM to obtain a fusion-based loss.
        '''
        pixel_values = kwargs["pixel_values"]
        messages = kwargs["messages"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]

        input_device = pixel_values.device
        B = len(messages)

        # 1) Extract visual features for the whole batch
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
        
        # ---- MoE routing ----
        if vision_feats.dim() == 2:
            # (Lv, D) → (1, Lv, D)
            vision_feats = vision_feats.unsqueeze(0)

        B, Lv, D = vision_feats.shape
        
        # -------- Vision global --------
        vision_global = vision_feats.mean(dim=1)  # (B, D)
        
        # -------- Text global --------
        with torch.no_grad():
            text_outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        text_hidden = text_outputs.hidden_states[-1]
        text_global = text_hidden[:, 0, :]  # (B, D)
        
        # -------- Semantic-aware Router --------
        gate = self.router(
            vision_global,
            text_global
        )

        z_all = []
        for i in range(B):
            
            # ---- experts ----
            z_views = []
            for view in self.view_names:
                q = self.q_view[view]                      # (N, D)
                z, _ = self.cross_attn[view](
                    q.unsqueeze(0),                        # (1, N, D)
                    vision_feats[i:i+1],                   # (1, Lv, D)
                    vision_feats[i:i+1]
                )
                z_views.append(z.squeeze(0))               # (N, D)

            Z = torch.stack(z_views, dim=0)                # (V, N, D)

            # weighted_sum = (gate[i].view(-1,1,1) * Z).sum(dim=0)
            # Z_mean = Z.mean(dim=0)   # baseline 평균
            # z_i = Z_mean + self.fusion * (weighted_sum - Z_mean)
            z_i = self.fusion(
                Z,
                gate[i]
            )
            z_all.append(z_i)
            
        z_all = torch.stack(z_all, dim=0)                  # (B, N, D)

        # 3) Replace vision tokens with [z_all[i], vision_feats] for each example
        input_embeds, attn_mask, labels_list = [], [], []
        for i in range(B):
            embed, attn, label = self.token_processor._replace_vision_tokens_stage2(
                self.vlm,
                input_ids[i],
                attention_mask[i],
                z_all[i],
                vision_feats[i],     # Note: original code passes full vision_feats
                input_device
            )
            input_embeds.append(embed)
            attn_mask.append(attn)
            labels_list.append(label)

        # 4) Pad to batch
        input_embeds = torch.nn.utils.rnn.pad_sequence(
            input_embeds, batch_first=True
        )
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_mask, batch_first=True
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True
        )

        # 5) Forward through the VLM language head
        outputs = self.vlm(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        
        # # ===== MoE auxiliary loss =====
        num_views = len(self.view_names)
        target = torch.full_like(gate, 1.0 / num_views)

        balance_loss = ((gate - target) ** 2).mean()
        entropy_loss = -(gate * torch.log(gate + 1e-8)).mean()

        loss = loss + 0.05 * balance_loss + 0.01 * entropy_loss
        # ==============================
    
        return loss, outputs

    def forward(self, **kwargs):
        '''
        Dispatch the forward pass to Stage 1 or Stage 2 logic.

        Args:
            **kwargs: Keyword arguments containing:
                - pixel_values
                - messages
                - image_grid_thw
                - input_ids
                - attention_mask
                - task (Stage 1 only)

        Returns:
            (loss, outputs): Same structure as returned by the underlying VLM.
        '''
        if self.stage == 'stage1':
            return self._forward_stage1(**kwargs)
        elif self.stage == 'stage2':
            return self._forward_stage2(**kwargs)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    @torch.no_grad()
    def _generate_stage1(self, **kwargs):
        '''
        Generation logic for Stage 1 (single-view prompt).

        Builds inputs_embeds + attention_mask by injecting a single view's z
        into the vision token region, then calls `vlm.generate`.
        '''
        pixel_values = kwargs["pixel_values"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        task = kwargs["task"]
        attention_mask = kwargs["attention_mask"]

        input_device = pixel_values.device

        # 1) Extract visual features and treat them as a "memory" for cross-attention
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
        vision_feats = vision_feats.unsqueeze(0)  # Match original code's behavior
        B = len(task)

        full_embeds, full_attn_masks = [], []

        # 2) For each sample, inject the corresponding view's z into the sequence
        for i in range(B):
            view_name = task[i]
            q = self.q_view[view_name]

            # Compute view-specific prompt z
            z, _ = self.cross_attn[view_name](
                q.unsqueeze(0),  # (1, N, D)
                vision_feats,
                vision_feats
            )
            z_embed = z.squeeze(0)  # (N, D)
            ids = input_ids[i]
            mask = attention_mask[i]

            # Replace vision region with z
            embed, attn, _ = self.token_processor._replace_vision_tokens_stage1(
                self.vlm,
                ids,
                mask,
                z_embed,
                input_device
            )
            full_embeds.append(embed.unsqueeze(0))
            full_attn_masks.append(attn.unsqueeze(0))

        full_embeds = torch.cat(full_embeds, dim=0)
        full_attn_masks = torch.cat(full_attn_masks, dim=0)

        # 3) Strip internal keywords from kwargs before calling generate
        generate_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {
                "pixel_values", "messages", "image_grid_thw",
                "input_ids", "task", "attention_mask"
            }
        }

        # 4) Call the underlying VLM's generate
        return self.vlm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_masks,
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True,
            top_p=1.0,
            **generate_kwargs
        )

    @torch.no_grad()
    def _generate_stage2(self, **kwargs):
        '''
        Generation logic for Stage 2 (multi-view fusion).

        Uses all views in VIEW_NAMES to build a fused prompt z, concatenates it
        with vision_feats, replaces the vision token region, and then calls
        `vlm.generate` for a single example.
        '''
        pixel_values = kwargs["pixel_values"]
        image_grid_thw = kwargs["image_grid_thw"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]

        input_device = pixel_values.device

        # 1) Extract visual features for the batch
        vision_feats = self.vlm.visual(pixel_values, grid_thw=image_grid_thw)
        # vision_feats = vision_feats.unsqueeze(0)  # Original code behavior
        if vision_feats.dim() == 2:
            vision_feats = vision_feats.unsqueeze(0)
        
        # -------- Vision global --------
        vision_global = vision_feats.mean(dim=1)  # (B, D)

        # -------- Text global --------
        with torch.no_grad():
            text_outputs = self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        text_hidden = text_outputs.hidden_states[-1]
        text_global = text_hidden[:, 0, :]  # (B, D)

        # -------- Semantic-aware Router --------
        gate = self.router(
            vision_global,
            text_global
        )

        # ---- experts ----
        z_views = []
        for view in self.view_names:
            q = self.q_view[view]
            z, _ = self.cross_attn[view](q.unsqueeze(0), vision_feats, vision_feats)
            z_views.append(z.squeeze(0))
            
        # ---- MoE aggregation ----
        Z = torch.stack(z_views, dim=0)
        gate_w = gate.view(-1, 1, 1)
        z_i = (gate_w * Z).sum(dim=0)

        # Take the first element of vision_feats for replacement
        vision_feats_ = vision_feats[0]
        ids = input_ids[0]
        mask = attention_mask[0]

        # 3) Replace vision region with [z_i, vision_feats_]
        embed, attn, _ = self.token_processor._replace_vision_tokens_stage2(
            self.vlm,
            ids,
            mask,
            z_i,
            vision_feats_,
            input_device
        )
        full_embeds = embed.unsqueeze(0)
        full_attn_mask = attn.unsqueeze(0)

        # 4) Strip internal keywords and call generate
        generate_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {
                "pixel_values", "messages", "image_grid_thw",
                "input_ids", "task", "attention_mask"
            }
        }

        return self.vlm.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_mask,
            temperature=0.8,
            do_sample=False,
            **generate_kwargs
        )

    @torch.no_grad()
    def generate(self, **kwargs):
        '''
        Dispatch the generation call to Stage 1 or Stage 2 logic.

        Args:
            **kwargs: Keyword arguments similar to forward() but used for generation.

        Returns:
            torch.LongTensor: Generated token sequences from the underlying VLM.
        '''
        if self.stage == 'stage1':
            return self._generate_stage1(**kwargs)
        elif self.stage == 'stage2':
            return self._generate_stage2(**kwargs)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def gradient_checkpointing_enable(self, **kwargs):
        '''
        Enable gradient checkpointing on the underlying VLM, if supported.

        This is useful when you want to reduce memory usage in training,
        especially with large VLMs.
        '''
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        '''
        Disable gradient checkpointing on the underlying VLM, if supported.
        '''
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
            
            
def build_stage1_model(train_config):
    """
    Build the Stage 1 SAVER model.

    Loads a frozen VLM and adds view-specific query prompts + cross-attention
    modules for single-view training.
    """
    model_name = train_config["model_name"]
    view_names = train_config["VIEW_NAMES"]

    base_model, processor = load_vlm_model(model_name)
    spmodel = SAVERModel(
        base_model,
        processor,
        view_names=view_names,
        stage="stage1"
    )
    return spmodel, processor


def build_stage2_model(train_config, base_model=None, processor=None):
    """
    Build the Stage 2 SAVER model.

    Optionally reuses the Stage 1 base_model/processor if provided.
    Loads pretrained q_view and cross-attention weights from checkpoint.
    """
    # Load VLM if not reused from Stage 1
    if base_model is None or processor is None:
        base_model, processor = load_vlm_model(train_config["model_name"])

    view_names = train_config["VIEW_NAMES"]
    spmodel = SAVERModel(
        base_model,
        processor,
        view_names=view_names,
        stage="stage2"
    )

    # Load pretrained soft-prompt weights from Stage 1 checkpoint
    ckpt = torch.load(train_config["ckpt_path"], map_location="cpu")

    # Load view-specific soft prompts
    for k in spmodel.q_view:
        spmodel.q_view[k].data.copy_(
            ckpt["q_view"][k]
        )

    # Load cross-attention weights
    for k in spmodel.cross_attn:
        spmodel.cross_attn[k].load_state_dict(ckpt["cross_attn"][k])

    return spmodel, processor
