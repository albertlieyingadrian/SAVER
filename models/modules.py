import torch
import torch.nn as nn

class SemanticRouter(nn.Module):
    '''
    Semantic-aware Mixture-of-Experts router.

    The router predicts view importance weights conditioned on:
        - global visual representation
        - global textual representation

    The output is a normalized gating vector over all views.
    '''
    def __init__(self, hidden_dim, num_views):
        super().__init__()

        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_views)
        )

    def forward(self, vision_global, text_global):

        router_input = torch.cat(
            [vision_global, text_global],
            dim=-1
        )

        gate = torch.softmax(
            self.router(router_input) / 1.5,
            dim=-1
        )

        return gate
    
class MoEFusion(nn.Module):
    '''
    Mixture-of-Experts fusion module for aggregating
    multiple view-specific prompt representations.
    '''
    def __init__(self):
        super().__init__()

        self.alpha = nn.Parameter(
            torch.tensor(0.5)
        )

    def forward(self, Z, gate):
        '''
        Args:
            Z:
                (V, N, D) stacked view-specific prompts

            gate:
                (V,) routing weights
        '''

        weighted_sum = (
            gate.view(-1,1,1) * Z
        ).sum(dim=0)

        Z_mean = Z.mean(dim=0)

        z_i = Z_mean + self.alpha * (
            weighted_sum - Z_mean
        )

        return z_i
    
class VisionTokenProcessor:
    def _find_vision_token_range(self, ids):
        '''
        Locate the start and end indices of the vision token region in input_ids.

        This assumes:
            - <vision_start> token id: 151652
            - <vision_end> token id:   151653

        Args:
            ids (torch.LongTensor): 1D tensor of token ids.

        Returns:
            (int, int): (vision_start_idx, vision_end_idx)
        '''
        vision_start_idx = (ids == 151652).nonzero(as_tuple=True)[0][0].item()
        vision_end_idx = (ids == 151653).nonzero(as_tuple=True)[0][-1].item()
        return vision_start_idx, vision_end_idx

    def _find_answer_token_range(self, ids):
        '''
        Locate the token range corresponding to the answer region in input_ids.

        This assumes:
            - Answer starts after a special "answer start" token id: 77091
            - Answer ends before the last occurrence of an "answer end" token id: 151645

        Args:
            ids (torch.LongTensor): 1D tensor of token ids.

        Returns:
            (int, int): (start_idx_orig, end_idx_orig)
        '''
        start_idx_orig = (ids == 77091).nonzero(as_tuple=True)[0].item() + 1
        end_idx_orig = (ids == 151645).nonzero(as_tuple=True)[0][-1].item()
        return start_idx_orig, end_idx_orig

    def _map_index_after_vision_replacement(self, old_idx, vision_start_idx, vision_end_idx, shift):
        '''
        Map an original token index to the new index after the vision token block
        has been replaced by a different-length segment.

        Args:
            old_idx (int): Original index.
            vision_start_idx (int): Start index of original vision region.
            vision_end_idx (int): End index of original vision region.
            shift (int): Number of tokens removed (positive) or added (negative)
                         when replacing the vision region.

        Returns:
            int or None: New index if it remains a valid non-vision index,
                         otherwise None if the original index was inside the vision region.
        '''
        if old_idx < vision_start_idx:
            return old_idx
        elif old_idx > vision_end_idx:
            return old_idx - shift
        else:
            # Token was inside the vision region and no longer maps to a single index
            return None

    def _replace_vision_tokens_stage1(self, vlm, ids, mask, z, input_device):
        '''
        Replace the vision token block with view-specific embeddings for Stage 1.

        For Stage 1, the replacement block is:
            [<vision_start>, z (view prompt tokens), <vision_end>]

        The answer tokens are restored into the label tensor while all other
        tokens (including vision-related tokens) are masked out with -100.

        Args:
            ids (torch.LongTensor): 1D input_ids for a single example.
            mask (torch.LongTensor): 1D attention_mask for the same example.
            z (torch.FloatTensor): 2D tensor of view-specific embeddings (Lz, D).
            input_device (torch.device): Device to allocate any new tensors.

        Returns:
            (input_embeds, attn_mask, labels):
                - input_embeds (torch.FloatTensor): Modified input embeddings.
                - attn_mask (torch.LongTensor): Modified attention mask.
                - labels (torch.LongTensor): Label tensor with answer tokens only.
        '''
        # 1) Locate original vision token span
        vision_start_idx, vision_end_idx = self._find_vision_token_range(ids)

        # 2) Split ids and compute embeddings around the vision region
        ids_before = ids[:vision_start_idx]
        ids_after = ids[vision_end_idx + 1:]

        embed_before = vlm.get_input_embeddings()(ids_before)
        embed_after = vlm.get_input_embeddings()(ids_after)
        embed_vs = vlm.get_input_embeddings()(ids[vision_start_idx].unsqueeze(0))  # <vision_start>
        embed_ve = vlm.get_input_embeddings()(ids[vision_end_idx].unsqueeze(0))    # <vision_end>

        # Concatenate: [before, <vs>, z, <ve>, after]
        embed = torch.cat([embed_before, embed_vs, z, embed_ve, embed_after], dim=0)

        # 3) Build new attention mask aligned with new embedding sequence
        mask_before = mask[:vision_start_idx]
        mask_after = mask[vision_end_idx + 1:]
        mask_vs = mask[vision_start_idx].unsqueeze(0)
        mask_ve = mask[vision_end_idx].unsqueeze(0)

        # All z tokens are visible (1s in attention mask)
        z_mask = torch.ones((z.size(0),), dtype=torch.long, device=input_device)
        attn = torch.cat([mask_before, mask_vs, z_mask, mask_ve, mask_after])

        # 4) Build label tensor: mask out everything except the final answer region
        label = ids.clone()
        label[:] = -100  # Start by masking everything

        z_len = z.size(0)
        label_vs = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_ve = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_z = torch.full((z_len,), -100, dtype=torch.long, device=input_device)

        # Concatenate labels in the same layout as embeddings
        label = torch.cat([
            label[:vision_start_idx],
            label_vs,
            label_z,
            label_ve,
            label[vision_end_idx + 1:]
        ], dim=0)

        # 5) Restore answer tokens in the new label positions
        start_idx_orig, end_idx_orig = self._find_answer_token_range(ids)

        num_vision_tokens = vision_end_idx - vision_start_idx + 1
        new_token_len = z_len + 2  # <vs> + z tokens + <ve>
        shift = num_vision_tokens - new_token_len

        # Map original answer index range onto the new sequence
        start_idx = self._map_index_after_vision_replacement(
            start_idx_orig, vision_start_idx, vision_end_idx, shift
        )
        end_idx = self._map_index_after_vision_replacement(
            end_idx_orig, vision_start_idx, vision_end_idx, shift
        )

        if start_idx is not None and end_idx is not None:
            label[start_idx:end_idx] = ids[start_idx_orig:end_idx_orig]

        return embed, attn, label

    def _replace_vision_tokens_stage2(self, vlm, ids, mask, z, vision_feats, input_device):
        '''
        Replace the vision token block with fused view prompts and visual features for Stage 2.

        For Stage 2, the replacement block is:
            [<vision_start>, z (all-view prompts) + vision_feats, <vision_end>]

        Both z and vision_feats are treated as prompt-like tokens that replace
        the original vision token region. Answer tokens are restored in the label
        tensor similarly to Stage 1.

        Args:
            ids (torch.LongTensor): 1D input_ids for a single example.
            mask (torch.LongTensor): 1D attention_mask for the same example.
            z (torch.FloatTensor): 2D tensor of fused view embeddings (Lz, D).
            vision_feats (torch.FloatTensor): 2D tensor of visual features (Lv, D).
            input_device (torch.device): Device to allocate any new tensors.

        Returns:
            (input_embeds, attn_mask, labels):
                - input_embeds (torch.FloatTensor): Modified input embeddings.
                - attn_mask (torch.LongTensor): Modified attention mask.
                - labels (torch.LongTensor): Label tensor with answer tokens only.
        '''
        # 1) Locate original vision token span
        vision_start_idx, vision_end_idx = self._find_vision_token_range(ids)

        # 2) Split ids and compute embeddings around the vision region
        ids_before = ids[:vision_start_idx]
        ids_after = ids[vision_end_idx + 1:]

        embed_before = vlm.get_input_embeddings()(ids_before)
        embed_after = vlm.get_input_embeddings()(ids_after)
        embed_vs = vlm.get_input_embeddings()(ids[vision_start_idx].unsqueeze(0))
        embed_ve = vlm.get_input_embeddings()(ids[vision_end_idx].unsqueeze(0))

        # Concatenate: [before, <vs>, z, vision_feats, <ve>, after]
        embed = torch.cat([
            embed_before,
            embed_vs,
            z,
            vision_feats,
            embed_ve,
            embed_after
        ], dim=0)

        # 3) Build new attention mask
        mask_before = mask[:vision_start_idx]
        mask_after = mask[vision_end_idx + 1:]
        mask_vs = mask[vision_start_idx].unsqueeze(0)
        mask_ve = mask[vision_end_idx].unsqueeze(0)

        prompt_mask = torch.ones(
            (z.size(0) + vision_feats.size(0),),
            dtype=torch.long,
            device=input_device
        )
        attn = torch.cat([mask_before, mask_vs, prompt_mask, mask_ve, mask_after])

        # 4) Build label tensor: mask out everything except the final answer region
        label = ids.clone()
        label[:] = -100

        label_vs = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_ve = torch.full((1,), -100, dtype=torch.long, device=input_device)
        label_z = torch.full(
            (z.size(0) + vision_feats.size(0),),
            -100,
            dtype=torch.long,
            device=input_device
        )

        label = torch.cat([
            label[:vision_start_idx],
            label_vs,
            label_z,
            label_ve,
            label[vision_end_idx + 1:]
        ], dim=0)

        # 5) Restore answer tokens in the new label positions
        start_idx_orig, end_idx_orig = self._find_answer_token_range(ids)

        old_vision_len = vision_end_idx - vision_start_idx + 1
        new_vision_len = z.size(0) + vision_feats.size(0) + 2  # <vs> + z + feats + <ve>
        shift = old_vision_len - new_vision_len

        start_idx = self._map_index_after_vision_replacement(
            start_idx_orig, vision_start_idx, vision_end_idx, shift
        )
        end_idx = self._map_index_after_vision_replacement(
            end_idx_orig, vision_start_idx, vision_end_idx, shift
        )

        if start_idx is not None and end_idx is not None:
            label[start_idx:end_idx] = ids[start_idx_orig:end_idx_orig]

        return embed, attn, label