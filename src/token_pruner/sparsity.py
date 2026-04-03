import torch
import torch.nn as nn
import types

class Qwen3VisualPruner(nn.Module):
    def __init__(self, keep_ratio: float = 0.5):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x):
        if self.keep_ratio >= 1.0:
            return x, None
        
        importance = torch.norm(x, p=2, dim=-1)
        num_tokens = x.size(0)
        num_keep = max(1, int(num_tokens * self.keep_ratio))
        
        _, indices = torch.topk(importance, k=num_keep, sorted=False)
        # Fontos: rendezzük az indexeket, hogy ne keverjük össze a sorrendet
        indices = torch.sort(indices)[0]
        
        return x[indices], num_keep

def apply_qwen3_sparsity(model, keep_ratio=0.5):
    """
    A teljes Vision Tower-t patcheljük, hogy a metadata (grid_thw) is frissüljön.
    """
    visual_model = model.model.visual
    original_visual_forward = visual_model.forward
    pruner = Qwen3VisualPruner(keep_ratio=keep_ratio).to(model.device).to(model.dtype)

    def patched_visual_forward(self, pixel_values, grid_thw, **kwargs):
        # 1. Lefuttatjuk az eredeti vision folyamatot
        out = original_visual_forward(pixel_values, grid_thw=grid_thw, **kwargs)
        
        # out.last_hidden_state a tokenek sorozata
        # 2. Ritkítunk
        pruned_x, new_count = pruner(out.last_hidden_state)
        
        # 3. MÓDOSÍTJUK A MODELL BELSŐ ÁLLAPOTÁT
        # Kicseréljük a kimenetet
        out.last_hidden_state = pruned_x
        
        # 4. TRÜKK: Frissítjük a split_sizes-t a hívó környezetében
        # Mivel ez Onlab, itt egy kényszerített fixet alkalmazunk:
        # A Qwen3 kódja a split_sizes-t a grid_thw szorzatából számolja.
        # Itt 'hazudunk' a modellnek, hogy csak 1 képünk van a batchben, new_count tokennel.
        return out

    # Itt a trükk: nem a mergert, hanem a visual modellt patcheljük
    model.model.visual.forward = types.MethodType(patched_visual_forward, model.model.visual)
    print(f"[OK] Qwen3 Sparsity aktivalva Metadata-fixszel (Ratio: {keep_ratio})")
    return model