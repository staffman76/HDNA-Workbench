"""
HDNA Workbench Demo — One-line inspectability for any PyTorch model.

Shows how to:
1. Take a standard model and make it inspectable
2. Run inference with full tracing
3. Query what happened inside
4. Set breakpoints and watchers
5. Revert back to standard (for production/saving)
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import workbench


# ============================================================
# Step 1: Create a standard PyTorch model (nothing special)
# ============================================================
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)
        self.layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                      batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


print("=" * 60)
print("HDNA Workbench Demo")
print("=" * 60)

# Create a standard model
model = SimpleTransformer()
print(f"\n1. Created standard model: {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================
# Step 2: One line — make it inspectable
# ============================================================
model = workbench.inspect(model)
print("2. workbench.inspect(model) -- done. Every layer is now traced.")

# ============================================================
# Step 3: Run inference (works exactly the same)
# ============================================================
input_ids = torch.randint(0, 1000, (2, 32))  # batch=2, seq_len=32
output = model(input_ids)
print(f"3. Inference: input {tuple(input_ids.shape)} -> output {tuple(output.shape)}")

# ============================================================
# Step 4: See what happened inside
# ============================================================
print("\n4. Full model trace:")
traces = workbench.trace(model)
for layer_name, trace_info in traces.items():
    shape_str = str(trace_info.get('last_output_shape', '?'))
    print(f"   {layer_name:40s}  calls={trace_info['calls']}  "
          f"shape={shape_str:>20s}  "
          f"time={trace_info.get('last_elapsed_ms', 0):.2f}ms")

# ============================================================
# Step 5: Inspect specific layers
# ============================================================
print("\n5. Detailed layer inspection:")
for name, module in model.named_modules():
    if hasattr(module, 'snapshot'):
        snap = module.snapshot()
        if 'attention' in snap:
            attn = snap['attention']
            print(f"   {name}:")
            print(f"      Head entropy:  {['%.3f' % e for e in attn.get('head_entropy', [])]}")
            print(f"      Head sharpness: {['%.3f' % s for s in attn.get('head_sharpness', [])]}")
            print(f"      Head redundancy: {attn.get('head_redundancy', 'N/A'):.4f}")
        if 'weight_stats' in snap:
            ws = snap['weight_stats']
            print(f"   {name}:")
            print(f"      Weights: mean={ws['mean']:.4f} std={ws['std']:.4f} "
                  f"sparsity={ws['sparsity']:.1%}")

# ============================================================
# Step 6: Check for anomalies
# ============================================================
print("\n6. Anomaly scan:")
anomalies = workbench.anomalies(model)
if anomalies:
    for a in anomalies:
        print(f"   WARNING: {a['layer']} — {a['issue']}")
else:
    print("   No anomalies detected (need more forward passes for detection)")

# ============================================================
# Step 7: Embedding usage tracking
# ============================================================
print("\n7. Embedding usage:")
for name, module in model.named_modules():
    if hasattr(module, 'most_accessed'):
        snap = module.snapshot()
        usage = snap.get('usage', {})
        print(f"   {name}: {usage.get('unique_accessed', 0)}/{snap['layer_config']['num_embeddings']} "
              f"tokens accessed, top tokens: {module.most_accessed(5)}")

# ============================================================
# Step 8: Set a breakpoint (fires if output explodes)
# ============================================================
print("\n8. Breakpoints:")
for name, module in model.named_modules():
    if hasattr(module, 'add_breakpoint') and 'head' in name:
        module.add_breakpoint(lambda l, inp, out: out.abs().max() > 1000)
        print(f"   Set breakpoint on '{name}': fires if |output| > 1000")
        break

# ============================================================
# Step 9: Revert (for production/saving)
# ============================================================
model_clean = workbench.revert(model)
print("\n9. workbench.revert(model) -- back to standard PyTorch. No workbench dependency.")

# Verify it still works
output2 = model_clean(input_ids)
print(f"   Reverted model output shape: {tuple(output2.shape)} [OK]")

print("\n" + "=" * 60)
print("Key insight: same model, same weights, same math.")
print("But now you can see EVERYTHING that happens inside.")
print("=" * 60)
