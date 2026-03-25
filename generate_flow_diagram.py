"""Generate a horizontal flow diagram of how Tokusan works internally."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 5.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5.5)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

ARROW_COLOR = '#555555'
COLORS = {
    'input':     '#4A90D9',
    'tokenize':  '#5BAE5B',
    'vectorize': '#D9A34A',
    'classify':  '#D96A4A',
    'lime':      '#9B59B6',
    'explain':   '#1ABC9C',
    'ai':        '#E74C3C',
    'output':    '#2C3E50',
}


def draw_box(ax, x, y, w, h, label, detail, color, fontsize=8.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.92, zorder=3
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.63, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=4)
    if detail:
        ax.text(x + w / 2, y + h * 0.25, detail,
                ha='center', va='center', fontsize=6,
                color='#EEEEEE', style='italic', zorder=4)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.6),
        zorder=2
    )


# --- Title ---
ax.text(8, 5.15, 'Tokusan Internal Flow', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2C3E50')
ax.text(8, 4.8, 'Japanese Text Classification with LIME Explanations',
        ha='center', va='center', fontsize=8, color='#7F8C8D')

# ============================================================
# Layout
# ============================================================
bw = 1.7
bh = 1.1
ry = 2.2
gap = 0.3

# 1. User Input
x = 0.15
draw_box(ax, x, ry, bw, bh, 'User Input',
         'Japanese text', COLORS['input'])
arrow(ax, x + bw, ry + bh / 2, x + bw + gap, ry + bh / 2)

# 2. Tokenizer
x += bw + gap
draw_box(ax, x, ry, bw, bh, 'Tokenizer\n(SudachiPy)',
         'Split + filter', COLORS['tokenize'])
arrow(ax, x + bw, ry + bh / 2, x + bw + gap, ry + bh / 2)

# 3. TF-IDF
x += bw + gap
draw_box(ax, x, ry, bw, bh, 'TF-IDF\nVectorizer',
         'Tokens → vectors', COLORS['vectorize'])
arrow(ax, x + bw, ry + bh / 2, x + bw + gap, ry + bh / 2)

# 4. Classifier
x += bw + gap
draw_box(ax, x, ry, bw, bh, 'Classifier\n(sklearn)',
         'LR / SVM / RF / NB', COLORS['classify'])
arrow(ax, x + bw, ry + bh / 2, x + bw + gap, ry + bh / 2)

# 5. LIME Explainer
x += bw + gap
draw_box(ax, x, ry, bw, bh, 'LIME\nExplainer',
         'Perturb → local model\n→ word weights', COLORS['lime'])
lime_end = x + bw
lime_mid = ry + bh / 2

# Split arrows
arrow(ax, lime_end, lime_mid + 0.2, lime_end + gap, lime_mid + 0.6)
arrow(ax, lime_end, lime_mid - 0.2, lime_end + gap, lime_mid - 0.6)

# 6a. Template Summary (top)
sx = lime_end + gap
sw = 1.8
sh = 0.85
sy_top = ry + bh / 2 + 0.22
draw_box(ax, sx, sy_top, sw, sh, 'Template\nSummary',
         'JP / EN', COLORS['explain'], fontsize=8)

# 6b. AI Interpreter (bottom)
sy_bot = ry + bh / 2 - sh - 0.22
draw_box(ax, sx, sy_bot, sw, sh, 'AI Interpreter',
         'Gemini (optional)', COLORS['ai'], fontsize=8)

# Arrows to output
out_x = sx + sw + gap
out_w = 2.0
out_h = 1.1
arrow(ax, sx + sw, sy_top + sh / 2, out_x, ry + bh / 2 + 0.12)
arrow(ax, sx + sw, sy_bot + sh / 2, out_x, ry + bh / 2 - 0.12)

# 7. PredictionResult
draw_box(ax, out_x, ry, out_w, out_h, 'Prediction\nResult',
         'class, probabilities\nexplanation, summary',
         COLORS['output'], fontsize=9.5)

plt.tight_layout()
plt.savefig('/Users/noyuritsuji/tokusan/tokusan_flow.png', dpi=180, bbox_inches='tight',
            facecolor='#FAFAFA', edgecolor='none')
print("Saved: tokusan_flow.png")
