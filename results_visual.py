import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

baseline_data = [ # this is from 03-full-baseline-evaluation notebook
    {"label": "BM25", "mrr": 0.4633, "f1": 0.7501},
    {"label": "DPR (bge-m3)", "mrr": 0.7307, "f1": 0.9804},
    {"label": "Vanilla GNN", "mrr": 0.6637, "f1": 0.8405},
]

with open("/home/sslab/24m0786/graph-enhanced-retrieval-qa/results/experiment_results.json", "r") as f:
    results = json.load(f)

df = pd.DataFrame(results)
config_df = pd.json_normalize(df['config'])
df = df.drop(columns=['config']).join(config_df)

df['label'] = df.apply(lambda row: f"GNN {row['num_layers']}-layer ({row['aggregation']})", axis=1)

gnn_variants = df[['label', 'mrr', 'f1']].to_dict(orient="records")
plot_data = baseline_data + gnn_variants
plot_df = pd.DataFrame(plot_data)

x = np.arange(len(plot_df))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))

bars1 = ax.bar(x - width/2, plot_df['mrr'], width, label='MRR', color='gold')
bars2 = ax.bar(x + width/2, plot_df['f1'], width, label='F1', color='orange')

ax.set_ylabel("Score")
ax.set_title("Baseline vs. Experimental GNN Configurations: MRR and F1")
ax.set_xticks(x)
ax.set_xticklabels(plot_df['label'], rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("/home/sslab/24m0786/graph-enhanced-retrieval-qa/results/report_figures/mrr_f1_comparison.png", dpi=300)
print("Saved mrr_f1_comparison.png")
