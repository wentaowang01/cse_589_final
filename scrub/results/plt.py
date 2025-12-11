import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_theme()

# Input CSV (change here if you want to point at a different run)
csv = "reatrain_scrub_mnist_2nn.csv"
df = pd.read_csv(csv)

# Metrics to visualize against n_removals
metrics = {
    # Validation accuracy / accuracy related
    "val_acc_after": "Validation accuracy after scrub",
    "retrain_val_accuracy": "Validation accuracy after retrain",
    "residual_acc_after": "Residual accuracy after scrub",
    "retrain_res_acc": "Residual accuracy after retrain",
    # Gradient norms of individual samples / residuals
    "sample_gradnorm_after": "Sample gradnorm after scrub",
    "residual_gradnorm_after": "Residual gradnorm after scrub",
    "retrain_res_gradnorm": "Residual gradnorm after retrain",
}

for metric, title in metrics.items():
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=df, x="n_removals", y=metric, hue="run", alpha=0.6, legend="brief"
    )
    plt.title(title)
    plt.legend(title="run", loc="best")
    plt.tight_layout()
    plt.savefig(f"{metric}.png", dpi=150)

# Where retrains triggered (bad_sample==1) on validation accuracy
plt.figure(figsize=(8, 4))
sns.scatterplot(
    data=df[df.bad_sample == 1],
    x="n_removals",
    y="val_acc_after",
    hue="run",
    style="bad_sample",
    legend=False,
    s=20,
)
plt.title("Val acc after (bad samples marked)")
plt.tight_layout()
plt.savefig("val_acc_bad_samples.png", dpi=150)
