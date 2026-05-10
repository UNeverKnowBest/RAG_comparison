import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import PROCESSED_DIR, RESULTS_DIR

sns.set_theme(style="ticks", context="paper")

plt.rcParams.update(
    {
        "font.family": "serif",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

colors = ["#85C3DC", "#8DAFAB", "#DAC6CF", "#E2C098"]


def plot_stats(data, save_dir="."):

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    cnt = Counter(i.get("type", "unknown") for i in data)

    x = list(cnt.keys())
    y = list(cnt.values())

    bars = ax[0].bar(
        x,
        y,
        color=colors[: len(x)],
        edgecolor="black",
    )

    ax[0].set_title("(a) Query Type Distribution")
    ax[0].set_ylabel("Number of Queries")

    total = sum(y)

    for b, v in zip(bars, y):
        ax[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v}\n({v / total * 100:.1f}%)",
            ha="center",
            va="bottom",
        )

    hops = [len(i.get("evidences", [])) for i in data]

    sns.histplot(
        hops,
        bins=range(1, max(hops) + 2) if hops else 10,
        discrete=True,
        color="#4C72B0",
        edgecolor="black",
        ax=ax[1],
    )

    if hops:
        mean_hop = np.mean(hops)
        ax[1].axvline(
            mean_hop,
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mean_hop:.2f}",
        )

    ax[1].set_title("(b) Evidence Hop Distribution")
    ax[1].set_xlabel("Number of Hops")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/dataset_stats.pdf")
    plt.show()


if __name__ == "__main__":
    file_path = str(PROCESSED_DIR) + "/validation_full.jsonl"

    with open(file_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    plot_stats(dataset, str(RESULTS_DIR))
