import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

History = dict[str, list[float]]

# Validated categorical palette, assigned in this fixed order (never cycled).
COLORS = [
    '#2a78d6',
    '#eb6834',
    '#1baf7a',
    '#eda100',
    '#e87ba4',
    '#008300',
    '#6250d6',
    '#e34948',
]
METRICS = {
    'train_loss': 'Train loss',
    'val_loss': 'Val loss',
    'val_accuracy': 'Val accuracy',
}


def best_epoch(history: History) -> int:
    val_loss = history['val_loss']

    return min(range(len(val_loss)), key=val_loss.__getitem__)


def summarize_histories(
    histories: dict[str, History],
) -> pd.DataFrame:
    rows = {}
    for name, history in histories.items():
        best = best_epoch(history)
        rows[name] = {
            'best_epoch': best + 1,
            'train_loss': history['train_loss'][best],
            'val_loss': history['val_loss'][best],
            'loss_gap': history['val_loss'][best] - history['train_loss'][best],
            'val_accuracy': history['val_accuracy'][best],
        }

    return pd.DataFrame.from_dict(rows, orient='index').sort_values('val_loss')


def plot_histories(
    histories: dict[str, History],
    baseline: float | None = None,
    clip_loss: float | None = 0.6,
) -> Figure:
    if len(histories) > len(COLORS):
        raise ValueError(f'At most {len(COLORS)} histories can be compared at once.')

    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(16, 4.5), sharex=True, layout='constrained'
    )
    axes[1].sharey(axes[0])

    for color, (name, history) in zip(COLORS, histories.items(), strict=False):
        epochs = range(1, len(history['train_loss']) + 1)
        best = best_epoch(history)

        for ax, metric in zip(axes, METRICS, strict=True):
            ax.plot(epochs, history[metric], color=color, linewidth=2, label=name)
            ax.plot(
                best + 1,
                history[metric][best],
                'o',
                color=color,
                markersize=8,
                markeredgecolor='white',
                markeredgewidth=2,
                zorder=3,
            )

    if clip_loss is not None:
        axes[0].set_ylim(top=clip_loss)

    if baseline is not None:
        baseline_line = axes[2].axhline(
            baseline,
            color='#8a8984',
            linestyle='--',
            linewidth=1,
            label='majority class',
        )
        axes[2].set_ylim(bottom=baseline - 0.05)
        axes[2].legend(handles=[baseline_line], frameon=False, loc='lower right')

    for ax, title in zip(axes, METRICS.values(), strict=True):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(color='#e4e3df', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='outside right upper',
        frameon=False,
        title='dot = best val loss',
    )

    return fig
