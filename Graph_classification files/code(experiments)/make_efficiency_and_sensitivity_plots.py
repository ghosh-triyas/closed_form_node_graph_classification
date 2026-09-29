"""
make_efficiency_and_sensitivity_plots.py

Two plots:
  1. Training-time bar chart, from the numbers already in the paper's
     efficiency table (Table~\\ref{tab:efficiency}) -- visualizes the
     "dramatically cheaper" claim.
  2. Sensitivity sweep line plots (accuracy vs. T), reading from the
     CSV saved by run_T_sensitivity_sweep.py.

Run run_T_sensitivity_sweep.py FIRST if you want plot 2 -- this
script will skip it gracefully with a clear message if the CSV isn't
found yet, and still produce plot 1.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import csv
import os

# ── Plot 1: Training time bar chart ─────────────────────────────────
# Values transcribed directly from Table~\ref{tab:efficiency} --
# update these if that table changes.
methods = ['WL Kernel', 'GCN', 'GraphSAGE', 'GIN', 'Ours']
mutag_times = [0.5, 4, 5, 6.5, 2]      # minutes (midpoint of ranges)
dhfr_times  = [2.5, 12.5, 15, 17.5, 12]
nci1_times  = [7.5, 75, 87.5, 105, 50]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
colors = ['#7f7f7f', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

for ax, times, title in zip(
        axes, [mutag_times, dhfr_times, nci1_times],
        ['MUTAG', 'DHFR', 'NCI1']):
    bar_colors = colors[:-1] + ['#d62728']
    bars = ax.bar(methods, times, color=bar_colors)
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(1.5)
    ax.set_ylabel('Training time (minutes, log scale)')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper_outputs/figures/efficiency_barchart.pdf', dpi=200, bbox_inches='tight')
plt.savefig('paper_outputs/figures/efficiency_barchart.png', dpi=200, bbox_inches='tight')
print("Saved paper_outputs/figures/efficiency_barchart.pdf (and .png)")


# ── Plot 2: Sensitivity sweep line plot ─────────────────────────────
csv_path = 'paper_outputs/data/T_sensitivity_results.csv'
if not os.path.exists(csv_path):
    print(f"\n{csv_path} not found -- run run_T_sensitivity_sweep.py "
          f"first if you want the sensitivity plot. Skipping plot 2.")
else:
    data = {}  # dataset -> (T_list, mean_list, std_list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row['dataset']
            if ds not in data:
                data[ds] = ([], [], [])
            data[ds][0].append(int(row['T']))
            data[ds][1].append(float(row['mean_acc']))
            data[ds][2].append(float(row['std_acc']))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = ['o', 's', '^', 'D']
    for (ds, (T_list, mean_list, std_list)), marker in zip(data.items(), markers):
        order = np.argsort(T_list)
        T_arr = np.array(T_list)[order]
        mean_arr = np.array(mean_list)[order]
        std_arr = np.array(std_list)[order]
        ax.errorbar(T_arr, mean_arr, yerr=std_arr, marker=marker,
                    label=ds, capsize=3, linewidth=2, markersize=7)

    ax.set_xlabel('Number of filtration steps $T$')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('PHS sensitivity to filtration granularity')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper_outputs/figures/T_sensitivity_plot.pdf', dpi=200, bbox_inches='tight')
    plt.savefig('paper_outputs/figures/T_sensitivity_plot.png', dpi=200, bbox_inches='tight')
    print("Saved paper_outputs/figures/T_sensitivity_plot.pdf (and .png)")
