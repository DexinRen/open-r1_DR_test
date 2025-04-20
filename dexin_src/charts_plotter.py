import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the CSV data
csv_path = "/home/adonis/proj/open-r1_DR_test/dexin_src/eval_output/eval_summary.csv"
df = pd.read_csv(csv_path)

# Convert 'checkpoint' to numeric step values, with 'untrained' as 0
def parse_checkpoint(val):
    if val == 'untrained':
        return 0
    if val.startswith("checkpoint-"):
        return int(val.split('-')[-1])
    return -1  # fallback

df['step'] = df['checkpoint'].apply(parse_checkpoint)
df = df[df['step'] >= 0].sort_values('step')

# Convert necessary columns to numeric
numeric_columns = [
    'train_accuracy', 'test_accuracy', 'math500_accuracy',
    'gpqa_diamond_accuracy_all', 'gpqa_diamond_accuracy_clean',
    'gpqa_extended_accuracy_all', 'gpqa_extended_accuracy_clean'
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Set up the figure
fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

# Chart 1: Train & Test Accuracy on math220k
axs[0].plot(df['step'], df['train_accuracy'], label='Train Accuracy', marker='o')
axs[0].plot(df['step'], df['test_accuracy'], label='Test Accuracy', marker='o')
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Math220k Train & Test Accuracy")
axs[0].legend()
axs[0].grid(True)

# Chart 2: Math500 and math220k-test Accuracy
axs[1].plot(df['step'], df['math500_accuracy'], label='Math500 Accuracy', marker='s')
axs[1].plot(df['step'], df['test_accuracy'], label='Math220k Test Accuracy', linestyle='--', marker='^')
axs[1].set_ylabel("Accuracy")
axs[1].set_title("Math Performance (Math500 and Math220k Test)")
axs[1].legend()
axs[1].grid(True)

# Chart 3: GPQA Diamond Accuracy
axs[2].plot(df['step'], df['gpqa_diamond_accuracy_all'], label='Diamond (All)', marker='x')
axs[2].plot(df['step'], df['gpqa_diamond_accuracy_clean'], label='Diamond (Clean)', marker='o')
axs[2].set_ylabel("Accuracy")
axs[2].set_title("GPQA Diamond Accuracy")
axs[2].legend()
axs[2].grid(True)

# Chart 4: GPQA Extended Accuracy
axs[3].plot(df['step'], df['gpqa_extended_accuracy_all'], label='Extended (All)', marker='x')
axs[3].plot(df['step'], df['gpqa_extended_accuracy_clean'], label='Extended (Clean)', marker='o')
axs[3].set_ylabel("Accuracy")
axs[3].set_title("GPQA Extended Accuracy")
axs[3].legend()
axs[3].grid(True)
axs[3].set_xlabel("Checkpoint Step")

# Format x-axis ticks
axs[3].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt_path = "/home/adonis/proj/open-r1_DR_test/dexin_src/eval_output/eval_summary_plots.png"
plt.savefig(plt_path)
plt_path
