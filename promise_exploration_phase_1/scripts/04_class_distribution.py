"""Inspect label distribution and save a bar chart."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_dataset, prepare_outputs, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, choose_text_column, choose_label_column


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-column', help='Actual label column name; overrides heuristic')
    args = parser.parse_args()
    prepare_outputs()
    df = load_dataset()
    col = args.label_column or choose_label_column(df, choose_text_column(df))
    if not col or col not in df.columns:
        raise ValueError(f'Cannot determine label column. Available columns: {list(df.columns)}. Use --label-column NAME')
    counts = df[col].fillna('[MISSING]').astype(str).value_counts()
    table = counts.rename_axis('class').reset_index(name='count')
    table['percentage'] = (table['count'] / len(df) * 100).round(2) if len(df) else 0
    table.to_csv(TABLES_DIR / '04_class_distribution.csv', index=False)
    fig, ax = plt.subplots(figsize=(max(8, min(16, len(counts) * 0.7)), 5))
    counts.plot.bar(ax=ax)
    ax.set_title(f'Class distribution: {col}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of records')
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / '04_class_distribution.png', dpi=200)
    plt.close(fig)
    summary = f'Label column: {col}\n\n{table.to_string(index=False)}\n'
    print(summary)
    (REPORTS_DIR / '04_class_distribution.txt').write_text(summary, encoding='utf-8')


if __name__ == '__main__':
    main()
