"""Inspect text length and save word-count histogram."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from common import load_dataset, prepare_outputs, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, choose_text_column


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-column', help='Actual requirement text column name; overrides heuristic')
    args = parser.parse_args()
    prepare_outputs()
    df = load_dataset()
    col = args.text_column or choose_text_column(df)
    if not col or col not in df.columns:
        raise ValueError(f'Cannot determine text column. Available columns: {list(df.columns)}. Use --text-column NAME')
    text = df[col].fillna('').astype(str)
    stats = pd.DataFrame({'character_count': text.str.len(), 'word_count': text.str.split().str.len()})
    stats.describe().round(2).to_csv(TABLES_DIR / '05_text_length_summary.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(stats['word_count'], bins=30, edgecolor='white')
    ax.set_title(f'Requirement word counts: {col}')
    ax.set_xlabel('Words per requirement')
    ax.set_ylabel('Number of records')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / '05_word_count_histogram.png', dpi=200)
    plt.close(fig)
    report = f'Text column: {col}\n\n{stats.describe().round(2).to_string()}\n'
    print(report)
    (REPORTS_DIR / '05_text_analysis.txt').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
