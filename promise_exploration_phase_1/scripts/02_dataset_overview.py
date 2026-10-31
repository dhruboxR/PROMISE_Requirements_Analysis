"""High-level overview, descriptive summary, and sample rows."""
import pandas as pd
from common import load_dataset, prepare_outputs, REPORTS_DIR, TABLES_DIR, choose_text_column, choose_label_column


def main():
    prepare_outputs()
    df = load_dataset()
    text_col = choose_text_column(df)
    label_col = choose_label_column(df, text_col)
    overview = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(t) for t in df.dtypes],
        'non_null': [int(df[c].notna().sum()) for c in df.columns],
        'unique': [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    overview.to_csv(TABLES_DIR / '02_column_overview.csv', index=False)
    df.head(20).to_csv(TABLES_DIR / '02_first_20_rows.csv', index=False)
    report = (f'Rows: {len(df)}\nColumns: {len(df.columns)}\n'
              f'Possible text column (heuristic): {text_col}\n'
              f'Possible label column (heuristic): {label_col}\n\n'
              'Column overview:\n' + overview.to_string(index=False))
    print(report)
    (REPORTS_DIR / '02_overview.txt').write_text(report + '\n', encoding='utf-8')
    print('\nConfirm the suggested text/label columns by inspecting the ARFF schema.')


if __name__ == '__main__':
    main()
