"""Check missing, blank, duplicate rows, and duplicate requirement text."""
import pandas as pd
from common import load_dataset, prepare_outputs, REPORTS_DIR, TABLES_DIR, choose_text_column


def main():
    prepare_outputs()
    df = load_dataset()
    text_col = choose_text_column(df)
    full_duplicates = int(df.duplicated().sum())
    quality = pd.DataFrame({
        'column': df.columns,
        'missing': [int(df[c].isna().sum()) for c in df.columns],
        'missing_pct': [round(df[c].isna().mean() * 100, 2) for c in df.columns],
    })
    quality.to_csv(TABLES_DIR / '03_missing_values.csv', index=False)
    report = [f'Rows: {len(df)}', f'Exact duplicate rows: {full_duplicates}']
    if text_col:
        texts = df[text_col].astype('string')
        blank = int(texts.fillna('').str.strip().eq('').sum())
        normalized = texts.str.strip().str.lower()
        duplicate_texts = int(normalized[normalized.notna() & normalized.ne('')].duplicated().sum())
        report += [f'Possible requirement text column: {text_col}', f'Blank requirement texts: {blank}',
                   f'Duplicate requirement texts (case/whitespace normalized): {duplicate_texts}']
    report += ['', quality.to_string(index=False)]
    text = '\n'.join(report)
    print(text)
    (REPORTS_DIR / '03_data_quality.txt').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
