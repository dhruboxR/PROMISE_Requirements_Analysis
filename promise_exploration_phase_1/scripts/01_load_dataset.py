"""Load PROMISE+ ARFF and report the actual schema."""
from common import load_dataset, DATA_PATH, prepare_outputs, REPORTS_DIR


def main():
    prepare_outputs()
    df = load_dataset()
    lines = [f'Source: {DATA_PATH}', f'Rows: {len(df)}', f'Columns: {len(df.columns)}', '', 'Columns and types:']
    lines.extend(f'- {col}: {df[col].dtype}' for col in df.columns)
    lines.extend(['', 'First five rows:', df.head(5).to_string(index=False)])
    report = '\n'.join(lines)
    print(report)
    (REPORTS_DIR / '01_loading_and_schema.txt').write_text(report + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
