
import glob
import os
import json
import pandas as pd

def main():
    # 1) locate results.csv one level under 'result/'
    paths = glob.glob(os.path.join('results.csv'))
    if not paths:
        print("Error: No 'results.csv' found under result/*/")
        return
    csv_path = paths[0]
    print(f"Loading data from {csv_path}")

    # 2) read the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # 3) detect precision & recall columns
    if 'metrics/precision' in df.columns and 'metrics/recall' in df.columns:
        prec_col = 'metrics/precision'
        rec_col = 'metrics/recall'
    elif 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        prec_col = 'metrics/precision(B)'
        rec_col = 'metrics/recall(B)'
    else:
        print("Error: Couldn't find precision/recall columns.")
        print("Available columns:", df.columns.tolist())
        return

    # 4) compute accuracy
    df['accuracy'] = (df[prec_col] + df[rec_col]) / 2

    # 5) find the row with maximum accuracy
    best_idx = df['accuracy'].idxmax()
    best_row = df.loc[best_idx, ['epoch', prec_col, rec_col, 'accuracy']].copy()
    best_row.index = ['epoch', 'precision', 'recall', 'accuracy']

    # 6) prepare output dict
    best_metrics = best_row.to_dict()

    # 7) write JSON
    out_dir = os.path.join('result', 'metrics_output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'best_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(best_metrics, f, indent=4)

    print(f"✅ Saved best metrics to {out_path}")
    print(best_metrics)

if __name__ == '__main__':
    main()

