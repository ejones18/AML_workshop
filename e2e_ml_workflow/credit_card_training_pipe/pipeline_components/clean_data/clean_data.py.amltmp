import os
import argparse
import pandas as pd
import logging
import mlflow

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--cleaned_data", type=str, help="out path to cleaned data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()
    credit_df_raw = pd.read_csv(args.data, header=0)
    credit_df_raw.dropna(inplace=True) 

    mlflow.log_metric("num_samples", credit_df_raw.shape[0])
    mlflow.log_metric("num_features", credit_df_raw.shape[1] - 1)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    credit_df_raw.to_csv(os.path.join(args.cleaned_data, "cleaned_data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
