"""Script to pre-process AML transaction data to be used in training and inference."""
import os
import argparse
import logging
from datetime import datetime

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


def main(in_path: str, out_path: str):
    df = pd.read_csv(in_path)

    # Define the format of your timestamp
    timestamp_format = "%Y/%m/%d %H:%M"

    # Convert the datetime object to Unix time (POSIX time)
    format_fn = lambda x: int(datetime.strptime(x, timestamp_format).timestamp())

    df['Timestamp'] = df['Timestamp'].apply(format_fn)

    df.rename(columns={'Account': 'From ID', 'Account.1': 'To ID'}, inplace=True)

    df['From Bank'] = df['From Bank'].apply(lambda b: f'B_{b}')
    df['To Bank'] = df['To Bank'].apply(lambda b: f'B_{b}')


    logger.info(f'Edge data:\n {str(df)}')
    logger.info(f'Saving edge data:\n {out_path}')
    df.to_csv(out_path, index=False)
    logger.info(f'Saved edge data in {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", required=True, type=str, help="Input transactions CSV file")
    parser.add_argument("-o", required=True, type=str, help="Output formatted transactions CSV file")

    args = parser.parse_args()

    main(args.i, args.o)