import pandas as pd
import os


def get_data(file_name: str) -> pd.DataFrame:
    current_directory = os.getcwd()  # Use the current working directory
    data_path = os.path.join(current_directory, file_name)
    return pd.read_csv(data_path)


def remove_irrelevant_rows(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    return df[df[column] != value]


def remove_irrelevant_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    return df.drop(columns=columns_to_remove, errors='ignore')


def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def main():

    file_name = 'us_chronic_disease_indicators.csv'
    df = get_data(file_name)

    # (1) Remove irrelevant rows
    df = remove_irrelevant_rows(df, 'locationdesc', 'United States')

    # (2) Remove irrelevant columns
    columns_to_remove = [
        'yearend', 'datavalueunit', 'datavaluetype', 'geolocation', 'topicid', 'questionid', 'datavaluetypeid',
        'stratificationcategoryid1', 'stratificationid1'
    ]
    df = remove_irrelevant_columns(df, columns_to_remove)

    # (3) Remove empty rows
    df = remove_empty_rows(df)

    # return data frame
    return df


if __name__ == '__main__':
    main()
