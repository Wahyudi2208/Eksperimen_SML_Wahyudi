import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):

    # Load dataset
    df = pd.read_excel(input_path)

    # Drop irrelevant columns
    columns_to_drop = [
        'CustomerID',
        'Count',
        'Country',
        'State',
        'City',
        'Zip Code',
        'Lat Long',
        'Latitude',
        'Longitude',
        'Churn Label',
        'Churn Reason',
        'Churn Score'
    ]

    df.drop(columns=columns_to_drop, inplace=True)

    # Convert Total Charges
    df['Total Charges'] = pd.to_numeric(
        df['Total Charges'],
        errors='coerce'
    )

    # Handle missing values
    df.dropna(inplace=True)

    # Encoding
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    # Scaling
    scaler = StandardScaler()

    numerical_cols = [
        'Tenure Months',
        'Monthly Charges',
        'Total Charges',
        'CLTV'
    ]

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    # Save result
    df.to_csv(
        output_path,
        index=False
    )

    print("Preprocessing completed.")
    print("Output shape:", df.shape)


if __name__ == "__main__":

    preprocess_data(
        "Dataset_telco_customer_churn.xlsx",
        "telco_preprocessed.csv"
    )