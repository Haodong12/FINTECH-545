import pandas as pd
from data_processing.missing_data_handler import MissingDataHandler
from covariance.skip_missing_row import Covariance
from correlation.skip_missing_row_corr import Correlation
from covariance.cov_pairwise import PairwiseCovariance
from correlation.corr_pairwise import PairwiseCorrelation

def main():
    # Load the dataset
    df = pd.read_csv('/Users/Lenovo/Desktop/Week5/test_files/test1.csv')  # Adjust the path as necessary
    
    # Initialize the MissingDataHandler and clean the data
    handler = MissingDataHandler(df)
    complete_data = handler.skip_missing_rows()
    
    # Calculate and print the covariance matrix
    cov_calculator = Covariance(complete_data)
    covariance_matrix = cov_calculator.calculate_covariance()
    print("Covariance Matrix:")
    print(covariance_matrix)
    
    # Calculate and print the correlation matrix
    corr_calculator = Correlation(complete_data)
    correlation_matrix = corr_calculator.calculate_correlation()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Initialize and calculate pairwise covariance
    cov_calculator = PairwiseCovariance(df)
    pairwise_covariance_matrix = cov_calculator.calculate_pairwise_covariance()
    print("Pairwise Covariance Matrix:")
    print(pairwise_covariance_matrix)
    
    # Initialize and calculate pairwise correlation
    corr_calculator = PairwiseCorrelation(df)
    pairwise_correlation_matrix = corr_calculator.calculate_pairwise_correlation()
    print("\nPairwise Correlation Matrix:")
    print(pairwise_correlation_matrix)


if __name__ == "__main__":
    main()
