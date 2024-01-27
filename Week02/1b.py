import pandas as pd

def parse_csv(file_name):
    try:
        data = pd.read_csv(file_name)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")


# To calculate Mean
def calc_mean(data):
    mean = data.mean()
    return mean


# To calculate Variance
def calc_var(data):
    var = data.var()
    return var


# To calculate Skewness
def calc_skew(data):
    skew = data.skew()
    return skew

# To calculate Kurtosis
def calc_kurt(data):
    kurt = data.kurtosis()
    return kurt

df = parse_csv('problem1.csv')
print("mean", calc_mean(df))
print("variance", calc_var(df))
print("skewness", calc_skew(df))
print("kurtosis", calc_kurt(df))


