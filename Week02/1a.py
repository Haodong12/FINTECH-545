import csv

def parse_csv(file_name):
    data = []
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
                data.append(float(row[0]))  
    return data
    
file = "problem1.csv"
data = parse_csv(file)

def calc_mean(data):
    mean = sum(data) / len(data) 
    return mean

def calc_variance(data):
    mean = calc_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1) 
    return variance

def calc_skewness(data):
    mean = calc_mean(data)
    variance = calc_variance(data)
    n = len(data)
    skewness = (sum((x - mean) ** 3 for x in data) / n) / (variance ** 1.5)
    return skewness

def calc_kurtosis(data):
    mean = calc_mean(data)
    variance = calc_variance(data)
    n = len(data)
    kurtosis = (sum((x - mean) ** 4 for x in data) / n) / (variance ** 2) - 3
    return kurtosis  


print("mean", calc_mean(data))
print("variance", calc_variance(data))
print("skewness", calc_skewness(data))
print("kurtosis", calc_kurtosis(data))