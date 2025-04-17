
import matplotlib.pyplot as mpl
import csv
import numpy as np 

def read_csv_to_array(file_path):
    """
    Reads a CSV file and returns its contents as a 2D array (list of lists).
    """
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

output = np.loadtxt(open(r"C:\Users\bubuc\Documents\Code\C\multigrid2D\Test4\FinalOutput.csv", "rb"), delimiter=",")

a = mpl.contourf(np.linspace(0,1,129),np.linspace(0,1,129),output)
cbar = mpl.colorbar()
mpl.show()