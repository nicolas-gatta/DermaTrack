from statistics import mean, median
from scipy.signal import medfilt

matrix = [50,47,49,50,51,51,45,50,47,48,46,50,48,49,48,50,51,48,48,49,48,52,50,49,52,48,49,47,50,50,51,51,50,47,48,51]

matrix_2 = [51,47,51,50,52,49,45,49,48,48,46,50,49,48,48,52,51,48,48,48,50,50,50,49,52,48,50,47,50,49,50,49,50,48,49,52]

matrix_3 = [30,49,50,30,52,30,47,50,49,47,47,51,49,46,49,52,51,50,47,49,49,53,50,49,52,48,47,48,51,49,50,50,40,40,40,40]

print(f"Matrix 1: mean = {mean(matrix)} | min = {min(matrix)} | median = {median(matrix)} | filter median = {median(medfilt(matrix))}")

print(f"Matrix 2: mean = {mean(matrix_2)} | min = {min(matrix_2)} | median = {median(matrix_2)} | filter median = {median(medfilt(matrix))}")

print(f"Matrix 3: mean = {mean(matrix_3)} | min = {min(matrix_3)} | median = {median(matrix_3)} | filter median = {median(medfilt(matrix))}")