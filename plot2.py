import numpy as np
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('whitegrid')
import os
import pandas as pd
from matplotlib import font_manager


matplotlib.rc('font', size=30)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['text.usetex'] = True
# matplotlib.rc('font', family='Times New Roman')

# files = [(0, 1), (1, 1), (1, 1.5), (3, 1.5), (5, 1.5), (7, 1.5), (10, 1.5)]

\

# FashionMNIST non-iid
# FedPPKT = [10.0, 10.0, 34.11, 43.79, 67.27, 72.63, 80.30, 84.05, 85.31, 86.11, 86.54, 85.76, 86.43, 86.47, 86.95, 85.26, 86.57, 86.56, 86.77, 86.94, 86.38, 87.06, 87.04, 87.13, 86.81, 87.42, 86.94, 86.84, 87.06, 87.66, 86.6, 87.43, 87.7, 86.67, 86.56, 87.02, 86.21, 86.51, 86.59, 87.35, 86.58, 86.71, 86.63, 86.54, 86.71, 86.6, 87.01, 86.3, 87.27, 86.38, 86.4, 87.51, 88.31, 87.29, 87.47, 87.19, 86.51, 87.39, 86.95, 86.18, 86.32, 87.29, 87.02, 87.03, 86.48, 87.01, 86.1, 86.41, 87.63, 86.59, 87.13, 87.85, 86.93, 87.12, 86.61, 86.52, 87.29, 86.33, 86.91, 86.47, 87.66, 86.09, 87.33, 86.85, 86.93, 86.97, 86.8, 87.36, 86.51, 86.62, 87.19, 86.46, 87.29, 86.73, 86.86, 87.14, 87.05, 87.1, 86.84, 86.68]
# FDP = [58.36, 66.51, 68.86, 69.59, 71.81, 72.68, 72.19, 73.58, 74.36, 74.14, 73.76, 74.92, 75.35, 75.02, 75.77, 76.0, 75.5, 76.74, 76.38, 77.07, 77.09, 76.47, 77.97, 77.01, 78.21, 77.91, 77.97, 78.78, 78.47, 78.5, 78.89, 78.87, 79.03, 79.48, 79.39, 79.76, 79.66, 80.04, 79.59, 80.03, 80.19, 79.89, 80.49, 80.87, 80.65, 80.57, 81.35, 81.48, 81.17, 81.26, 81.65, 81.39, 81.63, 81.89, 82.07, 81.93, 82.18, 82.04, 82.34, 82.06, 81.81, 82.81, 82.46, 82.38, 82.6, 82.7, 82.81, 82.59, 82.81, 82.89, 83.07, 82.95, 83.08, 83.38, 83.09, 83.49, 83.5, 82.77, 83.33, 83.68, 83.05, 83.47, 82.84, 83.28, 83.43, 83.85, 83.69, 83.37, 84.24, 84.17, 84.09, 84.16, 84.37, 84.1, 84.19, 84.04, 84.41, 84.63, 84.32, 84.14]
# DP_FedAvg = [29.21, 43.51, 50.61, 64.37, 63.54, 69.05, 67.39, 72.74, 69.8, 70.63, 73.63, 72.15, 72.61, 74.75, 74.61, 75.26, 75.71, 76.87, 70.6, 75.66, 76.54, 78.46, 78.03, 79.65, 78.61, 77.79, 78.56, 78.10, 78.23, 78.43, 78.74, 77.63, 77.58, 78.14, 79.97, 79.47, 79.24, 80.47, 81.47, 80.7, 79.65, 79.94, 80.96, 81.2, 79.34, 80.72, 79.55, 80.28, 79.42, 81.53, 79.88, 80.87, 79.9, 81.3, 81.25, 81.12, 81.65, 81.94, 82.18, 82.55, 82.21, 82.12, 82.99, 82.13, 82.59, 82.58, 82.99, 83.20, 81.98, 82.68, 82.37, 82.46, 83.01, 83.25, 83.66, 83.03, 83.43, 83.95, 84.15, 84.47, 84.50, 84.46, 83.86, 83.57, 84.24, 83.1, 84.03, 83.91, 83.27, 83.21, 82.71, 82.42, 82.70, 82.59, 82.71, 82.94, 82.1, 82.31, 81.99, 83.02]
# CENTAUR = [40.47, 49.75, 55.45, 58.47, 61.58, 64.05, 66.52, 67.82, 69.08, 70.95, 71.2, 73.07, 73.77, 73.97, 75.03, 75.48, 76.38, 76.3, 77.42, 77.38, 77.58, 77.5, 77.9, 78.67, 78.1, 78.15, 78.85, 79.3, 80.03, 80.95, 81.12, 81.98, 81.27, 81.48, 82.23, 82.27, 82.83, 83.33, 83.03, 83.73, 84.28, 84.1, 84.83, 84.52, 84.73, 85.33, 85.18, 85.35, 85.48, 85.27, 84.83, 84.56, 84.52, 84.1, 84.78, 84.95, 84.72, 85.08, 84.77, 84.47, 84.2, 84.73, 84.1, 83.3, 83.95, 84.12, 84.37, 84.72, 84.82, 85.12, 85.25, 85.48, 84.83, 84.18, 84.38, 84.48, 84.72, 84.82, 84.78, 84.62, 84.25, 84.17, 83.93, 83.73, 84.67, 84.37, 84.57, 84.9, 85.47, 84.72, 84.85, 85.32, 84.97, 84.62, 85.3, 85.17, 85.03, 85.48, 85.23, 85.45]
# PPSGD = [38.45, 46.78, 54.88, 57.23, 59.23, 61.27, 61.13, 63.13, 64.17, 65.03, 66.08, 65.87, 67.18, 69.42, 70.27, 71.0, 71.95, 72.65, 72.7, 72.78, 73.05, 73.88, 74.27, 74.8, 75.23, 75.6, 76.15, 76.68, 77.73, 77.8, 78.77, 79.37, 79.6, 79.85, 79.98, 80.38, 80.48, 81.63, 81.23, 81.77, 82.15, 82.28, 82.43, 82.77, 82.7, 83.38, 83.6, 83.82, 84.25, 84.2, 84.52, 84.3, 84.82, 84.75, 85.07, 85.55, 85.38, 85.75, 85.5, 86.17, 86.02, 86.33, 86.3, 86.55, 86.6, 86.57, 86.1, 86.28, 86.65, 85.88, 86.18, 85.87, 86.28, 86.28, 86.22, 85.7, 85.82, 85.55, 85.93, 86.2, 86.47, 86.57, 85.7, 85.47, 85.03, 84.98, 85.37, 85.65, 85.62, 85.93, 85.9, 85.57, 85.82, 85.93, 86.08, 86.52, 86.45, 86.2, 86.47, 86.33]
# FedSAM = [19.2, 25.48, 37.19, 47.67, 56.27, 62.06, 65.82, 68.11, 69.97, 72.41, 73.57, 73.79, 75.01, 75.54, 75.73, 76.17, 76.76, 76.77, 76.92, 77.37, 77.62, 78.02, 78.02, 78.05, 78.49, 79.08, 79.14, 79.34, 79.16, 79.64, 79.88, 80.23, 80.44, 80.27, 80.35, 80.54, 81.19, 81.39, 81.57, 81.88, 82.12, 82.58, 82.69, 82.79, 83.18, 83.54, 83.85, 83.77, 84.28, 84.13, 84.65, 84.23, 84.65, 84.63, 84.92, 84.81, 85.09, 85.3, 85.11, 85.2, 85.12, 85.1, 85.08, 85.23, 85.34, 85.66, 85.35, 85.32, 85.56, 85.41, 85.52, 85.78, 85.8, 86.18, 86.33, 86.22, 86.56, 86.75, 86.85, 86.83, 86.99, 87.09, 87.12, 87.06, 86.73, 86.19, 86.61, 86.12, 86.23, 86.45, 86.34, 86.5, 86.76, 87.02, 86.65, 86.32, 86.27, 86.19, 86.74, 86.67]
# FashionMNIST iid
# FedPPKT = [10.0, 38.89, 60.58, 78.48,83.08, 84.57, 87.54, 88.03, 87.92, 88.4, 88.55, 88.74, 88.84, 88.57, 89.16, 89.34, 89.26, 89.4, 89.37, 89.18, 89.64, 89.38, 89.6, 89.71, 89.57, 89.05, 89.83, 89.65, 89.43, 89.12, 89.42, 89.59, 89.28, 89.19, 89.31, 89.44, 89.6, 89.61, 89.58, 89.58, 89.59, 89.4, 89.11, 89.2, 89.05, 89.26, 89.14, 89.32, 89.42, 89.51, 89.45, 89.39, 89.32, 89.5, 89.12, 89.52, 89.28, 89.57, 88.89, 89.53, 88.95, 89.32, 89.45, 89.5, 89.16, 89.17, 89.82, 89.43, 89.57, 89.38, 89.16, 89.39, 89.61, 89.79, 89.75, 89.31, 89.61, 89.68, 89.66, 89.75, 89.21, 89.55, 89.07, 89.37, 89.51, 89.42, 89.2, 89.57, 89.43, 89.82, 89.81, 89.56, 89.29, 89.43, 89.57, 89.58, 89.53, 89.4, 89.53, 89.72]
# FDP = [34.0, 46.68, 54.84, 58.24, 62.37, 65.46, 67.87, 68.74, 69.55, 71.72, 72.18, 73.51, 73.62, 75.49, 74.31, 74.81, 75.4, 76.62, 77.47, 77.59, 78.34, 79.04, 80.5, 81.27, 82.0, 82.17, 82.33, 82.62, 83.55, 83.77, 83.41, 83.61, 83.15, 83.11, 83.83, 83.72, 84.19, 84.34,84.26, 84.53, 85.12, 84.9, 84.87, 85.33, 85.2, 85.03, 85.61, 84.97, 85.2, 85.67, 85.83, 85.59, 85.85, 86.32, 86.46, 86.59, 86.7, 86.52, 86.38, 86.69, 86.08, 86.41, 86.17, 86.56, 86.24, 85.74, 85.96, 85.47, 85.64, 85.39, 85.22, 85.98, 85.59, 85.44, 86.06, 86.48, 86.61, 86.29, 86.16, 85.61, 85.47, 85.66, 85.59, 85.22, 86.43, 86.04, 85.74, 86.14, 86.15, 85.16, 85.63, 85.35, 85.28, 85.37, 85.65, 85.54, 85.79, 85.58, 85.78, 85.39]
# DP_FedAvg = [36.15, 52.82, 58.49, 61.73, 63.72, 65.66, 68.85, 72.56, 73.46, 74.51, 74.59, 75.09, 75.27, 75.64, 76.51, 76.6, 75.49, 76.93, 78.49, 78.45, 78.72, 79.14, 79.41, 79.2, 79.23, 79.39, 79.86, 80.37, 80.31, 80.07, 80.54, 81.25, 81.39, 81.2, 81.44, 81.49, 81.5, 81.67, 82.37, 82.43, 82.52, 82.67, 82.96, 83.08, 83.89, 84.51, 84.71, 84.19, 83.89, 83.64, 83.44, 84.16, 84.06, 84.47, 84.29, 84.61, 84.47, 84.71, 85.09, 84.45, 84.46, 84.97, 85.04, 84.37, 84.42, 84.94, 85.0, 84.75, 84.11, 84.82, 84.56, 84.3, 82.83, 84.02, 84.65, 84.97, 84.83, 84.85, 84.71, 84.16, 84.55, 84.92, 84.52, 84.89, 84.75, 84.96, 84.11, 83.76, 83.95, 84.13, 84.51, 84.21, 84.79, 85.07, 84.73, 84.02, 84.41, 84.3, 83.98, 83.83]
# CENTAUR = [50.95, 55.03, 54.5, 56.45, 57.62, 56.67, 58.32, 57.92, 59.32, 60.4, 61.83, 62.5, 63.32, 63.7, 64.83, 65.43, 66.2, 66.9, 67.2, 67.42, 68.02, 68.33, 68.52, 68.1, 68.72, 68.35, 68.28, 69.88, 70.27, 69.63, 70.85, 71.67, 71.87, 72.65, 74.67, 74.35, 73.95, 74.78, 75.23, 75.28, 75.63, 77.9, 78.38, 79.35, 78.73, 80.17, 80.03, 80.08, 80.95, 81.12, 81.28, 81.95, 82.83, 82.42, 83.33, 83.57, 84.17, 84.27, 84.7, 85.28, 85.3, 85.4, 85.28, 85.67, 86.13, 86.22, 86.35, 85.92, 86.25, 87.07, 86.55, 86.98, 87.62, 87.4, 87.68, 87.7, 87.83, 87.62, 87.43, 87.88, 87.95, 88.35, 88.38, 88.3, 88.55, 88.37, 88.35, 88.43, 88.5, 88.4, 88.62, 88.6, 88.55, 88.14, 88.26, 88.21, 88.57, 88.27, 88.57, 88.33]
# PPSGD = [47.2, 47.88, 48.87, 51.7, 55.37, 57.8, 60.58, 65.3, 69.98, 69.22, 70.83, 73.13, 72.47, 74.42, 76.85, 79.48, 79.27, 81.08, 81.2, 82.4, 83.05, 83.73, 84.75, 85.13, 85.12, 85.12, 85.12, 85.18, 85.37, 85.37, 84.98, 85.37, 85.0, 84.72, 85.03, 85.45, 85.7, 85.17, 85.75, 85.95, 86.02, 86.25, 86.28, 86.5, 86.47, 86.75, 86.8, 86.47, 86.57, 86.68, 86.7, 86.58, 86.72, 86.7, 86.9, 87.13, 87.03, 87.13, 87.17, 87.37, 87.52, 87.52, 87.8, 87.85, 87.77, 87.87, 87.95, 88.0, 88.15, 88.2, 88.27, 88.3, 88.38, 88.12, 88.25, 88.28, 88.35, 88.25, 88.58, 88.4, 88.33, 88.32, 88.22, 88.22, 88.07, 88.27, 88.12, 88.18,88.28, 88.18, 87.98, 88.03, 87.98, 88.07, 88.35, 88.3, 88.12, 88.12, 88.42, 88.38]
# FedSAM = [19.2, 25.48, 37.19, 47.67, 56.27, 62.06, 65.82, 68.11, 69.97, 72.41, 73.57, 73.79, 75.01, 75.54, 75.73, 76.17, 76.76, 76.77, 76.92, 77.37, 77.62, 78.02, 78.02, 78.05, 78.49, 79.08, 79.14, 79.34, 79.16, 79.64, 79.88, 80.23, 80.44, 80.27, 80.35, 80.54, 81.19, 81.89, 82.27, 82.38, 82.82, 82.58, 82.69, 82.79, 82.78, 83.24, 83.85, 83.77, 84.28, 84.13, 84.65, 84.73, 84.65, 84.63,84.92, 85.31, 85.09, 85.3, 86.11, 86.2, 86.12, 86.7, 87.08, 87.23, 87.34, 87.66, 88.35, 88.32, 88.56, 88.61, 88.72, 88.78, 88.89, 88.68, 88.91, 88.72, 88.56, 87.95, 87.85, 87.83, 87.99, 88.09, 88.36, 88.06, 88.23, 88.19, 87.61, 87.12, 87.23, 86.45, 86.34, 86.5, 86.76, 87.02, 87.65, 87.32, 87.27, 87.19, 87.74, 87.67]
# CIFAR10 non-iid
FedPPKT = [10.00, 11.94, 31.32, 49.56, 56.10, 61.53, 63.77, 65.75, 65.27, 68.00, 68.03, 68.24, 69.47, 69.76, 69.40, 70.65, 70.60, 71.15, 72.27, 70.89, 71.66, 71.68, 71.66, 71.60, 71.18, 71.82, 72.45, 72.25, 72.50, 71.26, 71.67, 72.18, 71.58, 72.10, 72.95, 72.24, 71.91, 72.22, 71.24, 72.38, 71.52, 71.52, 72.10, 72.52, 71.55, 71.92, 71.79, 71.51, 72.69, 72.44, 71.45, 71.45, 71.36, 73.28, 73.52, 73.77, 73.96, 73.85, 71.65, 73.37, 73.80, 73.96, 74.55, 74.82, 75.43, 75.68, 74.79, 75.40, 75.27, 74.86, 75.18, 75.16, 74.58, 75.19, 75.53, 75.01, 74.72, 74.27, 73.48, 73.78, 73.43, 73.51, 72.89, 73.53, 73.25, 72.90, 73.91, 73.28, 73.41, 73.61, 72.80, 72.25, 73.52, 74.08, 74.20, 73.32, 72.43, 72.54, 72.96, 72.49]
FDP = [34.75, 36.54, 39.23, 43.59, 47.32, 49.27, 50.76, 52.42, 53.29, 55.04, 56.73, 57.04, 57.76, 57.98, 57.84, 58.52, 58.41, 59.53, 59.91, 60.01, 60.13, 60.34, 61.09, 61.16, 61.33, 61.29, 61.47, 62.54, 62.78, 62.94, 63.06, 63.55, 63.84, 63.59, 64.48, 64.94, 64.29, 64.69, 64.38, 64.65, 64.46, 65.38, 65.98, 66.05, 65.98, 66.62, 66.3, 66.03, 65.51, 65.77, 66.05, 66.33, 66.38, 66.16, 66.55, 66.61, 65.96, 65.94, 66.28, 66.55, 66.87, 66.64, 66.83, 66.61, 66.75, 66.69, 66.89, 66.93, 66.68, 66.79, 66.92, 67.16, 67.09, 67.04, 66.89, 67.0, 67.05, 66.82, 66.8, 66.59, 66.38, 66.3, 66.39, 66.04, 66.48, 66.16, 66.58, 65.93, 65.88, 66.83, 66.62, 67.12, 66.12, 66.84, 66.31, 67.11, 66.75, 67.12, 66.23, 66.18]
DP_FedAvg = [13.38, 19.13, 27.58, 35.67, 38.07, 41.07, 44.15, 46.12, 48.04, 49.91, 50.55, 52.04, 53.35, 53.88, 54.81, 55.52, 57.16, 60.89, 61.96, 62.74, 63.14, 64.04, 65.61, 65.48, 64.79, 65.2, 66.95, 66.68, 67.12, 68.11, 68.69, 69.81, 70.34, 71.29, 71.78,72.33, 72.76, 73.09, 72.9, 73.24, 72.7, 73.32, 73.06, 73.26, 72.77, 72.98, 72.78, 72.73, 73.06, 73.19, 73.26, 72.91, 73.21, 72.91, 72.79, 72.59, 72.12, 72.9, 73.03, 73.53, 73.69, 73.08, 72.96, 73.36, 73.55, 72.9, 73.33, 73.0, 72.61, 72.67, 72.34, 72.91, 73.33, 73.77, 73.06, 72.32, 72.65, 72.3, 72.51, 72.85, 72.5, 72.2, 71.98, 72.03, 72.08, 72.24, 72.85, 73.72, 73.82, 73.21, 72.35, 72.89, 72.35, 72.92, 73.19, 73.82, 72.64, 72.26, 70.89, 71.72]
PPSGD = [36.82, 43.08, 47.06, 51.52, 54.34, 53.64, 57.92, 60.96, 63.58, 65.22, 65.16, 66.18, 66.64, 66.42, 66.18, 67.26, 67.0, 67.18, 67.16, 67.5, 67.82, 66.92, 66.82, 67.2, 67.58, 68.04, 68.44, 68.2, 67.88, 67.52, 67.6, 68.46, 68.5, 68.54, 68.52, 68.82, 69.31, 69.19, 69.37, 69.83, 69.62, 69.56, 69.84, 69.9, 70.18, 70.04, 70.88, 70.74, 70.36, 70.82, 71.18, 71.35, 71.24, 71.62, 71.24, 71.36, 71.81, 71.44, 70.38, 70.82, 70.14, 70.2, 71.22, 71.72, 71.41, 71.39, 71.66, 71.3, 71.04, 70.94, 70.7, 71.16, 71.56, 71.76, 71.06, 70.85, 70.88, 70.7, 71.04, 71.56, 71.8, 71.34, 71.58, 70.54, 69.96, 69.72, 69.1, 69.94, 70.82, 70.84, 70.92, 70.94, 70.56, 70.26, 71.18, 70.64, 71.34, 71.22, 70.48, 69.86]
CENTAUR = [36.75, 41.39, 44.73, 46.79, 48.52, 50.47, 51.96, 51.62, 53.49, 57.24, 56.93, 58.24, 59.96, 60.18, 62.04, 63.72, 63.61, 63.73, 64.23, 62.21, 61.33, 62.54, 63.19, 63.26, 64.43, 64.39, 65.57, 65.64, 65.88, 66.74, 66.16, 66.63, 66.92, 66.68, 67.39, 67.81, 67.8, 67.58, 67.25, 67.52, 68.33, 68.26, 67.84, 68.15, 68.08, 67.72, 67.4, 68.13, 68.61, 68.87, 68.15, 68.43, 68.45, 68.23, 68.62, 67.79, 67.91, 68.08, 68.42, 68.69, 69.01, 69.64, 69.87, 69.65, 70.8, 70.69, 70.81, 71.13, 71.06, 71.79, 71.35, 71.78, 71.73, 71.94, 71.84, 72.05, 72.16, 72.34, 72.01, 71.17, 71.91, 71.92, 71.29, 71.57, 71.17, 71.45, 71.78, 71.73, 71.68, 71.35, 71.47, 71.97, 71.86, 71.64, 71.11, 70.95, 70.65, 71.02, 71.11, 71.28]
FedSAM = [10.03, 10.68, 27.44, 43.54, 52.34, 55.09, 57.4, 58.35, 60.85, 61.25, 62.91, 63.47, 64.49, 65.55, 65.07, 65.27, 65.57, 66.27, 66.59, 66.24, 66.73, 67.24, 67.45, 67.25, 67.93, 68.55, 68.78, 68.86, 68.87, 68.56, 68.61, 69.16, 69.19, 69.26, 69.28, 69.55, 69.64, 69.67, 69.87, 70.47, 70.44, 70.86, 70.84, 71.41, 71.56, 71.34, 71.98, 72.01, 72.47, 72.45, 72.65, 72.17, 72.16, 72.22, 72.31, 72.42, 72.59, 72.67, 72.71, 72.82, 72.9, 73.22, 73.1, 73.41, 73.17, 73.66, 73.85, 73.94, 73.97, 73.92, 73.81, 73.77, 73.41, 73.64, 73.06, 73.12, 72.73, 72.88, 72.71, 72.99, 72.75, 73.16, 73.44, 73.57, 73.04, 72.85, 72.76, 72.37, 72.38, 72.01, 71.82, 72.77, 72.87, 72.73, 72.48, 72.33, 72.58, 72.3, 72.13, 72.73]
# CIFAR10 iid
# FedPPKT = [10.0, 14.86, 44.11, 62.12, 68.12, 71.86, 74.02, 74.12, 74.86, 75.14, 75.47, 76.27, 76.36, 76.82, 77.89, 77.97, 78.43, 77.45, 78.35, 78.5, 77.24, 77.67, 78.04, 78.12, 77.19, 77.67, 77.74, 76.77, 77.91, 78.36, 78.36, 78.13, 77.37, 77.78, 77.56, 77.4, 77.77, 77.38, 76.96, 76.83, 77.27, 77.53, 77.59, 77.88, 78.44, 78.54, 78.75, 78.61, 78.06, 78.15, 78.68, 78.99, 78.94, 78.81, 79.17, 78.72, 78.54, 78.67, 78.31, 78.41, 78.23, 77.88, 77.51, 77.65, 77.95, 78.53, 77.49, 77.85, 77.25, 77.88, 77.99, 77.23, 76.59, 77.32, 77.14, 78.0, 76.85, 78.22, 77.65, 77.51,77.68,77.91,78.01,78.16,77.94,78.32,78.65,78.91,78.49,78.16,77.68, 77.35,76.98,77.37,77.14,76.59,76.34,76.38,76.69,77.15]
# FDP = [39.75, 49.73, 53.52, 55.96, 56.49, 56.93, 57.96, 58.04, 59.61, 60.23, 60.33, 60.59, 61.43, 61.57, 61.88, 62.16, 61.92, 62.39, 62.18, 62.25, 62.33, 62.84, 63.08, 63.4, 63.61, 63.15, 63.45, 63.62, 63.1, 63.42, 63.01, 63.87, 63.8, 63.81, 63.6, 63.35, 63.73, 63.84, 64.6, 64.37, 64.91, 64.92, 65.7, 65.78, 65.68, 65.47, 65.97, 66.11, 66.65, 66.11, 66.7, 66.65, 66.5, 66.39, 66.55, 66.91, 67.06, 67.44, 67.64, 67.26, 67.24, 67.77, 67.41, 67.52, 67.17, 67.64, 67.49, 67.88, 68.18, 68.04, 68.27, 68.01, 67.96, 68.21, 67.69, 67.88, 68.01, 67.63, 67.34, 67.62, 67.57, 67.43, 67.74, 67.94, 68.32, 68.16, 68.05, 67.7, 67.26, 67.65, 67.15, 67.32, 67.04, 67.4, 68.21, 67.73, 68.11, 67.72, 67.95, 67.81]
# DP_FedAvg = [17.03, 31.25, 39.70, 45.62, 49.64, 53.72, 58.74, 61.03, 63.84, 64.61, 65.87, 66.27, 67.25, 68.18, 68.13, 68.66, 69.57, 70.38, 71.06, 70.83, 70.69, 71.24, 70.95, 71.71, 71.37, 71.85, 72.30, 72.34, 73.02, 73.57, 73.23, 73.85, 74.01, 74.25, 74.60, 74.56, 74.33, 74.62, 74.49, 75.23, 75.17, 75.08, 74.96, 75.28, 75.10, 74.88, 74.83, 74.86, 75.63, 75.54, 75.13, 74.92, 74.94, 74.97, 75.16, 75.55, 75.52, 75.46, 75.80, 75.70, 75.66, 75.28, 75.17, 75.52, 75.52, 75.19, 75.45, 75.99, 76.11, 76.01, 75.64, 75.63, 75.29, 74.60, 75.51, 75.16, 74.81, 75.10, 75.62, 75.62, 75.60, 76.12, 75.51, 75.64, 75.29, 75.50, 75.05, 75.91, 76.19, 75.61, 76.19, 75.84, 75.57, 76.47, 76.12, 75.91, 76.03, 75.98, 76.16, 76.10]
# PPSGD = [38.02, 45.3, 51.44, 56.87, 59.68, 61.28, 63.02, 65.52, 65.16, 65.76, 66.61, 65.61, 66.15, 66.42, 66.62, 66.88, 67.32, 67.26, 67.71, 67.35, 67.89, 67.77, 67.98, 68.59, 68.53, 69.04, 69.56, 69.91, 70.17, 70.38, 70.62, 70.2, 70.6, 69.61, 69.33, 69.97, 69.94, 69.91, 70.17, 70.3, 70.36, 70.82, 70.92, 70.92, 71.12, 71.09, 70.74, 70.7, 70.98, 70.75, 70.98, 71.4, 71.68, 71.81, 71.51, 71.64, 71.37, 71.58, 70.68, 70.9, 70.56, 70.6, 69.86, 70.06, 70.11, 70.44, 70.15, 70.49, 71.12, 70.87, 70.89, 70.53, 70.76, 69.63, 69.6, 69.82, 69.50, 70.32, 70.47, 71.15, 71.71, 70.63, 70.18, 70.93, 71.16, 71.21, 71.81, 71.42, 71.02, 70.68, 71.43, 71.16, 71.69, 70.94, 71.16, 71.13, 70.78, 70.91, 70.36, 71.38]
# CENTAUR = [34.52, 38.58, 41.34, 45.58, 49.02, 55.32, 57.82, 58.58, 59.2, 60.24, 61.8, 60.58, 61.64, 62.74, 62.18, 62.68, 62.85, 63.96, 63.54, 64.08, 63.7, 64.3, 65.48, 66.04, 65.14, 64.96, 66.66, 66.42, 67.64, 67.6, 67.54, 67.95, 67.36, 67.83, 68.14, 68.26, 68.54, 68.44, 68.98, 69.28, 69.76, 69.12, 69.36, 70.46, 71.02, 71.38, 71.56, 71.34, 71.92, 72.58, 72.02, 72.22, 72.0, 72.38, 72.66, 73.38, 73.24, 73.62, 72.86, 73.56, 73.16, 73.44, 73.52, 73.82, 73.97, 74.12, 74.28, 74.28, 74.64, 74.34, 74.12, 74.82, 75.01, 74.88, 74.74, 74.72, 74.6, 73.5, 73.1, 73.36, 72.34, 72.06, 72.58, 73.54, 72.92, 73.26, 74.56, 74.2, 73.38, 73.36, 72.74, 72.8, 72.54, 72.6, 72.82, 71.5, 72.54, 71.88, 72.28, 72.88]
# FedSAM = [10.03, 27.68, 39.44, 45.54, 49.34, 52.09, 55.4, 61.35, 63.85, 64.25, 64.91,65.47, 66.49, 67.55, 67.07, 67.57, 68.27, 69.59, 69.24, 69.73, 70.24, 70.94, 71.25, 71.65, 71.93, 72.55, 72.78, 73.26, 73.57, 73.86, 73.61, 74.16, 74.39, 74.56, 74.68, 74.95, 75.64, 75.87, 75.54, 75.64, 75.95, 76.46, 76.54, 76.41, 76.56, 76.74, 76.98, 77.31, 77.12, 76.87, 76.95, 77.25, 77.17, 77.26, 76.92, 77.01, 76.82, 76.59, 76.87, 76.71, 76.92, 76.70, 76.52, 76.80, 76.91, 77.17, 77.06, 76.85, 76.94, 77.08, 77.32, 77.37, 77.14, 77.08, 76.84, 76.46, 76.54, 76.12, 76.73, 76.88, 76.35, 76.71, 76.99, 76.75, 76.16, 76.44, 76.57, 77.04, 76.85, 76.76, 77.13, 77.31, 77.21, 76.84, 76.67, 77.17, 76.97, 76.73, 76.48, 76.33 ]
plt.subplot(111)
plt.figure(figsize=(16,15),)
plt.title('CIFAR10 non-iid', weight='bold')
plt.plot(range(100), FDP, color=(30/255, 30/255, 32/255), label='FDP', linewidth=3)
plt.plot(range(100), DP_FedAvg, color=(224/255,210/255,163/255), label='DP-FedAvg', linewidth=3)
plt.plot(range(100), CENTAUR, color=(85/255, 150/255, 126/255), label='CENTAUR', linewidth=3)
plt.plot(range(100), PPSGD, color=(135/255,49/255,78/255), label='PPSGD', linewidth=3)
plt.plot(range(100), FedSAM, color=(124/255, 124/255, 186/255), label='DP-FedSAM', linewidth=3)
plt.plot(range(100), FedPPKT, color=(63/255, 160/255, 192/255), label='FedPPKT', linewidth=3)


plt.legend(loc=4, prop={ 'size': 24}, framealpha=1 )

plt.xlabel('Communication Rounds(T)', size=36)
plt.ylabel('Test Accuracy(%)', size=36)

# plt.xlim(1, 151)
# plt.ylim(0, 1)

# plt.show()
plt.savefig('CIFAR10-non-iid.eps', format='eps')
# plt.savefig(os.path.join('./figures/CIFAR10-non-iid.png') )