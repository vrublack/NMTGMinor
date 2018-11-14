import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


lines = []
while True:
    line = input()
    if line:
        lines.append(line)
    else:
        break

for fname in lines:

    log_fname = '/Users/valentin/BThesis/log/' + fname

    with open(log_fname) as f:
        log = f.read()

    matches = re.findall(r'(?<=accuracy: )(\d+\.\d+)', log)
    acc_train = [float(s) for s in matches[1::2]]
    acc_val = [float(s) for s in matches[2::2]]

    matches = re.findall(r'(?<=Reconstruction ppl: )(\d+\.\d+)', log)
    if 'BEST RESULTS:' in log:
        matches = matches[:-2]
    reconstr_train = [float(s) for s in matches[1::2]]
    reconstr_val = [float(s) for s in matches[2::2]]

    # job might not be finished so one result was written but another one not yet
    min_len = min(len(reconstr_val), len(reconstr_train), len(acc_train), len(acc_val))
    del reconstr_train[min_len:]
    del reconstr_val[min_len:]
    del acc_train[min_len:]
    del acc_val[min_len:]


    x_range = np.arange(1, min_len + 1)

    plt.plot(x_range, reconstr_train, label='Reconstr ppl train')
    plt.plot(x_range, reconstr_val, label='Reconstr ppl val')
    plt.plot(x_range, acc_train, label='Acc train')
    plt.plot(x_range, acc_val, label='Acc val')

    plt.ylim(0, 1.5)
    plt.grid(True)

    plt.xlabel('Epoch')
    plt.ylabel('Score')

    plt.title(Path(log_fname).name)

    plt.legend()

    plt.show()