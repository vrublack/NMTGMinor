import collections
import os
import re

LOGS_DIR = '/Users/valentin/BThesis/log'
EVAL_OUT_DIR = 'out'


def read_all_logs():
    all_logs = []
    for fname in os.listdir(LOGS_DIR):
        try:
            with open(os.path.join(LOGS_DIR, fname)) as l:
                all_logs.append((fname, l.read()))
        except UnicodeDecodeError:
            print('Skipping {}, probably .DS_STORE'.format(fname))
    return all_logs


def parse_job_id(log_fname):
    res = re.findall(r'(?<=_)\d+(?=\.)', log_fname)
    if len(res) == 0:
        return None
    else:
        if len(res) > 1:
            print('ERROR: job id parsing ambiguous with fname {}'.format(log_fname))
        return res[0]


def get_jobname(saved_model, logs):
    saved_model = saved_model.replace('%25j', '%j').replace('%2525j', '%j')
    jobname = None
    for log_name, log_content in logs:
        if saved_model in log_content:
            if jobname is not None:
                print('ERROR: multiple logs link to {} -> should probably ignore results for id {}'.format(saved_model, jobname))
            else:
                jobname = parse_job_id(log_name)
                if jobname is None:
                    print('ERROR: log {} doesnt have job name'.format(log_name))

    return jobname


def parse_model_name(fname):
    res = re.findall(r'(?<=mout_).+(?=_target)', fname)
    if len(res) == 0:
        return None
    else:
        if len(res) > 1:
            print('ERROR: job id parsing ambiguous for eval fname {}'.format(fname))
        return res[0]


def parse_target_index(fname):
    res = re.findall(r'(?<=_target_)\d(?=\.csv)', fname)
    if len(res) == 0:
        return None
    else:
        if len(res) > 1:
            print('ERROR: target parsing ambiguous for eval fname {}'.format(fname))
        return int(res[0]) - 1

def parse_metrics(fname):
    with open(fname) as f:
        last_line = f.readlines()[-1]
        if not last_line.startswith('AVG: '):
            print('ERROR: log file {} invalid'.format(fname))
            return -1, -1, -1
        last_line = last_line[len('AVG: '):]
        return [float(comp) for comp in last_line.split(',')]


logs = read_all_logs()

model_results = collections.defaultdict(lambda: [None, None])

for fname in os.listdir(EVAL_OUT_DIR):
    if not fname.startswith('eval'):
        continue

    model_name = parse_model_name(fname)
    if model_name is None:
        print('ERROR: model name not in {}'.format(fname))
        continue

    target_index = parse_target_index(fname)
    if target_index is None:
        print('ERROR: target index is None for fname {}'.format(fname))
        continue

    job_name = get_jobname(model_name, logs)

    if job_name is None:
        print('ERROR: no job name for eval out {}'.format(fname))

    style_strength, fluency, content_preservation = parse_metrics(os.path.join(EVAL_OUT_DIR, fname))

    model_results[job_name][target_index] = style_strength, fluency, content_preservation


def avg(tup1, tup2):
    res = []
    for i in range(len(tup1)):
        res.append(float('{:.2f}'.format((tup1[i] + tup2[i]) / 2)))
    return tuple(res)



for k, v in model_results.items():
    print('{}: avg {} and style 1 {} and style 2 {}'.format(k, avg(v[0], v[1]), v[0], v[1]))
