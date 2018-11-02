import sys
import re

# should be sorted!
translate_src_f = sys.argv[1]
all_avgs = []


def extract_lr(fname):
    res = re.findall(r'(?<=lr_).+(?=_target)', fname)
    if len(res) == 0:
        return None
    else:
        return float(res[0])


def read_translate_src():
    with open(translate_src_f) as f:
        return f.read().split('\n')


def read_eval():
    with open(eval_f) as f:
        data = f.read().split('\n')[1:-3]
    parsed = []
    for d in data:
        comps = d.split(',')
        parsed.append((float(comps[0]), float(comps[1]), float(comps[2])))
    return parsed


translate_src = read_translate_src()

for i in range(2, len(sys.argv)):
    eval_f = sys.argv[i]

    lr = extract_lr(eval_f)

    eval = read_eval()

    assert len(translate_src) == len(eval)

    length_eval = []

    len_steps = 10
    for i, s in enumerate(translate_src):
        if i == 0 or len(s) - length_eval[-1][0] >= len_steps:
            length_eval.append((len(s), []))
        length_eval[-1][1].append(eval[i])


    def avg(l):
        return sum(l) / len(l)


    avgs = []
    for les in length_eval:
        avgs.append((les[0], avg(list(zip(*les[1]))[0]), avg(list(zip(*les[1]))[1]), avg(list(zip(*les[1]))[2])))

    all_avgs.append((lr, avgs))

    # for i, a in enumerate(avgs):
    #     print('length {}: ts {:.3f}, fl {:.3f}, cp {:.3f}'.format(length_eval[i][0], a[0], a[1], a[2]))

all_avgs = sorted(all_avgs)

# digest
for i in range(len(all_avgs[0][1])):
    smallest_len = all_avgs[0][1][i][0]
    print('{} --> '.format(smallest_len), end='')
    for lr_d in all_avgs:
        print('({:.2f}, {:.2f}, {:.2f}), '.format(lr_d[1][i][1], lr_d[1][i][2], lr_d[1][i][3]), end='')
    print('\n')

print('Done.')
