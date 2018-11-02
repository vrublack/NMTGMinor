import sys

translate_src_f = sys.argv[1]
# should be sorted!
eval_f = sys.argv[2]


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
eval = read_eval()

assert len(translate_src) == len(eval)

length_eval = []

for i, s in enumerate(translate_src):
    if i == 0 or len(s) > len(translate_src[i - 1]):
        length_eval.append((len(s), []))
    length_eval[-1][1].append(eval[i])

def avg(l):
    return sum(l) / len(l)

avgs = []
for les in length_eval:
    avgs.append((avg(list(zip(*les[1]))[0]), avg(list(zip(*les[1]))[1]), avg(list(zip(*les[1]))[2])))


for i, a in enumerate(avgs):
    print('length {}: ts {:.3f}, fl {:.3f}, cp {:.3f}'.format(length_eval[i][0], a[0], a[1], a[2]))

