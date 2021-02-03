class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':.6f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self._list.append(self.avg)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_inout_seq(seq, len_seq, len_pred):
    inout_seq = []
    L = len(seq)
    for i in range(0, L - len_pred, len_pred):
        input_seq = seq