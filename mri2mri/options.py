import argparse
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--input', type=str, help='path to the input file, should be nii.gz', required=True)
        self.parser.add_argument('--transform', type=str, help='specify which transform. t1w2t2w, t2w2t1w', required=True)
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--model', type=str, default='unet_256', help='selects model to use for net')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--output', type=str, default='output', help='name of the output file')
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print more information')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        return self.opt
