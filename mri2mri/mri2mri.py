import torch
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from torchvision import transforms
import scipy.misc
from .options import Options
from .model import Model
from PIL import Image

def _toTensor(nibImg):

    img = Image.fromarray(nibImg).convert('RGB')     
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    return img

def _RGBtoGray(A):
    gray = A[:,0, ...] * 0.299 + A[:,1, ...] * 0.587 + A[:,2, ...] * 0.114
    return gray

def main():
    opt = Options().parse()
    assert(opt.input.endswith('nii.gz'))
    inputVolume = nib.load(opt.input)
    N = inputVolume.shape[2]
    target_shape = (opt.fineSize, opt.fineSize, N)
    data = resample_img(inputVolume, inputVolume.affine, target_shape=target_shape).get_data()

    model = Model()
    model.initialize(opt)
    output = torch.FloatTensor(N, 3, opt.fineSize, opt.fineSize)
    for i in range(N):
        if opt.verbose:
            print('process slice %d' % i)
        model.set_input({'A': _toTensor(data[:,:,i])})
        model.forward()
        output[i] = model.fake_B.detach().cpu()

    output = _RGBtoGray(output)
    outputImg = nib.Nifti1Image(output.permute(1,2,0).numpy(), inputVolume.affine)
    outputfile = opt.output
    if not outputfile.endswith("nii.gz"):
        outputfile = "%s.nii.gz" % (outputfile)
    print('save output as %s' % outputfile)
    nib.save(outputImg, outputfile)

if __name__ == "__main__":
    main()




