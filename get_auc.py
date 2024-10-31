from autometric import *



if __name__=="__main__":
	gt = fileList('./datasets/COVID-19-Inf/TestingSet/LungInfection-Test/GT/', '*.png')
	unet = fileList('./outputs/Inf/UNet/crossVal0/', '*.png')
	unext = fileList('./outputs/Inf/UNeXt/crossVal0/', '*.png')
	transunet = fileList('./outputs/Inf/TransUNet/crossVal0/', '*.png')
	swinunet = fileList('./outputs/Inf/SwinUNet/crossVal0/', '*.png')
	swint = fileList('./outputs/Inf/swinT/crossVal0/', '*.png')
	segnet = fileList('./outputs/Inf/SegNet/crossVal0/', '*.png')
	resnet = fileList('./outputs/Inf/resnet/crossVal0/', '*.png')
	pspnet = fileList('./outputs/Inf/PSPNet/crossVal0/', '*.png')
	poolformer = fileList('./outputs/Inf/poolformer/crossVal0/', '*.png')
	nextvit = fileList('./outputs/Inf/nextvit/crossVal0/', '*.png')
	miniseg = fileList('./outputs/Inf/MiniSeg/crossVal0/', '*.png')
	infnet = fileList('./outputs/Inf/InfNet/crossVal0/', '*.png')
	fcn = fileList('./outputs/Inf/FCN/crossVal0/', '*.png')
	deeplab = fileList('./outputs/Inf/DeepLabv3/crossVal0/', '*.png')
	cswin = fileList('./outputs/Inf/cswin/crossVal0/', '*.png')
	ctformer_t = fileList('./outputs/Inf/ctformer_t/crossVal0/', '*.png')
	ctformer_s = fileList('./outputs/Inf/ctformer_s/crossVal0/', '*.png')
	ctformer_b = fileList('./outputs/Inf/ctformer_b/crossVal0/', '*.png')

	modelName=['unet','unext','transunet', 'swinunet', 'swint','segnet', 'resnet', 'pspnet', 'poolformer', 'nextvit', 'miniseg', 'infnet', 'fcn', 'deeplab', 'cswin', 'ctformer_t', 'ctformer_s', 'ctformer_b']

	drawCurve(gt,[unet,unext,transunet, swinunet, swint,segnet, resnet, pspnet, poolformer, nextvit, miniseg, infnet, fcn, deeplab, cswin, ctformer_t, ctformer_s, ctformer_b],modelName,'kaggle')