import numpy as np
import scipy.misc
import torch
import os
import termcolor
import visdom
import torchvision

def mkdir(path):
	if not os.path.exists(path): os.mkdir(path)
def imread(fname):
	return scipy.misc.imread(fname)/255.0
def imsave(fname,array):
	scipy.misc.toimage(array,cmin=0.0,cmax=1.0).save(fname)
# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])
# restore model
def restoreModel(opt,geometric,classifier,it):
	geometric.load_state_dict(torch.load("models_{0}/{1}_it{2}_GP.npy".format(opt.group,opt.model,it)))
	classifier.load_state_dict(torch.load("models_{0}/{1}_it{2}_C.npy".format(opt.group,opt.model,it)))
# save model
def saveModel(opt,geometric,classifier,it):
	torch.save(geometric.state_dict(),"models_{0}/{1}_it{2}_GP.npy".format(opt.group,opt.model,it))
	torch.save(classifier.state_dict(),"models_{0}/{1}_it{2}_C.npy".format(opt.group,opt.model,it))

class Visdom():
	def __init__(self,opt):
		self.vis = visdom.Visdom(port=opt.port,use_incoming_socket=False)
		self.trainLossInit = True
		self.testLossInit = True
		self.meanVarInit = True
	def tileImages(self,opt,images,H,W,HN,WN):
		assert(len(images)==HN*WN)
		images = images.reshape([HN,WN,-1,H,W])
		images = [list(i) for i in images]
		imageBlocks = np.concatenate([np.concatenate(row,axis=2) for row in images],axis=1)
		return imageBlocks
	def trainLoss(self,opt,it,loss):
		loss = float(loss.detach().cpu().numpy())
		if self.trainLossInit:
			self.vis.line(Y=np.array([loss]),X=np.array([it]),win="{0}_trainloss".format(opt.model),
						  opts={ "title": "{0} (TRAIN_loss)".format(opt.model) })
			self.trainLossInit = False
		else: self.vis.line(Y=np.array([loss]),X=np.array([it]),win=opt.model+"_trainloss",update="append")
	def testLoss(self,opt,it,loss):
		if self.testLossInit:
			self.vis.line(Y=np.array([loss]),X=np.array([it]),win="{0}_testloss".format(opt.model),
						  opts={ "title": "{0} (TEST_error)".format(opt.model) })
			self.testLossInit = False
		else: self.vis.line(Y=np.array([loss]),X=np.array([it]),win=opt.model+"_testloss",update="append")
	def meanVar(self,opt,mean,var):
		mean = [self.tileImages(opt,m,opt.H,opt.W,1,10) for m in mean]
		var = [self.tileImages(opt,v,opt.H,opt.W,1,10)*3 for v in var]
		self.vis.image(mean[0].clip(0,1),win="{0}_meaninit".format(opt.model), opts={ "title": "{0} (TEST_mean_init)".format(opt.model) })
		self.vis.image(mean[1].clip(0,1),win="{0}_meanwarped".format(opt.model), opts={ "title": "{0} (TEST_mean_warped)".format(opt.model) })
		self.vis.image(var[0].clip(0,1),win="{0}_varinit".format(opt.model), opts={ "title": "{0} (TEST_var_init)".format(opt.model) })
		self.vis.image(var[1].clip(0,1),win="{0}_varwarped".format(opt.model), opts={ "title": "{0} (TEST_var_warped)".format(opt.model) })

	def HomograhpyImgs(self,opt, transdImg):
		transdImg1 = transdImg[0,0:10,::]
		row1 = transdImg1[0,::]
		for ii in range(1,len(transdImg1)):
			row1 = torch.cat((row1,transdImg1[ii,::]),1)

		transdImg2 = transdImg[0, 10:20, ::]
		row2 = transdImg2[0,::]
		for ii in range(1,len(transdImg2)):
			row2 = torch.cat((row2,transdImg2[ii,::]),1)

		image_display = torch.cat((row1,row2),0)
		self.vis.image(image_display,win="{0}_Perspective".format(opt.trainingset))

	def SingleImgs(self,opt, transdImg):
		transdImg = torch.from_numpy(transdImg).unsqueeze(0).unsqueeze(0)
		trdimg2 = [self.tileImages(opt,transdImg,opt.H,opt.W,1,1)]
		self.vis.image(trdimg2[0].clip(0,255),win="{0}_transdimg".format(opt.trainingset))
