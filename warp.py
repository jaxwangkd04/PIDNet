import numpy as np
import scipy.linalg
import torch
import util_visdom

def fit(Xsrc,Xdst):
	ptsN = len(Xsrc)
	X,Y,U,V,O,I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
	A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1),
						np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
	b = np.concatenate((U,V),axis=0)
	p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
	pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=torch.float32)
	return pMtrx

# compute composition of warp parameters
def compose(opt,p,dp):
	pMtrx = vec2mtrx(opt,p)
	dpMtrx = vec2mtrx(opt,dp)
	pMtrxNew = dpMtrx.matmul(pMtrx)
	pMtrxNew = pMtrxNew/pMtrxNew[:,2:3,2:3]
	pNew = mtrx2vec(opt,pMtrxNew)
	return pNew

# compute inverse of warp parameters
def inverse(opt,p):
	pMtrx = vec2mtrx(opt,p)
	pInvMtrx = pMtrx.inverse()
	pInv = mtrx2vec(opt,pInvMtrx)
	return pInv

# convert warp parameters to matrix
def vec2mtrx(opt,p):
	O = torch.zeros(opt.batch_size,dtype=torch.float32).cuda()
	I = torch.ones(opt.batch_size,dtype=torch.float32).cuda()
	p1,p2,p3,p4,p5,p6,p7,p8 = torch.unbind(p,dim=1)
	pMtrx = torch.stack([torch.stack([I+p1,p2,p3],dim=-1),
						 torch.stack([p4,I+p5,p6],dim=-1),
						 torch.stack([p7,p8,I],dim=-1)],dim=1)
	return pMtrx

# convert warp matrix to parameters
def mtrx2vec(opt,pMtrx):
	[row0,row1,row2] = torch.unbind(pMtrx,dim=1)
	[e00,e01,e02] = torch.unbind(row0,dim=1)
	[e10,e11,e12] = torch.unbind(row1,dim=1)
	[e20,e21,e22] = torch.unbind(row2,dim=1)
	if opt.warpType=="translation": p = torch.stack([e02,e12],dim=1)
	if opt.warpType=="similarity": p = torch.stack([e00-1,e10,e02,e12],dim=1)
	if opt.warpType=="affine": p = torch.stack([e00-1,e01,e02,e10,e11-1,e12],dim=1)
	if opt.warpType=="homography": p = torch.stack([e00-1,e01,e02,e10,e11-1,e12,e20,e21],dim=1)
	return p

# warp the image
def transformImage(opt,image,pMtrx):
	refMtrx = torch.from_numpy(opt.refMtrx).cuda()
	refMtrx = refMtrx.repeat(opt.batch_size,1,1)
	transMtrx = refMtrx.matmul(pMtrx)
	X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
	X,Y = X.flatten(),Y.flatten()
	XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
	XYhom = np.tile(XYhom,[opt.batch_size,1,1]).astype(np.float32)
	XYhom = torch.from_numpy(XYhom).cuda()
	XYwarpHom = transMtrx.matmul(XYhom)
	XwarpHom,YwarpHom,ZwarpHom = torch.unbind(XYwarpHom,dim=1)
	Xwarp = (XwarpHom/(ZwarpHom+1e-8)).reshape(opt.batch_size,opt.H,opt.W)
	Ywarp = (YwarpHom/(ZwarpHom+1e-8)).reshape(opt.batch_size,opt.H,opt.W)
	grid = torch.stack([Xwarp,Ywarp],dim=-1)

	imageWarp = torch.nn.functional.grid_sample(image.cuda(),grid,mode="bilinear",align_corners=True)
	return imageWarp

def jx_transSingleImage(opt,image,pMtrx):
	bcsz = 1
	singleIm = torch.from_numpy(image).cuda().unsqueeze(0).unsqueeze(0)
	refMtrx = torch.from_numpy(np.eye(3).astype(np.float32)).cuda()
	transMtrx = refMtrx.matmul(torch.from_numpy(pMtrx).cuda().float())
	X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
	X,Y = X.flatten(),Y.flatten()
	XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
	XYhom = np.tile(XYhom,[bcsz,1,1]).astype(np.float32)
	XYhom = torch.from_numpy(XYhom).cuda()
	XYwarpHom = transMtrx.matmul(XYhom)
	XwarpHom,YwarpHom,ZwarpHom = torch.unbind(XYwarpHom,dim=1)
	Xwarp = (XwarpHom/(ZwarpHom+1e-8)).reshape(bcsz,opt.H,opt.W)
	Ywarp = (YwarpHom/(ZwarpHom+1e-8)).reshape(bcsz,opt.H,opt.W)
	grid = torch.stack([Xwarp,Ywarp],dim=-1)
	imageWarp = torch.nn.functional.grid_sample(singleIm, grid,mode="bilinear",align_corners=True)
	return imageWarp

def genPerturbations(opt):
	X = np.tile(opt.canon4pts[:,0],[opt.batch_size,1])
	Y = np.tile(opt.canon4pts[:,1],[opt.batch_size,1])
	O = np.zeros([opt.batch_size,4],dtype=np.float32)
	I = np.ones([opt.batch_size,4],dtype=np.float32)
	dX = np.random.randn(opt.batch_size,4)*opt.pertScale + np.random.randn(opt.batch_size,1)*opt.transScale
	dY = np.random.randn(opt.batch_size,4)*opt.pertScale + np.random.randn(opt.batch_size,1)*opt.transScale
	dX,dY = dX.astype(np.float32),dY.astype(np.float32)
	A = np.concatenate([np.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
						np.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)],axis=1)
	b = np.expand_dims(np.concatenate([X+dX,Y+dY],axis=1),axis=-1)
	pPert = np.matmul(np.linalg.inv(A),b).squeeze()
	pPert -= np.array([1,0,0,0,1,0,0,0])
	pInit = torch.from_numpy(pPert).cuda()
	return pInit