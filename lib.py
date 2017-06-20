import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time
from scipy import asarray as ar,exp
from PIL import Image
from scipy.special import erf




def read_and_plot_all(dir,fn,Nsm, rejection, xs, ys, Dy, xs2, ys2, Dy2):
	
	
	start_time = time.time()

	#DOWN
	#read the file as numpy array
	
	down = np.zeros(1001)
	delay = np.arange(0, len(down), 1)
	#~ print down[1]
	rej=0
	passed = 0
	x_pas = np.zeros(1001)
	y_pas = np.zeros(1001)
	i = 0
	#read the file as a string
	for a in range(0,fn+1):
		a = str(a)
		down_add = np.loadtxt(dir+'/results_down_'+a+'.txt')
		fin = len(down_add)-1
		if  np.average(down_add[0:5])/6-np.average(down_add[fin/2-100:fin/2+100])/100>=rejection and \
			np.average(down_add[fin-5:fin])/6-np.average(down_add[fin/2-100:fin/2+100])/100>=rejection and\
			np.amin(down_add[0:fin])>100 and np.amax(down_add[0:fin])<500:
			down = down + down_add
			passed = passed + 1
			#print'this is i=', i
			x_pas[i] = xs2 + int(a) / Dy2
			y_pas [i] = ys2 + int(a) % Dy2
			i=i+1
			plt.figure(13)
			plt.title('down passed')
			plt.plot(delay, down_add,  label = a)
		else:
			rej = rej + 1
			plt.figure(12)
			plt.title('down rejected')
			plt.plot(delay, down_add,  label = a)
		
	if passed >0:
		down = down / fn
	
	
	print '#####################'
	print '       DOWN           '
	print 'rejection parameter', rejection
	print 'rejected scans:', rej
	print 'passed scans:', passed
	rej=0
	passed = 0
	
	#UP
	#read the file	
	up = np.zeros(1001)
	
	#~ print down[1]
	x_pas2 = np.zeros(1001)
	y_pas2 = np.zeros(1001)
	i = 0
	#read the file as a string
	for a in range(0,fn+1):
		a = str(a)
		up_add = np.loadtxt(dir+'/results_up_'+a+'.txt')
		fin = len(up_add)-1
		if  np.average(down_add[0:5])/6-np.average(down_add[fin/2-100:fin/2+100])/100>=rejection and \
			np.average(down_add[fin-5:fin])/6-np.average(down_add[fin/2-100:fin/2+100])/100>=rejection and\
			np.amin(down_add[0:fin])>100 and np.amax(down_add[0:fin])<500 :
			down = down + down_add
			passed = passed + 1
			x_pas2[i] = xs + int(a) / Dy
			y_pas2[i] = ys + int(a) % Dy
			i=i+1
			plt.figure(12)
			plt.title('up passed')
			plt.plot(delay, up_add,  label = a)
		else:
			rej = rej + 1
			plt.figure(11)
			plt.title('up rejected')
			plt.plot(delay, up_add,  label = a)
		
	if passed >0:
		up = up / passed
	plt.figure(10)
	plt.plot(x_pas, y_pas, 'ro', label = 'down')
	plt.plot(x_pas2, y_pas2, 'bs',label = 'up')
	#print 'left', np.average(up_add[0:5])/6
	#print 'middle', np.average(up_add[fin/2-100:fin/2])/100
	#print 'right', np.average(up_add[fin-5:fin])/6
	
	#print 'condition', np.average(up_add[0:5])/6-np.average(up_add[fin/2-100:fin/2])/100
	
	#print 'condition 2', np.average(up_add[fin-5:fin])/6-np.average(up_add[fin/2-100:fin/2])/100
	
	print '#####################'
	print '       UP           '
	print 'rejection parameter', rejection
	print 'rejected scans:', rej
	print 'passed scans:', passed
	
	
	#Summed together
	inten = (up+down)/2.0

	
	#doing the math

	xint = np.arange(0, len(inten), 1)
	xder = np.empty(len(inten))
	der = np.empty(len(inten))
	
	# smooth		
	xsmooth = np.zeros(len(inten)/Nsm+1)
	smooth = np.zeros(len(inten)/Nsm+1)
	
	for x in range(0,len(inten)):
		xsmooth[int(x/Nsm)] = int(x)
		smooth[int(x/Nsm)] += inten[x]
	
	sder = np.zeros(len(inten)/Nsm+1)
	
	
	for x in range(1,len(inten)-1):
	#for x in inten:	
		der[x] = inten[x]-inten[x-1]
	
	#print 'number', (len(inten)-1)/Nsm-1, len(smooth), len(inten)
	for x in range(1,(len(inten)-1)/Nsm-1):
	#for x in smooth:	
		sder[x] = smooth[x]-smooth[x-1]
		
	elapsed_time = time.time() - start_time
	print 'Time for execution:', elapsed_time
		
	return xsmooth, smooth, sder
	
	
	
	
	
	
	
	
	

def tiff_read_and_extract(dir,pic_name, first_im,n_stack, xpix_in, xpix_fin, ypix_in, ypix_fin, fin):
	'''this function read a stack of tif files and extract the values of certain pixels through
	the stack and saves them in a nice matrix
	
	dir = 'directory of the files (name of the files has to be changed in the functionn
		for the moment)'
	pic_name = 'the name of the tif image without the number'
	
	first_im = number of the first image to analyze
	
	n_stack = number of stacks(number of scans)
	
	xpix_in = initial pixel for loop on x coordinate
	xpix_fin = final pixel for loop on x coordinate
	
	ypix_in = initial pixel for loop on y coordinate
	ypix_fin = final pixel for loop on y coordinate
	
	fin = lenght of a delay scan(or how many images for each scan)
	'''
	

	#coordinates of the pixel to analyze

	dx = xpix_fin - xpix_in + 1
	dy = ypix_fin - ypix_in + 1
	
	
	#declare two zeros arraay for later
	delay= np.zeros(1001)
	intensity= np.zeros(shape=(1001, dx*dy+1))
	
	for i in range(0, fin):
		num = str((n_stack*fin)+(first_im+i))
		#print 'n_stack', n_stack
		#print 'num', num
		delay[i] = i
		im = Image.open(dir + pic_name+num+'.tif')
		pix = im.load()
		pos=0
		for x in range(xpix_in, xpix_fin+1):
			for y in range(ypix_in, ypix_fin+1):
				#print i, pos
				intensity[i,pos] = pix[x,y]
				pos = pos +1
	
	imarray = np.array(im)
	
	#plotting the delay scan at the pixel position xpix,ypix
	#plt.figure(2)
	#plt.plot(delay[0:fin-1], intensity[0:fin-1])
	
	#printng the .tif file with the pixel position
	#this part will be used to see where the rejected and passed scans come from. 
	#I am using imarray because I still have no idea how to plot pix, because I have no idea what it is
	
	#plt.figure(1)
	#plt.matshow(imarray)
	#plt.plot(xpix,ypix, 'ro')
	
	
	
	#plt.show()
	#im.show()
	
	return intensity, imarray
	
	
	
def select_good_scans(n_stack,intensity, xpix_in, xpix_fin, ypix_in, ypix_fin, fin, rejection):
	
	dx = xpix_fin - xpix_in + 1
	dy = ypix_fin - ypix_in + 1
	

	
	x_pas = np.zeros(dx*dy*(n_stack+1))
	y_pas = np.zeros(dx*dy*(n_stack+1))

	x_rej = np.zeros(dx*dy*(n_stack+1))
	y_rej = np.zeros(dx*dy*(n_stack+1))

	i_pas = 0
	i_rej = 0
	rej=0
	for i in range(0,dx*dy*(n_stack)):
		l_av = np.average(intensity[[0,5], i])
		c_av = np.average(intensity[[fin/2-100,fin/2+100], i])
		r_av= np.average(intensity[[fin-6,fin-1], i])
		if l_av - c_av>rejection and r_av - c_av>rejection and \
			np.amin(intensity[:, i])>100 and np.amax(intensity[:, i])<500:
				plt.figure(5+i)
				plt.plot(intensity[:,i])
				try:
					intensity_sum
				except NameError:
					#print "running for the first time"
					intensity_sum = intensity[:, i]
					x_pas[0] = xpix_in + int(i) / dy 
					y_pas [0] = ypix_in + int(i) % dy
					pas = 1
					
				else:
					#print "running for the", i+1, " time"
					#print 
					intensity_sum = (intensity[:,i] + intensity_sum)
					x_pas[pas] = xpix_in + int(i) / dy 
					y_pas [pas] = ypix_in + int(i) % dy
					pas = pas + 1

		else:
			x_rej[rej] = xpix_in + i / dy  
			y_rej [rej] = ypix_in + i % dy
			rej = rej+1
			
	intensity_sum = intensity_sum / len(x_pas)
	x_pas_short = x_pas[0:pas]
	y_pas_short = y_pas[0:pas]
	x_rej_short = x_rej[0:rej]
	y_rej_short = y_rej[0:rej]
	#print 'xrej', x_rej_short
	#print 'y rej', y_rej_short
	print 'rejected', rej
	print 'passed', pas
	return intensity_sum, x_pas_short, y_pas_short, x_rej_short, y_rej_short
	
	
def smooth(inten, Nsm):

	#doing the math

	xint = np.arange(0, len(inten), 1)
	xder = np.empty(len(inten))
	der = np.empty(len(inten))
	
	# smooth		
	xsmooth = np.zeros(len(inten)/Nsm+1)
	smooth = np.zeros(len(inten)/Nsm+1)
	
	for x in range(0,len(inten)):
		xsmooth[int(x/Nsm)] = int(x)
		smooth[int(x/Nsm)] += inten[x]
	
	sder = np.zeros(len(inten)/Nsm+1)
	
	
	for x in range(1,len(inten)-1):
	#for x in inten:	
		der[x] = inten[x]-inten[x-1]
	
	#print 'number', (len(inten)-1)/Nsm-1, len(smooth), len(inten)
	for x in range(1,(len(inten)-1)/Nsm-1):
	#for x in smooth:	
		sder[x] = smooth[x]-smooth[x-1]
		
		
	return xsmooth[0:len(xsmooth)-1], smooth[0:len(xsmooth)-1], sder[0:len(xsmooth)-1]
	


def norm(xsmooth, smooth, smooth_der,Nsm):
	#Original dataset shows an increasing amplitude during the scan that is not related
	#width the effect we want to observe. we then normalize it:
	#averaging five points to the left and 5 to the right and drawing a line
	left_av = np.average(smooth[0:Nsm/3])
	right_av = np.average(smooth[-Nsm/3:-1])
	norm_line = left_av + (right_av-left_av)/(len(xsmooth)*Nsm)*xsmooth
	
	#correcting for the pendenza della retta
	smooth = smooth/norm_line
	
	#shifting to zero 
	smooth = smooth-np.amin(smooth)
	#print 'ss max', np.amin(ss)
	
	#normalizing to 1
	smooth = smooth/np.amax(smooth)
	smooth_der = smooth_der/np.amax(smooth_der)
	
	return xsmooth, smooth, smooth_der

	
def gaus(x,a,x0,sigma):
	
	return a*exp(-(x-x0)**2/(2*sigma**2))
	
def sb(x,x0_l, x0_r, x0_sb,sigma, amp):
	sb = (1-erf((x-x0_l)/(np.sqrt(2)*sigma)))/2 \
		+ (1+erf((x-x0_r)/(np.sqrt(2)*sigma)))/2 \
		+amp*np.exp(-(x-x0_sb)**2/(2*sigma**2))
	return sb
		