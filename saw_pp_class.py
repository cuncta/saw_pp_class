#~ from lib import *
import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from PIL import Image
import time
import sys, os, re
from lib import *



class time_resolved_analysis():
	''' 	This class provides methods to do a complete analysis of pulse picker. 
		The raw data consists of tif files. One scan consists of several tif files recorded
		at different delay. In the example there are 10 scans, each one consisting of 1001 picture. 
		
		The Tiff files to be analyzed must be in the data folder: since the original data are
		approximately 10.000 tif files, I do not upload them on github. To run the test function I use the files
		produced by tiff_extract_n_scans.
	
		sample = string, the name of the sample, will be used when saving data/picture relative to the sample

		pic_name = string,  'the name of the tif image without the number'
	
		first_im = int, number of the first image to analyze
	
		n_scans = int, number of stacks(number of scans)'''
	
	def __init__(self, sample, pic_name, first_im, scan_length, n_scans):
		
		self.sample = sample
		self.pic_name = pic_name
		self.first_im = first_im
		self.scan_length = scan_length 
		self.n_scans = n_scans

		if not os.path.exists('intermediate'):
			os.mkdir('intermediate')
		return	
	
	def create_name_array(self):
		'''This method creates an array with the names of all the tiff files used for n scans. 
		At the moment this method is not used by other classes, this will be implemented in future'''
		file_names = []
		for m in range(0,self.n_scans):
			for i in range(0, self.scan_length):
				num = str((m*self.scan_length)+(self.first_im+i))
				file_names.append(self.pic_name+num+'.tif')
		return file_names
		
		
	def tiff_extract_n_scans(self, file_names, xpix_in, xpix_fin, ypix_in, ypix_fin, up_down):
		'''this method read n_scans composed  of scan_length tif files and extract the values 
		of certain pixels and saves them in matrix
	
		file_names list of strings, the name of the images to be analyzed, this is still not used at the moment
		
		xpix_in = initial pixel for loop on x coordinate
		xpix_fin = final pixel for loop on x coordinate
	
		ypix_in = initial pixel for loop on y coordinate
		ypix_fin = final pixel for loop on y coordinate
	
		up_down = string, will be used to save the data and differ between plus/minus first order'''
		
		self.file_names = file_names
		self.xpix_in = xpix_in
		self.xpix_fin = xpix_fin
		self.ypix_in = ypix_in
		self.ypix_fin = ypix_fin
		self.up_down = up_down
		
		dir = 'data/'
		#coordinates of the pixel to analyze
	
		dx = self.xpix_fin - self.xpix_in + 1
		dy = self.ypix_fin - self.ypix_in + 1
		
		
		#declare two zeros array for later
		delay= np.zeros(self.scan_length)
		intensity= np.zeros(shape=(self.scan_length, dx*dy+1))
		for m in range(0,self.n_scans):
			for i in range(0, self.scan_length):
				num = str((m*self.scan_length)+(self.first_im+i))
				delay[i] = i
				im = Image.open(dir + self.pic_name+num+'.tif')
				pix = im.load()
				pos=0
				for x in range(self.xpix_in, self.xpix_fin+1):
					for y in range(self.ypix_in, self.ypix_fin+1):
						#print i, pos
						intensity[i,pos] = pix[x,y]
						pos = pos +1
			try:
				intensity_all
			except NameError:
				print "extracted scan number 1"
				intensity_all = intensity
			else:
				print "extracted scan number", m+1
				intensity_all = np.concatenate((intensity_all, intensity), axis=1)
		imarray = np.array(im)

	
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_imarray.txt', (imarray))   
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_intensity.txt', (intensity_all))   
		return intensity_all#, imarray
	
	def select_good_scans(self, intensity, xpix_in, xpix_fin, ypix_in, ypix_fin, rejection, up_down, plot ):
		'''this method provides an easy way to differ between pixel scans. It differers between the pixels hit by the 
		plus minus first order and the others. TO DO: draw a picture to explain this. 
	
		file_names list of strings, the name of the images to be analyzed, this is still not used at the moment
		
		xpix_in = initial pixel for loop on x coordinate
		xpix_fin = final pixel for loop on x coordinate
	
		ypix_in = initial pixel for loop on y coordinate
		ypix_fin = final pixel for loop on y coordinate
	
		up_down = string, will be used to save the data and differ between plus/minus first order'''
		self.intensity = intensity
		self.xpix_in = xpix_in
		self.xpix_fin = xpix_fin
		self.ypix_in = ypix_in
		self.ypix_fin = ypix_fin
		self.rejection = rejection
		self.up_down = up_down
		
		#~ intensity = np.loadtxt('intermediate/'+self.sample+'_'+self.up_down+'_intensity.txt')   

		dx = self.xpix_fin - self.xpix_in + 1
		dy = self.ypix_fin - self.ypix_in + 1
		x_pas = np.zeros(dx*dy*(self.n_scans+1))
		y_pas = np.zeros(dx*dy*(self.n_scans+1))
		x_rej = np.zeros(dx*dy*(self.n_scans+1))
		y_rej = np.zeros(dx*dy*(self.n_scans+1))
	
		i_pas = 0
		i_rej = 0
		rej=0
		
		for i in range(0,dx*dy*(self.n_scans)):
			l_av = np.average(self.intensity[[0,5], i])
			c_av = np.average(self.intensity[[self.scan_length/2-100,self.scan_length/2+100], i])
			r_av= np.average(self.intensity[[self.scan_length-6,self.scan_length-1], i])
			if l_av - c_av>self.rejection and r_av - c_av>self.rejection and \
				np.amin(self.intensity[:, i])>100 and np.amax(self.intensity[:, i])<500:
					#~ plt.figure(5+i)
					#~ plt.plot(intensity[:,i])
					try:
						intensity_sum
					except NameError:
						#print "running for the first time"
						intensity_sum = self.intensity[:, i]
						x_pas[0] = self.xpix_in + int(i) / dy 
						y_pas [0] = self.ypix_in + int(i) % dy
						pas = 1
						
					else:
						#print "running for the", i+1, " time"
						#print 
						intensity_sum = (self.intensity[:,i] + intensity_sum)
						x_pas[pas] = self.xpix_in + int(i) / dy 
						y_pas [pas] = self.ypix_in + int(i) % dy
						pas = pas + 1
	
			else:
				x_rej[rej] = self.xpix_in + i / dy  
				y_rej [rej] = self.ypix_in + i % dy
				rej = rej+1
				
		intensity_sum = intensity_sum / len(x_pas)
		x_pas_short = x_pas[0:pas]
		y_pas_short = y_pas[0:pas]
		x_rej_short = x_rej[0:rej]
		y_rej_short = y_rej[0:rej]
		#print 'xrej', x_rej_short
		#print 'y rej', y_rej_short
		print 'selecting scans '+up_down
		print 'rejected', rej
		print 'passed', pas
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_selected_intensity.txt', (intensity_sum))
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_xpas.txt', x_pas)
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_xrej.txt', x_rej)   
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_ypas.txt', y_pas)
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_yrej.txt', y_rej) 
		if plot == 1:
			plt.figure(1)
			plt.title('Average of good scans '+self.up_down)
			plt.plot(intensity_sum[0: self.scan_length])
			plt.show()
		return  intensity_sum	
	
	def smooth(self, intensity, Nsm):
		'''this method provides an easy way to smooth the data. the data points of a signal are modified so 
		individual points (presumably because of noise) are reduced, and points that are lower than the adjacent 
		points are increased leading to a smoother signal. 
	
		intensity =array, scan to smooth
		Nsm = int, number of data points to average to smooth intensity
		
		the method return the intensity array, and an arbitrary x array smoothed by Nsm'''
		self.intensity = intensity
		self.Nsm = Nsm

		delay_smooth = np.zeros(len(self.intensity)/self.Nsm+1)
		intensity_smooth = np.zeros(len(self.intensity)/self.Nsm+1)
		
		for x in range(0,len(self.intensity)):
			delay_smooth[int(x/self.Nsm)] = int(x)
			intensity_smooth[int(x/self.Nsm)] += self.intensity[x]
			
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_intensity_smooth.txt', intensity_smooth) 
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_x_smooth.txt', delay_smooth) 
		return delay_smooth[0:len(delay_smooth)-1], intensity_smooth[0:len(delay_smooth)-1]
		
	def norm(self, delay, intensity, intensity_der,Nsm):
		self.delay = delay
		self.intensity = intensity
		self.intensity_der = intensity_der
		self.Nsm = Nsm
		#Original dataset shows an increasing amplitude during the scan that is not related
		#width the effect we want to observe. we then normalize it:
		#averaging five points to the left and 5 to the right and drawing a line
		left_av = np.average(self.intensity[0:Nsm/3])
		right_av = np.average(self.intensity[-Nsm/3:-1])
		norm_line = left_av + (right_av-left_av)/(len(self.delay)*self.Nsm)*self.delay
		
		#correcting for the pendenza della retta
		self.intensity = self.intensity/norm_line
		
		#shifting to zero 
		self.intensity = self.intensity-np.amin(self.intensity)
		#print 'ss max', np.amin(ss)
		
		#normalizing to 1
		self.intensity = self.intensity/np.amax(self.intensity)
		self.intensity_der = self.intensity_der/np.amax(self.intensity_der)
		
		return self.delay, self.intensity, self.intensity_der

	def derive(self, intensity):
		'''this method provides an easy way to derive a data set 
	
		intensity =array, scan to smooth
		
		the method return the derived intensity array, and saves it in the intermediate folder'''
		self.intensity = intensity
		intensity_der = np.empty(len(self.intensity))
		for x in range(1,len(self.intensity)):
		#for x in smooth:	
			#~ print 'x', x
			#~ print self.intensity[x]
			#~ print self.intensity[x-1]
			#~ print self.intensity[x]-self.intensity[x-1]
			intensity_der[x] = self.intensity[x]-self.intensity[x-1]
		#~ print sder	
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_intensity_der.txt', intensity_der) 
		return intensity_der[0:len(intensity_der)+1]
	
	def fit_edge(self, delay, intensity_der, plot):
		'''this method provides an easy way to fit the edges. Of course if used to fit the delay scan, this has to be 
			derived.
		
		delay = array, the delay relative to intensity_der
		intensity_der =array, the intensity scan derived, so that the there are peaks instead of edges
		plot = integer, if plot == 1 the results of the fit are plot and saved.
		
		the method returns the position and the fwhm of the left and right edge.
		Additionally it saves the parameters and the pcov matrix resulting from the fit in the intermediate folder,
		as well as a graph with the fit if plot = 1'''
		self.delay = delay
		self.intensity_der = intensity_der
		self.plot = plot 
		
		#LEFT PEAK
		x_l = self.delay[0:len(self.delay)/2]
		y = self.intensity_der[0:len(self.intensity_der)/2] * -1
		n = len(x_l)                          			#the number of data
		mean = x_l[np.argmax(y)]				#guessing mean value
		sigma = np.sqrt(sum(y*(x_l-mean)**2)/n )#guessing the fwhm
		#fitting
		[amp_l,mean_l,sigma_l],pcov_l = curve_fit(gaus,x_l,y,p0=[1,mean,sigma])
		fwhm_l = sigma_l * 2.35

		#RIGHT PEAK
		x_r = self.delay[len(self.delay)/2:len(self.delay)]
		y = self.intensity_der[len(self.intensity_der)/2:len(self.intensity_der)] 
		n = len(x_r)                          #the number of data
		mean = x_r[np.argmax(y)]                   #note this correction
		#No need to guess sigma again, taking the value from the previous fit
		#fitting
		[amp_r,mean_r,sigma_r],pcov_r = curve_fit(gaus,x_r,y,p0=[1,mean,sigma_l])
		fwhm_r = sigma_r * 2.35
		#SAVING FIT PARAMETER
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'left_edge_fit_param.txt', [amp_l,mean_l,sigma_l] )
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'left_edge_fit_pcov.txt', pcov_l )
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'right_edge_fit_param.txt', [amp_r,mean_r,sigma_r] )
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'right_edge_fit_pcov.txt', pcov_r )

		#PLOTTING
		if self.plot == 1:
			xplot_l = np.arange(0,self.delay[len(self.delay)/2], 0.001)
			xplot_r = np.arange(self.delay[len(self.delay)/2], self.delay[len(self.delay)-1], 0.001)
			plt.figure(10)
			plt.plot(self.delay,self.intensity_der,'b+:',label='data')
			plt.plot(xplot_r,gaus(xplot_r,amp_r,mean_r,sigma_r),color = 'red',label='gaussian fit right')
			plt.plot(xplot_l,-1*gaus(xplot_l,amp_l,mean_l,sigma_l),color = 'c',label='gaussian fit left')
			plt.legend(loc = 2)
			plt.title('Edges Fit')
			plt.xlabel('Delay (ns)')
			plt.ylabel('Normalized Intensity (a.u.)')
			plt.savefig('intermediate/'+self.sample+'_edge_fitting.pdf', bbox_inches="tight") 
			plt.show()
		return mean_l, fwhm_l, mean_r, fwhm_r

	def fit_single_bunch(self, delay, intensity,left_edge, right_edge, fwhm, Nsm, plot):
		'''this method provides an easy way to fit the edges. Of course if used to fit the delay scan, this has to be 
			derived.
		
		delay = array, the delay relative to intensity_der
		intensity_der =array, the intensity scan derived, so that the there are peaks instead of edges
		left_edge = int, the position of the left edge
		right_edge = int, the position of the right edge

		the method returns the position and the fwhm of single bunch.
		Additionally it saves the parameters and the pcov matrix resulting from the fit in the intermediate folder,
		as well as a graph with the fit if plot = 1'''
		self.delay = delay
		self.intensity = intensity
		self.left_edge = left_edge
		self.right_edge = right_edge
		self.fwhm = fwhm
		self.Nsm = Nsm
		self.plot = plot 
		
		#defining the region to fit
		mean = np.int((self.left_edge + self.right_edge)/2/self.Nsm)               
		h_w = fwhm/Nsm
		l = np.int(mean - h_w)
		r = np.int(mean + h_w)
		x = self.delay[l : r]
		y = self.intensity[l : r] 
		#Guessing the parameter for the fit
		n = len(x)                #the number of data
		mean = (self.left_edge + self.right_edge)/2 #               #note this correction
		sigma = self.fwhm       #note this correction
		amp = np.amax(y)
		#fitting
		print mean, sigma, amp
		plt.figure(200)
		plt.plot(x,y)
		[amp_sb,mean_sb,sigma_sb],pcov_sb = curve_fit(gaus,x,y,p0=[amp,mean,sigma])
		print mean_sb, sigma_sb, amp_sb
		fwhm_sb = sigma_sb * 2.35
		#SAVING FIT PARAMETER
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'single_bunch_fit_param.txt', [amp_sb,mean_sb,sigma_sb] )
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'single_bunch_fit_pcov.txt', pcov_sb )
		if plot ==1:
			plt.figure(11)
			plt.plot(self.delay, self.intensity,'b+:',label='data')
			plt.plot(x,gaus(x,amp_sb,mean_sb,sigma_sb),color = 'red',label='single bunch position')
			plt.legend(loc=9)
			plt.title('Delay scan')
			plt.xlabel('Delay (ns)')
			plt.ylabel('Normalized Intensity (a.u.)')
			plt.yscale('log')
			plt.savefig('intermediate/'+self.sample+'__single_bunch_fitting.pdf', bbox_inches="tight") 
			plt.show()
		print 'fwhm', fwhm_sb
		return mean_sb, fwhm_sb
		
def test_time_resolved_analysis():
	test = time_resolved_analysis('test', 'ipp', 2426, 1001, 10)
	file_names = test.create_name_array()
	
	xpix_in = 855
	xpix_fin = 859
	ypix_in = 211
	ypix_fin = 219
	rejection = 140
	#~ intensity_down = test.tiff_extract_n_scans(file_names, xpix_in, xpix_fin, ypix_in, ypix_fin, 'down')
	intensity_down = np.loadtxt('intermediate/test_down_intensity.txt')
	intensity_down = test.select_good_scans(intensity_down, xpix_in, xpix_fin, ypix_in, ypix_fin, rejection, 'down', 1)

	xpix_in = 860
	xpix_fin = 864
	ypix_in = 156
	ypix_fin = 163
	rejection = 150
	#~ intensity_up = test.tiff_extract_n_scans(file_names, xpix_in, xpix_fin, ypix_in, ypix_fin, 'up')
	intensity_up = np.loadtxt('intermediate/test_up_intensity.txt')
	intensity_up = test.select_good_scans(intensity_up, xpix_in, xpix_fin, ypix_in, ypix_fin, rejection, 'up', 1)

	intensity = (intensity_down + intensity_up)/2
	plt.plot(intensity, 'r')
	plt.plot(intensity_down, 'b')
	plt.plot(intensity_up, 'g')

	
	plt.show()
	#~ intensity = intensity_up 

	Nsm = 9
	delay_smooth, intensity_smooth = test.smooth(intensity, Nsm)
	intensity_der = test.derive(intensity_smooth)
	delay_smooth, intensity_smooth, intensity_der = test.norm(delay_smooth, intensity_smooth, intensity_der,Nsm)	
	mean_l, fwhm_l, mean_r, fwhm_r = test.fit_edge(delay_smooth, intensity_der, 1)
	fwhm_edges = (fwhm_l + fwhm_r)/2
	mean_sb, fwhm_sb = test.fit_single_bunch(delay_smooth, intensity_smooth, mean_l, mean_r, fwhm_edges, Nsm, 1)
	print 'FWHM right edge:', fwhm_r, 'ns'
	print 'FWHM left edge:', fwhm_l, 'ns'
	print 'FWHM single bunch:', fwhm_sb, 'ns'
	print 'FWHM average:', (fwhm_r+fwhm_l+fwhm_sb)/3, 'ns'

	#plt.show()



if __name__ == "__main__":
	test_time_resolved_analysis()