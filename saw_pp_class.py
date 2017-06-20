import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os 
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
	
	def __init__(self, sample, pic_name, first_im, scan_length, n_scans, pixel = 3):
		
		self.sample = sample
		self.pic_name = pic_name
		self.first_im = first_im
		self.scan_length = scan_length 
		self.n_scans = n_scans
		#variables that will be defined later
		self.pixel = None
		self.xpix_in = None
		self.xpix_fin= None
		self.ypix_in = None 
		self.ypix_fin = None 
		self.up_down = None
		self.rejection = None

		if not os.path.exists('intermediate'):
			os.mkdir('intermediate')
		return	
	
	def set_region(self, x1, x2, y1, y2, label):
		self.xpix_in = x1
		self.xpix_fin = x2
		self.ypix_in = y1
		self.ypix_fin = y2
		self.up_down = label
	def set_rejection(self, rejection):
		self.rejection = rejection
	def set_Nsm(self, Nsm):
		self.Nsm = Nsm
	
	def create_name_array(self):
		'''This method creates an array with the names of all the tiff files used for n scans. 
		At the moment this method is not used by other classes, this will be implemented in future'''
		file_names = []
		for m in range(0,self.n_scans):
			for i in range(0, self.scan_length):
				num = str((m*self.scan_length)+(self.first_im+i))
				file_names.append(self.pic_name+num+'.tif')
		return file_names
		
		
	def tiff_extract_n_scans(self):
		'''this method read n_scans composed  of scan_length tif files and extract the values 
		of certain pixels and saves them in matrix
	
		file_names list of strings, the name of the images to be analyzed, this is still not used at the moment
		
		xpix_in = initial pixel for loop on x coordinate
		xpix_fin = final pixel for loop on x coordinate
	
		ypix_in = initial pixel for loop on y coordinate
		ypix_fin = final pixel for loop on y coordinate
	
		up_down = string, will be used to save the data and differ between plus/minus first order'''
		

		
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

		
	def select_good_scans(self, intensity, plot ):
		'''this method provides an easy way to differ between pixel scans. It differs between the pixels hit by the 
		plus minus first order and the others. It return one single scan, which is the average of the selected scans.
		TO DO: draw a picture to explain this. 
	
		file_names list of strings, the name of the images to be analyzed, this is still not used at the moment
		
		xpix_in = initial pixel for loop on x coordinate
		xpix_fin = final pixel for loop on x coordinate
	
		ypix_in = initial pixel for loop on y coordinate
		ypix_fin = final pixel for loop on y coordinate
	
		up_down = string, will be used to save the data and differ between plus/minus first order'''
		self.intensity = intensity

		if self.rejection == None:
			sys.exit("Please define the rejection with set_rejection()")
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
		print '################################'
		print 'selecting scans '+ self.up_down
		print 'rejected', rej
		print 'passed', pas
		print '################################'
		np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_selected_intensity.txt', (intensity_sum))
		#np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_xpas.txt', x_pas)
		#np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_xrej.txt', x_rej)   
		#np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_ypas.txt', y_pas)
		#np.savetxt('intermediate/'+self.sample+'_'+self.up_down+'_yrej.txt', y_rej) 
		if plot == 1:
			plt.figure(1)
			plt.title('Average of good scans '+self.up_down)
			plt.plot(intensity_sum[0: self.scan_length])
			plt.show()
		return  intensity_sum	
	
	def smooth(self, intensity):
		'''this method provides an easy way to smooth the data. the data points of a signal are modified so 
		individual points (presumably because of noise) are reduced, and points that are lower than the adjacent 
		points are increased leading to a smoother signal. 
	
		intensity =array, scan to smooth
		Nsm = int, number of data points to average to smooth intensity
		
		the method return the intensity array, and an arbitrary x array smoothed by Nsm'''
		self.intensity = intensity

		delay_smooth = np.zeros(len(self.intensity)/self.Nsm+1)
		intensity_smooth = np.zeros(len(self.intensity)/self.Nsm+1)
		
		for x in range(0,len(self.intensity)):
			delay_smooth[int(x/self.Nsm)] = int(x)
			intensity_smooth[int(x/self.Nsm)] += self.intensity[x]
			
		np.savetxt('intermediate/'+self.sample+'_intensity_smooth.txt', intensity_smooth) 
		np.savetxt('intermediate/'+self.sample+'_delay_smooth.txt', delay_smooth) 
		return delay_smooth[0:len(delay_smooth)-1], intensity_smooth[0:len(delay_smooth)-1]
		
	def norm(self, delay, intensity):
		'''Original dataset shows an increasing amplitude during the scan that is not related
		width the effect we want to observe. we then normalize it:
		averaging five points to the left and 5 to the right and drawing a line, and then normalizing to one
		
		delay = array, the delay relative to intensity'''
		
		self.delay = delay
		self.intensity = intensity
		
		#calculating the line
		left_av = np.average(self.intensity[0:self.Nsm/3])
		right_av = np.average(self.intensity[-self.Nsm/3:-1])
		norm_line = left_av + (right_av-left_av)/(len(self.delay)*self.Nsm)*self.delay
		
		#correcting for the pendenza della retta
		self.intensity = self.intensity/norm_line
		
		#shifting to zero 
		self.intensity = self.intensity-np.amin(self.intensity)
		#print 'ss max', np.amin(ss)
		
		#normalizing to 1
		self.intensity = self.intensity/np.amax(self.intensity)
		np.savetxt('intermediate/'+self.sample+'_intensity_norm.txt', self.intensity) 

		return self.delay, self.intensity

	def derive(self, intensity):
		'''this method provides an easy way to derive a data set 
		intensity =array, the data to derive
		the method returns the derived intensity array, and saves it in the intermediate folder'''
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
		np.savetxt('intermediate/'+self.sample+'_intensity_derived.txt', intensity_der) 
		return intensity_der[1:len(intensity_der)+2]
	
	def estimate_edge(self, delay, intensity_der, plot):
		'''this method provides an easy way to fit the edges. Of course if used to fit the delay scan, this has to be 
			derived.
		
		delay = array, the delay relative to intensity_der
		intensity_der =array, the intensity scan derived, so that the there are peaks instead of edges
		plot = integer, if plot == 1 the results of the fit are plot and saved.
		
		the method returns the position and the fwhm of the left and right edge.
		Additionally it saves the parameters and the pcov matrix resulting from the fit in the intermediate folder,
		as well as a graph with the fit if plot = 1'''
		self.delay = delay[1:len(delay)] #necessary because derived array has one less value
		self.intensity_der = intensity_der
		self.plot = plot 
		
		#LEFT PEAK
		x_l = self.delay[0:len(self.delay)/2]
		y = self.intensity_der[0:len(self.intensity_der)/2] * -1
		n = len(x_l)   		#the number of data
		amp = np.amax(y)
		mean = x_l[np.argmax(y)]				#guessing mean value
		sigma = np.sqrt(sum(y*(x_l-mean)**2)/n )#guessing the fwhm
		#~ plt.figure(302)
		#~ plt.plot(x_l,y)
		#~ print mean, sigma*2.3548, 
		#~ plt.show()
		
		#fitting
		[amp_l,mean_l,sigma_l],pcov_l = curve_fit(gaus,x_l,y,p0=[amp,mean,sigma])
		fwhm_l = sigma_l * 2.3548

		#RIGHT PEAK
		x_r = self.delay[len(self.delay)/2:len(self.delay)]
		y = self.intensity_der[len(self.intensity_der)/2:len(self.intensity_der)] 
		n = len(x_r)                          #the number of data
		amp = np.amax(y)
		mean = x_r[np.argmax(y)]                   #note this correction
		#No need to guess sigma again, taking the value from the previous fit
		#fitting
		[amp_r,mean_r,sigma_r],pcov_r = curve_fit(gaus,x_r,y,p0=[amp,mean,sigma_l])
		fwhm_r = sigma_r * 2.3548
		#SAVING FIT PARAMETER
		np.savetxt('intermediate/'+self.sample+'_left_edge_fit_param.txt', [amp_l,mean_l,sigma_l] )
		np.savetxt('intermediate/'+self.sample+'_left_edge_fit_pcov.txt', pcov_l )
		np.savetxt('intermediate/'+self.sample+'_right_edge_fit_param.txt', [amp_r,mean_r,sigma_r] )
		np.savetxt('intermediate/'+self.sample+'_right_edge_fit_pcov.txt', pcov_r )

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

	def estimate_single_bunch(self, delay, intensity,left_edge, right_edge, fwhm, shift, plot):
		'''this method provides an easy way to fit the edges. Of course if used to fit the delay scan, this has to be 
			derived.
		
		delay = array, the delay relative to intensity_der
		intensity_der =array, the intensity scan derived, so that the there are peaks instead of edges
		left_edge = int, the position of the left edge
		right_edge = int, the position of the right edge
		fwhm = int, the expected fwhm, use the average of the edges if available
		Nsm = int, same as for smoothing
		shift = float, to plot logaritmic, shift the data to avoid zeros (0.01?)

		the method returns the position and the fwhm of single bunch.
		Additionally it saves the parameters and the pcov matrix resulting from the fit in the intermediate folder,
		as well as a graph with the fit if plot = 1
		
		TO DO: I do not understand why the single bunch fwhm is 141 ns, it should be 125 ns, according to the script
			with functions.'''
		self.delay = delay
		self.intensity = intensity
		self.left_edge = left_edge
		self.right_edge = right_edge
		self.fwhm = fwhm
		self.shift = shift
		self.plot = plot 
		
		#defining the region to fit
		mean = np.int((self.left_edge + self.right_edge)/2/self.Nsm)               
		h_w = fwhm/self.Nsm
		l = np.int(mean - h_w)
		r = np.int(mean + h_w)
		#~ l = 40
		#~ r = 62
		x = self.delay[l : r]
		y = self.intensity[l : r]
		#~ print 'l, r', l, r
		#~ plt.figure(122)
		#~ plt.title('fitting region for single bunch')
		#~ plt.plot(x, y)
		#Guessing the parameter for the fit
		n = len(x)                #the number of data
		mean = (self.left_edge + self.right_edge)/2 #               #note this correction
		sigma = np.sqrt(sum(y*(x-mean)**2)/n )#self.fwhm       #note this correction
		amp = np.amax(y)
		#fitting
		[amp_sb,mean_sb,sigma_sb],pcov_sb = curve_fit(gaus,x,y,p0=[amp,mean,sigma])
		fwhm_sb = sigma_sb * 2.3548
		#SAVING FIT PARAMETER
		np.savetxt('intermediate/'+self.sample+'_single_bunch_fit_param.txt', [amp_sb,mean_sb,sigma_sb] )
		np.savetxt('intermediate/'+self.sample+'_single_bunch_fit_pcov.txt', pcov_sb )
		if plot ==1:
			plt.figure(11)
			plt.plot(self.delay, self.shift+self.intensity,'b+:',label='data')
			plt.plot(x,self.shift+gaus(x,amp_sb,mean_sb,sigma_sb),color = 'red',label='single bunch position')
			plt.legend(loc=9)
			plt.title('Delay scan')
			plt.xlabel('Delay (ns)')
			plt.ylabel('Normalized Intensity (a.u.)')
			#~ plt.yscale('log')
			plt.ylim(0.01,1+self.shift)
			plt.savefig('intermediate/'+self.sample+'_single_bunch_fitting.pdf', bbox_inches="tight") 
			plt.show()
		print 'fwhm', fwhm_sb
		return mean_sb, fwhm_sb, amp_sb
	
	
	def set_fit_delay_scan_parameter(self, mean_l, mean_r, mean_sb, amp_sb, fwhm):
		'''Use this method to set the initial parameters for the fit delay scan. One can use the estimate_edges and
		estimate_single bunch to guess them.
		
		mean_l  = int, position of the left edge
		mean_r  = int, position of the right edge
		mean_s  = int, position of the single bunch
		amp_sb = int, amplitude of the single bunch
		fwhm = int, fwhm of the sb or of the derivative of the edge'''
		
		self.mean_l = mean_l
		self.mean_r = mean_r
		self.mean_sb = mean_sb
		self.amp_sb = amp_sb
		self.fwhm = fwhm
		
	
	def fit_delay_scan(self, delay, intensity, plot):
		'''This method provides an easy way to fit the complete delay scan. set the initial parameters for the fit
		using the set_fit_delay_scan_parameter method. The fit is carried out using the sb function defined in lib.py
		
		it return the fwhm, and saves the parameters and the pcov matrix in text files in the intermediate folder.
		it saves a pdf with the plot of the data and the fit'''
		self.delay = delay
		self.intensity = intensity
		
		sigma = self.fwhm / 2.3548
		[self.mean_l, self.mean_r, self.mean_sb,sigma, self.amp_sb],pcov_del = curve_fit(sb,self.delay,self.intensity,p0=[self.mean_l, self.mean_r, self.mean_sb,sigma, self.amp_sb])
		np.savetxt('intermediate/'+self.sample+'_delay_scan_fit_param.txt', [self.mean_l, self.mean_r, self.mean_sb,sigma, self.amp_sb] )
		np.savetxt('intermediate/'+self.sample+'_delay_scan_fit_pcov.txt', pcov_del )
		if plot ==1:
			gauss_sum_fit = sb(self. delay,self.mean_l, self.mean_r, self.mean_sb,sigma, self.amp_sb)
			plt.figure(12)
			plt.plot(self.delay,gauss_sum_fit, 'r', label = 'fit')
			plt.plot(self.delay, self.shift+self.intensity,'b+:',label='data')
			plt.legend(loc=9)
			plt.title('Delay scan')
			plt.xlabel('Delay (ns)')
			plt.ylabel('Normalized Intensity (a.u.)')
			plt.ylim(0.01,1.3+self.shift)
			plt.savefig('intermediate/'+self.sample+'_delay_scan_fit.pdf', bbox_inches="tight") 
			plt.show()
		fwhm = 2.3548 * sigma
		return fwhm
		
		
		

		
def test_time_resolved_analysis():
	test = time_resolved_analysis('test', 'ipp', 2426, 1001, 10)
	file_names = test.create_name_array()
	
	xpix_in = 855
	xpix_fin = 859
	ypix_in = 211
	ypix_fin = 219
	test.set_region(xpix_in, xpix_fin,ypix_in,ypix_fin,'down')
	#~ intensity_down = test.tiff_extract_n_scans()
	intensity_down = np.loadtxt('intermediate/test_down_intensity.txt')
	test.set_rejection(140)
	intensity_down = test.select_good_scans(intensity_down, 1)

	xpix_in = 860
	xpix_fin = 864
	ypix_in = 156
	ypix_fin = 163
	test.set_region(xpix_in, xpix_fin,ypix_in,ypix_fin,'up')
	#~ intensity_up = test.tiff_extract_n_scans()
	intensity_up = np.loadtxt('intermediate/test_up_intensity.txt')
	test.set_rejection(150)
	intensity_up = test.select_good_scans(intensity_up, 1)

	intensity = (intensity_down + intensity_up)/2

	test.set_Nsm(9)
	delay_smooth, intensity_smooth = test.smooth(intensity)
	delay_smooth, intensity_smooth = test.norm(delay_smooth, intensity_smooth)	
	#~ plt.plot(intensity_smooth)
	intensity_der = test.derive(intensity_smooth)
	mean_l, fwhm_l, mean_r, fwhm_r = test.estimate_edge(delay_smooth, intensity_der, 1)
	fwhm_edges = (fwhm_l + fwhm_r)/2
	mean_sb, fwhm_sb, amp_sb = test.estimate_single_bunch(delay_smooth, intensity_smooth, mean_l, mean_r, fwhm_edges,0.0, 1)
	print 'FWHM right edge, mean:', fwhm_r, 'ns', mean_r, 'ns'
	print 'FWHM left edge, mean:', fwhm_l, 'ns', mean_l, 'ns'
	print 'FWHM single bunch,mean, amp:', fwhm_sb, 'ns', mean_sb, 'ns', amp_sb, 'ns'
	print 'FWHM average:', (fwhm_r+fwhm_l+fwhm_sb)/3, 'ns'
	test.set_fit_delay_scan_parameter(mean_l, mean_r, mean_sb, amp_sb, fwhm_sb)
	fwhm = test.fit_delay_scan(delay_smooth,intensity_smooth, 1)
	print 'FWHM from delay scan:', fwhm, 'ns'

	#~ plt.show()



if __name__ == "__main__":
	test_time_resolved_analysis()