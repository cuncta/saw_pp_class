import numpy as np
from scipy.special import erf





	
def gaus(x,a,x0,sigma):
	
	return a*np.exp(-(x-x0)**2/(2*sigma**2))
	
def sb(x,x0_l, x0_r, x0_sb,sigma, amp):
	sb = (1-erf((x-x0_l)/(np.sqrt(2)*sigma)))/2 \
		+ (1+erf((x-x0_r)/(np.sqrt(2)*sigma)))/2 \
		+amp*np.exp(-(x-x0_sb)**2/(2*sigma**2))
	return sb
		