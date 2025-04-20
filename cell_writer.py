import tensorflow as tf
std=0

f = open("./digital_igzo.cir",'w')
f.write('*editor khb\n')
for waynum in range(0,5):   #(0,1) makes one cell (cell0)
	for cellnum in range(0,2048):
		waynum=str(waynum)
		cellnum=str(cellnum)
		###m1###
		gaus_dis=tf.random.normal([1],mean=1.0,stddev=std)
		mod_width=((1e-6)*gaus_dis).numpy()		
		mod_width=str(mod_width[0])
		#m1='M1_'+'way'+waynum+'_'+'cell'+cellnum+' '+'A_'+'way'+waynum+'_'+'cell'+cellnum+' '+'NSL_'+'way'+waynum+'_'+'cell'+cellnum+' '+'GND'+' '+'GND'+' '+'nigzo'+' L=1u W=1u'
		m1='M1_'+'way'+waynum+'_'+'cell'+cellnum+' '+'A_'+'way'+waynum+'_'+'cell'+cellnum+' '+'NSL_'+'way'+waynum+'_'+'cell'+cellnum+' '+'GND'+' '+'GND'+' '+'nigzo'+' L=1u W='+mod_width
		f.write(m1)
		f.write('\n')
		###m2###
		gaus_dis=tf.random.normal([1],mean=1.0,stddev=std)
		mod_width=((1e-6)*gaus_dis).numpy()		
		mod_width=str(mod_width[0])
		#m2='M2_'+'way'+waynum+'_'+'cell'+cellnum+' '+'ML'+waynum+' '+'SN_'+'way'+waynum+'_'+'cell'+cellnum+' '+'A_'+'way'+waynum+'_'+'cell'+cellnum+' '+' '+'A_'+'way'+waynum+'_'+'cell'+cellnum+' '+'nigzo'+' '+'L=1u W=1u'
		m2='M2_'+'way'+waynum+'_'+'cell'+cellnum+' '+'ML'+waynum+' '+'SN_'+'way'+waynum+'_'+'cell'+cellnum+' '+'A_'+'way'+waynum+'_'+'cell'+cellnum+' '+' '+'A_'+'way'+waynum+'_'+'cell'+cellnum+' '+'nigzo'+' '+'L=1u W='+mod_width
		f.write(m2)
		f.write('\n')
		###m3###
		gaus_dis=tf.random.normal([1],mean=1.0,stddev=std)
		mod_width=((1e-6)*gaus_dis).numpy()		
		mod_width=str(mod_width[0])
		#m3='M3_'+'way'+waynum+'_'+'cell'+cellnum+' '+'SN_'+'way'+waynum+'_'+'cell'+cellnum+' '+'WL '+'WL '+'WL '+'nigzo'+' L=1u W=1u'
		m3='M3_'+'way'+waynum+'_'+'cell'+cellnum+' '+'SN_'+'way'+waynum+'_'+'cell'+cellnum+' '+'WL '+'WL '+'WL '+'nigzo'+' L=1u W='+mod_width
		f.write(m3)
		f.write('\n')
		###m4###
		gaus_dis=tf.random.normal([1],mean=1.0,stddev=std)
		mod_width=((1e-6)*gaus_dis).numpy()		
		mod_width=str(mod_width[0])
		#m4='M4_'+'way'+waynum+'_'+'cell'+cellnum+' '+'SNB_'+'way'+waynum+'_'+'cell'+cellnum+' '+'WL '+'WL '+'WL '+'nigzo'+' L=1u W=1u'
		m4='M4_'+'way'+waynum+'_'+'cell'+cellnum+' '+'SNB_'+'way'+waynum+'_'+'cell'+cellnum+' '+'WL '+'WL '+'WL '+'nigzo'+' L=1u W='+mod_width
		f.write(m4)
		f.write('\n')
		###m5###
		gaus_dis=tf.random.normal([1],mean=1.0,stddev=std)
		mod_width=((1e-6)*gaus_dis).numpy()		
		mod_width=str(mod_width[0])
		#m5='M5_'+'way'+waynum+'_'+'cell'+cellnum+' '+'ML'+waynum+' '+'SNB_'+'way'+waynum+'_'+'cell'+cellnum+' '+'B_'+'way'+waynum+'_'+'cell'+cellnum+' '+' '+'B_'+'way'+waynum+'_'+'cell'+cellnum+' '+' '+'nigzo'+' L=1u W=1u'
		m5='M5_'+'way'+waynum+'_'+'cell'+cellnum+' '+'ML'+waynum+' '+'SNB_'+'way'+waynum+'_'+'cell'+cellnum+' '+'B_'+'way'+waynum+'_'+'cell'+cellnum+' '+' '+'B_'+'way'+waynum+'_'+'cell'+cellnum+' '+' '+'nigzo'+' L=1u W='+mod_width
		f.write(m5)
		f.write('\n')
		###m6###
		gaus_dis=tf.random.normal([1],mean=1.0,stddev=std)
		mod_width=((1e-6)*gaus_dis).numpy()		
		mod_width=str(mod_width[0])
		#m6='M6_'+'way'+waynum+'_'+'cell'+cellnum+' '+'B_'+'way'+waynum+'_'+'cell'+cellnum+' '+'NSLB_'+'way'+waynum+'_'+'cell'+cellnum+' '+'GND'+' '+'GND'+' '+'nigzo'+' L=1u W=1u'
		m6='M6_'+'way'+waynum+'_'+'cell'+cellnum+' '+'B_'+'way'+waynum+'_'+'cell'+cellnum+' '+'NSLB_'+'way'+waynum+'_'+'cell'+cellnum+' '+'GND'+' '+'GND'+' '+'nigzo'+' L=1u W='+mod_width
		f.write(m6)
		f.write('\n')
###WL###
wl='Vwl1 '+'WL '+'GND '+'0'
f.write(wl)
f.write('\n')
