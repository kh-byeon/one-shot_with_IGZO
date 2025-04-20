f = open("./infer_sim_mem_dist_cell.cir",'w')
f.write('*editor khb\n')
for waynum in range(0,5):   #(0,1) makes one cell (cell0)
	for cellnum in range(0,100):
		waynum=str(waynum)
		cellnum=str(cellnum)
		#nmos#
		mn='mn'+'way'+waynum+'_cell'+cellnum+' '+'ML'+waynum+' '+'gate_'+'way'+waynum+'_cell'+cellnum+' '+'GND '+'GND '+'n1 '+'W=1e-6 '+'L=1e-6'
		f.write(mn)
		f.write('\n')
		#pmos#
		mp='mp'+'way'+waynum+'_cell'+cellnum+' '+'ML'+waynum+' '+'gate_'+'way'+waynum+'_cell'+cellnum+' '+'GND '+'ML'+waynum+' p1 '+'W=2.4e-6 '+'L=1e-6'
		f.write(mp)
		f.write('\n')
		#
