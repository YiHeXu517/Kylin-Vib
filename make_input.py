import numpy as np

N,d = 10,16
lattice,pdims = [],[]
lattice_ref = dict()
for i in range(N):
    lattice.append("e{}L".format(i+1))
    pdims.append(2)
    lattice.append("v{}L".format(i+1))
    pdims.append(d)
lattice.insert(N//2*2,"caL")
pdims.insert(N//2*2,d)
for i in range(len(lattice)):
    lattice_ref[lattice[i]] = i

f1 = open("naive_polariton.inp",'w')
dims = pdims.__str__()[1:-1]
dims = dims.replace(","," ").replace("\'"," ")
EE,Freq,CE,lam,gCE,tE = 2.0,0.15,2.0,-0.2121,0.07,0.0
print(dims,file=f1)
for i in range(N):
    print("{} {} {} 0.0".format(lattice_ref["e{}L".format(i+1)],'N',EE),file=f1)
    print("{} {} {} 0.0".format(lattice_ref["v{}L".format(i+1)],'N',Freq),file=f1)
    print("{} {} {} {} {} 0.0".format(lattice_ref["e{}L".format(i+1)],lattice_ref["v{}L".format(i+1)],'N','Q',lam),file=f1)
    print("{} {} {} {} {} 0.0".format(lattice_ref["e{}L".format(i+1)],lattice_ref["caL"],'+','-',gCE),file=f1)
    print("{} {} {} {} {} 0.0".format(lattice_ref["e{}L".format(i+1)],lattice_ref["caL"],'-','+',gCE),file=f1)
    
f1.close()