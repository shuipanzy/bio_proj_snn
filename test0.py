import scipy.io as sio
import numpy as np
import random as random
from neuron import h,gui

name='./MNIST/photos_0.mat'
vals=sio.loadmat(name)
images=vals['images']

name1='./MNIST/photos_1.mat'
vals=sio.loadmat(name1)
images1=vals['images']

answer='./MNIST/photos_test_label.mat'
v=sio.loadmat(answer)
labels=v['labels']

h('''load_file("stdp.hoc")
objref snn
create foo
access foo
snn=new stdpnn()''')

h('numInputs=1')
h.numInputs=len(images[0])
h('double img[numInputs]')

#f=open("out.txt", "w")
#f1=open("weight_before.txt", "w")
#f2=open("weight_after.txt", "w")

#testing set/////////////////////////////
'''
count=[]
for i in range(10):
    count.append(0)

nimages=[]
index=[]     
for i in range(600):#TRAIN
    lab=labels[0][i]
    if i>300:
      if count[lab]<50:
        nimages.append(images[i])
        index.append(lab)
        for j in range(10):
            if lab==j:
                count[j]+=1
    else:
        nimages.append(images[i])
        index.append(lab)
        for j in range(10):
            if lab==j:
                count[j]+=1

print "figure number", count
print len(nimages)

for i in range(1000): #TEST
    lab=labels[0][i+2000]
    nimages.append(images[i+2000])
    index.append(lab)
print "total number:", len(index)
'''

nimages=[]
index=[]
for j in range(200):
    for i in range(10): 
        nimages.append(images[0])
        index.append(labels[0][0])

#print index
#for i in range(100):
#    nimages.append(images1[i+10])
#    index.append(labels[0][i+10])

labnum=[]
rate=[]
for x in range(10):
    labnum.append(0)
    rate.append(0)

h("objref outputCounts[10]") #change
h('nWeights=snn.num_in')

h('double update[nWeights]')
h('k=0')

Numout=10 #change 
outputs=[0]*Numout
freq=[0]*Numout
cur=0

#load weight
#weights = sio.loadmat('./trained_weights')
#weights = weights['allWeights']

neuronResult=[[0 for row in range(0,10)] for col in range(0,Numout)]

#add inhibitory neuron
h('snn.inhibi')

#//////////////////////////////////////////////////////////////////////////
for item in nimages:
    
    #read figure in
    for i in range(len(item)):
            h.img[i]=item[i]
    h('snn.input(&img)')
    
    #set threshold for spike
    threshold=0
    h('z=0')
    for i in range(Numout):
            h.z=i
            h('snn.post[z].soma outputCounts[z]=new APCount(0.5)')
            h.outputCounts[i].thresh=threshold
    
    #print weight pre run
    '''
    matrix=[]
    for i in range(196):
	for j in range(Numout):		
		#print >>f1, (i, j, h.snn.wa[i*Numout+j].wsyn)
                print >>f1, (h.snn.wa[i*Numout+j].wsyn)
                lala=h.snn.wa[i*Numout+j].wsyn
                matrix.append(lala)
    '''
    '''
    #update weight////////////////////////////////////////////////////////
    for i in range(Numout):
        h.k=i
        for j in range(int(196)):
            h.update[j]= weights[i][j]
            h('snn.post[k].setWeights(&update)')
        print h.update[j]
    '''
    #running............////////////
    h.tstop=200
    h.run()
    
    #print weigt post run
    '''
    for i in range(196):
	for j in range(10):		
		print >>f2, (i, j, h.snn.wa[i*10+j].wsyn)
    '''

    # Save the results///////////////////////////////////////////////////
    '''
    allWeights = [[0 for row in range(0,196)] for col in range(0,Numout)]
    for i in range(Numout):
        allWeights[i] = list(h.snn.post[i].getWeights())

    allWeights = {'allWeights': allWeights}
    sio.savemat('trained_weights', allWeights)
    '''

    #getting results for one iter//////////////////////////////////////////
    for i in range(len(outputs)):
            outputs[i]=h.outputCounts[i].n
    cur+=1
    
    #print >>f, "iter", cur
    #print >>f, outputs
    print "iter", cur, "label",index[cur-1]
    print outputs

    best_freq=max(outputs)
    winners=list()
    for i in range(len(outputs)) :  
        if (outputs[i]==best_freq):
        #if (outputs[i]>4):   
            outputs[i]=1
            winners.append(i)
            freq[i]+=1
        else:
            outputs[i]=0

    best=max(freq)
    result=list()
    for i in range(len(outputs)) :
        if (freq[i]==best):
           result.append(i) 
        

    #print >>f, "WINNERS: ", winners
    #print >>f, freq
    #print >>f, "best neuron:", result   
    
    print  "WINNERS: ", winners
    #print  freq
    #print  "best neuron:", result
    
    #counting/////////////////////////////////////////////////////////////
    '''   
    ind=index[cur-1] #or lables[0][cur-1]
    labnum[ind]+=1
    print "label", ind
    
    imgtotal=[]
    for i in range(10):
        imgtotal.append(0)
    
    for i in range(len(outputs)):
        neuronResult[i][ind]+=outputs[i]

    if cur%10 ==0:
        for i in range(len(outputs)):
            print i,neuronResult[i]
            for j in range(10):
                imgtotal[j]+=neuronResult[i][j]
    
        for x in range(10):
          if labnum[x]!=0:  
            rate[x]=imgtotal[x]/labnum[x]

        print imgtotal
        print labnum
        print rate
     '''   
    #update weight////////////////////////////////////////////////////////
    '''
    for i in range(Numout):
        h.k=i
        for j in range(int(196)):
            w = np.random.normal(0.01, 0.02)
            while w <= 0 or w>0.01 :
               w = np.random.normal(0.01, 0.02) 
            
            h.update[j]=w
            #h.update[j]= matrix[i*10+j]
            h('snn.post[k].setWeights(&update)')
            #print h.update[j]
    '''
    #print figure/////////////////////////////////////////////////////////
'''
v_vec=[]
for i in range(10):
	v_vec0=h.Vector()
	v_vec.append(v_vec0)
	v_vec[i].record(h.snn.post[i].soma(0.5)._ref_v)
        
a_vec=h.Vector()
a_vec.record(h.snn.pre[36].soma(0.5)._ref_v)

b_vec=h.Vector()
b_vec.record(h.snn.pre[37].soma(0.5)._ref_v)


c_vec=h.Vector()
c_vec.record(h.snn.pre[49].soma(0.5)._ref_v)

d_vec=h.Vector()
d_vec.record(h.snn.pre[50].soma(0.5)._ref_v)


w_vec=h.Vector()
t_vec=h.Vector()
w_vec.record(h.snn.wa[360]._ref_wsyn)
t_vec.record(h._ref_t)


pyplot.figure(figsize=(8,4))
pyplot.plot(
        t_vec, v_vec[0],
	t_vec, v_vec[1],
	t_vec, v_vec[2])
pyplot.xlabel('time(ms)')
pyplot.ylabel('mV')
pyplot.show()

pyplot.figure(figsize=(8,4))
pyplot.plot(t_vec,w_vec)
pyplot.xlabel('time(ms)')
pyplot.ylabel('weight')
pyplot.show()

pyplot.figure(figsize=(8,4))
pyplot.subplot(221)
pyplot.plot(t_vec,a_vec)
pyplot.subplot(222)
pyplot.plot(t_vec,b_vec)
pyplot.subplot(223)
pyplot.plot(t_vec,c_vec)
pyplot.subplot(224)
'''

'''
cur=0
for i in range(196):
	for j in range(10):		
		print >>f, (i, j, h.snn.wa[i*10+j].wsyn)
print >>f, "DONE"
h.run() 
for i in range(196):
	for j in range(10):		
		print >>f_t, (i, j, h.snn.wa[i*10+j].wsyn)
'''

