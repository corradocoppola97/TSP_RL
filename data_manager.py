import math
import random
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class data_manager():
    def __init__(self, stacklength = 10000, seed = None):
        self._memoryx = []
        self._memoryy = []
        self._stacklength = stacklength
        self._length = 0
        self.seed = seed
        if seed != None:
            random.seed(a= seed)


    def memoryrecoverCSV(self, path):
        # read csv and add data to memory
        return True


    def add(self, instancex, instancey):
        self._length += 1
        #print(type(self._memoryx))
        #print(self._memoryx)
        #print(type(self._memoryy))
        #print(self._memoryy)
        if self._length <= self._stacklength:
            #print('X',type(instancex))
            #print('Y',type(float(instancey)))
            #print(len(self._memoryy))
            #print(len(self._memoryx))
            self._memoryx.append(instancex)
            self._memoryy.append(instancey)
        else:
            #print('MEMORIA PIENA')
            ind = self._length % self._stacklength
            self._memoryx[ind] = instancex
            self._memoryy[ind] = instancey


    def get_batch(self, batchsize, last = None):
        batchsize = min(self._length, batchsize, self._stacklength)
        if last is None:
            indices = random.sample(range(min(self._stacklength, self._length)), batchsize)
        else:
            bsize = round(batchsize*last)
            bbsize = batchsize-bsize
            indices = random.sample(range(min(self._stacklength, self._length)), bsize) + list(range(max(0,min(self._stacklength, self._length)-bbsize),min(self._stacklength, self._length)))
        xx = [self._memoryx[i] for i in indices]
        yy = [self._memoryy[i] for i in indices]
        return xx, yy

    def clear(self, newlength=None):
        self._memoryx = []
        self._memoryy = []
        self._length = 0

        if newlength is not None:
            self._stacklength = newlength
    def size(self):
        return self._length/self._stacklength

class PPG_data_manager():

    def __init__(self,stacklenght):
        self.memory = {'state': [], 'actions_probs': [], 'advantages':[],'values':[],'masks':[],'actions':[],'actions_ind':[]}
        self.stacklenght = stacklenght

    def add(self,B):
        if self.get_lenght()<=self.stacklenght:
            for key in B.keys():
                self.memory[key] += B[key]
        else:
            raise BufferError('Memoria Piena')

    def restart(self):
        self.memory = {'state': [], 'actions_probs': [], 'advantages':[],'values':[],'masks':[],'actions':[],'actions_ind':[]}

    def get_batch(self,batch_size,list_index=None):
        if list_index == None:
            list_index = random.sample(range(min(self.stacklenght,self.get_lenght())),batch_size)

        oldprobs = [self.memory['actions_probs'][j] for j in list_index]
        values = [self.memory['values'][j] for j in list_index]
        actinds = [self.memory['actions_ind'][j] for j in list_index]
        adv = [self.memory['advantages'][j] for j in list_index]
        states = [self.memory['state'][j] for j in list_index]
        masks = [self.memory['masks'][j] for j in list_index]

        return oldprobs,values,actinds,adv,states,masks

    def get_lenght(self):
        return len(self.memory['state'])


















