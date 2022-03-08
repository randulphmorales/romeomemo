import numpy as np
from nptdms import TdmsFile, tdms
from datetime import datetime,timedelta

class readtdms:
    def __init__(self,filename):
        self.filename = filename
        self.tdms_file = TdmsFile(filename)
        self.groups = self.tdms_file.groups()
    
    def monitorquantities(self,groupindex):
        variab = self.tdms_file.group_channels(self.groups[groupindex])

        chans=[]
        for ii in range(len(variab)):
            #remove <TdmsObject with path /'Monitoring Values'/'
            channel = (str(variab[ii])).split("/")[-1][1:-2]
            chans.extend([channel])
        return chans
    
    def monitordata(self,groupindex,monitorquant):
        tdms_groups = self.groups
        MessageData_channel_1 = self.tdms_file.object(tdms_groups[groupindex], monitorquant)
        MessageData_data_1 = MessageData_channel_1.data
        return MessageData_data_1

    def __call__(self,val):
        if val=="timestamp":
            #milliseconds
            rawtime = self.monitordata(0,'t-stamp')
            return np.asarray([int(datetime.timestamp(rawtime[i])*1000000) for i in range(len(rawtime))])
        if val=="pressure":
            return self.monitordata(0,'p1-Cell')
        if val=="temperature":
            return self.monitordata(0,'T1-Cell')
