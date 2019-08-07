import sys
#sys.path.insert(0, '/home/cug/ldp7/.local/lib/python2.7/site-packages/librosa/')
import os
import re
import numpy as np
from general import *
import numpy as np
import librosa
import soundfile as sf

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

def get_data_mel(data_dir, store_dir, opt):
   
   n_mel = 128 
   n_win = 1200 #0.025 s, fs=48000, 
   n_hop = 230  #0.0048 s, fs=48000 
   n_fft = 4096
   f_min = 10
   f_max = None
   htk   = False
   eps   = np.spacing(1)

   # Get list of class
   org_class_name_list = os.listdir(data_dir)
   class_name_list = []
   for i in range(0,len(org_class_name_list)):
      isHidden=re.match("\.",org_class_name_list[i])
      if (isHidden is None):
         class_name_list.append(org_class_name_list[i])
   class_name_list = sorted(class_name_list)        
   class_name_num  = len(class_name_list)

   # For every class
   for nClass in range(0, class_name_num): 
   #for nClass in range(0, 1): 
       #3.1 Collect the file Name List
       class_name  = class_name_list[nClass]
       class_open  = data_dir + class_name + '/'

       class_store = store_dir + '/' + class_name
       if not os.path.exists(class_store):
          os.makedirs(class_store)

       org_file_name_list = os.listdir(class_open)
       file_name_list = []  #remove .file
       for i in range(0,len(org_file_name_list)):
          isHidden=re.match("\.",org_file_name_list[i])
          if (isHidden is None):
             file_name_list.append(org_file_name_list[i])
       file_name_list = sorted(file_name_list)
       file_name_num  = len(file_name_list)

       # For every file in class
       for nFile in range(0, file_name_num):
       #for nFile in range(0, 1):
          file_name  = file_name_list[nFile]
          file_open  = class_open + file_name
          org_wav, fs = sf.read(file_open)
          if(org_wav.ndim > 1):
             wav = org_wav[:,0]
          else:
             wav = org_wav
          

          #gammatone filter 
          #spec_data = gtgram(wav, fs, window_size, hop_size, gam_filter, min_req)

          #mel
          stft_res = librosa.core.stft(wav, 
                                  n_fft      = n_fft,
                                  win_length = n_win,
                                  hop_length = n_hop,
                                  center     = True
                                 )

          mel_basis = librosa.filters.mel( sr     = fs,
                                           n_fft  = n_fft,
                                           n_mels = n_mel,
                                           fmin   = f_min,
                                           fmax   = f_max,
                                           htk    = htk
                                          )
          spec_data = np.dot(mel_basis, stft_res)
          spec_data = np.log(np.abs(spec_data))
          
          plt.figure(figsize=(12, 12))
          #plt.subplot(2,1,1)
          #imshow(spec_data, aspect = 'auto', cmap='jet')
          #plt.colorbar()
          #plt.title('Pre-Normalize')

          # Normalization of whole 2-D image 
          min_spec_data = min(min(element) for element in spec_data)
          max_spec_data = max(max(element) for element in spec_data)
          data01 = spec_data
          data01 = data01 - min_spec_data
          data01 = data01/(max_spec_data - min_spec_data)
          #print np.shape(data01)

          #plt.subplot(2,1,2)
          imshow(data01, aspect='auto', cmap='jet')
          plt.colorbar()
          plt.title('Pos-Normalize')
          plt.show()

          #exit()

          #Store data
          file_des = class_store + '/' + str(nFile)

          np.save(file_des, data01)
      
       if opt==1:
          print ("Done extracting training feature for class {} \n".format(nClass))   
       else:   
          print ("Done extracting testing feature for class {} \n".format(nClass))   
          
   if opt==1:
      print ("==================== Done extracting training feature \n\n")   
   else:
      print ("==================== Done extracting testing feature \n\n")   
        


