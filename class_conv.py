

''' original classes in VisDrone-2019 dataset
0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
'''

import glob
import os

for i in glob.glob("validation/*.txt"):
    base_name = os.path.basename(i)
    with open(i,'r') as f:
        lines = f.readlines()
        for line in lines:
            
            #class 0 - people, cls 1 - tricyle, cls 2 - bicycle
            cls, x, y, w, h = line.split(" ")
            # if cls == '3' or '4' or '5' or '8' or '9':
            #     pass
                
            # else:
            if cls == '0':
                cls = '0'
                with open('validation_label/'+base_name, 'a') as g:
                    g.writelines(cls+" "+x+" "+y+" "+w+" "+h)
                
            if cls == '1':
                cls = '0'
                with open('validation_label/'+base_name, 'a') as g:
                    g.writelines(cls+" "+x+" "+y+" "+w+" "+h)
                
            if cls == '6':
                cls = '1'
                with open('validation_label/'+base_name, 'a') as g:
                    g.writelines(cls+" "+x+" "+y+" "+w+" "+h)
                
            if cls == '7':
                cls = '1'
                with open('validation_label/'+base_name, 'a') as g:
                    g.writelines(cls+" "+x+" "+y+" "+w+" "+h)
                
            if cls == '2':
                cls = '2'
                with open('validation_label/'+base_name, 'a') as g:
                    g.writelines(cls+" "+x+" "+y+" "+w+" "+h)
                
            
            print("processing")
                
            
                
            
                
        
