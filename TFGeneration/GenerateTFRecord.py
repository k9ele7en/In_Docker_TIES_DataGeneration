import warnings
warnings.filterwarnings("ignore")

import numpy as np
import traceback
import cv2
import os
import string
import pickle
from multiprocessing import Process,Lock
from TableGeneration.Table import Table
from multiprocessing import Process,Pool,cpu_count
import random
import argparse
from TableGeneration.tools import *
import numpy as np
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS
import warnings
from TableGeneration.Transformation import *
from icecream import ic
import json

def warn(*args,**kwargs):
    pass

class Logger:
    def __init__(self):
        pass
        #self.file=open('logtxt.txt','a+')

    def write(self,txt):
        file = open('logfile.txt', 'a+')
        file.write(txt)
        file.close()

class GenerateTFRecord:
    def __init__(self, outpath,filesize,unlvimagespath,unlvocrpath,unlvtablepath,distributionfilepath):
        self.outtfpath = outpath                        #directory to store tfrecords
        self.filesize=filesize                          #number of images in each tfrecord
        self.unlvocrpath=unlvocrpath                    #unlv ocr ground truth files
        self.unlvimagespath=unlvimagespath              #unlv images
        self.unlvtablepath=unlvtablepath                #unlv ground truth of tabls
        self.distributionfile=distributionfilepath      #pickle file containing UNLV distribution
        self.logger=Logger()                            #if we want to use logger and store output to file
        #self.logdir = 'logdir/'
        #self.create_dir(self.logdir)
        #logging.basicConfig(filename=os.path.join(self.logdir,'Log.log'), filemode='a+', format='%(name)s - %(levelname)s - %(message)s')
        self.num_of_max_vertices=900                    #number of vertices (maximum number of words in any table)
        self.max_length_of_word=30                      #max possible length of each word
        self.row_min=3                                  #minimum number of rows in a table (includes headers)
        self.row_max=1000                                 #maximum number of rows in a table, df=15
        self.col_min=3                                  #minimum number of columns in a table
        self.col_max=1000                                  #maximum number of columns in a table, df=9
        self.minshearval=-0.1                           #minimum value of shear to apply to images
        self.maxshearval=0.1                            #maxmimum value of shear to apply to images
        self.minrotval=-0.01                            #minimum rotation applied to images
        self.maxrotval=0.01                             #maximum rotation applied to images
        self.num_data_dims=5                            #data dimensions to store in tfrecord
        self.max_height=768                             #max image height
        self.max_width=1366                             #max image width
        self.tables_cat_dist = self.get_category_distribution(self.filesize)

    def get_category_distribution(self,filesize):
        tables_cat_dist=[0,0,0,0]
        firstdiv=filesize//2
        tables_cat_dist[0]=firstdiv//2
        tables_cat_dist[1]=firstdiv-tables_cat_dist[0]

        seconddiv=filesize-firstdiv
        tables_cat_dist[2]=seconddiv//2
        tables_cat_dist[3]=seconddiv-tables_cat_dist[2]
        return tables_cat_dist

    def create_dir(self,fpath):                         #creates directory fpath if it does not exist
        if(not os.path.exists(fpath)):
            os.mkdir(fpath)

    def str_to_int(self,str):                           #converts each character in a word to equivalent int
        intsarr=np.array([ord(chr) for chr in str])
        padded_arr=np.zeros(shape=(self.max_length_of_word),dtype=np.int64)
        padded_arr[:len(intsarr)]=intsarr
        return padded_arr

    def convert_to_int(self, arr):                      #simply converts array to a string
        return [int(val) for val in arr]

    def pad_with_zeros(self,arr,shape):                 #will pad the input array with zeros to make it equal to 'shape'
        dummy=np.zeros(shape,dtype=np.int64)
        dummy[:arr.shape[0],:arr.shape[1]]=arr
        return dummy

    def generate_tables(self,driver,N_imgs):
        row_col_min=[self.row_min,self.col_min]                 #to randomly select number of rows
        row_col_max=[self.row_max,self.col_max]                 #to randomly select number of columns
        rc_arr = np.random.uniform(low=row_col_min, high=row_col_max, size=(N_imgs, 2))        #random row and col selection for N images
        all_table_categories=[0,0,0,0]                         #These 4 values will count the number of images for each of the category
        rc_arr[:,0]=rc_arr[:,0]+2                                     #increasing the number of rows by a fix 2. (We can comment out this line. Does not affect much)
        data_arr=[]
        exceptioncount=0

        rc_count=0                                              #for iterating through row and col array
        print('total: ', self.tables_cat_dist)
        for assigned_category,cat_count in enumerate(self.tables_cat_dist):
            print('cat: ', assigned_category)
            for _ in range(cat_count):
                print('count: ',_)
                rows = int(round(rc_arr[rc_count][0]))
                cols = int(round(rc_arr[rc_count][1]))
                exceptcount=0

                ###out of while####
                table = Table(rows,cols,self.unlvimagespath,self.unlvocrpath,self.unlvtablepath,assigned_category+1,self.distributionfile)
                #get table of rows and cols based on unlv distribution and get features of this table
                #(same row, col and cell matrices, total unique ids, html conversion of table and its category)
                same_cell_matrix,same_col_matrix,same_row_matrix, id_count, html_content,tablecategory= table.create()

                #convert this html code to image using selenium webdriver. Get equivalent bounding boxes
                #for each word in the table. This will generate ground truth for our problem
                im,bboxes = html_to_img(driver, html_content, id_count)
                print('loop, current: ', assigned_category, _, cat_count)
                
                if(assigned_category+1==4):
                    #randomly select shear and rotation levels
                    while(True):
                        shearval = np.random.uniform(self.minshearval, self.maxshearval)
                        rotval = np.random.uniform(self.minrotval, self.maxrotval)
                        if(shearval!=0.0 or rotval!=0.0):
                            break

                    #If the image is transformed, then its categorycategory is 4

                    #transform image and bounding boxes of the words
                    # im, bboxes = Transform(im, bboxes, shearval, rotval, self.max_width, self.max_height)
                    # ic('pass transform')
                    # tablecategory=4

                #######################
                im=np.asarray(im,np.int64)[:,:,0]
            
                colmatrix = np.array(same_col_matrix,dtype=np.int64)
                cellmatrix = np.array(same_cell_matrix,dtype=np.int64)
                rowmatrix = np.array(same_row_matrix,dtype=np.int64)
                arr = np.array(bboxes)

                # save json and img
                cellmatrix=self.pad_with_zeros(same_cell_matrix,(self.num_of_max_vertices,self.num_of_max_vertices))
                colmatrix = self.pad_with_zeros(same_col_matrix, (self.num_of_max_vertices, self.num_of_max_vertices))
                rowmatrix = self.pad_with_zeros(same_row_matrix, (self.num_of_max_vertices, self.num_of_max_vertices))

                img_height, img_width=im.shape
                words_arr = arr[:, 1].tolist()
                no_of_words = len(words_arr)

                lengths_arr = self.convert_to_int(arr[:, 0])
                vertex_features=np.zeros(shape=(self.num_of_max_vertices,self.num_data_dims),dtype=np.int64)
                lengths_arr=np.array(lengths_arr).reshape(len(lengths_arr),-1)
                sample_out=np.array(np.concatenate((arr[:,2:],lengths_arr),axis=1))
                vertex_features[:no_of_words,:]=sample_out

                vertex_text = np.zeros((self.num_of_max_vertices,self.max_length_of_word), dtype=np.int64)
                vertex_text[:no_of_words]=np.array(list(map(self.str_to_int,words_arr)))

                # json
                featurejs = dict()
                filename = 'cat'+str(tablecategory)+'_'+str(_)
                cv2.imwrite(os.path.join(self.outtfpath,'images/'+filename+'.jpg'),im)

                featurejs['img_i'] = filename
                featurejs['bboxes'] = arr.tolist()
                
                featurejs['global_features'] = np.array([img_height, img_width,no_of_words,tablecategory]).astype(np.float32).flatten().tolist()
                featurejs['vertex_features_shp'] = vertex_features.shape
                featurejs['vertex_features'] = vertex_features.astype(np.float32).flatten().tolist()
                featurejs['adjacency_matrix_cells_shp'] = cellmatrix.shape
                featurejs['adjacency_matrix_cells'] = cellmatrix.astype(np.int64).flatten().tolist()
                featurejs['adjacency_matrix_cols_shp'] = colmatrix.shape
                featurejs['adjacency_matrix_cols'] = colmatrix.astype(np.int64).flatten().tolist()
                featurejs['adjacency_matrix_rows_shp'] = rowmatrix.shape
                featurejs['adjacency_matrix_rows'] = rowmatrix.astype(np.int64).flatten().tolist()
                featurejs['vertex_text_shp'] = vertex_text.shape
                featurejs['vertex_text'] = vertex_text.astype(np.int64).flatten().tolist()
                
                # a_file = open(os.path.join(self.outtfpath,filename+"_matrix.txt", "w")
                # for row in cellmatrix:
                #     np.savetxt(a_file, row)
                # a_file.close()

                jsonString = json.dumps(featurejs)

                jsonFile = open(os.path.join(self.outtfpath,'jsons/'+filename+'.json'), "w")
                jsonFile.write(jsonString)
                jsonFile.close()
                ###############
                print('Assigned category: ',assigned_category+1,', generated category: ',tablecategory)
                ##############
                
                rc_count+=1
                all_table_categories[tablecategory-1]+=1
        
        return data_arr,all_table_categories

    def write_tf(self,filesize,threadnum):
        '''This function writes tfrecords. Input parameters are: filesize (number of images in one tfrecord), threadnum(thread id)'''
        opts = Options()
        opts.set_headless()
        assert opts.headless
        #driver=PhantomJS()
        driver = Firefox(options=opts)
        # while(True):
            # starttime = time.time()

        print('\nThread: ',threadnum,' Started.')
        #data_arr contains the images of generated tables and all_table_categories contains the table category of each of the table
        data_arr,all_table_categories = self.generate_tables(driver, filesize)
        ic(all_table_categories)
        driver.stop_client()
        driver.quit()


    def write_to_tf(self,max_threads):
        '''This function starts tfrecords generation with number of threads = max_threads with each thread
        working on a single tfrecord'''
        
        if(not os.path.exists(self.distributionfile)):
            if((not os.path.exists(self.unlvtablepath)) or (not os.path.exists(self.unlvimagespath)) or (not os.path.exists(self.unlvocrpath))):
                print('UNLV dataset folders do not exist.')
                return
            
        #create all directories here
        self.create_dir(self.outtfpath)
        dirname=self.outtfpath
        self.create_dir(os.path.join(dirname,'images'))
        self.create_dir(os.path.join(dirname, 'jsons'))

        self.create_dir(self.outtfpath)                 #create output directory if it does not exist

        starttime=time.time()
        threads=[]
        for threadnum in range(max_threads):
            proc = Process(target=self.write_tf, args=(self.filesize, threadnum,))
            proc.start()
            threads.append(proc)

        for proc in threads:
            proc.join()
        print("Synth table completed, check at: ", self.outtfpath, ". time length: ", time.time()-starttime)
