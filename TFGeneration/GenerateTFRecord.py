import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
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
    def __init__(self, outpath,filesize,unlvimagespath,unlvocrpath,unlvtablepath,visualizeimgs,visualizebboxes,distributionfilepath):
        self.outtfpath = outpath                        #directory to store tfrecords
        self.filesize=filesize                          #number of images in each tfrecord
        self.unlvocrpath=unlvocrpath                    #unlv ocr ground truth files
        self.unlvimagespath=unlvimagespath              #unlv images
        self.unlvtablepath=unlvtablepath                #unlv ground truth of tabls
        self.visualizeimgs=visualizeimgs                #wheter to store images separately or not
        self.distributionfile=distributionfilepath      #pickle file containing UNLV distribution
        self.logger=Logger()                            #if we want to use logger and store output to file
        #self.logdir = 'logdir/'
        #self.create_dir(self.logdir)
        #logging.basicConfig(filename=os.path.join(self.logdir,'Log.log'), filemode='a+', format='%(name)s - %(levelname)s - %(message)s')
        self.num_of_max_vertices=900                    #number of vertices (maximum number of words in any table)
        self.max_length_of_word=30                      #max possible length of each word
        self.row_min=3                                  #minimum number of rows in a table (includes headers)
        self.row_max=15                                 #maximum number of rows in a table
        self.col_min=3                                  #minimum number of columns in a table
        self.col_max=9                                  #maximum number of columns in a table
        self.minshearval=-0.1                           #minimum value of shear to apply to images
        self.maxshearval=0.1                            #maxmimum value of shear to apply to images
        self.minrotval=-0.01                            #minimum rotation applied to images
        self.maxrotval=0.01                             #maximum rotation applied to images
        self.num_data_dims=5                            #data dimensions to store in tfrecord
        self.max_height=768                             #max image height
        self.max_width=1366                             #max image width
        self.tables_cat_dist = self.get_category_distribution(self.filesize)
        self.visualizebboxes=visualizebboxes

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

    def generate_tables(self,driver,N_imgs,output_file_name):
        row_col_min=[self.row_min,self.col_min]                 #to randomly select number of rows
        row_col_max=[self.row_max,self.col_max]                 #to randomly select number of columns
        rc_arr = np.random.uniform(low=row_col_min, high=row_col_max, size=(N_imgs, 2))        #random row and col selection for N images
        all_table_categories=[0,0,0,0]                         #These 4 values will count the number of images for each of the category
        rc_arr[:,0]=rc_arr[:,0]+2                                     #increasing the number of rows by a fix 2. (We can comment out this line. Does not affect much)
        data_arr=[]
        exceptioncount=0

        rc_count=0                                              #for iterating through row and col array

        for assigned_category,cat_count in enumerate(self.tables_cat_dist):
            for _ in range(cat_count):
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
                ic('loop, current: ', assigned_category, _, cat_count)
                
                if(assigned_category+1==4):
                    #randomly select shear and rotation levels
                    while(True):
                        ic('cat=4')
                        shearval = np.random.uniform(self.minshearval, self.maxshearval)
                        rotval = np.random.uniform(self.minrotval, self.maxrotval)
                        if(shearval!=0.0 or rotval!=0.0):
                            break

                    #If the image is transformed, then its categorycategory is 4

                    #transform image and bounding boxes of the words
                    # im, bboxes = Transform(im, bboxes, shearval, rotval, self.max_width, self.max_height)
                    # ic('pass transform')
                    tablecategory=4
                        

                if(self.visualizeimgs):
                    #if the image and equivalent html is need to be stored
                    dirname=os.path.join('visualizeimgs/','category'+str(tablecategory))
                    f=open(os.path.join(dirname,'html',str(rc_count)+output_file_name.replace('.tfrecord','.html')),'w')
                    f.write(html_content)
                    f.close()

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

                # feature = dict()
                # feature['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=im.astype(np.float32).flatten()))
                # feature['global_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array([img_height, img_width,no_of_words,tablecategory]).astype(np.float32).flatten()))
                # feature['vertex_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=vertex_features.astype(np.float32).flatten()))
                # feature['adjacency_matrix_cells'] = tf.train.Feature(int64_list=tf.train.Int64List(value=cellmatrix.astype(np.int64).flatten()))
                # feature['adjacency_matrix_cols'] = tf.train.Feature(int64_list=tf.train.Int64List(value=colmatrix.astype(np.int64).flatten()))
                # feature['adjacency_matrix_rows'] = tf.train.Feature(int64_list=tf.train.Int64List(value=rowmatrix.astype(np.int64).flatten()))
                # feature['vertex_text'] = tf.train.Feature(int64_list=tf.train.Int64List(value=vertex_text.astype(np.int64).flatten()))
                
                # json
                featurejs = dict()
                cv2.imwrite('visualizeimgs/cat'+str(tablecategory)+'_'+str(rc_count)+'.jpg',im)

                # featurejs['image'] = im.astype(np.float32).flatten().tolist()
                featurejs['img_i'] = 'cat'+str(tablecategory)+'_'+str(rc_count)
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
                
                a_file = open('visualizeimgs/cat'+str(tablecategory)+'_'+str(rc_count)+"_matrix.txt", "w")
                for row in cellmatrix:
                    np.savetxt(a_file, row)
                a_file.close()

                jsonString = json.dumps(featurejs)
                output_file_name=output_file_name.replace('.tfrecord','.json')

                jsonFile = open('visualizeimgs/cat'+str(tablecategory)+'_'+str(rc_count)+'.json', "w")
                jsonFile.write(jsonString)
                jsonFile.close()
                ###############
                print('Assigned category: ',assigned_category+1,', generated category: ',tablecategory)
                ##############
                
                rc_count+=1
        if(len(data_arr)!=N_imgs):
            #If total number of images are not generated, then return None.
            print('Images not equal to the required size.')
            return None
        ic('gentb5')
        
        return data_arr,all_table_categories

    def draw_matrices(self,img,arr,matrices,imgindex,output_file_name):
        '''Call this fucntion to draw visualizations of a matrix on image'''
        no_of_words=len(arr)
        colors = np.random.randint(0, 255, (no_of_words, 3))
        arr = arr[:, 2:]

        img=img.astype(np.uint8)
        img=np.dstack((img,img,img))

        mat_names=['row','col','cell']
        output_file_name=output_file_name.replace('.tfrecord','')

        for matname,matrix in zip(mat_names,matrices):
            im=img.copy()
            x=1
            indices = np.argwhere(matrix[x] == 1)
            for index in indices:
                cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
                              (int(arr[index, 2])+3, int(arr[index, 3])+3),
                              (0,255,0), 1)

            x = 4
            indices = np.argwhere(matrix[x] == 1)
            for index in indices:
                cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
                              (int(arr[index, 2])+3, int(arr[index, 3])+3),
                              (0, 0, 255), 1)

            img_name=os.path.join('bboxes/',output_file_name+'_'+str(imgindex)+'_'+matname+'.jpg')
            cv2.imwrite(img_name,im)



    def write_tf(self,filesize,threadnum):
        '''This function writes tfrecords. Input parameters are: filesize (number of images in one tfrecord), threadnum(thread id)'''
        options = tf.compat.v1.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
        opts = Options()
        opts.set_headless()
        assert opts.headless
        #driver=PhantomJS()
        driver = Firefox(options=opts)
        # while(True):
            # starttime = time.time()

            #randomly select a name of length=20 for tfrecords file.
        output_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) + '.tfrecord'
        print('\nThread: ',threadnum,' Started:', output_file_name)
        #data_arr contains the images of generated tables and all_table_categories contains the table category of each of the table
        data_arr,all_table_categories = self.generate_tables(driver, filesize, output_file_name)

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
        if(self.visualizeimgs):
            self.create_dir('visualizeimgs')
            for tablecategory in range(1,5):
                dirname=os.path.join('visualizeimgs/','category'+str(tablecategory))
                self.create_dir(dirname)
                self.create_dir(os.path.join(dirname,'html'))
                self.create_dir(os.path.join(dirname, 'img'))



        if(self.visualizebboxes):
            self.create_dir('bboxes')

        self.create_dir(self.outtfpath)                 #create output directory if it does not exist

        starttime=time.time()
        threads=[]
        for threadnum in range(max_threads):
            proc = Process(target=self.write_tf, args=(self.filesize, threadnum,))
            proc.start()
            threads.append(proc)

        for proc in threads:
            proc.join()
        print(time.time()-starttime)
