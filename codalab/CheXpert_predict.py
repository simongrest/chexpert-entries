import cv2
import sys
import pandas as pd
from joblib import Parallel, delayed
from fastai.vision import *
from torchvision.models import *

def load_and_resize_img(path):
    """
    Load and convert the full resolution images on CodaLab to
    low resolution used in the small dataset.
    """
    img = cv2.imread(path, 0) 

    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    
    if max_ind == 0:
        # width fixed at 320
        wpercent = (320 / float(size[0]))
        hsize = int((size[1] * wpercent))
        new_size = (hsize, 320)
    
    else:
        # height fixed at 320
        hpercent = (320 / float(size[1]))
        wsize = int((size[0] * hpercent))
        new_size = (320, wsize)

    resized_img = cv2.resize(img, new_size)

    cv2.imwrite(path, resized_img)


def main():

    #python src/<path-to-prediction-program> <input-data-csv-filename> <output-prediction-csv-path>
    
    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    print(infile)
    print(outfile)
    
    test_df = pd.read_csv(infile)
    
    Parallel(n_jobs=-1)(delayed(load_and_resize_img)(path) for path in test_df.Path.values)
    
    chexpert_learn = load_learner('src/','chexpert_densenet.pkl')
    
    test_data_src = (ImageList.from_df(test_df, '.','Path'))
    
    chexpert_learn.data.add_test(test_data_src)
    
    chexpert_learn.data.batch_size = 8

    test_preds=chexpert_learn.get_preds(ds_type=DatasetType.Test)[0]
   
    i = 0
    for c in chexpert_learn.data.classes:
       
        test_df[c] = test_preds[:,i]
        i = i+1
        
    #CheXpert-v1.0/{valid,test}/<PATIENT>/<STUDY>
    
    test_df.Path.str.split('/')
    
    def get_study(path):
        return path[0:path.rfind('/')]
    
    test_df['Study'] = test_df.Path.apply(get_study)
    
    study_df = test_df.drop('Path',axis=1).groupby('Study').max().reset_index()
    
    study_df.to_csv(outfile,index=False)
    
if __name__ == '__main__':
    main()