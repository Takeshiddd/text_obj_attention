import csv
import json
from Levenshtein import distance
import nltk
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
from glob import glob
import os 
import pandas as pd

def get_dictionary(path, model):
    stop_words = nltk.corpus.stopwords.words('english')
    words = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] not in stop_words and row[0] in model.wv.vocab:
                words.append(row[0].lower())
    return words

def get_proposed_word(word, dictionary, thresh):
    proposed_words = [element for element in dictionary if distance(word, element) / max(len(word), len(element)) <= thresh]
    if len(proposed_words) == 0:
        return None

    else:
        most_proposed = proposed_words[0]
        print('replace \'' + word + '\' to \'' + most_proposed + ' \' in {}'.format(proposed_words))
    return most_proposed

def preprocess_anns(text_anns, model, gsl_path='dictionary/ngsl.csv', thresh_distance=0.15):
    gsl_words = get_dictionary(gsl_path, model)
    del_list = []
    for key in tqdm(text_anns):
        ann = text_anns[key]
        new_texts = []
        mask = []
        if 'texts' not in ann:
            del_list.append(key)
            continue
        for text in ann['texts']:
            if text not in model.wv.vocab:
                text = get_proposed_word(text, gsl_words, thresh_distance)
                if text is None:
                    mask.append(False)
                    continue
            if text in gsl_words:
                mask.append(True)
                new_texts.append(text)
            else :
                mask.append(False)

        ann['texts'] = new_texts
        ann['bboxes'] = np.array(ann['bboxes'])[mask]
        if len(ann['bboxes']) == 0:
            del_list.append(key)
    
    for key in del_list:
        del text_anns[key]



if __name__ == "__main__":

    annootation_dir = '/media/kouki/kouki/scene_text_data/bboxes_data/OpenImages'
    dataset_type = 'train'
    annootation_paths = glob(os.path.join(annootation_dir, dataset_type, '*.json'))

    wv_model_path = '/media/kouki/kouki/scene_text_data/wv_models/self_train_200.bin'
    wv_model = KeyedVectors.load_word2vec_format(wv_model_path, binary=True)

    for path in annootation_paths:
        with open(path) as f:
            ann = json.load(f)

        preprocess_anns(ann, wv_model, '/media/kouki/kouki/scene_text_data/dictionary/ngsl.csv')
        columns=['ImageID', 'Height', 'Width', 'Text', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']
        data = {k:[] for k in columns}
        for key in tqdm(ann):
            image_id = key
            H, W = ann[key]['image_shape']
            bboxes = ann[key]['bboxes']
            texts = ann[key]['texts']
            for bbox, text in zip(bboxes, texts):
                data['ImageID'].append(image_id)
                data['Height'].append(H)
                data['Width'].append(W)
                data['Text'].append(text)
                x0, y0 = bbox[0]
                x1, y1 = bbox[1]
                x2, y2 = bbox[2]
                x3, y3 = bbox[3]
                data['x0'].append(x0)
                data['y0'].append(y0)
                data['x1'].append(x1)
                data['y1'].append(y1)
                data['x2'].append(x2)
                data['y2'].append(y2)
                data['x3'].append(x3)
                data['y3'].append(y3)
            
        export_path = os.path.join(os.path.splitext(path)[0]+'_preprocessed.pkl')
        new_ann = pd.DataFrame(data)
        new_ann.to_pickle(export_path)
