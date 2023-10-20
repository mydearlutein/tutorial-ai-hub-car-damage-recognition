import glob
import json
import os
import pandas as pd
from collections import Counter


def check_csv(task: str = None):
    
    if not task or (task == 'part'):
        file_name = 'code/part_labeling.csv'
        print(f'Start checking {file_name}')
        
        label_df = pd.read_csv(file_name)

        dir_name_img = 'data/Dataset/1.원천데이터/damage_part'
        dir_name_label = 'data/Dataset/2.라벨링데이터/damage_part'
    
        count_img, count_label = 0, 0
        for i, row in label_df.iterrows():
            if os.path.exists(os.path.join(dir_name_img, row['img_id'])):
                count_img += 1
            if os.path.exists(os.path.join(dir_name_label, row['img_id'].replace('jpg', 'json'))):
                count_label += 1
        
        print(f'Count image: {count_img}, Count label: {count_label}')

    if not task or (task == 'damage'):
        file_name = 'code/damage_labeling.csv'
        print(f'Start checking {file_name}')
        
        label_df = pd.read_csv(file_name)

        dir_name_img = 'data/Dataset/1.원천데이터/damage'
        dir_name_label = 'data/Dataset/2.라벨링데이터/damage'
    
        count_img, count_label = 0, 0
        for i, row in label_df.iterrows():
            if os.path.exists(os.path.join(dir_name_img, row['index'])):
                count_img += 1
            if os.path.exists(os.path.join(dir_name_label, row['index'].replace('jpg', 'json'))):
                count_label += 1
        
        print(f'Count image: {count_img}, Count label: {count_label}')


def make_new_label(task: str = None):

    if not task or (task == 'part'):
        print('Make a new labeling file for damaged parts')

        dir_name_img = 'data/Dataset/1.원천데이터/damage_part'
        
        # images = glob.glob(os.path.join(dir_name_img, r'*.jpg'))
        images = [file for file in os.listdir(dir_name_img) if file.endswith('.jpg')]
        print(f'Image name samples: {images[:3]}')
        
        df = pd.DataFrame(images, columns=['img_id'])
        
        # 전체 Training dataset 중 random하게 8:2로 train:val 구성 
        df.loc[df.index.to_series().sample(frac=0.8).index, 'dataset'] = 'train'
        df['dataset'].fillna('val', inplace=True)
        
        # validation 데이터를 다시 5:5로 val:test 구성
        sample_indices = df[df['dataset'] == 'val'].sample(frac=0.5, replace=True).index
        df.loc[sample_indices, 'dataset'] = 'test'
        print(df['dataset'].describe())
        
        df.to_csv('code/part_labeling_new.csv', index=False)
        print('Done.')
    
    if not task or (task == 'damage'):
        print('Make a new labeling file for damages')

        dir_name_img = 'data/Dataset/1.원천데이터/damage'
        dir_name_label = 'data/Dataset/2.라벨링데이터/damage'
        
        annos = []
        for i, file in enumerate(os.listdir(dir_name_img)):
            if file.endswith('.jpg'):
                anno_short = {'index': file}
                label_file = os.path.join(dir_name_label, file.replace('jpg', 'json'))
                if not os.path.exists(label_file):
                    continue

                with open(label_file, 'r') as f:
                    json_data = json.load(f)
                    damages = [anno['damage'] for anno in json_data['annotations']]
                    anno_short.update(dict(Counter(damages)))
                    anno_short['total_anns'] = len(damages)

                    # EDA용 column 추가
                    anno_short['image_width'] = json_data['images']['width']
                    anno_short['image_height'] = json_data['images']['height']
                    anno_short['car_size'] = json_data['categories']['supercategory_name']
                    anno_short['color'] = json_data['annotations'][0]['color']
                    anno_short['repair'] = json_data['annotations'][0]['repair']
                    anno_short['year'] = json_data['annotations'][0]['year']
                    
                    annos.append(anno_short)
                    
        print(f'New annotation samples: {annos[:3]}')
        
        df = pd.DataFrame(annos)
        for col in df.columns[1:]:
            if col in ["Scratched","Separated","Crushed","Breakage", "total_anns"]:
                print(col)
                df[col] = df[col].astype(float)
        
        df.fillna(0, inplace=True)
        print(df)
        
        # 전체 Training dataset 중 random하게 8:2로 train:val 구성 
        df.loc[df.index.to_series().sample(frac=0.8).index, 'dataset'] = 'train'
        df['dataset'].fillna('val', inplace=True)
        
        # validation 데이터를 다시 5:5로 val:test 구성
        sample_indices = df[df['dataset'] == 'val'].sample(frac=0.5, replace=True).index
        df.loc[sample_indices, 'dataset'] = 'test'
        print(df['dataset'].describe())
        
        df.to_csv('code/damage_labeling_new2.csv', index=False)
        print('Done.')


def check_anno(task: str = None):
    if not task or (task == 'part'):
        anno_file = 'data/datainfo/part_train.json'
        
        with open(anno_file, 'r') as f:
            json_data = json.load(f)
            
            wrong_count = 0
            for anno in json_data['annotations']:
                if (not isinstance(anno['segmentation'][0][0], int)) and (not isinstance(anno['segmentation'], float)):
                    print(anno)
                    break
                    wrong_count += 1
        
        print('Total count', len(json_data['annotations']))
        print('Wrong count', wrong_count)
            

if __name__ == '__main__':
    # check_csv()
    make_new_label('damage')
    # check_anno('part')