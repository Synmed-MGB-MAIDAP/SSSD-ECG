import random
import torch
from tqdm.auto import tqdm

# common practive thresholds 91
thresholds_15 = {
    "age": [12, 17, 34, 54, 74],
    "weight": [50, 70, 90, 110],
    "height": [150, 159, 169, 179],
    "hr": [60, 70, 80, 90, 100],
}


def map_age(age):
    """
    Map age to an on hot vector based on the provided thresholds.
    """
    age_threshold = thresholds_15['age']
    # Create a one-hot vector
    one_hot_vector = [0] * (len(age_threshold) + 1)
    if not isinstance(age, int):
        # fall in the average age threshold box
        # if the number of boxes is even
        if len(age_threshold) % 2 == 0:
            one_hot_vector[len(age_threshold) // 2] = 1
        else:
            # randomly select one of the two middle boxes
            if random.randint(0, 1) == 0:
                one_hot_vector[len(age_threshold) // 2] = 1
            else:
                one_hot_vector[len(age_threshold) // 2 + 1] = 1
        return one_hot_vector
    
    age = int(age)
    age_threshold = [int(x) for x in age_threshold]
     
    # Find the appropriate index for the one-hot vector
    for i, threshold in enumerate(age_threshold):
        if age <= threshold:
            one_hot_vector[i] = 1
            break
    else:
        one_hot_vector[-1] = 1  # Last category if above all thresholds
    
    return one_hot_vector


def map_gender(gender):
    """
    map gender to 2 dim one hot vector
    """
    if not isinstance(gender, str):
        # random select one
        if random.randint(0, 1) == 0:
            return [1, 0]
        else:
            return [0, 1]

    if gender == 'F':
        return [1, 0]
    if gender == 'M':
        return [0, 1]
    else:
        # random select one
        if random.randint(0, 1) == 0:
            return [1, 0]
        else:
            return [0, 1]

def map_heartrate(hr):
    """
    Map heart rate to an on hot vector based on the provided thresholds.
    """
    hr_threshold = thresholds_15['hr']
    # Create a one-hot vector
    one_hot_vector = [0] * (len(hr_threshold) + 1)
    
    if not isinstance(hr, int):
        # fall in the average hr threshold box
        # if the number of boxes is even
        if len(hr_threshold) % 2 == 0:
            one_hot_vector[len(hr_threshold) // 2] = 1
        else:
            # randomly select one of the two middle boxes
            if random.randint(0, 1) == 0:
                one_hot_vector[len(hr_threshold) // 2] = 1
            else:
                one_hot_vector[len(hr_threshold) // 2 + 1] = 1
        return one_hot_vector
    
    hr = int(hr)
    hr_threshold = [int(x) for x in hr_threshold]
    
    # Find the appropriate index for the one-hot vector
    for i, threshold in enumerate(hr_threshold):
        if hr <= threshold:
            one_hot_vector[i] = 1
            break
    else:
        one_hot_vector[-1] = 1  # Last category if above all thresholds
    
    return one_hot_vector

def categorize_demographics_for_one(sample, include_text_embedding=False):
    x, label_dict = sample
    x = torch.tensor(x).transpose(0, 1)
    disease = label_dict['encoded_label']
    age = torch.tensor(map_age(label_dict['age']))
    gender = torch.tensor(map_gender(label_dict['gender']))
    hr = torch.tensor(map_heartrate(label_dict['hr']))
    label_vec = torch.cat((disease, age, gender, hr), dim=0)
    
    # print("Shape of age vector:", age.shape)
    # print("Shape of gender vector:", gender.shape)
    # print("Shape of heart rate vector:", hr.shape)
    # print("Shape of disease vector:", disease.shape)
    # print("Shape of label vector:", label_vec.shape)
    
    if include_text_embedding:
        embedding_vec = label_dict.get('text_embedding')
        label_vec = torch.cat((label_vec, embedding_vec), dim=0)
    return x, label_vec

def categorize_demographics(data, include_text_embedding=False):
    """
    Categorize demographics data into one-hot vectors.
    """
    new_data = []
    for sample in tqdm(data, desc='Categorizing demographics'):
        # print(sample)
        x, label_vec = categorize_demographics_for_one(sample, include_text_embedding)
        new_data.append([x, label_vec])
    return new_data

if __name__ == "__main__":
    # Test the mapping functions
    age = 'NULL'
    print(map_age(age))
    gender = 'Null'
    print(map_gender(gender))
    hr = 120
    print(map_heartrate(hr))




