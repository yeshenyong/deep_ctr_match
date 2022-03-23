import requests
from time import time
import json
import numpy

url = 'http://localhost:8501/v1/models/mind_user/versions/1:predict'

predict_item = {
    'instances': [{
        'movie_id':[3, 4, 1, 5]
    }]
}

predict_user ={
        'instances': [{
            'user_id':[1], 
            'hist_movie_id':
       [120,  19, 101,   4,  13, 136,  42, 129, 189, 111, 157,  20, 106,
        115, 181,  17, 170,   9,  22,  44, 153,  94, 103,  91, 114,  37, 
        151,  90, 141, 130, 146,  96,  99,   2, 188,  15, 107,  23,   8,  
        192,  18, 171,  30, 132,  21, 155, 182, 206,  82,  16], 
            'hist_len': [ 52], 
            'gender': [1] ,
            'age': [1], 
            'occupation': [1], 
            'zip': [1]
        }]  
    }

print(predict_user)
r = requests.post(url, data=json.dumps(predict_user))
print(r.content)


