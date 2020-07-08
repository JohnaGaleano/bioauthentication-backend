from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler # for preprocessing the data
import uuid
from sklearn.ensemble import IsolationForest
from sklearn import svm 



class PinPredict(BaseModel):
    user_id: str
    data:List[str]= []

class PinTrain(BaseModel):
    model_type: int
    data:List[List[str]]



app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Authentication Backend"}

@app.post("/predict/pin")
async def predict(*, pinp:PinPredict ):    
    #Data transform
    x = np.array(pinp.data)
    x = x.astype(np.float)
    
    #Normalization
    #PASO EXTRA    
    
    #ReSize
    x = x.reshape(1,x.shape[0])
    print("Before Scaler: ", x)
    #Load Scaler
    PATH_SCALER = "./scalers/pin/"+pinp.user_id
    print(PATH_SCALER)
    scaler = load(PATH_SCALER)
    
    #Apply scaler
    # x = scaler.transform(x)
    # print("Scaler: ", x)

    #Load Model
    PATH_MODE = "./models/pin/"+pinp.user_id+".pkl"
    model = load(PATH_MODE) 
    predict =  model.predict(x)[0]
    print(predict)
    return {"real":int(predict)}

@app.post("/predict/pattern")
async def predict(*, pinp:PinPredict ):    
    #Data transform
    x = np.array(pinp.data)
    x = x.astype(np.float)
    
    #Normalization
    #PASO EXTRA    
    
    #ReSize
    x = x.reshape(1,x.shape[0])
    print("Before Scaler: ", x)
    #Load Scaler
    PATH_SCALER = "./scalers/pattern/"+pinp.user_id
    print(PATH_SCALER)
    scaler = load(PATH_SCALER)
    
    #Apply scaler
    x = scaler.transform(x)
    print("Scaler: ", x)

    #Load Model
    PATH_MODE = "./models/pattern/"+pinp.user_id+".pkl"
    model = load(PATH_MODE) 
    predict =  model.predict(x)[0]
    print(predict)
    return {"real":int(predict)}


@app.post("/train/pin")
async def predict(*,pinT:PinTrain ):    
    
    #Data transform
    x = np.array(pinT.data)
    x = x.astype(np.float)

    #Normalization
    scaler = StandardScaler().fit(x)
    Xtrain = scaler.transform(x)
       
    #Model
    if(pinT.model_type == 0):
        
        params = {'bootstrap': True,
                        'contamination': 0,
                        'max_features': 5,
                        'max_samples': 'auto',
                        'n_estimators': 13,
                        'n_jobs': -1
                    }
        
        model = IsolationForest()


    else:

        params = {'gamma': 'scale', 'kernel': 'sigmoid', 'nu': 0.01}        
        model =  svm.OneClassSVM()
        
    model.set_params(**params)

    #Train
    model.fit(x)

    #Model Id
    model_id = uuid.uuid4()
    
    #Save Model
    dump(model, "models/"+str(model_id)+".pkl")
    
    
    return {"model_id":str(model_id)}
