
from flask import Flask,request,jsonify
import joblib, numpy as np
from tensorflow.keras.models import load_model
import os

app=Flask(__name__)
models={
 "log": joblib.load("models/log_reg.pkl"),
 "rf": joblib.load("models/rf.pkl"),
 "xgb": joblib.load("models/xgb.pkl")
}
lstm=load_model("models/lstm.h5")
scaler=joblib.load("models/scaler.pkl")

@app.get("/health")
def h(): return {"status":"ok"}

@app.post("/predict")
def p():
    data=request.json
    model=data.get("model","log")
    feats=np.array(data["features"])
    feats=scaler.transform([feats])[0]
    if model=="lstm":
        seq=np.array(data["sequence"]).reshape(1,5,len(feats))
        pr=float(lstm.predict(seq)[0][0])
        return {"prediction":int(pr>0.5)}
    pr=models[model].predict([feats])[0]
    return {"prediction":int(pr)}

app.run(host="0.0.0.0",port=5000)
