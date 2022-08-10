import joblib 
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("credit_scoring")

class Item(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int 
    DebtRatio: float 
    MonthlyIncome: float 
    NumberOfOpenCreditLinesAndLoans: int 
    NumberOfTimes90DaysLate: int 
    NumberRealEstateLoansOrLines: int 
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: float


@app.post("/predict")
def read_root(item: Item):
    df = pd.DataFrame.from_dict(dict(item), orient='index').transpose()      
    result = model.predict(df)
    return {"result": int(result[0])}