from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
# Create your views here.
import joblib
relodModel=joblib.load('./models/Modelfordiabetes.pkl')


def index(request):
    context={'a':'helloword'}
    return render(request,'index.html',context)



def predictdiabetes(request):
    if request.method=="POST":
        temp={}
        temp['Pregnancies']=request.POST.get('Pregnancies')
        temp['Glucose'] = request.POST.get('Glucose')
        temp['BloodPressure'] = request.POST.get('BloodPressure')
        temp['SkinThickness'] = request.POST.get('SkinThickness')
        temp['Insulin'] = request.POST.get('Insulin')
        temp['BMI'] = request.POST.get('BMI')
        temp['DiabetesPedigreeFunction'] = request.POST.get('DiabetesPedigreeFunction')
        temp['Age'] = request.POST.get('Age')
    testdata=pd.DataFrame({'x':temp}).transpose()
    scoreval=relodModel.predict(testdata)[-1]
    if scoreval==1:
        scoreval="Diabetic"
    else:
        scoreval="Non Diabetic"
    print(scoreval)
    con = {'scoreval': scoreval}
    return render(request,'index.html', con)