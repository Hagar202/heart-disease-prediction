
import pandas as pd  # read data
from sklearn import preprocessing #using in encoding with data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import os

def dataPreparation():
    global Data_set
    # Data Preparation
    print(Data_set.shape)  # Dimensions
    print(Data_set.head(0)) # Column name
    print(Data_set.nunique())  # elemnts with column
    print(Data_set.dtypes)    # Data type     
    print(Data_set.dropna().shape) # look found empty cell or no 
    print(Data_set.duplicated())  # look found dublicated or no with data
    # print("------------------------------------------")

def dataEncoding():
  global Data_set , encoding_mapping
  # Data_encoding because found in data (object)
  de_types=Data_set.dtypes
  encoding_mapping = {} 
  for i in range(Data_set.shape[1]):
    if de_types[i]=='object':
      Encoding_data = preprocessing.LabelEncoder()
      original_column = Data_set[Data_set.columns[i]]
      encoded_column = Encoding_data.fit_transform(original_column)
      Data_set[Data_set.columns[i]] = encoded_column
      encoding_mapping[Data_set.columns[i]] = dict(zip(original_column, encoded_column))

def dataScailing():
  # Data_Scaling 
  Data_set_scaling=preprocessing.MinMaxScaler()
  X = Data_set.iloc[:, :-1]
  scailingModel=Data_set_scaling.fit_transform(X.values)
  scaled=pd.DataFrame(scailingModel,columns=X.columns)
  return Data_set_scaling,scaled
  # -----------------------------------------------------------------------------
def trainmodel():
  # using best parameter and using graid search
  parameter_grid = {'C': [0.1, 1, 10, 100,1000],
                'gamma': [0.001, 0.01, 0.1, 1,10],
                'kernel':['rbf','linear']}
  # building Model
  Model_svm = SVC()
  grid_search = GridSearchCV(Model_svm, parameter_grid, cv=5)   
  grid_search.fit(X_train, y_train)     #training the model

  print("Best Parameters: ", grid_search.best_params_) #here best_parameter in data
  print("Best Score: ", grid_search.best_score_) #best score with parameters

  K , G , C = grid_search.best_params_['kernel'] , grid_search.best_params_['gamma'] , grid_search.best_params_['C'] 
  model = SVC(kernel=K,gamma=G,C=C)
  model.fit(X_train, y_train)
  accuracy_model1 = model.score(X_test, y_test)
  print("Test Accuracy: ", accuracy_model1)
  # performance Evaluation
  Y_Pridict = model.predict(X_test) # Predict values for the test suite
  # Evaluate the model in more detail
  print(classification_report(y_test, Y_Pridict))
  return model


def getData(scailingModel):
    # User Input
    age = int(input("Please enter the Age: "))
    gender_input = int(input(f"Please enter the Gender {encoding_mapping['Sex']}: "))
    chest_pain_type = int(input(f"Please enter the chest pain type {encoding_mapping['ChestPainType']} "))
    resting_bp = float(input("Please enter the RestingBP: "))
    cholestero = float(input("Please enter the Cholestero: "))
    fasting_bs = float(input("Please enter the FastingBS: "))
    resting_ecg_input = int(input(f"Please enter the RestingECG {encoding_mapping['RestingECG']}: "))
    max_hr = float(input("Please enter the maximum heart rate achieved: "))
    exercise_angina_input = int(input(f"Please enter the exercise-induced angina ({encoding_mapping['ExerciseAngina']}): "))
    oldpeak = float(input("Please enter the Numeric value measured in depression: "))
    st_slope = int(input(F"Please enter the slope of the peak exercise ST segment ({encoding_mapping['ST_Slope']}): "))
    user_data = [[age,gender_input,chest_pain_type,resting_bp,cholestero,fasting_bs,resting_ecg_input,max_hr,exercise_angina_input,oldpeak,st_slope]]
    scaled_data = scailingModel.fit_transform(user_data)
    return scaled_data
  
if __name__ == '__main__':
  Data_set = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/1582403/2603715/heart.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20231216%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231216T150739Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=c373eb8d28fb23d49086a383ce8241d3d532b33909f304a2268ce0759467e0c0d32fbad3d84d42df556cbf18d11a751fe27e5fe68e134658183f9d76d4424dd44487ed98cf72aa626913a2fc0489961cd87787ae2878fd6b31ee1ea25b558a716ca59f3a3ce196e45ece5dc1f19537e21ae8565e6a60860bbabbce0c58c0eecd056b24cfec9b1080cfea30425e9d2e4e5cfe1e1367bfe95e267bfa59b63ffdd427af145f3f2f9123433650778d0c7e3e2afd86f4449cce63b6675bdd56302201bcb3a94b00f3d49b57ab6ae965fc2b8396a57500cf3b055d81d217308de9d4e70ba9ec7952318597a7f109bc4c9220c8ee9108e992a360fbb425da687d214c4a ')
  X = Data_set.iloc[:, :-1].values
  Y = Data_set.iloc[:, -1].values
  dataPreparation()
  dataEncoding()
  scailingModel , X = dataScailing()
  # Feature Selection
  # Divide the data into a training and test skagglesdsdata/datasets/1582403/uite
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  print(X_train,X_test)
  SVMModel = trainmodel()
  userInput = getData(scailingModel)
  prediction = SVMModel.predict(userInput)
  if prediction[0] == 1:
      print("You seem to have a potential for heart failure")
  else:
      print("You don't seem to have a potential for heart failure")
