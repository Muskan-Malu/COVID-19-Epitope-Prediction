

import numpy as np
import pandas as pd

import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

import pickle

gmail_list=[]
password_list=[]
gmail_list1=[]
password_list1=[]



from flask import Flask, request, jsonify, render_template

import joblib

with open('parent_protein_id.pkl', 'rb') as handle1:
        protein1=pickle.load(handle1)

with open('protein_seq.pkl', 'rb') as handle2:
        protein12=pickle.load(handle2)

with open('peptide_seq.pkl', 'rb') as handle3:
        protein123=pickle.load(handle3)
        
# Load the model from the file 
model= joblib.load('final_pickle_model.pkl')  
  
# Use the loaded model to make predictions 



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('register.html')

        
@app.route('/register',methods=['POST'])
def register():
    

    int_features2 = [str(x) for x in request.form.values()]

    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

  
    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()


    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      

                      
    print(gmail_list1)
    if logu1 in gmail_list1:
        return render_template('register.html',text="This Username is Already in Use ")

    else:

                  
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                  return render_template('register.html',text="Succesfully Registered")
        


@app.route('/login')
def login(): 
    return render_template('login.html')         
                      


@app.route('/logedin',methods=['POST'])
def logedin():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]


    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()

    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      

                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()

    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                    
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('index.html')
    else:
        return render_template('login.html',text='Use Proper Username and Password')
  
                                               

   

    
   


@app.route('/production')
def production(): 
    return render_template('index.html')


@app.route('/production/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    a=int_features

  
# intialise data of lists. 
    data = {'start_position':[float(a[0])],'end_position':[float(a[1])],'chou_fasman':[float(a[2])],'emini':[float(a[3])],'kolaskar_tongaonkar':[float(a[4])],'parker':[float(a[5])],'isoelectric_point':[float(a[6])],'aromaticity':[float(a[7])],'hydrophobicity':[float(a[8])],'stability':[float(a[9])]} 
 
# Create DataFrame 
    df = pd.DataFrame(data) 
  
# Print the output. 
    print(df )
    
    pred=model.predict(df)
    pred=int(pred)
    print(pred)


    

    if pred==1:
        return render_template('index.html', prediction_text='This can be used for Vaccine Production')

    else:
         return render_template('index.html', prediction_text='This can not be used for Vaccine Production')       
     

if __name__ == "__main__":
    app.run(debug=False)
