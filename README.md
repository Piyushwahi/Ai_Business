//for backend
cd backend                                                                                                                                                                                                                    
env\Scripts\activate                                                                                                          
 npm start

//for ai-service
cd ai-service 
venv\Scripts\activate 
python app.py 


//for frontend
deactivate
cd frontend   
 npm start


 //for facerecognition model
 venv/Scripts/Activate
 python config.py 
 python run.py

