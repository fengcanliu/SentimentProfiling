### **Submission Sentiment Analysis**

To setup local envirment:
1. Download python 3.8.8 from https://www.python.org/downloads/
2. Install required packages using `pip install -r requirements.txt`
3. Create a data directory and copy the dataset over 
4. To train a model on the dataset, simply run `python train_and_predict.py` and this will show the score of the model and save the model.
5. To load a saved model and test it against a dataset simply run `python load_and_score.py`
6. To load a saved model and use it to predict a text, simply run `python load_and_predict.py`
