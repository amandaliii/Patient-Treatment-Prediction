# Patient-Treatment-Prediction
Model to predict ICU patient treatment

1. processedVocab - turns the mimic 3 itemID sequences into a flat list
2. itemID - maps itemIDs to their labels string format
3. model - the LSTM model
4. train - trains the model using precision, accuracy, recall, and f1 score
5. predictNext - predicts the next sequence in a 2 hour time frame
6. main - runs the whole program
