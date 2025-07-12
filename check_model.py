import pickle

# Load the .pkl file
with open('best_rf_model.pkl', 'rb') as f:
    obj = pickle.load(f)

# Print the type of object inside
print("Type of object in best_rf_model.pkl:", type(obj))
