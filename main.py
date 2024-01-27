from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    Rooms = sorted(data['numberOfRooms'].unique())
    garage = sorted(data['garage'].unique())
    sizes = sorted(data['squareMeters'].unique())
    city_code= sorted(data['cityCode'].unique())

    return render_template('index.html', Rooms=Rooms, garage=garage, sizes=sizes, city_code=city_code)

@app.route('/predict', methods=['POST'])
def predict():
    Rooms = request.form.get('numberOfRooms')
    garage= request.form.get('garage')
    sizes = request.form.get('squareMeters')
    citycode = request.form.get('cityCode')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[Rooms, garage, sizes, citycode]],
                               columns=['Rooms', 'garage', 'sizes', 'cityCode'])

    print("Input Data:")
    print(input_data)

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    print("Processed Input Data:")
    print(input_data)

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=6000)


if __name__ == "__main__":
    app.run(debug=True, port=6000)