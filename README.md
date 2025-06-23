# 🩺 Diabetes Predictor

A machine learning-based application that predicts the likelihood of a person having diabetes based on health parameters. This project uses a trained classification model and provides a user-friendly interface for predictions.

## Screenshots

![App Screenshot1](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20203520.png)
![App Screenshot2](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20203600.png)
![App Screenshot3](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20203616.png)
![App Screenshot4](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20203634.png)
![App Screenshot5](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20203653.png)
![App Screenshot6](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20203753.png)
![App Screenshot7](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20203948.png)
![App Screenshot8](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20204001.png)
![App Screenshot9](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20204143.png)
![App Screenshot10](https://github.com/Soujanya-R/Diabetes-Predictor/blob/main/images/Screenshot%202025-06-23%20204227.png)


---

## 📁 Project Structure
DiabetesPredictorPro
- app.py              # Main app runner
- simple_app.py       # Lightweight version of the app
- data_loader.py      # Loads and preprocesses dataset
- model_trainer.py    # Trains the ML model
- predictor.py        # Performs predictions using the trained model
-  diabetes_app.py     # Streamlit or UI logic
- pyproject.toml      # Project dependencies
- README.md           # Project documentation

---

## ⚙️ How It Works

1. **Data Loading**  
   Reads the diabetes dataset (likely from CSV or internal source).

2. **Preprocessing**  
   Cleans the data and splits it into training/testing sets.

3. **Model Training**  
   Trains a classification model (e.g., Logistic Regression, Random Forest).

4. **Prediction**  
   Takes user input and returns a diabetes prediction.

5. **Interface**  
   Run via `app.py` or `simple_app.py`, likely using Streamlit or command-line.

---

## 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/DiabetesPredictorPro.git
   cd DiabetesPredictorPro

1. **Create a Virtual Environment**
   ```bash
   python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate.bat  # Windows

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

## Usage/Examples

**Run Main App**

```bash
python app.py
```
**Run Lightweight App**

```bash
streamlit run diabetes_app.py
```
## 🧠 Model Info


Input features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age




## 📦 Dependencies
Python ≥ 3.7

NumPy, Pandas

scikit-learn

Streamlit 


## 📈 Example Output

Input:
  Glucose: 150
  BMI: 33
  Age: 45

Output:
  ⚠️ High chance of diabetes.



## Authors

Soujanya Ravi Kumar
BTech AI & ML | UVCE
https://github.com/Soujanya-R

## License

[MIT](https://choosealicense.com/licenses/mit/)
This project is licensed under the MIT License.

## Screenshots

![App Screenshot1](https://github.com/Soujanya-R/Flight-Ticket-Booking-System/blob/main/Screenshot%202025-04-16%20085423.png)