# ArnavGoyal_IML_midterm

## Description
This project implements various machine learning algorithms from scratch, including Random Forest, Linear Regression, and Logistic Regression, without using any machine learning libraries. The project focuses on data preprocessing, model training, and evaluation.

## Data
The initial data is placed in the `data` directory. It includes FASTA files and protein properties CSV files.

## Data Preprocessing
Run the `data_preprocessing.py` script to process the initial data. This script generates the `pipeline` folder, which contains preprocessed data ready for model training.

## Algorithms Implemented
We have implemented the following algorithms from scratch:
- **Random Forest Regressor**: Provides the best performance among the implemented models.
- **Linear Regression Classifier**
- **Logistic Regression Classifier**

## Usage
The main script to run the Random Forest model is `rforrest.py`. It loads data, preprocesses it, trains the model, and evaluates its performance.

## Code Overview
- `rforrest.py`: Contains the implementation of the `RandomForestRegressor` class, data loading, preprocessing, model training, and evaluation.
- `dtrees.py`: Implements the `DecisionTreeRegressor` used by the Random Forest.
- `data_preprocessing.py`: Handles data preprocessing tasks.
- `linear.py`: Implements the Linear Regression Classifier.
- `logistic.py`: Implements the Logistic Regression Classifier.

## Running the Project
Execute the main script:
```bash
python rforrest.py
```

## Evaluation
The model's performance is evaluated using accuracy and a confusion matrix, which are saved in the `reports` folder. Among the models, the Random Forest Regressor achieved the highest accuracy.

## Dependencies
The project requires the following Python packages:
- pandas
- numpy
- math

## License
Specify the license under which the project is distributed.

## Contributing
Guidelines for contributing to the project.

## Contact
Contact information for further inquiries.
