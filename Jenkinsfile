pipeline {
  agent any
  stages {
    stage('Install dependencies') {
      steps { bat 'py -3.11 -m pip install -r Productionisation_Travel_ML_System\\Productionisation_ML_Systems\\requirements.txt' }
    }
    stage('Train model') {
      steps { bat 'py -3.11 Productionisation_Travel_ML_System\\Productionisation_ML_Systems\\flight-price-pred-mlflow.py' }
    }
    stage('Smoke test API') {
      steps { bat 'py -3.11 -m compileall Productionisation_Travel_ML_System\\Productionisation_ML_Systems' }
    }
  }
}
