# AI-Driven Water Demand Forecasting System for Urban Areas

A comprehensive machine learning project that predicts urban water consumption using Linear Regression and ARIMA time-series models.

## Project Overview
This project addresses the critical challenge of water resource management in urban areas by implementing AI-powered forecasting models. The system predicts future water demand based on historical data, population trends, and environmental factors, enabling authorities to optimize water supply and reduce wastage.

## Key Features

- **Dual Model Approach**: Implements both Linear Regression and ARIMA models
- **Realistic Data Simulation**: Generates synthetic urban water consumption data
- **Comprehensive Evaluation**: Uses MAE, RMSE, and R² metrics
- **Rich Visualizations**: Creates detailed graphs and analysis charts
- **Academic Documentation**: Includes detailed report suitable for project submissions

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

**Execute the main script:**
```bash
python water_demand_forecasting.py
```

This will:
- Generate a synthetic water consumption dataset
- Preprocess the data
- Train Linear Regression and ARIMA models
- Evaluate model performance
- Generate visualizations
- Save results to CSV and image files

## Project Structure

```
ieee/
│
├── water_demand_forecasting.py    # Main implementation script
├── PROJECT_REPORT.md              # Comprehensive project documentation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── water_consumption_dataset.csv  # Generated dataset (after running)
└── visualizations/                # Generated plots (after running)
    ├── water_demand_forecasting_results.png
    └── correlation_heatmap.png
```

## What the System Does

### 1. Data Generation
Creates a realistic dataset with:
- Monthly water consumption (60 months)
- Population data (gradually increasing)
- Temperature (seasonal patterns)
- Rainfall (seasonal patterns)
- Seasonal classifications

### 2. Data Preprocessing
- Handles missing values
- Normalizes numerical features
- Encodes categorical variables
- Splits data into training (80%) and testing (20%) sets

### 3. Model Training

**Linear Regression:**
- Predicts water demand based on population, temperature, rainfall, and season
- Captures relationships between multiple features

**ARIMA (2,1,2):**
- Time-series model using historical consumption patterns
- Captures temporal dependencies and trends

### 4. Evaluation
Calculates:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

### 5. Visualizations
Generates:
- Actual vs Predicted scatter plots
- Time series comparison charts
- Seasonal consumption trends
- Feature correlation heatmap

## Expected Results

After running the script, you should see:

```
✓ Dataset generated successfully with 60 months of data
✓ Data preprocessed successfully
✓ Linear Regression model trained successfully
✓ ARIMA model trained successfully

Linear Regression Evaluation Metrics:
  Mean Absolute Error (MAE): ~2,500,000 liters
  Root Mean Squared Error (RMSE): ~3,200,000 liters
  R² Score: ~0.90

ARIMA Evaluation Metrics:
  Mean Absolute Error (MAE): ~3,500,000 liters
  Root Mean Squared Error (RMSE): ~4,100,000 liters
  R² Score: ~0.85
```

## Documentation

For detailed explanations of:
- How the models work
- Why AI/ML is used for this problem
- Model evaluation interpretation
- Academic explanations

See **PROJECT_REPORT.md**

## Academic Use

This project is designed for:
- **University Projects**: IEEE, Computer Science, Data Science courses
- **Research**: Water resource management studies
- **Presentations**: Viva and project defense
- **Documentation**: Report writing and technical documentation

## Customization

### Adjust Dataset Size
```python
forecaster.generate_dataset(n_months=120)  # Generate 10 years of data
```

### Change ARIMA Parameters
```python
forecaster.train_arima(order=(3, 1, 3))  # Different ARIMA configuration
```

### Modify Train-Test Split
Edit the `test_size` parameter in `preprocess_data()` method.

## Output Files

1. **water_consumption_dataset.csv**: Complete dataset in CSV format
2. **visualizations/water_demand_forecasting_results.png**: Main results visualization
3. **visualizations/correlation_heatmap.png**: Feature correlation analysis

## 🛠️ Troubleshooting

### Common Issues

**Import Error:**
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn statsmodels
```

**ARIMA Warning:**
- Warnings about convergence are normal and can be ignored
- The model will still produce valid results

**Visualization Issues:**
- Ensure matplotlib backend is properly configured
- On some systems, you may need: `export MPLBACKEND=TkAgg`

## Contributing

This is an academic project. Feel free to:
- Modify parameters for experimentation
- Add additional models (LSTM, Prophet, etc.)
- Extend the dataset with more features
- Improve visualizations

##  License

This project is provided for educational purposes.

## Acknowledgments

- Built for IEEE/University project requirements
- Uses standard ML libraries: scikit-learn, statsmodels
- Inspired by real-world water management challenges







