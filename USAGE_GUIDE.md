# Quick Usage Guide

## Running the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Main Script
```bash
python water_demand_forecasting.py
```

### Step 3: View Results
After execution, you'll find:
- `water_consumption_dataset.csv` - Your dataset
- `visualizations/` folder - All generated graphs

---

## Understanding the Output

### Console Output
The script prints:
1. Dataset generation confirmation
2. Preprocessing status
3. Model training progress
4. Evaluation metrics for both models
5. File save confirmations

### Generated Files

**1. water_consumption_dataset.csv**
- Complete dataset with all features
- Ready for analysis or reporting
- Can be opened in Excel or any data analysis tool

**2. visualizations/water_demand_forecasting_results.png**
Contains 4 subplots:
- **Top Left**: Linear Regression - Actual vs Predicted scatter plot
- **Top Right**: Linear Regression - Time series comparison
- **Bottom Left**: Seasonal consumption trends (bar chart)
- **Bottom Right**: ARIMA - Time series forecast

**3. visualizations/correlation_heatmap.png**
- Shows correlation between all numerical features
- Helps understand feature relationships

---

##  For Project Reports

### What to Include in Your Report

1. **Introduction Section**
   - Copy from `PROJECT_REPORT.md` Section 1 & 2

2. **Dataset Description**
   - Show sample from `water_consumption_dataset.csv`
   - Explain features (from Section 3 of report)

3. **Methodology**
   - Linear Regression explanation (Section 5.1)
   - ARIMA explanation (Section 5.2)

4. **Results**
   - Include evaluation metrics from console output
   - Add visualization images
   - Interpret results (Section 6.2)

5. **Conclusion**
   - Use Section 8 from `PROJECT_REPORT.md`

---

## Customization Examples

### Generate More Data
```python
# In water_demand_forecasting.py, modify:
forecaster.generate_dataset(n_months=120)  # 10 years instead of 5
```

### Try Different ARIMA Orders
```python
# Common ARIMA orders to try:
forecaster.train_arima(order=(1, 1, 1))  # Simple ARIMA
forecaster.train_arima(order=(3, 1, 3))  # More complex
forecaster.train_arima(order=(2, 1, 0))  # AR only
```

### Change Visualization Style
```python
# In visualize_results(), modify:
plt.style.use('ggplot')  # Different style
sns.set_palette("Set2")  # Different color palette
```

---

## For Presentations/Viva

### Key Points to Explain

1. **Why Time-Series?**
   - Water demand has temporal patterns
   - Historical data predicts future demand

2. **Why Two Models?**
   - **Linear Regression**: Uses external factors (population, weather)
   - **ARIMA**: Uses only historical consumption patterns
   - Compare which performs better

3. **Evaluation Metrics**
   - **MAE/RMSE**: Error in liters (lower is better)
   - **R²**: How much variance is explained (higher is better, max 1.0)

4. **Real-World Impact**
   - Helps plan water supply
   - Reduces shortages and wastage
   - Saves costs and resources

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Install missing package
```bash
pip install <package_name>
```

### Issue: ARIMA warnings
**Solution**: These are normal, can be ignored. The model still works.

### Issue: Visualization not showing
**Solution**: Check if `visualizations/` folder was created. Images are saved automatically.

### Issue: Low R² scores
**Possible reasons**:
- Try adjusting ARIMA order
- Check if data relationships are realistic
- Consider adding more features

---

## Tips for Best Results

1. **For Better Accuracy**:
   - Generate more data (increase `n_months`)
   - Experiment with different ARIMA orders
   - Add more features (e.g., economic indicators, holidays)

2. **For Reports**:
   - Use high-resolution images (already set to 300 DPI)
   - Include actual numbers from evaluation metrics
   - Explain both models' strengths and weaknesses

3. **For Presentations**:
   - Focus on the visualization graphs
   - Explain the seasonal trends chart
   - Highlight the R² score (model accuracy)



## Next Steps

After running the basic project:

1. **Experiment**: Try different model parameters
2. **Extend**: Add more features or models
3. **Compare**: Test which model performs better for your data
4. **Document**: Use the generated outputs in your report

---

**Need Help?** Refer to `PROJECT_REPORT.md` for detailed explanations!



