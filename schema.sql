CREATE TABLE IF NOT EXISTS water_demand (
    id SERIAL PRIMARY KEY,
    date DATE,
    zone_id VARCHAR(50),
    demand_liters FLOAT,
    population INT,
    avg_temp_c FLOAT,
    rainfall_mm FLOAT,
    is_holiday BOOLEAN
);

-- Table to store ML predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    zone_id VARCHAR(50) NOT NULL,
    predicted_demand FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
