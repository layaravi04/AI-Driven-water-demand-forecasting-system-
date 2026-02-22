require("dotenv").config({ path: __dirname + "/.env" });
const express = require('express');
const cors = require("cors");
const pool = require('./db');

const app = express();
app.use(cors());
app.use(express.json());

// Import prediction routes (they will use the same pool)
const predictionsRouter = require('./routes/predictions');

app.get("/", (req, res) => {
  res.send("Backend is Running......");
});

app.get("/api/water-demand", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT * FROM water_demand ORDER BY date ASC"
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Mount prediction routes
app.use('/api/predictions', predictionsRouter);

app.listen(5000, () => {
  console.log("Server running on port 5000");
});
