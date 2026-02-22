const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
require('dotenv').config();

const db = require('./db');
const CSV_FILE = path.join(__dirname, 'data', 'water_demand.csv');

async function seed_database() {
  const rows = [];
  try {
    if (typeof db.ensureSchema === 'function') await db.ensureSchema();

    await new Promise((resolve, reject) => {
      fs.createReadStream(CSV_FILE)
        .pipe(csv({ separator: '\t' }))
        .on('data', (data) => {
          rows.push({
            date: data.date,
            zone_id: data.zone_id,
            demand_liters: parseFloat(data.demand_liters),
            population: parseInt(data.population),
            avg_temp_c: parseFloat(data.avg_temp_c),
            rainfall_mm: parseFloat(data.rainfall_mm),
            is_holiday: data.is_holiday === 'true' || data.is_holiday === '1'
          });
        })
        .on('end', resolve)
        .on('error', reject);
    });

    // clear existing rows then insert
    await db.query('DELETE FROM water_demand');

    for (const r of rows) {
      await db.query(
        'INSERT INTO water_demand(date,zone_id,demand_liters,population,avg_temp_c,rainfall_mm,is_holiday) VALUES ($1,$2,$3,$4,$5,$6,$7)',
        [r.date, r.zone_id, r.demand_liters, r.population, r.avg_temp_c, r.rainfall_mm, r.is_holiday]
      );
    }

    console.log('Database seeding completed!');
  } catch (err) {
    console.error('Seeding error:', err.message || err);
  } finally {
    if (typeof db.close === 'function') await db.close();
  }
}

seed_database();
