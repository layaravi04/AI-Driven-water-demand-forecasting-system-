require('dotenv').config({ path: __dirname + '/../.env' });
const fs = require('fs');
const path = require('path');

// If DB env present, use Postgres; otherwise use sqlite fallback
const useSqlite = !process.env.DB_HOST || process.env.DB_HOST === '';

if (!useSqlite) {
  // Postgres adapter (export pool with query)
  const { Pool } = require('pg');
  const pool = new Pool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'water_demand_db',
    port: process.env.DB_PORT || 5432,
  });

  async function ensureSchema() {
    const schemaFile = path.join(__dirname, 'schema.sql');
    if (fs.existsSync(schemaFile)) {
      const sql = fs.readFileSync(schemaFile, 'utf8');
      await pool.query(sql);
    }
  }

  module.exports = {
    query: (text, params) => pool.query(text, params),
    ensureSchema,
    close: () => pool.end(),
  };

} else {
  // SQLite fallback
  const sqlite3 = require('sqlite3');
  const { open } = require('sqlite');

  const DB_FILE = path.join(__dirname, 'waterdb.sqlite');

  let dbPromise = open({ filename: DB_FILE, driver: sqlite3.Database });

  async function ensureSchema() {
    const db = await dbPromise;
    // Create compatible tables for sqlite
    await db.exec(`
      CREATE TABLE IF NOT EXISTS water_demand (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        zone_id TEXT,
        demand_liters REAL,
        population INTEGER,
        avg_temp_c REAL,
        rainfall_mm REAL,
        is_holiday INTEGER
      );
    `);
    await db.exec(`
      CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        zone_id TEXT NOT NULL,
        predicted_demand REAL NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
      );
    `);
  }

  async function query(sql, params = []) {
    const db = await dbPromise;
    const trimmed = sql.trim().toUpperCase();
    // Convert Postgres-style $1,$2 params to sqlite '?' placeholders
    const hasDollarParams = /\$\d+/.test(sql);
    if (hasDollarParams) {
      // replace $1, $2 ... with ?
      sql = sql.replace(/\$\d+/g, '?');
    }
    if (trimmed.startsWith('SELECT')) {
      const rows = await db.all(sql, params);
      return { rows };
    } else {
      const res = await db.run(sql, params);
      return { lastID: res.lastID, changes: res.changes };
    }
  }

  async function close() {
    const db = await dbPromise;
    await db.close();
  }

  module.exports = { query, ensureSchema, close };
}
