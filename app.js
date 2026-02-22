const API_BASE = 'http://localhost:5000';

let actualData = [];
let predictions = [];
let demandChart = null;
let temperatureChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadData();
});

function initializeEventListeners() {
    document.getElementById('refreshBtn').addEventListener('click', loadData);
    document.getElementById('exportBtn').addEventListener('click', exportToCSV);
    document.getElementById('zoneSelect').addEventListener('change', updateCharts);
}

async function loadData() {
    try {
        console.log('Fetching data from backend...');
        
        // Fetch actual data
        const response = await fetch(`${API_BASE}/api/water-demand`);
        actualData = await response.json();
        
        // Fetch predictions
        try {
            const predResponse = await fetch(`${API_BASE}/api/predictions`);
            predictions = await predResponse.json();
        } catch (e) {
            console.log('Predictions endpoint not available yet');
            predictions = [];
        }
        
        updateUI();
        updateCharts();
        updateTable();
        updateLastUpdate();
    } catch (error) {
        console.error('Error loading data:', error);
        alert('Failed to load data. Ensure backend is running on http://localhost:5000');
    }
}

function getFilteredData() {
    const zone = document.getElementById('zoneSelect').value;
    if (!zone) return actualData;
    return actualData.filter(d => d.zone_id === zone);
}

function updateUI() {
    const filtered = getFilteredData();
    
    if (filtered.length === 0) return;
    
    const demands = filtered.map(d => d.demand_liters);
    const avgDemand = (demands.reduce((a, b) => a + b, 0) / demands.length).toFixed(0);
    const peakDemand = Math.max(...demands);
    const minDemand = Math.min(...demands);
    
    document.getElementById('avgDemand').textContent = formatNumber(avgDemand);
    document.getElementById('peakDemand').textContent = formatNumber(peakDemand);
    document.getElementById('minDemand').textContent = formatNumber(minDemand);
    
    // Calculate simple accuracy
    if (predictions.length > 0) {
        const errors = filtered.map(d => {
            const pred = predictions.find(p => p.date === d.date && p.zone_id === d.zone_id);
            if (pred) return Math.abs(d.demand_liters - pred.predicted_demand) / d.demand_liters;
            return 0;
        }).filter(e => e > 0);
        
        const accuracy = errors.length > 0 ? (100 - (errors.reduce((a, b) => a + b) / errors.length * 100)).toFixed(1) : 'N/A';
        document.getElementById('accuracy').textContent = accuracy;
    }
}

function updateCharts() {
    const filtered = getFilteredData();
    const dates = filtered.map(d => d.date).slice(0, 30);
    
    // Demand Chart
    const demands = filtered.map(d => d.demand_liters).slice(0, 30);
    const predictedDemands = predictions
        .filter(p => dates.includes(p.date))
        .map(p => p.predicted_demand);
    
    if (demandChart) {
        demandChart.destroy();
    }
    
    const ctx1 = document.getElementById('demandChart').getContext('2d');
    demandChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Actual Demand',
                    data: demands,
                    borderColor: '#1e40af',
                    backgroundColor: 'rgba(30, 64, 175, 0.1)',
                    tension: 0.3,
                    fill: true,
                    pointRadius: 3
                },
                {
                    label: 'Predicted Demand',
                    data: predictedDemands,
                    borderColor: '#dc2626',
                    backgroundColor: 'rgba(220, 38, 38, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.3,
                    fill: false,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Demand (Liters)'
                    }
                }
            }
        }
    });
    
    // Temperature Chart
    const temperatures = filtered.map(d => d.avg_temp_c).slice(0, 30);
    
    if (temperatureChart) {
        temperatureChart.destroy();
    }
    
    const ctx2 = document.getElementById('temperatureChart').getContext('2d');
    temperatureChart = new Chart(ctx2, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Temperature vs Demand',
                    data: filtered.slice(0, 30).map((d, i) => ({
                        x: d.avg_temp_c,
                        y: d.demand_liters
                    })),
                    backgroundColor: 'rgba(34, 197, 94, 0.6)',
                    borderColor: '#16a34a',
                    borderWidth: 1,
                    pointRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Demand (Liters)'
                    }
                }
            }
        }
    });
}

function updateTable() {
    const filtered = getFilteredData().slice(0, 20);
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    
    filtered.forEach(row => {
        const pred = predictions.find(p => p.date === row.date && p.zone_id === row.zone_id);
        const errorPercent = pred ? ((Math.abs(row.demand_liters - pred.predicted_demand) / row.demand_liters * 100).toFixed(2)) : 'N/A';
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.date}</td>
            <td>${row.zone_id}</td>
            <td>${formatNumber(row.demand_liters)}</td>
            <td>${pred ? formatNumber(pred.predicted_demand) : 'N/A'}</td>
            <td>${errorPercent}%</td>
            <td>${row.avg_temp_c}°C</td>
        `;
        tbody.appendChild(tr);
    });
}

function updateLastUpdate() {
    const now = new Date().toLocaleString();
    document.getElementById('lastUpdate').textContent = now;
}

function exportToCSV() {
    const filtered = getFilteredData();
    let csv = 'Date,Zone,Demand,Temperature,Rainfall,Population\n';
    
    filtered.forEach(row => {
        csv += `${row.date},${row.zone_id},${row.demand_liters},${row.avg_temp_c},${row.rainfall_mm},${row.population}\n`;
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `water_demand_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(parseInt(num));
}
