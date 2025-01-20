// chart_modal.js
// Must be loaded after Bootstrap & Chart.js in your HTML

// We'll store a reference to the current Chart instance
let myChart = null;

function openChartModal(modelName, validationLosses, trainingLosses) {
  // 1. Destroy existing chart if it exists, to prevent duplicates
  if (myChart) {
    myChart.destroy();
  }

  // 2. Get the canvas context
  const ctx = document.getElementById('chartCanvas').getContext('2d');

  // 3. Create a new Chart.js line chart
  myChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [...Array(validationLosses.length).keys()].map(i => i + 1),
      datasets: [
        {
          label: 'Validation Loss',
          data: validationLosses,
          borderColor: 'rgba(75, 192, 192, 1)',
          fill: false,
          tension: 0.1
        },
        {
          label: 'Training Loss',
          data: trainingLosses,
          borderColor: 'rgba(255, 99, 132, 1)',
          fill: false,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: `Losses for ${modelName}`
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epoch'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Loss'
          }
        }
      }
    }
  });

  // 4. Show the Bootstrap modal
  const chartModal = new bootstrap.Modal(document.getElementById('chartModal'));
  chartModal.show();
}

// On page load, set up the "View Chart" button handlers
document.addEventListener('DOMContentLoaded', () => {
    console.log("coucou");
    const viewChartButtons = document.querySelectorAll('.view-chart-btn');
    viewChartButtons.forEach(btn => {
        btn.addEventListener('click', () => {
        // 1. Get the relevant row's hidden inputs
        const row = btn.closest('tr');
        const valLossEl = row.querySelector('.validation-losses');
        const trainLossEl = row.querySelector('.training-losses');

        // 2. Parse them as arrays from JSON
        let validationLosses = [];
        let trainingLosses = [];
        try {
            validationLosses = JSON.parse(valLossEl.value);
            trainingLosses = JSON.parse(trainLossEl.value);
        } catch (e) {
            console.error('Invalid JSON in hidden fields:', e);
        }

        // 3. Grab the model name from the button
        const modelName = btn.getAttribute('data-modelname') || 'Unknown Model';

        // 4. Open the modal with our chart
        openChartModal(modelName, validationLosses, trainingLosses);
        });
    });
});
