var viewChartButtons = document.querySelectorAll(".view-chart-btn");

viewChartButtons.forEach((button) => {
  button.addEventListener("click", function () {

    const currentRow = this.closest("tr");

    const validationLosses = JSON.parse(
      currentRow.querySelector("input[name='validation_losses']").value
    );
    const trainingLosses = JSON.parse(
      currentRow.querySelector("input[name='training_losses']").value
    );

    const nextRow = currentRow.nextElementSibling;
    if (nextRow && nextRow.classList.contains("chart-row")) {
      nextRow.remove();
      return;
    }

    const chartRow = document.createElement("tr");
    chartRow.classList.add("chart-row");

    const chartCell = document.createElement("td");
    chartCell.colSpan = 9;

    const canvasId = `chart-${Math.random().toString(36).substr(2, 9)}`;

    chartCell.innerHTML = `
      <div style="position: relative; margin: 1rem 0; height:40vh; width:80vw">
        <canvas id="${canvasId}" height="100%"></canvas>
      </div>
    `;

    chartRow.appendChild(chartCell);
    currentRow.insertAdjacentElement("afterend", chartRow);

    const ctx = document.getElementById(canvasId).getContext("2d");

    new Chart(ctx, {
      plugins: [ChartDataLabels],
      type: "line",
      data: {
        labels: validationLosses.map((_, idx) => `Epoch ${idx + 1}`),
        datasets: [
          {
            label: "Validation Loss",
            data: validationLosses,
            borderColor: "red",
            fill: false,
            tension: 0.1,
            datalabels: {
              color: "red",
              align: "end",
              anchor: "end"
            }
          },
          {
            label: "Training Loss",
            data: trainingLosses,
            borderColor: "blue",
            fill: false,
            tension: 0.1,
            datalabels: {
              color: "blue",
              align: "top",
              anchor: "top"
            }
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            title: {
              display: true,
              text: "Loss"
            }
          },
          x: {
            title: {
              display: true,
              text: "Epoch"
            }
          }
        }
      }
    });
  });
});
