function deleteCanvas(){
  if (document.getElementsByClassName("chart-row").length != 0) {
    document.querySelectorAll(".chart-row").forEach(el => el.remove());
  }
}

document.querySelectorAll(".view-chart-btn").forEach((button) => {
  const currentRow = button.closest("tr");
  button.addEventListener("click", function () {
    if(currentRow.nextElementSibling == null || !currentRow.nextElementSibling.className.includes("chart-row")){

      deleteCanvas();

      const validationLosses = JSON.parse(
        currentRow.querySelector("input[name='validation_losses']").value
      );
      const trainingLosses = JSON.parse(
        currentRow.querySelector("input[name='training_losses']").value
      );
  
      const chartRow = document.createElement("tr");
      chartRow.classList.add("chart-row");
  
      const chartCell = document.createElement("td");
      chartCell.colSpan = 19;
  
      chartCell.innerHTML = `
        <div style="position: relative; margin: 1rem 0; height:20rem">
          <canvas></canvas>
        </div>
      `;
  
      chartRow.appendChild(chartCell);
      currentRow.insertAdjacentElement("afterend", chartRow);

      const ctx = document.querySelector("canvas").getContext("2d");
  
      new Chart(ctx, {
        type: "line",
        data: {
          labels: validationLosses.map((_, idx) => `Epoch ${idx + 1}`),
          datasets: [
            {
              label: "Validation Loss",
              data: validationLosses,
              borderColor: "red",
              fill: false,
              tension: 0.1
            },
            {
              label: "Training Loss",
              data: trainingLosses,
              borderColor: "blue",
              fill: false,
              tension: 0.1
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
    }else{
      deleteCanvas();
    }
  });
});
