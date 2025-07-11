let activeButton = null;

async function loadBodyContentCore(button) {

    if (button.id === "patient_button") {
        htmx.ajax("GET", "patient_list", "#dynamic-body");

    } else if (button.id === "visit_button") {
        htmx.ajax("GET", "visit_list", "#dynamic-body");
    
    } else if (button.id === "doctor_button") {
        htmx.ajax("GET", "doctor_list", "#dynamic-body");
    }
        
}

function changingButtonState(button) {
    
    if(activeButton){
        activeButton.classList.remove("active");
    }

    button.classList.add("active");

    localStorage.setItem("core_activeButton", button.id);

    activeButton = button;
}

function initializeStateCore() {
    button = document.getElementById(localStorage.getItem("core_activeButton"));
    if (button == null){
        button = document.getElementById("patient");
    }else{
        activeButton = button;
    }

    changingButtonState(button);
    loadBodyContentCore(button);
} 

window.addEventListener("DOMContentLoaded", initializeStateCore, { once: true });