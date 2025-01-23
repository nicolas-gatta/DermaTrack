let activeButton = null;

/**
 * Function to load body content dynamically based on the state.
 * @param {HTMLElement} button - The button that was clicked.
 */

async function loadBodyContentCore(button) {

    if (button.id === "patient") {
        htmx.ajax("GET", "patient_list", "#dynamic-body");

    } else if (button.id === "visit") {
        htmx.ajax("GET", "visit_list", "#dynamic-body");
    
    } else if (button.id === "doctor") {
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

function initializeStateCore(){
    button = document.getElementById(localStorage.getItem("core_activeButton"));
    if (button == null){
        button = document.getElementById("patient");
    }else{
        activeButton = button;
    }

    changingButtonState(button);
    loadBodyContentCore(button);
} 

// Initialize the state when the page loads
window.addEventListener("DOMContentLoaded", initializeStateCore);