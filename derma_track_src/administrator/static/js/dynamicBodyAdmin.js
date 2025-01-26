let activeButton = null;

/**
 * Function to load body content dynamically based on the state.
 * @param {HTMLElement} button - The button that was clicked.
 */

async function loadBodyContentAdmin(button) {

    if (button.id === "patient") {
        htmx.ajax("GET", "patient_list", "#dynamic-body");

    } else if (button.id === "visit") {
        htmx.ajax("GET", "visit_list", "#dynamic-body");
    
    } else if (button.id === "doctor") {
        htmx.ajax("GET", "doctor_list", "#dynamic-body");

    } else if (button.id === "model_form") {
        htmx.ajax("GET", "model_form", "#dynamic-body");
    
    }else if (button.id === "show_models") {
        htmx.ajax("GET", "show_models", "#dynamic-body");
    
    }else if (button.id === "test_model") {
        htmx.ajax("GET", "test_model", "#dynamic-body");
    
    }    
}

function changingAdminButtonState(button) {
    
    if(activeButton){
        activeButton.classList.remove("active");
    }

    button.classList.add("active");

    localStorage.setItem("admin_activeButton", button.id);

    activeButton = button;
}

function initializeStateAdmin(){
    button = document.getElementById(localStorage.getItem("admin_activeButton"));
    if (button == null){
        button = document.getElementById("patient");
    }else{
        activeButton = button;
    }

    changingAdminButtonState(button);
    loadBodyContentAdmin(button);
} 

// Initialize the state when the page loads
window.addEventListener("DOMContentLoaded", initializeStateAdmin);