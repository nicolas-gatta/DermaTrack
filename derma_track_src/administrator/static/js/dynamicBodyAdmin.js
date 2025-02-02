let activeButton = null;

/**
 * Function to load body content dynamically based on the state.
 * @param {HTMLElement} button - The button that was clicked.
 */

async function loadBodyContentAdmin(button) {

    if (button.id === "patient") {
        htmx.ajax("GET", "/core/patient_list", "#dynamic-body");

    } else if (button.id === "visit") {
        htmx.ajax("GET", "/core/visit_list", "#dynamic-body");
    
    } else if (button.id === "doctor") {
        htmx.ajax("GET", "/core/doctor_list", "#dynamic-body");

    } else if (button.id === "model_form") {
        htmx.ajax("GET", "/super_resolution/model_form", "#dynamic-body");
    
    }else if (button.id === "show_models") {
        htmx.ajax("GET", "/super_resolution/show_models", "#dynamic-body");
    
    }else if (button.id === "test_model") {
        htmx.ajax("GET", "/super_resolution/test_model", "#dynamic-body");
        
    }else if (button.id == "dataset"){
        htmx.ajax("GET", "/super_resolution/dataset_form", "#dynamic-body");
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

window.addEventListener("DOMContentLoaded", initializeStateAdmin);