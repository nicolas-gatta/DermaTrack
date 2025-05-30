let activeButton = null;

/**
 * Function to load body content dynamically based on the state.
 * @param {HTMLElement} button - The button that was clicked.
 */

async function loadBodyContentAdmin(button) {

    if (button.id === "patient_button") {
        htmx.ajax("GET", "/core/patient_list", "#dynamic-body");

    } else if (button.id === "visit_button") {
        htmx.ajax("GET", "/core/visit_list", "#dynamic-body");
    
    } else if (button.id === "doctor_button") {
        htmx.ajax("GET", "/core/doctor_list", "#dynamic-body");

    } else if (button.id === "model_form_button") {
        htmx.ajax("GET", "/super_resolution/model_form", "#dynamic-body");
    
    }else if (button.id === "show_models_button") {
        htmx.ajax("GET", "/super_resolution/show_models", "#dynamic-body");
    
    }else if (button.id === "test_model_button") {
        htmx.ajax("GET", "/super_resolution/test_model", "#dynamic-body");
    }
    else if (button.id === "dataset_button"){
        htmx.ajax("GET", "/super_resolution/dataset_form", "#dynamic-body");

    }else if (button.id === "evaluation_button"){
        htmx.ajax("GET", "/super_resolution/evaluation_form", "#dynamic-body");
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