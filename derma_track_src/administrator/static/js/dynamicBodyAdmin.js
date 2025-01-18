let activeButton = null;
let response = null;

/**
 * Function to load body content dynamically based on the state.
 * @param {number|string} state - The state indicating which body content to load.
 * @param {HTMLElement} button - The button that was clicked.
 */

async function loadBodyContentAdmin(state, button) {
    try {
        // Send a GET request to the server to fetch the partial HTML
        if(activeButton){
            activeButton.classList.remove('active');
        }

        switch (state) {
            case 1:
                break;

            case 2:
                break;

            case 3:
                break;
                
            case 4:
                response = await fetch('patient_list');
                break;
            
            case 5:
                response = await fetch('visit_list');
                break;

            case 6:
                response = await fetch('doctor_list');
                break;

            case 7:
                response =  await fetch('doctor_list');
                break;

            default:
                response = await fetch('patient_list');
                break;
        }


        button.classList.add('active');

        localStorage.setItem('admin_currentState', state);

        localStorage.setItem('admin_activeButton', button.id);

        activeButton = button;

        // Update the dynamic-body div with the new content
        document.getElementById('dynamic-body').innerHTML = await response.text();
        
    } catch (error) {
        console.error('Error loading body content:', error);
        document.getElementById('dynamic-body').innerHTML = `<p>Error loading content. Please try again later.</p>`;
    }
}

function initializeStateAdmin(){
    state = parseInt(localStorage.getItem('admin_currentState'));
    button = document.getElementById(localStorage.getItem('admin_activeButton'));
    if (state == null || button == null){
        state = 1;
        button = document.getElementById("train");
    }else{
        activeButton = button;
    }
    loadBodyContentAdmin(state, button);
} 

// Initialize the state when the page loads
window.addEventListener('DOMContentLoaded', initializeStateAdmin);