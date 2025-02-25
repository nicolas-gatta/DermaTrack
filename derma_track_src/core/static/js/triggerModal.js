document.body.addEventListener('htmx:afterSwap', function (event) {
    if (event.detail.target.id === 'pop-up') {

        modalElement = document.querySelector("#mainModal");

        let modalInstance = new bootstrap.Modal(modalElement, {
            keyboard: false,
            backdrop: "static"
        });

        modalInstance.show();
    }
});


function clearPopUp() {
    document.getElementById('pop-up').innerHTML = "";
}