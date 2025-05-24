document.body.addEventListener("htmx:afterRequest", handleModalToggleOnPostSuccess);

function handleModalToggleOnPostSuccess(event) {
    const xhr = event.detail.xhr;

    try {
        const response = JSON.parse(xhr.responseText);
        if (response.success === true) {
            const modalMain = bootstrap.Modal.getInstance(document.querySelector("#mainModal"));
            if (modalMain) {
                modalMain.hide();
            }
            document.querySelector("#pop-up").innerHTML = "";
        }
    } catch (error) {
        
    }
}
