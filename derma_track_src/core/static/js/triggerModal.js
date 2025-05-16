function addEventListener(){
    document.body.addEventListener('htmx:afterSwap', handleModal)
}

function handleModal(event){
    if (event.detail.target.id === 'pop-up') {

        modalElement = document.querySelector("#mainModal");

        let modalInstance = new bootstrap.Modal(modalElement, {
            keyboard: false,
            backdrop: "static"
        });

        modalInstance.show();

        const visitId = modalElement.dataset.visitId;

        if (visitId){
            createFileExplorer(visitId);
        }
    }
}


function clearPopUp() {
    document.getElementById('pop-up').innerHTML = "";
}

function createFileExplorer(id){

    document.getElementById("backButton").classList.add("d-none");

    var fileContainer = document.getElementById("file-container");

    fileContainer.innerHTML = ""

    fetch(`/core/visit_list/${id}/folders/`)
        .then(response => response.json())
        .then(data => {
            const folders = data.folders;

            folders.forEach(folderName => {
                const figure = document.createElement("figure");
                const img_1 = document.createElement("img");
                const img_2 = document.createElement("img");
                const caption = document.createElement("figcaption");
                
                img_1.src = "/static/images/"+folderName+".svg";
                img_1.classList.add("body-icon");

                img_2.src = "/static/images/folder.svg";
                img_2.classList.add("folder-icon");

                caption.textContent = folderName;

                figure.classList.add("file-item", "text-center", "svg-wrapper");
                figure.appendChild(img_1);
                figure.appendChild(img_2);
                figure.appendChild(caption);

                figure.addEventListener("dblclick", () => {
                    createFileExplorerImage(id, folderName);
                  });

                fileContainer.appendChild(figure);

            });
        })
        .catch(error => {
            console.error("Error fetching folder list:", error);
        });

}

function createFileExplorerImage(id, body_part){

    document.getElementById("backButton").classList.remove("d-none");

    var fileContainer = document.getElementById("file-container");

    fileContainer.innerHTML = ""

    fetch(`/core/visit_list/${id}/${body_part}/images`)
        .then(response => response.json())
        .then(data => {
            const images = data.images;

            images.forEach(imageName => {
                const figure = document.createElement("figure");
                const img = document.createElement("img");
                const caption = document.createElement("figcaption");
                
                img.src = "/media/visits/visit_"+id+"/"+body_part+"/"+imageName;

                caption.textContent = imageName;

                figure.classList.add("file-item", "text-center", "svg-wrapper");
                figure.appendChild(img);
                figure.appendChild(caption);

                figure.addEventListener("dblclick", () => {
                    showPreview(img.src);
                });

                fileContainer.appendChild(figure);

            });
        })
        .catch(error => {
            console.error("Error fetching folder list:", error);
        });
}

function showPreview(src) {

    // const previewImage = document.getElementById("preview-image");

    // previewImage.src = src;

    setBackgroundImage(src);

    const modalImage = new bootstrap.Modal(document.querySelector("#imagePreviewModal"), {
        keyboard: false,
        backdrop: "static"
    });

    const modalMain = bootstrap.Modal.getInstance(document.querySelector("#mainModal"));

    modalImage.toggle();

    modalMain.toggle();
}

function setBackgroundImage(url) {
    document.getElementById("image-preview").style.backgroundImage = `url('${url}')`;
    const img = new Image();

    img.onload = () => {
        let init_canvas = document.getElementById('canvas-annotation');
        init_canvas.width = img.width;
        init_canvas.height = img.height;
    }

    img.src = url;
}

window.addEventListener("DOMContentLoaded", addEventListener, { once: true });

