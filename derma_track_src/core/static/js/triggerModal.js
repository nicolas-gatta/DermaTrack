let selectedImageId = null;

function dismissModal() {
    const modalElement = document.getElementById('mainModal');
    const modal = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
    modal.hide();
}

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

async function createFileExplorer(id = null){

    if (id == null){
        id = document.querySelector("#mainModal").dataset.visitId;
    }
    
    document.getElementById("backButton").classList.add("d-none");
    document.getElementById("delete-button-image").classList.add("d-none");

    var fileContainer = document.getElementById("file-container");

    fileContainer.innerHTML = ""
    try{
        await fetch(`/core/visit_list/${id}/folders/`)
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
    }catch (error) {
        console.error("Error fetching folder list:", error);
        return null;
    }
}

async function createFileExplorerImage(id, body_part){

    document.getElementById("backButton").classList.remove("d-none");
    document.getElementById("delete-button-image").classList.remove("d-none");

    var fileContainer = document.getElementById("file-container");
    var deleteButton = document.getElementById("delete-button-image");
 

    fileContainer.innerHTML = ""
    deleteButton.disabled = true;

    try{
        await fetch(`/core/visit_list/${id}/${body_part}/images`)
        .then(response => response.json())
        .then(data => {
            const images = data.images;

            if (images.length != 0){
                images.forEach(image => {
                    const figure = document.createElement("figure");
                    const img = document.createElement("img");
                    const caption = document.createElement("figcaption");
                    
                    img.src = "/media/visits/visit_"+id+"/"+body_part+"/"+image["image_preview_name"];
                    caption.textContent = image["image_name"];

                    figure.classList.add("file-item", "text-center", "svg-wrapper");
                    figure.appendChild(img);
                    figure.appendChild(caption);
                    
                    figure.addEventListener("click", () => {
                        const allFigures = document.querySelectorAll(".file-item");
                        allFigures.forEach(f => f.classList.remove("selected"));

                        if (selectedImageId === image["pk"]) {
                            selectedImageId = null;
                            deleteButton.disabled = true;
                        } else {
                            selectedImageId = image["pk"];
                            figure.classList.add("selected");
                            deleteButton.disabled = false;
                        }
                    });

                    figure.addEventListener("dblclick", () => {
                        showPreview(image["pk"]);
                    });

                    fileContainer.appendChild(figure);

                });
            }else{
                createFileExplorer(id);
            }
        });
    }catch (error) {
        console.error("Error fetching Images list:", error);
        return null;
    }
}

async function showPreview(id) {

    try{
        await fetch(`/core/visit_list/get_image/${id}`)
        .then(response => response.json())
        .then(data => {
            setBackgroundImage(id, data.image, data.pixel_size, data.distance, data.focal);
            document.getElementById("superSwitch").disabled = !data.has_super;
        });

        const modalImage = new bootstrap.Modal(document.querySelector("#imagePreviewModal"), {
            keyboard: false,
            backdrop: "static"
        });

        const modalMain = bootstrap.Modal.getInstance(document.querySelector("#mainModal"));

        modalImage.toggle();

        modalMain.toggle();

    } catch (error) {
        console.error("Error fetching the Image:", error);
        return null;
    }
}

function setBackgroundImage(id, url, pixel_size, distance, focal) {
    document.getElementById("image-preview").style.backgroundImage = `url(data:image/png;base64,${url})`;
    const img = new Image();

    img.onload = () => {
        let canvas = document.getElementById('canvas-annotation');
        let image = document.getElementById("image-preview");
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.dataset.pixel_size = pixel_size;
        canvas.dataset.distance = distance;
        canvas.dataset.focal = focal;
        canvas.dataset.id = id;
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
        image.style.width = img.width + "px";
        image.style.height = img.height + "px";
    }

    img.src = `data:image/png;base64,${url}`;
}

async function deleteImage(){
    if (!selectedImageId) return;

    try{
        await fetch(`/core/visit_list/delete_image/${selectedImageId}/`, {
            method: "DELETE"
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                const visitId = data.visit_id;
                const bodyPart = data.body_part;
                createFileExplorerImage(visitId, bodyPart);
            } else {
                console.error("Failed to delete image");
            }
        });
    } catch (error) {
        console.error("Failed to delete image:", error);
        return null;
    }
}

window.addEventListener("DOMContentLoaded", addEventListener, { once: true });

