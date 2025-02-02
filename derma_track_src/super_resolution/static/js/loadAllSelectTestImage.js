async function loadAllSelectTestImage(){
    
    const myModule = await import("./addSelectOption.js");

    fetch("/super_resolution/get_test_images")
    .then(response => response.json())
    .then(data => {
        myModule.addSelectOption(data.images, document.getElementById("image-select"));
    })
}

loadAllSelectTestImage();