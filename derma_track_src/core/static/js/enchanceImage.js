var showEnchancedImage = false;
var canvas = document.getElementById('canvas-annotation');

async function enchancedImage(){
    try {
        showLoading("Enchancing of the image, please wait...", "Feel free to grab a coffee or a nice cup of tea just like our British mates would!");
        await fetch(`/super_resolution/apply_sr/`, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            },
            body : JSON.stringify({visit_body_part_id: canvas.dataset.id})
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                console.error(`Failed to enchanced the image${canvas.dataset.id}: ${data.message}`);
            }else{
                document.getElementById("superSwitch").disabled = false;
            }
        })

    } catch (error) {
        console.error(`Failed to enchanced the image ${canvas.dataset.id}:`, error);
    } finally {
      hideLoading();
    }
}

async function toggleEnchancedImage(){
    showEnchancedImage = !showEnchancedImage;

    url_request = showEnchancedImage ? "/core/visit_list/get_enchanced_image/" : "/core/visit_list/get_image/"
    
    try {
        await fetch(`${url_request}${canvas.dataset.id}`, {
            method: "GET",
            headers: { 
                "Content-Type": "application/json",
                "X-CSRFToken": csrftoken
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'error') {
                console.error(`Failed to get the enchanced image${canvas.dataset.id}: ${data.message}`);
            }else{
                document.getElementById("annotationSwitch").disabled = showEnchancedImage;
                setBackgroundImage(canvas.dataset.id, data.image, data.pixel_size, data.distance, data.focal);
            }
        })

    } catch (error) {
        console.error(`Error get the enchanced image ${canvas.dataset.id}:`, error);
    }
}