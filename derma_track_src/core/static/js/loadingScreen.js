function showLoading(action, text) {
    let overlay = document.getElementById('loading-overlay');

    if (!overlay){
        return
    }
    overlay.classList.remove('d-none');
    document.getElementById('loading-text').innerHTML = action + '<br>'+ text;
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('d-none');
    document.getElementById('loading-text').innerHTML = "";
}