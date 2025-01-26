document.body.addEventListener('htmx:afterSwap', function(event) {
    if (event.detail.target.id === 'pop-up') {
        const modalElement = document.querySelector('#visitModal');
        const modalInstance = new bootstrap.Modal(modalElement);
        modalInstance.show();
    }
});
