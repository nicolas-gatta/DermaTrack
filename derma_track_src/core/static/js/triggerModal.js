document.body.addEventListener('htmx:afterSwap', function (event) {
    if (event.detail.target.id === 'pop-up') {
        const modalElement = document.querySelector('#visitModal');

        if (modalElement) {
            if (!modalElement.dataset.modalInitialized) {
                modalElement.dataset.modalInitialized = true;

                const modalInstance = new bootstrap.Modal(modalElement);
                modalInstance.show();

                modalElement.addEventListener('hidden.bs.modal', () => {
                    modalElement.dataset.modalInitialized = false;
                });
            }
        }
    }
});
