export function addSelectOption(items, element) {
    items.forEach(item => {
        let option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        element.appendChild(option);
    });
}