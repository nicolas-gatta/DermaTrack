var selectElement = document.getElementById("login-form");

selectElement.addEventListener("submit", (event) => {
    event.preventDefault();

    form = new FormData(event.target)

    for (let pair of form.entries()) {
        console.log(pair[0] + ": " + pair[1]);
    }

    fetch('/', {
        method: 'POST',
        headers: {
            "X-CSRFToken": form["csrfmiddlewaretoken"],
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        },
        body:{
            "login": form["login"],
            "password" : form["password"]
        }
    })
    .then(res => res.json())
    .then(data => document.getElementById("message").innerHTML = data);
})



