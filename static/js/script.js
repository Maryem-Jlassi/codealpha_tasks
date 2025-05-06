document.getElementById("translation-form").onsubmit = async function (e) {
    e.preventDefault();
    const sourceLang = document.getElementById("source-lang").value;
    const targetLang = document.getElementById("target-lang").value;
    const text = document.querySelector("textarea[name='text']").value;

    if (sourceLang === targetLang) {
        alert("Source and target languages must be different.");
        return;
    }

    const response = await fetch("/translation", {
        method: "POST",
        body: JSON.stringify({ source_lang: sourceLang, target_lang: targetLang, text: text }),
        headers: { "Content-Type": "application/json" },
    });

    const data = await response.json();
    if (data.translated_text) {
        document.getElementById("result").innerHTML = `
            <h3>Translated Text:</h3>
            <p>${data.translated_text}</p>
            <audio controls>
                <source src="${data.audio_url}" type="audio/mpeg">
            </audio>
        `;
    } else {
        alert(data.error || "Translation failed.");
    }
};
document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const chatWindow = document.getElementById("chat-messages");
    const userQueryInput = document.getElementById("user-query");

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const userQuery = userQueryInput.value.trim();
        if (!userQuery) return;

        // Display the user query
        const userMessageDiv = document.createElement("div");
        userMessageDiv.className = "user-message";
        userMessageDiv.innerText = userQuery;
        chatWindow.appendChild(userMessageDiv);

        // Clear the input field
        userQueryInput.value = "";

        // Send the query to the Flask backend
        const response = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: userQuery }),
        });

        const data = await response.json();

        // Display the bot response
        const botMessageDiv = document.createElement("div");
        botMessageDiv.className = "bot-message";

        if (data.answer) {
            botMessageDiv.innerText = data.answer;
        } else {
            botMessageDiv.innerText = "Error: " + (data.error || "Unable to process your query.");
        }

        chatWindow.appendChild(botMessageDiv);

        // Scroll to the latest message
        chatWindow.scrollTop = chatWindow.scrollHeight;
    });
});