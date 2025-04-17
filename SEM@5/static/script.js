document.getElementById('question-form').addEventListener('submit', async function (e) {
    e.preventDefault(); // Prevent form submission

    const question = document.getElementById('question').value;
    const outputElement = document.getElementById('output');

    // Clear previous results
    outputElement.textContent = 'Loading...';

    // Send the question to the server
    const response = await fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
    });

    if (response.ok) {
        const data = await response.json();
        // Display context and answer without labels
        outputElement.innerHTML = `${data.context} ${data.answer}`;
    } else {
        outputElement.textContent = 'Error: Could not retrieve answer.';
    }
});
