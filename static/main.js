document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");
    const resultDiv = document.getElementById("result");
    const predictedClassEl = document.getElementById("predicted-class");
    const predictedProbEl = document.getElementById("predicted-probability");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const submitBtn = form.querySelector(".submit-btn");
        submitBtn.disabled = true;
        submitBtn.textContent = "Predicting…";
        resultDiv.classList.add("hidden");

        // Remove previous error if any
        const prevError = form.parentElement.querySelector(".error-msg");
        if (prevError) prevError.remove();

        const payload = {
            age: parseInt(document.getElementById("age").value, 10),
            ever_married: document.getElementById("ever_married").checked ? "Yes" : "No",
            smoking_status: document.getElementById("smoking_status").value,
            heart_disease: document.getElementById("heart_disease").checked ? 1 : 0,
            hypertension: document.getElementById("hypertension").checked ? 1 : 0,
        };

        try {
            const res = await fetch("/v1/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Prediction failed");
            }

            const data = await res.json();
            predictedClassEl.textContent = data.predicted_class;
            predictedClassEl.className = "predicted-class " +
                (data.predicted_class === "Stroke" ? "stroke" : "no-stroke");
            predictedProbEl.textContent = (data.predicted_probability * 100).toFixed(2) + "%";
            resultDiv.classList.remove("hidden");
        } catch (err) {
            const errorEl = document.createElement("p");
            errorEl.className = "error-msg";
            errorEl.textContent = err.message;
            form.parentElement.appendChild(errorEl);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Predict";
        }
    });
});
