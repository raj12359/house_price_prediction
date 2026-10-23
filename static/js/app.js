/**
 * EstateIQ - Interactive House Price Prediction App
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const form = document.getElementById("predictionForm");
    const predictBtn = document.getElementById("predictBtn");
    const btnText = predictBtn.querySelector(".btn-text");
    const btnLoader = predictBtn.querySelector(".btn-loader");
    const btnIcon = predictBtn.querySelector(".btn-icon");

    const locationSelect = document.getElementById("locationSelect");
    const sqftRange = document.getElementById("sqftRange");
    const sqftInput = document.getElementById("totalSqftInput");
    const sqftDisplay = document.getElementById("sqftDisplay");

    const bhkGroup = document.getElementById("bhkGroup");
    const bhkInput = document.getElementById("bhkInput");

    const bathroomsGroup = document.getElementById("bathroomsGroup");
    const bathroomsInput = document.getElementById("bathroomsInput");

    const ageRange = document.getElementById("ageRange");
    const ageDisplay = document.getElementById("ageDisplay");

    const parkingGroup = document.getElementById("parkingGroup");
    const parkingInput = document.getElementById("parkingInput");

    // Output elements
    const rfPriceVal = document.getElementById("rfPriceVal");
    const rfPriceExact = document.getElementById("rfPriceExact");
    const rateSqftVal = document.getElementById("rateSqftVal");

    const lrPriceVal = document.getElementById("lrPriceVal");
    const lrPriceExact = document.getElementById("lrPriceExact");
    const insightText = document.getElementById("insightText");

    // Format number with commas
    const formatNumber = (num) => new Intl.NumberFormat("en-IN").format(num);

    // 1. Sync Area Slider & Direct Input
    const updateSqft = (val) => {
        val = Math.max(400, Math.min(5000, Number(val)));
        sqftRange.value = val;
        sqftInput.value = val;
        sqftDisplay.textContent = formatNumber(val);
    };

    sqftRange.addEventListener("input", (e) => updateSqft(e.target.value));
    sqftInput.addEventListener("input", (e) => updateSqft(e.target.value));

    // 2. Sync Property Age Slider
    ageRange.addEventListener("input", (e) => {
        const val = e.target.value;
        ageDisplay.textContent = val === "0" ? "Brand New (0 Yrs)" : `${val} Years`;
    });

    // 3. Pill Button Group Helper (BHK & Bathrooms)
    const setupPillGroup = (groupElem, hiddenInput) => {
        const buttons = groupElem.querySelectorAll(".pill-btn");
        buttons.forEach((btn) => {
            btn.addEventListener("click", () => {
                buttons.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                hiddenInput.value = btn.dataset.value;
            });
        });
    };

    setupPillGroup(bhkGroup, bhkInput);
    setupPillGroup(bathroomsGroup, bathroomsInput);

    // 4. Toggle Button Group (Parking)
    const toggleBtns = parkingGroup.querySelectorAll(".toggle-btn");
    toggleBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            toggleBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            parkingInput.value = btn.dataset.value;
        });
    });

    // Helper to manually set pill state
    const setPillActive = (groupElem, hiddenInput, value) => {
        const buttons = groupElem.querySelectorAll(".pill-btn");
        buttons.forEach((btn) => {
            if (btn.dataset.value === String(value)) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
        hiddenInput.value = value;
    };

    // 5. Presets
    const presets = [
        { id: "presetCompact", bhk: 1, sqft: 650, bath: 1, age: 2, loc: "Electronic City" },
        { id: "presetFamily", bhk: 3, sqft: 1650, bath: 3, age: 5, loc: "HSR Layout" },
        { id: "presetLuxury", bhk: 4, sqft: 2800, bath: 4, age: 1, loc: "Indiranagar" }
    ];

    presets.forEach((p) => {
        const el = document.getElementById(p.id);
        if (el) {
            el.addEventListener("click", () => {
                locationSelect.value = p.loc;
                updateSqft(p.sqft);
                setPillActive(bhkGroup, bhkInput, p.bhk);
                setPillActive(bathroomsGroup, bathroomsInput, p.bath);
                ageRange.value = p.age;
                ageDisplay.textContent = `${p.age} Years`;
                // Trigger form prediction immediately
                handlePrediction();
            });
        }
    });

    // 6. Handle Prediction Request
    const handlePrediction = async (e) => {
        if (e) e.preventDefault();

        const loc = locationSelect.value;
        if (!loc) {
            alert("Please select a location from the dropdown first.");
            locationSelect.focus();
            return;
        }

        const payload = {
            location: loc,
            total_sqft: parseFloat(sqftInput.value),
            bhk: parseInt(bhkInput.value),
            bathrooms: parseInt(bathroomsInput.value),
            property_age: parseInt(ageRange.value),
            parking: parseInt(parkingInput.value)
        };

        // Loading state
        btnText.textContent = "Estimating Valuation...";
        btnLoader.style.display = "block";
        btnIcon.style.display = "none";
        predictBtn.disabled = true;

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const result = await res.json();

            if (result.status === "success") {
                const data = result.data;
                const rf = data.predictions.random_forest;
                const lr = data.predictions.linear_regression;
                const rateSqft = data.predictions.price_per_sqft;

                // Format price displays
                const formatCroreLakh = (lakhs) => {
                    if (lakhs >= 100) {
                        return `${(lakhs / 100).toFixed(2)} Cr`;
                    }
                    return `${lakhs.toFixed(2)} Lakh`;
                };

                rfPriceVal.textContent = formatCroreLakh(rf.price_lakhs);
                rfPriceExact.textContent = `₹ ${rf.price_lakhs.toFixed(2)} Lakhs`;

                lrPriceVal.textContent = formatCroreLakh(lr.price_lakhs);
                lrPriceExact.textContent = `₹ ${lr.price_lakhs.toFixed(2)} Lakhs`;

                rateSqftVal.textContent = `₹ ${formatNumber(Math.round(rateSqft))}/sqft`;

                // Insight calculation
                const diffPct = Math.abs((rf.price_lakhs - lr.price_lakhs) / rf.price_lakhs) * 100;
                let insightMsg = `<strong>Model Consensus:</strong> Both models project consistent valuations for ${payload.bhk} BHK in ${loc} with a ${diffPct.toFixed(1)}% variance. `;
                
                if (diffPct < 8) {
                    insightMsg += `High confidence prediction — Linear Regression and Random Forest show tight agreement.`;
                } else {
                    insightMsg += `Random Forest captures localized non-linear premium for this configuration, offering higher precision.`;
                }
                insightText.innerHTML = insightMsg;
            } else {
                alert(`Error: ${result.message}`);
            }
        } catch (err) {
            console.error("Prediction error:", err);
            alert("Failed to communicate with prediction server.");
        } finally {
            btnText.textContent = "Estimate Property Value";
            btnLoader.style.display = "none";
            btnIcon.style.display = "flex";
            predictBtn.disabled = false;
        }
    };

    form.addEventListener("submit", handlePrediction);

    // 7. Tab Switching Logic
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabButtons.forEach((b) => b.classList.remove("active"));
            tabPanes.forEach((p) => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPaneId = btn.dataset.tab;
            const targetPane = document.getElementById(targetPaneId);
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });

    // Run initial prediction on page load with default values
    locationSelect.value = "Indiranagar";
    handlePrediction();
});
