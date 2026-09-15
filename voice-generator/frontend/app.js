(function () {
  "use strict";

  const voiceSelect = document.getElementById("voice-select");
  const speedSelect = document.getElementById("speed-select");
  const exaggerationSelect = document.getElementById("exaggeration-select");
  const scriptInput = document.getElementById("script-input");
  const charCount = document.getElementById("char-count");
  const generateButton = document.getElementById("generate-button");
  const errorBanner = document.getElementById("error-banner");
  const resultArea = document.getElementById("result-area");
  const player = document.getElementById("player");
  const resultStats = document.getElementById("result-stats");
  const downloadLink = document.getElementById("download-link");

  const MAX_CHARS = 5000;

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  async function loadVoices() {
    try {
      const res = await fetch("/api/voices");
      const data = await res.json();
      voiceSelect.innerHTML = "";
      for (const v of data.voices) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = v.display_name;
        voiceSelect.appendChild(opt);
      }
      if (data.voices.length === 0) {
        showError("No voices configured (no voices/*/reference.wav found).");
      }
    } catch (e) {
      showError("Could not reach the server to load the voice list.");
    }
  }

  scriptInput.addEventListener("input", () => {
    charCount.textContent = `${scriptInput.value.length} / ${MAX_CHARS}`;
  });

  generateButton.addEventListener("click", async () => {
    const text = scriptInput.value.trim();
    if (!text) {
      showError("Enter some script text first.");
      return;
    }

    clearError();
    generateButton.disabled = true;
    generateButton.textContent = "Generating…";
    resultArea.hidden = true;

    const form = new FormData();
    form.append("voice", voiceSelect.value);
    form.append("speed", speedSelect.value);
    form.append("exaggeration", exaggerationSelect.value);
    form.append("text", text);

    try {
      const res = await fetch("/api/generate", { method: "POST", body: form });
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        throw new Error(errText || `Server error (${res.status})`);
      }
      const data = await res.json();

      player.src = data.audio_url;
      downloadLink.href = data.audio_url;
      resultStats.textContent =
        `${data.sentence_count} sentence(s) · ${data.duration_seconds}s of audio · ` +
        `generated in ${data.generation_seconds}s · pace ${data.speed}x · expressiveness ${data.exaggeration}`;
      resultArea.hidden = false;
    } catch (e) {
      showError("Error: " + e.message);
    } finally {
      generateButton.disabled = false;
      generateButton.textContent = "Generate";
    }
  });

  loadVoices();
})();
