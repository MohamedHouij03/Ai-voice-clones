(function () {
  "use strict";

  const micButton = document.getElementById("mic-button");
  const micIcon = document.getElementById("mic-icon");
  const micStatus = document.getElementById("mic-status");
  const conversation = document.getElementById("conversation");
  const errorBanner = document.getElementById("error-banner");
  const player = document.getElementById("player");
  const teacherModal = document.getElementById("teacher-modal");
  const teacherCards = document.getElementById("teacher-cards");
  const levelSelect = document.getElementById("level-select");
  const textForm = document.getElementById("text-fallback-form");
  const textInput = document.getElementById("text-fallback-input");
  const startButton = document.getElementById("start-button");
  const micArea = document.getElementById("mic-area");

  // Mode selection (practice conversation vs. direct sentence generation)
  const modeSelect = document.getElementById("mode-select");
  const switchModeButton = document.getElementById("switch-mode-button");
  const practiceMain = document.getElementById("practice-main");
  const generateMain = document.getElementById("generate-main");

  // Generate-mode ("Générer des phrases") elements
  const generateErrorBanner = document.getElementById("generate-error-banner");
  const generateTeacherSelect = document.getElementById("generate-teacher-select");
  const generateTextInput = document.getElementById("generate-text-input");
  const generateButton = document.getElementById("generate-button");
  const generateStatus = document.getElementById("generate-status");
  const generateResult = document.getElementById("generate-result");
  const generatePlayer = document.getElementById("generate-player");
  const generateDownload = document.getElementById("generate-download");

  let sessionId = null;
  let mediaRecorder = null;
  let recordedChunks = [];
  let state = "ready"; // ready | listening | processing | speaking
  let selectedTeacherId = null;
  let selectedTeacherName = "Professeur";

  function setState(next) {
    state = next;
    micButton.classList.remove("listening", "processing", "speaking");
    micButton.disabled = false;

    if (next === "ready") {
      micIcon.textContent = "🎤";
      micStatus.textContent = "Maintenez pour parler";
    } else if (next === "listening") {
      micButton.classList.add("listening");
      micIcon.textContent = "🔴";
      micStatus.textContent = "Je vous écoute… (relâchez pour envoyer)";
    } else if (next === "processing") {
      micButton.classList.add("processing");
      micButton.disabled = true;
      micIcon.textContent = "…";
      micStatus.textContent = "Je réfléchis…";
    } else if (next === "speaking") {
      micButton.classList.add("speaking");
      micButton.disabled = true;
      micIcon.textContent = "🔊";
      micStatus.textContent = "Le professeur parle…";
    }
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  function clearEmptyState() {
    const empty = conversation.querySelector(".empty-state");
    if (empty) empty.remove();
  }

  function addBubble(speaker, text) {
    clearEmptyState();
    const bubble = document.createElement("div");
    bubble.className = "bubble " + (speaker === "student" ? "student" : "teacher");
    const label = document.createElement("span");
    label.className = "speaker";
    label.textContent = speaker === "student" ? "Vous" : selectedTeacherName;
    bubble.appendChild(label);
    bubble.appendChild(document.createTextNode(text));
    conversation.appendChild(bubble);
    conversation.scrollTop = conversation.scrollHeight;
    return bubble;
  }

  function appendToBubble(bubble, text) {
    bubble.appendChild(document.createTextNode(" " + text));
    conversation.scrollTop = conversation.scrollHeight;
  }

  // Shown when the teacher references a real video from the local catalog
  // (see backend/services/video_library.py) -- a clickable card, not just
  // text, since the teacher only says the title aloud (a spoken URL would be
  // useless) and the actual link needs somewhere to live. `label` lets
  // addVideoRecommendations reuse this same card shape with different
  // wording ("Vidéo du professeur" instead of "Vidéo recommandée").
  function buildVideoCard(video, label) {
    const card = document.createElement("a");
    card.className = "video-card";
    card.href = video.url;
    card.target = "_blank";
    card.rel = "noopener noreferrer";

    if (video.thumbnail_url) {
      const thumb = document.createElement("img");
      thumb.className = "video-card-thumb";
      thumb.src = video.thumbnail_url;
      thumb.alt = "";
      thumb.loading = "lazy";
      card.appendChild(thumb);
    } else {
      const icon = document.createElement("span");
      icon.className = "video-card-icon";
      icon.textContent = "▶";
      card.appendChild(icon);
    }

    const text = document.createElement("span");
    text.className = "video-card-text";
    const labelEl = document.createElement("span");
    labelEl.className = "video-card-label";
    labelEl.textContent = label;
    const title = document.createElement("span");
    title.className = "video-card-title";
    title.textContent = video.title;
    text.appendChild(labelEl);
    text.appendChild(title);
    card.appendChild(text);

    return card;
  }

  function addVideoCard(video) {
    clearEmptyState();
    const card = buildVideoCard(video, "Vidéo recommandée");
    conversation.appendChild(card);
    conversation.scrollTop = conversation.scrollHeight;
    return card;
  }

  // Shown once a topic quiz wraps up (see backend/services/topics.py
  // advance_topic_quiz) -- that teacher's own videos from the catalog, laid
  // out as a small row rather than stacked full-width cards like a single
  // referenced video, since there can be several at once.
  function addVideoRecommendations(videos) {
    if (!videos || !videos.length) return;
    clearEmptyState();
    const row = document.createElement("div");
    row.className = "video-recommendations";
    for (const video of videos) {
      row.appendChild(buildVideoCard(video, "Vidéo du professeur"));
    }
    conversation.appendChild(row);
    conversation.scrollTop = conversation.scrollHeight;
    return row;
  }

  // Sentence audio arrives one at a time as the server generates it (see
  // sendToServer); this queue plays each clip back-to-back in order so the
  // student hears the first sentence within a few seconds instead of waiting
  // for the whole multi-sentence reply to finish synthesizing.
  //
  // Chunk playback and chunk arrival are two independent races: a short first
  // chunk can easily finish PLAYING before the next chunk has even arrived
  // from the server (still being generated). streamEnded tracks whether the
  // server has said it's done sending sentence events at all -- only once
  // that's true AND the queue is empty AND nothing is playing is the teacher
  // actually finished, so only then should the mic re-enable. Without this,
  // the queue draining mid-reply (queue empty, but more is coming) would
  // otherwise be indistinguishable from the teacher truly being done.
  let audioQueue = [];
  let isPlayingQueue = false;
  let streamEnded = true;

  function enqueueSentenceAudio(url) {
    audioQueue.push(url);
    if (!isPlayingQueue) playNextInQueue();
  }

  function playNextInQueue() {
    if (audioQueue.length === 0) {
      isPlayingQueue = false;
      if (streamEnded) setState("ready");
      // else: caught up with playback but the server is still streaming more
      // of the reply -- leave the state (and disabled mic button) as-is
      // rather than flashing "ready" mid-reply; enqueueSentenceAudio will
      // resume playback the moment the next chunk arrives.
      return;
    }
    isPlayingQueue = true;
    setState("speaking");
    const url = audioQueue.shift();
    player.src = url;
    player.hidden = false;
    player.onended = playNextInQueue;
    player.onerror = () => {
      showError("La lecture audio a échoué.");
      playNextInQueue();
    };
    player.play().catch(() => {
      // Autoplay can be blocked until the user interacts with the page; the mic
      // click that triggered this flow should already count as interaction, but
      // fail safe rather than leaving the UI stuck in "speaking".
      showError("Cliquez sur la page puis réessayez pour entendre la réponse.");
      audioQueue.length = 0;
      isPlayingQueue = false;
      setState("ready");
    });
  }

  function selectTeacherCard(id, name) {
    selectedTeacherId = id;
    selectedTeacherName = name;
    for (const card of teacherCards.children) {
      card.classList.toggle("selected", card.dataset.teacherId === id);
    }
    startButton.disabled = false;
  }

  async function loadTeachers() {
    try {
      const res = await fetch("/api/teachers");
      const data = await res.json();
      teacherCards.innerHTML = "";
      generateTeacherSelect.innerHTML = "";
      for (const t of data.teachers) {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "teacher-card";
        card.dataset.teacherId = t.id;

        const avatar = document.createElement("span");
        avatar.className = "avatar";
        avatar.textContent = t.display_name.charAt(0).toUpperCase();
        card.appendChild(avatar);
        card.appendChild(document.createTextNode(t.display_name));

        card.addEventListener("click", () => selectTeacherCard(t.id, t.display_name));
        teacherCards.appendChild(card);

        const option = document.createElement("option");
        option.value = t.id;
        option.textContent = t.display_name;
        generateTeacherSelect.appendChild(option);
      }
      if (data.teachers.length === 0) {
        showError("Aucun professeur configuré (aucun voices/*/reference.wav trouvé).");
      } else {
        const defaultTeacher = data.teachers.find((t) => t.id === data.active_default) || data.teachers[0];
        selectTeacherCard(defaultTeacher.id, defaultTeacher.display_name);
        generateTeacherSelect.value = defaultTeacher.id;
      }
    } catch (e) {
      showError("Impossible de contacter le serveur pour charger la liste des professeurs.");
    }
  }

  function revealConversationArea() {
    teacherModal.hidden = true;
    micArea.hidden = false;
  }

  async function startConversation() {
    setState("processing");

    const form = new FormData();
    form.append("level", levelSelect.value);
    form.append("teacher", selectedTeacherId);

    let data = null;
    try {
      const res = await fetch("/api/conversation/start", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
      data = await res.json();
    } catch (e) {
      // Silent fallback: reveal the mic anyway so the student can still speak/type
      // even if the teacher's greeting audio failed to generate.
    }

    revealConversationArea();

    if (!data) {
      setState("ready");
      return;
    }

    sessionId = data.session_id;
    addBubble("teacher", data.response_text);

    setState("speaking");
    player.src = data.audio_url;
    player.hidden = false;
    player.onended = () => setState("ready");
    player.onerror = () => setState("ready");
    try {
      await player.play();
    } catch (e) {
      // Autoplay is commonly blocked before any user interaction; the mic-button
      // click that led here should already count, but fail safe regardless.
      setState("ready");
    }
  }

  async function sendToServer({ audioBlob, text }) {
    setState("processing");
    clearError();
    streamEnded = false;

    const form = new FormData();
    if (sessionId) form.append("session_id", sessionId);
    form.append("level", levelSelect.value);
    form.append("teacher", selectedTeacherId);
    if (audioBlob) {
      form.append("audio", audioBlob, "recording.webm");
    } else {
      form.append("text", text);
    }

    let res;
    try {
      res = await fetch("/api/conversation/message/stream", { method: "POST", body: form });
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        throw new Error(errText || `Erreur serveur (${res.status})`);
      }
    } catch (e) {
      showError("Erreur : " + e.message);
      streamEnded = true;
      setState("ready");
      return;
    }

    // Server streams newline-delimited JSON events as each sentence finishes
    // synthesizing (see process_message_stream on the backend) instead of one
    // big response at the end -- each "sentence" event is queued for playback
    // as soon as it arrives.
    let teacherBubble = null;
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let newlineIndex;
        while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          if (!line.trim()) continue;
          const event = JSON.parse(line);

          if (event.type === "transcript") {
            sessionId = event.session_id;
            addBubble("student", event.transcript);
          } else if (event.type === "sentence") {
            if (!teacherBubble) teacherBubble = addBubble("teacher", event.text);
            else appendToBubble(teacherBubble, event.text);
            enqueueSentenceAudio(event.audio_url);
          } else if (event.type === "error") {
            showError(event.message);
          } else if (event.type === "done") {
            if (event.video) addVideoCard(event.video);
            if (event.topic_videos) addVideoRecommendations(event.topic_videos);
          }
        }
      }
    } catch (e) {
      showError("Erreur : " + e.message);
    }

    // The server is done sending events; if playback has already caught up
    // (queue empty, nothing playing) the teacher is truly finished. If not,
    // leave the state alone -- playNextInQueue's onended chain will reach
    // "ready" on its own once the last queued clip finishes, now that
    // streamEnded is true.
    streamEnded = true;
    if (!isPlayingQueue && audioQueue.length === 0) {
      setState("ready");
    }
  }

  async function startRecording() {
    clearError();
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      showError("Micro refusé ou indisponible. Vérifiez les autorisations du navigateur, ou écrivez votre message ci-dessous.");
      return;
    }

    if (typeof MediaRecorder === "undefined") {
      showError("Votre navigateur ne prend pas en charge l'enregistrement audio. Essayez Chrome, Edge ou Firefox récents.");
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    recordedChunks = [];
    try {
      mediaRecorder = new MediaRecorder(stream);
    } catch (e) {
      showError("Impossible de démarrer l'enregistrement sur ce navigateur.");
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      if (blob.size < 1000) {
        showError("Enregistrement vide ou trop court. Réessayez.");
        setState("ready");
        return;
      }
      sendToServer({ audioBlob: blob });
    };

    if (!pressActive) {
      // The button was already released while awaiting mic permission/setup
      // above (e.g. a very quick tap, or a slow permission prompt) --
      // onPressEnd already ran and found state still "ready", so it couldn't
      // call stopRecording(). Don't start recording at all, or the mic would
      // be stuck "listening" forever with nothing left to release.
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    mediaRecorder.start();
    setState("listening");
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  }

  // Push-to-talk: hold the button to record, release to send -- rather than
  // click-to-start / click-to-stop. Both mouse and touch are handled since
  // this needs to work on desktop and mobile; mouseleave/touchcancel are
  // included so dragging off the button (or an interrupted touch) still
  // stops recording instead of leaving it stuck listening forever.
  let pressActive = false;

  function onPressStart(e) {
    e.preventDefault();
    if (state !== "ready") return;
    pressActive = true;
    startRecording();
  }
  function onPressEnd(e) {
    e.preventDefault();
    pressActive = false;
    if (state === "listening") stopRecording();
  }
  micButton.addEventListener("mousedown", onPressStart);
  micButton.addEventListener("mouseup", onPressEnd);
  micButton.addEventListener("mouseleave", onPressEnd);
  micButton.addEventListener("touchstart", onPressStart, { passive: false });
  micButton.addEventListener("touchend", onPressEnd);
  micButton.addEventListener("touchcancel", onPressEnd);

  textForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = textInput.value.trim();
    if (!text) return;
    textInput.value = "";
    sendToServer({ text });
  });

  startButton.addEventListener("click", () => {
    startButton.disabled = true;
    startButton.textContent = "Préparation…";
    startConversation();
  });

  // --- Mode selection: practice conversation vs. direct sentence generation ---
  // The two modes are otherwise fully independent (separate backend routes,
  // separate UI regions) -- this just decides which one is visible.

  function enterPracticeMode() {
    modeSelect.hidden = true;
    practiceMain.hidden = false;
    teacherModal.hidden = false;
    switchModeButton.hidden = false;
  }

  function enterGenerateMode() {
    modeSelect.hidden = true;
    generateMain.hidden = false;
    switchModeButton.hidden = false;
  }

  for (const optionButton of document.querySelectorAll(".mode-option")) {
    optionButton.addEventListener("click", () => {
      if (optionButton.dataset.mode === "practice") enterPracticeMode();
      else enterGenerateMode();
    });
  }

  switchModeButton.addEventListener("click", () => {
    // A full reload is the simplest reliable way to reset both modes' state
    // (active recording, session, generated-audio preview, etc.) cleanly,
    // rather than hand-resetting every variable in both code paths.
    if (confirm("Changer de mode ? La session actuelle sera fermée.")) {
      location.reload();
    }
  });

  // --- Generate mode: direct text -> cloned-voice audio (see backend/api/tts.py) ---

  function showGenerateError(message) {
    generateErrorBanner.textContent = message;
    generateErrorBanner.hidden = false;
  }

  function clearGenerateError() {
    generateErrorBanner.hidden = true;
    generateErrorBanner.textContent = "";
  }

  function setGenerateStatus(message) {
    if (message) {
      generateStatus.textContent = message;
      generateStatus.hidden = false;
    } else {
      generateStatus.hidden = true;
      generateStatus.textContent = "";
    }
  }

  // A short, filesystem-safe suggested filename for the download link,
  // derived from the sentence itself rather than the server's random
  // storage filename -- purely a client-side courtesy via <a download>.
  function slugifyForFilename(text) {
    const slug = text
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60);
    return (slug || "phrase") + ".wav";
  }

  generateButton.addEventListener("click", async () => {
    const text = generateTextInput.value.trim();
    clearGenerateError();
    generateResult.hidden = true;

    if (!text) {
      showGenerateError("Écris une phrase à générer.");
      return;
    }
    if (!generateTeacherSelect.value) {
      showGenerateError("Choisis un professeur.");
      return;
    }

    generateButton.disabled = true;
    setGenerateStatus("Génération en cours…");

    const form = new FormData();
    form.append("text", text);
    form.append("teacher", generateTeacherSelect.value);

    try {
      const res = await fetch("/api/tts/generate", { method: "POST", body: form });
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        throw new Error(errText || `Erreur serveur (${res.status})`);
      }
      const data = await res.json();
      generatePlayer.src = data.audio_url;
      generateDownload.href = data.audio_url;
      generateDownload.download = slugifyForFilename(text);
      generateResult.hidden = false;
      setGenerateStatus(null);
    } catch (e) {
      showGenerateError("Erreur : " + e.message);
      setGenerateStatus(null);
    } finally {
      generateButton.disabled = false;
    }
  });

  // Best-effort cleanup when the tab closes/navigates away -- sendBeacon is
  // fire-and-forget and survives page teardown, unlike a normal fetch.
  // URLSearchParams (not a plain string) is required so it serializes as
  // application/x-www-form-urlencoded, which is what the backend's Form(...)
  // parameter parses; a plain string would send as text/plain and silently
  // fail to populate session_id.
  window.addEventListener("pagehide", () => {
    if (sessionId) {
      navigator.sendBeacon("/api/conversation/end", new URLSearchParams({ session_id: sessionId }));
    }
  });

  setState("ready");
  loadTeachers();
})();
