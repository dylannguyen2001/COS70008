// Load the two JSON files
Promise.all([
  fetch("data/threads/Threads_internal_multi_9902.json").then(r => r.json()),
  fetch("data/threads/ThreadText_internal_multi_9902.json").then(r => r.json())
]).then(([threadsData, textData]) => {
  const params = new URLSearchParams(window.location.search);
  const threadId = params.get("id");

  const metaBox = document.getElementById("thread-meta");
  const textBox = document.getElementById("thread-text");

  if (!threadId) {
    metaBox.innerHTML = "<h3>No thread ID provided</h3>";
    textBox.innerHTML = "";
    return;
  }

  // Find metadata
  const threadMeta = threadsData.threads.find(t => t.id === threadId);
  if (!threadMeta) {
    metaBox.innerHTML = `<h3>Thread ${threadId}</h3><p>No metadata found.</p>`;
  } else {

    const participants = (threadMeta.participants && threadMeta.participants.length > 0)
    ? threadMeta.participants
        .map(p => 
          `<a href="node_detail.html?id=${encodeURIComponent(p)}" 
              class="text-decoration-none text-primary me-2">${p}</a>`
        )
        .join(", ")
    : "<span class='text-muted'>None</span>";

    metaBox.innerHTML = `
      <h3>${threadId}</h3>
      <p><b>Risk Label:</b> ${threadMeta.risk_label_zeroshot}</p>
      <p><b>Mean Risk:</b> ${threadMeta.mean_risk.toFixed(2)}</p>
      <p><b>Sentiment Label:</b> ${threadMeta.dominant_sentiment}</p>
      <p><b>Mean Sentiment:</b> ${threadMeta.mean_sentiment_score.toFixed(2)}</p>
      <p><b>Emotion Label:</b> ${threadMeta.dominant_emotion}</p>
      <p><b>Mean Emotion:</b> ${threadMeta.mean_emotion_score.toFixed(2)}</p>
      <p><b>Email Count:</b> ${threadMeta.n_emails}</p>
      <p><b>Participants:</b> ${participants}</p>
    `;
  }

  // Find text content
  const textArray = Array.isArray(textData) ? textData : textData.threads;
  const threadText = textArray.find(t => t.thread_id === threadId);

  //const threadText = textData.threads.find(t => t.thread_id === threadId);
  if (!threadText) {
    textBox.innerHTML = `<p>No text available for this thread.</p>`;
  } else {
    textBox.innerHTML = `
      <h4>${threadText.subject_norm || "(No Subject)"}</h4>
      <hr>
      <p>${threadText.body_concat}</p>
      <hr>
      <p class="">
        Tokens: ${threadText.n_tokens} | Has Text: ${threadText.has_text}
      </p>
    `;
  }
}).catch(err => {
  console.error("Error loading thread details:", err);
  document.getElementById("thread-meta").innerHTML = `<p>Error loading data.</p>`;
});
