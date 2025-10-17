fetch("data/threads/Threads_internal_multi_9902.json")
  .then(res => {
    if (!res.ok) throw new Error("Failed to load threads.json");
    return res.json();
  })
  .then(threadsData => {
    const tbody = document.querySelector("#threadTable tbody");

    console.log("Loaded threads:", threadsData?.threads?.length);

    if (!threadsData || !threadsData.threads || threadsData.threads.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;">No thread data found</td></tr>`;
      return;
    }

    const threads = [...threadsData.threads].sort((a, b) => b.mean_risk - a.mean_risk);

    threads.forEach(t => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><a href="threads_detail.html?id=${t.id}" class="text-decoration-none">${t.id}</a></td>
        <td>${t.dominant_emotion}</td>
        <td>${t.mean_risk}</td>
        <td>${t.mean_emotion_score}</td>
        <td>${t.n_emails ?? "?"}</td>
      `;
      tbody.appendChild(tr);
    });
  })
  .catch(err => {
    console.error("Error loading threads.json:", err);
    document.querySelector("#threadTable tbody").innerHTML =
      `<tr><td colspan="5" style="text-align:center;">Error loading threads</td></tr>`;
  });
