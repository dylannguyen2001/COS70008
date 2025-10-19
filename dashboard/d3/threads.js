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

    const threads = [...threadsData.threads].sort((a, b) => {
      const nonRiskLabel = "This email appears routine, compliant, and shows no indication of risk or wrongdoing.";
      const aIsNonRisk = a.risk_label_zeroshot === nonRiskLabel;
      const bIsNonRisk = b.risk_label_zeroshot === nonRiskLabel;

      // 1. Prioritize non-risky emails first
      if (aIsNonRisk !== bIsNonRisk) return aIsNonRisk ? 1 : -1;

      // 2. Higher mean_risk next
      const rA = a.mean_risk ?? 0;
      const rB = b.mean_risk ?? 0;
      if (rB !== rA) return rB - rA;

      // 3. Then higher email count
      const nA = a.n_emails ?? 0;
      const nB = b.n_emails ?? 0;
      return nB - nA;
    });

    threads.forEach(t => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><a href="threads_detail.html?id=${t.id}" class="text-decoration-none">${t.id}</a></td>
        <td>${t.risk_label_zeroshot}</td>
        <td>${(t.mean_risk ?? 0).toFixed(2)}</td>
        <td>${t.dominant_sentiment}</td>
        <td>${(t.mean_sentiment_score ?? 0).toFixed(2)}</td>
        <td>${t.dominant_emotion}</td>
        <td>${(t.mean_emotion_score ?? 0).toFixed(2)}</td>
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
