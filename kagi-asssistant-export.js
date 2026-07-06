// Kagi Assistant exporter.
// Paste into the browser console on https://assistant.kagi.com

(async () => {
  const CONVERSATION_PAGE_LIMIT = 100;
  const REQUEST_DELAY_MS = 150;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const assertOnAssistantSite = () => {
    if (location.origin !== "https://assistant.kagi.com") {
      throw new Error(
        `Open https://assistant.kagi.com first. Current origin is ${location.origin}`,
      );
    }
  };

  const apiGet = async (path) => {
    const response = await fetch(path, {
      credentials: "include",
      headers: { Accept: "application/json" },
    });

    const data = await response.json();

    if (!response.ok) {
      const message =
        data?.error?.message || response.statusText || "Request failed";
      throw new Error(`${path} failed with ${response.status}: ${message}`);
    }

    return data;
  };

  const fetchConversationList = async () => {
    const conversations = [];
    const seen = new Set();
    let afterId = null;
    let page = 1;

    while (true) {
      const params = new URLSearchParams({
        limit: String(CONVERSATION_PAGE_LIMIT),
      });
      if (afterId) params.set("after_id", afterId);

      const data = await apiGet(`/api/conversations?${params}`);
      const items = Array.isArray(data?.items) ? data.items : [];

      for (const item of items) {
        if (!item?.uuid || seen.has(item.uuid)) continue;
        seen.add(item.uuid);
        conversations.push(item);
      }

      console.log(
        `Fetched conversation page ${page}: ${items.length} items, ${conversations.length} total`,
      );

      if (!data?.has_more || items.length === 0) break;

      afterId = items.at(-1)?.uuid;
      if (!afterId) {
        console.warn("Pagination stopped because the last item had no uuid.");
        break;
      }

      page += 1;
      await sleep(REQUEST_DELAY_MS);
    }

    return conversations;
  };

  const fetchConversationExport = async (conversation, index, total) => {
    console.log(
      `Exporting ${index + 1}/${total}: ${conversation.title || conversation.uuid}`,
    );
    const init = await apiGet(`/api/conversations/${conversation.uuid}/init`);

    return {
      conversation,
      init,
    };
  };

  const downloadExportData = (data) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    const dateStamp = new Date().toISOString().slice(0, 10);
    anchor.download = `${dateStamp}-kagi-assistant-export.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    console.log(
      `Download complete: ${data.conversations.length} conversation exports.`,
    );
  };

  const fetchAllConversationData = async () => {
    assertOnAssistantSite();

    const init = await apiGet("/api/init");
    const conversations = await fetchConversationList();
    const conversationExports = [];

    for (let index = 0; index < conversations.length; index += 1) {
      const conversation = conversations[index];
      conversationExports.push(
        await fetchConversationExport(
          conversation,
          index,
          conversations.length,
        ),
      );

      await sleep(REQUEST_DELAY_MS);
    }

    return {
      init,
      conversations: conversationExports,
    };
  };

  const allData = await fetchAllConversationData();
  downloadExportData(allData);
})();
