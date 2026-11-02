// Minimal browser chat client for Store Chat. Plain JS, no build step, no
// framework: fetches /api/chat, renders messages, and remembers the
// conversation id in localStorage so a page reload resumes the same
// conversation.
(function () {
  "use strict";

  var STORAGE_KEY = "storeChatConversationId";

  var messagesEl = document.getElementById("messages");
  var formEl = document.getElementById("chat-form");
  var inputEl = document.getElementById("message-input");
  var statusEl = document.getElementById("status");
  var submitButton = formEl.querySelector("button[type=\"submit\"]");

  function getConversationId() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return null;
    }
  }

  function setConversationId(id) {
    try {
      window.localStorage.setItem(STORAGE_KEY, id);
    } catch (err) {
      // Storage may be unavailable (e.g. private browsing); the chat still
      // works for the current page load, it just won't resume on reload.
    }
  }

  function appendMessage(role, text) {
    var wrapper = document.createElement("div");
    wrapper.className = "message message-" + role;

    var label = document.createElement("span");
    label.className = "message-role";
    label.textContent = role === "user" ? "You" : "Assistant";

    var body = document.createElement("p");
    body.className = "message-text";
    // Always set as text content, never innerHTML: a message (from the
    // user, or echoed back from the model) is rendered as plain text, never
    // interpreted as HTML/script.
    body.textContent = text;

    wrapper.appendChild(label);
    wrapper.appendChild(body);
    messagesEl.appendChild(wrapper);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("error", Boolean(isError));
  }

  function extractErrorMessage(body) {
    if (body && typeof body.detail === "string") {
      return body.detail;
    }
    return "Something went wrong. Please try again.";
  }

  function sendMessage(message) {
    var payload = { message: message };
    var conversationId = getConversationId();
    if (conversationId) {
      payload.conversation_id = conversationId;
    }

    return fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (response) {
      return response.json().catch(function () {
        return {};
      }).then(function (body) {
        if (!response.ok) {
          throw new Error(extractErrorMessage(body));
        }
        return body;
      });
    });
  }

  formEl.addEventListener("submit", function (event) {
    event.preventDefault();
    var message = inputEl.value.trim();
    if (!message) {
      return;
    }

    appendMessage("user", message);
    inputEl.value = "";
    inputEl.disabled = true;
    submitButton.disabled = true;
    setStatus("Waiting for a response...", false);

    sendMessage(message)
      .then(function (data) {
        setConversationId(data.conversation_id);
        appendMessage("assistant", data.reply);
        setStatus("", false);
      })
      .catch(function (error) {
        setStatus(error.message || "Something went wrong. Please try again.", true);
      })
      .then(function () {
        inputEl.disabled = false;
        submitButton.disabled = false;
        inputEl.focus();
      });
  });
})();
