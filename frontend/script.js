const API_BASE_URL = "https://url-answers.vercel.app";

// State
let chats = [];
let currentChatId = null;
let isSending = false;
let idToken = null;

// Elements
const loginScreen = document.getElementById("login-screen");
const loginForm = document.getElementById("login-form");
const loginError = document.getElementById("login-error");
const loginButton = document.getElementById("login-button");
const logoutBtn = document.getElementById("logout-btn");

const app = document.getElementById("app");
const chatListEl = document.getElementById("chat-list");
const chatTitleEl = document.getElementById("chat-title");
const chatSubtitleEl = document.getElementById("chat-subtitle");
const chatMessagesEl = document.getElementById("chat-messages");
const messageInputEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const loadingIndicator = document.getElementById("loading-indicator");
const chatError = document.getElementById("chat-error");
const refreshChatsBtn = document.getElementById("refresh-chats-btn");

const newChatBtn = document.getElementById("new-chat-btn");
const modalBackdrop = document.getElementById("modal-backdrop");
const closeModalBtn = document.getElementById("close-modal-btn");
const urlsInput = document.getElementById("urls-input");
const createChatBtn = document.getElementById("create-chat-btn");
const newChatError = document.getElementById("new-chat-error");

// ---------- Helpers ----------
function setSending(state) {
  isSending = state;
  sendBtn.disabled = state || !messageInputEl.value.trim() || !currentChatId;
  loadingIndicator.classList.toggle("hidden", !state);
}

function autoResizeTextarea(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 130) + "px";
}

function scrollMessagesToBottom() {
  requestAnimationFrame(() => {
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  });
}

function safeJson(response) {
  return response
    .json()
    .catch(() => ({}))
    .then((data) => ({ ok: response.ok, status: response.status, data }));
}

function getAuthHeaders() {
  return idToken ? { Authorization: `Bearer ${idToken}` } : {};
}

// ---------- API ----------
async function apiListChats() {
  const res = await fetch(`${API_BASE_URL}/chats`, {
    method: "GET",
    headers: {
      ...getAuthHeaders(),
    },
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) {
    throw new Error(wrapped.data.detail || "Failed to load chats");
  }
  return wrapped.data.chats || [];
}

async function apiCreateChat(urls) {
  const res = await fetch(`${API_BASE_URL}/process_urls`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ urls }),
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) {
    throw new Error(wrapped.data.detail || "Failed to create chat");
  }
  return wrapped.data;
}

async function apiGetChat(chatId) {
  const res = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(chatId)}`, {
    method: "GET",
    headers: {
      ...getAuthHeaders(),
    },
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) {
    throw new Error(wrapped.data.detail || "Failed to load chat");
  }
  return wrapped.data;
}

async function apiSendMessage(chatId, message) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ chat_id: chatId, message }),
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) {
    throw new Error(wrapped.data.detail || "Failed to send message");
  }
  return wrapped.data; // { response: "..." }
}

async function apiDeleteChat(chatId) {
  const res = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(chatId)}`, {
    method: "DELETE",
    headers: {
      ...getAuthHeaders(),
    },
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) {
    throw new Error(wrapped.data.detail || "Failed to delete chat");
  }
  return wrapped.data;
}

// ---------- Render ----------
function renderChatList() {
  chatListEl.innerHTML = "";

  if (!Array.isArray(chats) || chats.length === 0) {
    const empty = document.createElement("div");
    empty.className = "helper-text";
    empty.textContent = "No chats yet. Create one to get started.";
    chatListEl.appendChild(empty);
    return;
  }

  chats.forEach((chat) => {
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === currentChatId ? " active" : "");

    const titleSpan = document.createElement("span");
    titleSpan.className = "chat-item-title";
    titleSpan.textContent = chat.title || "Research Chat";

    const actions = document.createElement("div");
    actions.className = "chat-item-actions";

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "icon-btn chat-delete-btn";
    deleteBtn.type = "button";
    deleteBtn.title = "Delete chat";
    deleteBtn.textContent = "✕";

    deleteBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      handleDeleteChat(chat.id);
    });

    actions.appendChild(deleteBtn);

    item.appendChild(titleSpan);
    item.appendChild(actions);

    item.addEventListener("click", () => {
      if (chat.id !== currentChatId) {
        loadChat(chat.id);
      }
    });

    chatListEl.appendChild(item);
  });
}

function renderMessages(messages) {
  chatMessagesEl.innerHTML = "";

  if (!messages || messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "helper-text center";
    empty.textContent = "Ask a question about the URLs to start the conversation.";
    chatMessagesEl.appendChild(empty);
    return;
  }

  messages.forEach((msg) => {
    const row = document.createElement("div");
    const isUser = msg.role === "user";
    row.className = "message-row " + (isUser ? "user" : "ai");

    const bubble = document.createElement("div");
    bubble.className = "message-bubble " + (isUser ? "message-user" : "message-ai");
    bubble.textContent = msg.content || msg.text || "";

    row.appendChild(bubble);
    chatMessagesEl.appendChild(row);
  });

  scrollMessagesToBottom();
}

// ---------- Chat actions ----------
async function loadChats() {
  try {
    const data = await apiListChats();
    // Normalize into internal chat objects: { id, title, urls }
    chats = (data || []).map((c) => ({
      id: c.chat_id || c.id,
      title: c.title || c.first_question || null,
      urls: c.urls || [],
    }));
    renderChatList();
  } catch (err) {
    console.error(err);
  }
}

async function loadChat(chatId) {
  chatError.textContent = "";
  setSending(false);
  currentChatId = chatId;
  renderChatList();

  try {
    const data = await apiGetChat(chatId);
    const messages = data.messages || [];

    // Derive title from first user question, if available
    const firstUser = messages.find((m) => m.role === "user");
    const derivedTitle = firstUser?.content ? firstUser.content.trim().slice(0, 80) : "Research Chat";

    chatTitleEl.textContent = derivedTitle;
    chatSubtitleEl.textContent = "Ask detailed questions about these URLs.";
    renderMessages(messages);

    // Update sidebar chat title for this chat
    const idx = chats.findIndex((c) => c.id === chatId);
    if (idx !== -1) {
      chats[idx].title = derivedTitle;
      renderChatList();
    }

    sendBtn.disabled = !messageInputEl.value.trim();
  } catch (err) {
    console.error(err);
    chatError.textContent = err.message || "Unable to load chat.";
  }
}

async function handleSendMessage() {
  const text = messageInputEl.value.trim();
  if (!text || !currentChatId || isSending) return;

  chatError.textContent = "";
  setSending(true);

  // Build current messages from DOM
  const currentMessages = Array.from(chatMessagesEl.querySelectorAll(".message-row")).map(
    (row) => {
      const bubble = row.querySelector(".message-bubble");
      const isUser = row.classList.contains("user");
      return { role: isUser ? "user" : "assistant", content: bubble.textContent };
    }
  );

  // Add user message
  currentMessages.push({ role: "user", content: text });
  renderMessages(currentMessages);
  messageInputEl.value = "";
  autoResizeTextarea(messageInputEl);
  sendBtn.disabled = true;

  // Show temporary "AI is thinking..." bubble
  const thinkingRow = document.createElement("div");
  thinkingRow.className = "message-row ai";
  const thinkingBubble = document.createElement("div");
  thinkingBubble.className = "message-bubble message-ai";
  thinkingBubble.textContent = "AI is thinking...";
  thinkingRow.appendChild(thinkingBubble);
  chatMessagesEl.appendChild(thinkingRow);
  scrollMessagesToBottom();

  try {
    const data = await apiSendMessage(currentChatId, text);
    const reply = data.response || "";
    currentMessages.push({ role: "assistant", content: reply });
    renderMessages(currentMessages);

    // If this is the first user question in the chat, use it as the chat title
    const firstUser = currentMessages.find((m) => m.role === "user");
    if (firstUser && firstUser.content) {
      const derivedTitle = firstUser.content.trim().slice(0, 80);
      chatTitleEl.textContent = derivedTitle;
      // Do NOT update sidebar titles here; let refresh button pull latest
      // titles from the backend via /chats so names change only on refresh.
    }
  } catch (err) {
    console.error(err);
    chatError.textContent = err.message || "Failed to send message.";
    thinkingBubble.textContent = "Something went wrong. Please try again.";
  } finally {
    setSending(false);
  }
}

// ---------- Modal ----------
function openModal() {
  newChatError.textContent = "";
  urlsInput.value = "";
  modalBackdrop.classList.remove("hidden");
  urlsInput.focus();
}

function closeModal() {
  modalBackdrop.classList.add("hidden");
}

async function handleCreateChat() {
  const raw = urlsInput.value.trim();
  if (!raw) {
    newChatError.textContent = "Please enter at least one URL.";
    return;
  }

  const urls = raw
    .split(",")
    .map((u) => u.trim())
    .filter(Boolean);

  if (urls.length === 0) {
    newChatError.textContent = "Please enter at least one valid URL.";
    return;
  }

  createChatBtn.disabled = true;
  newChatError.textContent = "";

  try {
    const data = await apiCreateChat(urls);
    const newChat = {
      id: data.chat_id || data.id,
      title: null,
    };
    if (!Array.isArray(chats)) {
      chats = [];
    }
    chats.unshift(newChat);
    renderChatList();
    closeModal();
    if (newChat.id) {
      await loadChat(newChat.id);
    }
  } catch (err) {
    console.error(err);
    newChatError.textContent = err.message || "Unable to create chat.";
  } finally {
    createChatBtn.disabled = false;
  }
}

async function handleDeleteChat(chatId) {
  if (!chatId) return;
  const confirmed = window.confirm("Delete this chat? This cannot be undone.");
  if (!confirmed) return;

  try {
    await apiDeleteChat(chatId);
    chats = chats.filter((c) => c.id !== chatId);

    if (currentChatId === chatId) {
      currentChatId = null;
      chatTitleEl.textContent = "Welcome";
      chatSubtitleEl.textContent = "Create a new research chat with URLs to begin.";
      chatMessagesEl.innerHTML =
        '<div class="helper-text center">No chat selected. Create a new research chat from the sidebar.</div>';
      sendBtn.disabled = true;
    }

    renderChatList();
  } catch (err) {
    console.error(err);
    alert(err.message || "Failed to delete chat.");
  }
}

// ---------- Auth (Firebase) ----------
async function handleLogin(event) {
  event.preventDefault();
  const formData = new FormData(loginForm);
  const email = formData.get("email").trim();
  const password = formData.get("password").trim();

  if (!email || !password) {
    loginError.textContent = "Please fill in both email and password.";
    return;
  }

  loginError.textContent = "";
  loginButton.disabled = true;
  loginButton.textContent = "Logging in...";

  try {
    const userCredential = await firebase.auth().signInWithEmailAndPassword(email, password);
    const user = userCredential.user;
    idToken = await user.getIdToken();

    loginScreen.classList.add("hidden");
    app.classList.remove("hidden");
    await loadChats();
  } catch (err) {
    console.error(err);
    // Show friendly message instead of raw Firebase error
    loginError.textContent = "Incorrect email or password. Please check your details and try again.";
  } finally {
    loginButton.disabled = false;
    loginButton.textContent = "Continue";
  }
}

async function handleLogout() {
  try {
    await firebase.auth().signOut();
  } catch (err) {
    console.error(err);
  } finally {
    idToken = null;
    chats = [];
    currentChatId = null;
    renderChatList();
    chatMessagesEl.innerHTML =
      '<div class="helper-text center">Signed out. Log in to continue.</div>';
    app.classList.add("hidden");
    loginScreen.classList.remove("hidden");
  }
}

firebase.auth().onAuthStateChanged(async (user) => {
  if (user) {
    idToken = await user.getIdToken();
    loginScreen.classList.add("hidden");
    app.classList.remove("hidden");
    await loadChats();
  } else {
    idToken = null;
    app.classList.add("hidden");
    loginScreen.classList.remove("hidden");
  }
});

// ---------- Event listeners ----------
loginForm.addEventListener("submit", handleLogin);
logoutBtn.addEventListener("click", handleLogout);

messageInputEl.addEventListener("input", () => {
  autoResizeTextarea(messageInputEl);
  sendBtn.disabled = isSending || !messageInputEl.value.trim() || !currentChatId;
});

messageInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleSendMessage();
  }
});

sendBtn.addEventListener("click", handleSendMessage);

newChatBtn.addEventListener("click", openModal);
closeModalBtn.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", (event) => {
  if (event.target === modalBackdrop) {
    closeModal();
  }
});

createChatBtn.addEventListener("click", handleCreateChat);
refreshChatsBtn.addEventListener("click", () => {
  loadChats();
});

// Init
autoResizeTextarea(messageInputEl);

