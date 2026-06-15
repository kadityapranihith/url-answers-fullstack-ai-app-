const API_BASE_URL = "http://127.0.0.1:8000";
// ─── State ────────────────────────────────────────────────────────────────────
let chats = [];
let currentChatId = null;
let isSending = false;
let idToken = null;
let apiCheckInterval = null;
let isSignUp = false;

// ─── Element refs ─────────────────────────────────────────────────────────────
const loginScreen      = document.getElementById("login-screen");
const loginForm        = document.getElementById("login-form");
const loginError       = document.getElementById("login-error");
const loginButton      = document.getElementById("login-button");
const loginBtnText     = document.getElementById("login-btn-text");
const loginSpinner     = document.getElementById("login-spinner");
const confirmPwGroup   = document.getElementById("confirm-password-group");
const confirmPwInput   = document.getElementById("confirm-password");
const authToggleBtn    = document.getElementById("auth-toggle-btn");
const authToggleText   = document.getElementById("auth-toggle-text");
const loginTitle       = document.querySelector(".login-title");

const app              = document.getElementById("app");
const chatListEl       = document.getElementById("chat-list");
const chatTitleEl      = document.getElementById("chat-title");
const chatSubtitleEl   = document.getElementById("chat-subtitle");
const chatMessagesEl   = document.getElementById("chat-messages");
const welcomeScreen    = document.getElementById("welcome-screen");
const messageInputEl   = document.getElementById("message-input");
const sendBtn          = document.getElementById("send-btn");
const loadingIndicator = document.getElementById("loading-indicator");
const chatErrorBanner  = document.getElementById("chat-error");
const chatErrorText    = document.getElementById("chat-error-text");
const refreshChatsBtn  = document.getElementById("refresh-chats-btn");

const newChatBtn       = document.getElementById("new-chat-btn");
const modalBackdrop    = document.getElementById("modal-backdrop");
const closeModalBtn    = document.getElementById("close-modal-btn");
const cancelModalBtn   = document.getElementById("cancel-modal-btn");
const urlsInput        = document.getElementById("urls-input");
const createChatBtn    = document.getElementById("create-chat-btn");
const createBtnText    = document.getElementById("create-btn-text");
const createSpinner    = document.getElementById("create-spinner");
const newChatError     = document.getElementById("new-chat-error");

const sidebarToggle    = document.getElementById("sidebar-toggle");
const sidebar          = document.getElementById("sidebar");
const logoutBtn        = document.getElementById("logout-btn");

const userAvatarEl     = document.getElementById("user-avatar");
const userNameEl       = document.getElementById("user-name");
const userEmailEl      = document.getElementById("user-email");

// API status elements (sidebar + header)
const apiDotSidebar    = document.getElementById("api-status-dot");
const apiTextSidebar   = document.getElementById("api-status-text");
const apiDotHeader     = document.getElementById("api-dot-header");
const apiLabelHeader   = document.getElementById("api-label-header");

// ─── Helpers ──────────────────────────────────────────────────────────────────
function setSending(state) {
  isSending = state;
  sendBtn.disabled = state || !messageInputEl.value.trim() || !currentChatId;
  loadingIndicator.classList.toggle("hidden", !state);
}

function autoResizeTextarea(el) {
  el.style.height = "auto";
  const sh = el.scrollHeight;
  if (sh > 0) {
    el.style.height = Math.min(sh, 150) + "px";
  } else {
    el.style.height = "";
  }
}

function scrollToBottom() {
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

function showChatError(msg) {
  chatErrorText.textContent = msg;
  chatErrorBanner.classList.remove("hidden");
  setTimeout(() => chatErrorBanner.classList.add("hidden"), 5000);
}

function hideChatError() {
  chatErrorBanner.classList.add("hidden");
}

// ─── API Status Check ─────────────────────────────────────────────────────────
function setApiStatus(status) {
  // status: 'connected' | 'disconnected' | 'checking'
  const labels = {
    connected:    "API Connected",
    disconnected: "API Offline",
    checking:     "Checking API...",
  };
  const headerLabels = {
    connected:    "Connected",
    disconnected: "Offline",
    checking:     "Checking...",
  };

  [apiDotSidebar, apiDotHeader].forEach((dot) => {
    dot.classList.remove("connected", "disconnected", "checking");
    dot.classList.add(status);
  });

  apiTextSidebar.textContent  = labels[status] || status;
  apiLabelHeader.textContent  = headerLabels[status] || status;
}

async function checkApiStatus() {
  setApiStatus("checking");
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    const res = await fetch(`${API_BASE_URL}/`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeout);
    setApiStatus(res.ok ? "connected" : "disconnected");
  } catch {
    setApiStatus("disconnected");
  }
}

function startApiPolling() {
  checkApiStatus();
  if (apiCheckInterval) clearInterval(apiCheckInterval);
  apiCheckInterval = setInterval(checkApiStatus, 30000); // every 30s
}

// ─── API calls ────────────────────────────────────────────────────────────────
async function apiListChats() {
  const res = await fetch(`${API_BASE_URL}/chats`, {
    method: "GET",
    headers: { ...getAuthHeaders() },
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) throw new Error(wrapped.data.detail || "Failed to load chats");
  return wrapped.data.chats || [];
}

async function apiCreateChat(urls) {
  const res = await fetch(`${API_BASE_URL}/process_urls`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify({ urls }),
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) throw new Error(wrapped.data.detail || "Failed to create chat");
  return wrapped.data;
}

async function apiGetChat(chatId) {
  const res = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(chatId)}`, {
    method: "GET",
    headers: { ...getAuthHeaders() },
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) throw new Error(wrapped.data.detail || "Failed to load chat");
  return wrapped.data;
}

async function apiSendMessage(chatId, message) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify({ chat_id: chatId, message }),
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) throw new Error(wrapped.data.detail || "Failed to send message");
  return wrapped.data;
}

async function apiDeleteChat(chatId) {
  const res = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(chatId)}`, {
    method: "DELETE",
    headers: { ...getAuthHeaders() },
  });
  const wrapped = await safeJson(res);
  if (!wrapped.ok) throw new Error(wrapped.data.detail || "Failed to delete chat");
  return wrapped.data;
}

// ─── Render ───────────────────────────────────────────────────────────────────
function renderChatList() {
  chatListEl.innerHTML = "";

  if (!Array.isArray(chats) || chats.length === 0) {
    const empty = document.createElement("div");
    empty.className = "chat-empty";
    empty.textContent = "No chats yet. Create one to get started.";
    chatListEl.appendChild(empty);
    return;
  }

  chats.forEach((chat) => {
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === currentChatId ? " active" : "");
    item.setAttribute("data-chat-id", chat.id);

    // Icon
    const iconEl = document.createElement("div");
    iconEl.className = "chat-item-icon";
    iconEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;

    // Title
    const titleSpan = document.createElement("span");
    titleSpan.className = "chat-item-title";
    titleSpan.textContent = chat.title || "Research Chat";
    titleSpan.title = chat.title || "Research Chat";

    // Delete button
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "chat-delete-btn";
    deleteBtn.type = "button";
    deleteBtn.title = "Delete chat";
    deleteBtn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>`;
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleDeleteChat(chat.id);
    });

    item.appendChild(iconEl);
    item.appendChild(titleSpan);
    item.appendChild(deleteBtn);

    item.addEventListener("click", () => {
      if (chat.id !== currentChatId) loadChat(chat.id);
    });

    chatListEl.appendChild(item);
  });
}

function renderMessages(messages) {
  chatMessagesEl.innerHTML = "";

  if (!messages || messages.length === 0) {
    const empty = document.createElement("div");
    empty.style.cssText = "text-align:center;color:var(--text-muted);font-size:0.85rem;padding:2rem 1rem;";
    empty.textContent = "Ask a question about the URLs to start the conversation.";
    chatMessagesEl.appendChild(empty);
    return;
  }

  messages.forEach((msg) => {
    const isUser = msg.role === "user";
    const row = document.createElement("div");
    row.className = "message-row " + (isUser ? "user" : "ai");

    if (!isUser) {
      const avatar = document.createElement("div");
      avatar.className = "ai-avatar";
      avatar.innerHTML = `<svg width="13" height="13" viewBox="0 0 28 28" fill="none"><path d="M14 2L26 8V20L14 26L2 20V8L14 2Z" stroke="#a78bfa" stroke-width="2" fill="none"/><circle cx="14" cy="14" r="4" fill="#a78bfa"/></svg>`;
      row.appendChild(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = "message-bubble " + (isUser ? "message-user" : "message-ai");
    bubble.textContent = msg.content || msg.text || "";

    row.appendChild(bubble);
    chatMessagesEl.appendChild(row);
  });

  scrollToBottom();
}

// Show / hide the welcome screen vs message list
function showWelcome() {
  welcomeScreen.classList.remove("hidden");
  chatMessagesEl.classList.add("hidden");
  loadingIndicator.classList.add("hidden");
}

function showMessages() {
  welcomeScreen.classList.add("hidden");
  chatMessagesEl.classList.remove("hidden");
}

// ─── Chat Actions ─────────────────────────────────────────────────────────────
async function loadChats() {
  try {
    const data = await apiListChats();
    chats = (data || []).map((c) => ({
      id: c.chat_id || c.id,
      title: c.title || c.first_question || null,
      urls: c.urls || [],
    }));
    renderChatList();
  } catch (err) {
    console.error("loadChats:", err);
  }
}

async function loadChat(chatId) {
  hideChatError();
  setSending(false);
  currentChatId = chatId;
  renderChatList();
  showMessages();

  try {
    const data = await apiGetChat(chatId);
    const messages = data.messages || [];

    const firstUser = messages.find((m) => m.role === "user");
    const derivedTitle = firstUser?.content
      ? firstUser.content.trim().slice(0, 80)
      : "Research Chat";

    chatTitleEl.textContent = derivedTitle;
    chatSubtitleEl.textContent = "Ask questions about these URLs.";
    renderMessages(messages);

    const idx = chats.findIndex((c) => c.id === chatId);
    if (idx !== -1) {
      chats[idx].title = derivedTitle;
      renderChatList();
    }

    autoResizeTextarea(messageInputEl);
    sendBtn.disabled = !messageInputEl.value.trim();
  } catch (err) {
    console.error("loadChat:", err);
    showChatError(err.message || "Unable to load chat.");
  }
}

async function handleSendMessage() {
  const text = messageInputEl.value.trim();
  if (!text || !currentChatId || isSending) return;

  hideChatError();
  setSending(true);
  showMessages();

  const currentMessages = Array.from(chatMessagesEl.querySelectorAll(".message-row")).map((row) => {
    const bubble = row.querySelector(".message-bubble");
    return {
      role: row.classList.contains("user") ? "user" : "assistant",
      content: bubble?.textContent || "",
    };
  });

  currentMessages.push({ role: "user", content: text });
  renderMessages(currentMessages);
  messageInputEl.value = "";
  autoResizeTextarea(messageInputEl);
  sendBtn.disabled = true;
  scrollToBottom();

  try {
    const data = await apiSendMessage(currentChatId, text);
    const reply = data.response || "";
    currentMessages.push({ role: "assistant", content: reply });
    renderMessages(currentMessages);

    const firstUser = currentMessages.find((m) => m.role === "user");
    if (firstUser?.content) {
      chatTitleEl.textContent = firstUser.content.trim().slice(0, 80);
    }
  } catch (err) {
    console.error("sendMessage:", err);
    showChatError(err.message || "Failed to send message. Please try again.");
  } finally {
    setSending(false);
  }
}

// ─── Modal ────────────────────────────────────────────────────────────────────
function openModal() {
  newChatError.textContent = "";
  urlsInput.value = "";
  modalBackdrop.classList.remove("hidden");
  setTimeout(() => urlsInput.focus(), 80);
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

  const urls = raw.split(",").map((u) => u.trim()).filter(Boolean);
  if (urls.length === 0) {
    newChatError.textContent = "Please enter at least one valid URL.";
    return;
  }

  createChatBtn.disabled = true;
  createBtnText.textContent = "Creating...";
  createSpinner.classList.remove("hidden");
  newChatError.textContent = "";

  try {
    const data = await apiCreateChat(urls);
    const newChat = { id: data.chat_id || data.id, title: null };
    if (!Array.isArray(chats)) chats = [];
    chats.unshift(newChat);
    renderChatList();
    closeModal();
    if (newChat.id) await loadChat(newChat.id);
  } catch (err) {
    console.error("createChat:", err);
    newChatError.textContent = err.message || "Unable to create chat.";
  } finally {
    createChatBtn.disabled = false;
    createBtnText.textContent = "Create Chat";
    createSpinner.classList.add("hidden");
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
      chatSubtitleEl.textContent = "Paste URLs and ask anything";
      showWelcome();
      sendBtn.disabled = true;
    }
    renderChatList();
  } catch (err) {
    console.error("deleteChat:", err);
    alert(err.message || "Failed to delete chat.");
  }
}

// ─── User Profile ─────────────────────────────────────────────────────────────
function updateUserProfile(user) {
  if (!user) return;
  const email = user.email || "";
  const displayName = user.displayName || email.split("@")[0] || "User";
  const initials = displayName.slice(0, 1).toUpperCase();

  userAvatarEl.textContent = initials;
  userNameEl.textContent   = displayName;
  userEmailEl.textContent  = email;
  userEmailEl.title        = email;
}

// ─── Sidebar Toggle ───────────────────────────────────────────────────────────
let sidebarOpen = true;

function toggleSidebar() {
  const isMobile = window.innerWidth <= 780;
  if (isMobile) {
    sidebar.classList.toggle("mobile-open");
  } else {
    sidebarOpen = !sidebarOpen;
    sidebar.classList.toggle("collapsed", !sidebarOpen);
  }
}

// ─── Auth (Firebase) ──────────────────────────────────────────────────────────
function toggleAuthMode() {
  isSignUp = !isSignUp;
  loginError.textContent = "";

  if (isSignUp) {
    confirmPwGroup.classList.remove("hidden");
    confirmPwInput.setAttribute("required", "");
    loginBtnText.textContent = "Create Account";
    authToggleText.textContent = "Already have an account?";
    authToggleBtn.textContent = "Sign In";
    loginTitle.textContent = "Create Account";
  } else {
    confirmPwGroup.classList.add("hidden");
    confirmPwInput.removeAttribute("required");
    confirmPwInput.value = "";
    loginBtnText.textContent = "Sign In";
    authToggleText.textContent = "Don't have an account?";
    authToggleBtn.textContent = "Create Account";
    loginTitle.textContent = "URL-Answers";
  }
}

async function handleLogin(event) {
  event.preventDefault();
  const formData = new FormData(loginForm);
  const email    = formData.get("email").trim();
  const password = formData.get("password").trim();

  if (!email || !password) {
    loginError.textContent = "Please fill in both email and password.";
    return;
  }

  if (isSignUp) {
    const confirmPw = confirmPwInput.value.trim();
    if (!confirmPw) {
      loginError.textContent = "Please confirm your password.";
      return;
    }
    if (password !== confirmPw) {
      loginError.textContent = "Passwords do not match.";
      return;
    }
    if (password.length < 6) {
      loginError.textContent = "Password must be at least 6 characters.";
      return;
    }
  }

  loginError.textContent = "";
  loginButton.disabled   = true;
  loginBtnText.textContent = isSignUp ? "Creating account..." : "Signing in...";
  loginSpinner.classList.remove("hidden");

  try {
    let userCredential;
    if (isSignUp) {
      userCredential = await firebase.auth().createUserWithEmailAndPassword(email, password);
    } else {
      userCredential = await firebase.auth().signInWithEmailAndPassword(email, password);
    }
    const user = userCredential.user;
    idToken = await user.getIdToken();
    updateUserProfile(user);

    loginScreen.classList.add("hidden");
    app.classList.remove("hidden");
    autoResizeTextarea(messageInputEl);
    startApiPolling();
    await loadChats();
  } catch (err) {
    console.error("auth:", err);
    const code = err.code || "";
    if (code === "auth/email-already-in-use") {
      loginError.textContent = "This email is already registered. Try signing in instead.";
    } else if (code === "auth/weak-password") {
      loginError.textContent = "Password is too weak. Use at least 6 characters.";
    } else if (code === "auth/invalid-email") {
      loginError.textContent = "Please enter a valid email address.";
    } else if (isSignUp) {
      loginError.textContent = "Could not create account. Please try again.";
    } else {
      loginError.textContent = "Incorrect email or password. Please check your details and try again.";
    }
  } finally {
    loginButton.disabled = false;
    loginBtnText.textContent = isSignUp ? "Create Account" : "Sign In";
    loginSpinner.classList.add("hidden");
  }
}

async function handleLogout() {
  try {
    await firebase.auth().signOut();
  } catch (err) {
    console.error("logout:", err);
  } finally {
    idToken = null;
    chats   = [];
    currentChatId = null;

    if (apiCheckInterval) {
      clearInterval(apiCheckInterval);
      apiCheckInterval = null;
    }

    renderChatList();
    showWelcome();
    chatTitleEl.textContent    = "Welcome";
    chatSubtitleEl.textContent = "Paste URLs and ask anything";
    app.classList.add("hidden");
    loginScreen.classList.remove("hidden");
  }
}

firebase.auth().onAuthStateChanged(async (user) => {
  if (user) {
    idToken = await user.getIdToken();
    updateUserProfile(user);
    loginScreen.classList.add("hidden");
    app.classList.remove("hidden");
    autoResizeTextarea(messageInputEl);
    startApiPolling();
    await loadChats();
  } else {
    idToken = null;
    app.classList.add("hidden");
    loginScreen.classList.remove("hidden");
    setApiStatus("checking");
  }
});

// ─── Event Listeners ──────────────────────────────────────────────────────────
loginForm.addEventListener("submit", handleLogin);
logoutBtn.addEventListener("click", handleLogout);
authToggleBtn.addEventListener("click", toggleAuthMode);

messageInputEl.addEventListener("input", () => {
  autoResizeTextarea(messageInputEl);
  sendBtn.disabled = isSending || !messageInputEl.value.trim() || !currentChatId;
});

messageInputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSendMessage();
  }
});

sendBtn.addEventListener("click", handleSendMessage);

newChatBtn.addEventListener("click", openModal);
closeModalBtn.addEventListener("click", closeModal);
cancelModalBtn.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", (e) => {
  if (e.target === modalBackdrop) closeModal();
});
createChatBtn.addEventListener("click", handleCreateChat);
refreshChatsBtn.addEventListener("click", loadChats);
sidebarToggle.addEventListener("click", toggleSidebar);

// Close sidebar on outside click (mobile)
document.addEventListener("click", (e) => {
  if (
    window.innerWidth <= 780 &&
    sidebar.classList.contains("mobile-open") &&
    !sidebar.contains(e.target) &&
    e.target !== sidebarToggle
  ) {
    sidebar.classList.remove("mobile-open");
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
autoResizeTextarea(messageInputEl);
