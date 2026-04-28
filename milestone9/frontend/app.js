/**
 * app.js — Frontend JavaScript for TalentTwin Chat UI
 * =====================================================
 * This file:
 * 1. Loads the candidate profile from backend and fills the sidebar
 * 2. Handles sending messages to the AI (POST /chat)
 * 3. Displays messages in the chat area
 * 4. Manages the Calendly popup modal
 *
 * HOW IT WORKS:
 * - When page loads → fetch profile → fill sidebar
 * - When recruiter types + clicks Send → call POST /chat → show answer
 * - All communication with backend uses fetch() (built-in browser API)
 */

// ══════════════════════════════════════════════════════
// ⚠️  IMPORTANT: UPDATE THIS URL AFTER DEPLOYMENT
//     During local development → http://localhost:8000
//     After deploying to Render → https://your-app-name.onrender.com
// ══════════════════════════════════════════════════════
const BACKEND_URL = "http://localhost:8000";

// Unique ID for this browser session (so conversation memory works)
// Math.random generates something like "session_0.847392..."
const SESSION_ID = "session_" + Math.random().toString(36).substr(2, 9);

// Track if AI is currently responding (prevent double-sending)
let isLoading = false;

// Store Calendly URL after fetching from backend
let calendlyUrl = "";


// ══════════════════════════════════════════════════════
// 🚀 STARTUP — Run when the page loads
// ══════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  loadProfile();       // Fill sidebar with profile data
  fetchCalendlyUrl();  // Load Calendly URL for the button
  
  // Show a welcome message in the chat
  addMessage(
    "ai",
    "👋 Hello! I'm an AI agent representing this candidate. " +
    "I can answer questions about their skills, projects, experience, and more. " +
    "Feel free to ask anything you'd like to know!"
  );
});


// ══════════════════════════════════════════════════════
// 📊 PROFILE LOADING — Fill the sidebar
// ══════════════════════════════════════════════════════

async function loadProfile() {
  /**
   * Fetches profile data from GET /profile and populates the sidebar.
   * If the backend is not running, shows "Could not load profile".
   */
  try {
    const response = await fetch(`${BACKEND_URL}/profile`);
    
    if (!response.ok) {
      // Profile not loaded on backend yet
      document.getElementById("candidate-name").textContent = "TalentTwin Agent";
      document.getElementById("bio-text").textContent = "Profile not yet loaded. Start the backend and call /refresh.";
      return;
    }
    
    const data = await response.json();
    fillSidebar(data);
    
  } catch (error) {
    // Backend is not running at all
    console.error("Could not reach backend:", error);
    document.getElementById("candidate-name").textContent = "TalentTwin AI Agent";
    document.getElementById("bio-text").textContent = "⚠️ Backend not connected. Please start the server.";
  }
}


function fillSidebar(data) {
  /**
   * Takes the profile data object and fills all the sidebar elements.
   * Uses getElementById to find the HTML elements we defined in index.html.
   */
  const profile = data.profile || {};
  const repos   = data.repositories || [];
  const extra   = data.extra_info || {};
  const langs   = data.languages || [];
  
  // ── Name & Avatar ──
  const name = profile.name || "AI Candidate";
  document.getElementById("candidate-name").textContent = name;
  document.getElementById("candidate-title").textContent =
    extra.preferred_roles || "Software Engineer · AI/ML";
  
  // Avatar: first letter of first name + first letter of last name
  const initials = name.split(" ")
    .map(word => word[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);
  document.getElementById("avatar").textContent = initials;
  
  // ── Bio ──
  document.getElementById("bio-text").textContent =
    profile.bio || extra.about_me || "No bio available";
  
  // ── Location ──
  document.getElementById("location-text").textContent =
    profile.location || "Location not specified";
  
  // ── Skills (programming languages + extra skills) ──
  const skillsContainer = document.getElementById("skills-tags");
  const allSkills = [...langs, ...(extra.skills || [])];
  const uniqueSkills = [...new Set(allSkills)].slice(0, 12); // Max 12 tags
  
  skillsContainer.innerHTML = uniqueSkills
    .map(skill => `<span class="tag">${skill}</span>`)
    .join("");
  
  // ── Links ──
  const linksContainer = document.getElementById("links-container");
  const links = [];
  
  if (profile.github_url) links.push({ label: "GitHub", url: profile.github_url, icon: "🐙" });
  if (extra.linkedin)     links.push({ label: "LinkedIn", url: extra.linkedin,   icon: "💼" });
  if (profile.blog)       links.push({ label: "Portfolio", url: profile.blog,    icon: "🌐" });
  
  linksContainer.innerHTML = links
    .map(link => `
      <a href="${link.url}" target="_blank" rel="noopener" class="link-item">
        <span>${link.icon}</span>
        <span>${link.label}</span>
      </a>
    `)
    .join("") || "<p style='color:var(--text-muted); font-size:13px'>No links available</p>";
  
  // ── Top Projects ──
  const projectsList = document.getElementById("projects-list");
  const topRepos = repos.slice(0, 5); // Show top 5
  
  projectsList.innerHTML = topRepos
    .map(repo => `
      <div class="project-item">
        <a href="${repo.url}" target="_blank" rel="noopener" class="project-name">
          ${repo.name}
        </a>
        <p class="project-desc">${truncate(repo.description, 80)}</p>
        <div class="project-meta">
          <span>💻 ${repo.language}</span>
          <span>⭐ ${repo.stars}</span>
        </div>
      </div>
    `)
    .join("") || "<p style='color:var(--text-muted); font-size:13px'>No projects loaded</p>";
}


// ══════════════════════════════════════════════════════
// 💬 CHAT — Send messages and display responses
// ══════════════════════════════════════════════════════

async function sendMessage() {
  /**
   * Called when the user clicks the Send button (or presses Enter).
   * 
   * Flow:
   * 1. Get the question from the textarea
   * 2. Show the user's message in the chat
   * 3. Show a typing indicator (three dots)
   * 4. POST to /chat endpoint
   * 5. Replace typing indicator with the AI's answer
   */
  
  if (isLoading) return; // Prevent sending while AI is still thinking
  
  const input = document.getElementById("question-input");
  const question = input.value.trim();
  
  if (!question) return; // Don't send empty messages
  
  // Clear input and hide suggestions (they only show once)
  input.value = "";
  autoResize(input);
  hideSuggestions();
  
  // Display user's message
  addMessage("user", question);
  
  // Show typing indicator
  const typingId = showTyping();
  
  // Disable send button while loading
  setLoading(true);
  
  try {
    // ── Call the backend /chat endpoint ──
    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",                           // POST request
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({                    // Send as JSON
        question: question,
        session_id: SESSION_ID
      })
    });
    
    if (!response.ok) {
      // Backend returned an error (e.g. 500 Internal Server Error)
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    const data = await response.json();
    
    // Remove typing indicator and show real answer
    removeTyping(typingId);
    addMessage("ai", data.answer);
    
  } catch (error) {
    // Network error or backend crashed
    removeTyping(typingId);
    addMessage("ai", 
      "⚠️ Could not reach the AI agent. Please make sure the backend server is running. " +
      `Error: ${error.message}`
    );
  }
  
  setLoading(false);
}


function sendSuggestion(text) {
  /**
   * When a recruiter clicks a suggestion button, fill the input and send.
   */
  const input = document.getElementById("question-input");
  input.value = text;
  sendMessage();
}


function addMessage(role, content) {
  /**
   * Creates a message bubble and appends it to the chat area.
   * role: "user" (recruiter) or "ai" (assistant)
   * content: the text to display
   */
  const container = document.getElementById("chat-messages");
  
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;
  
  const avatar = role === "user" ? "👤" : "🤖";
  
  // Convert newlines to <br> tags so multi-line responses look right
  const formattedContent = escapeHtml(content).replace(/\n/g, "<br>");
  
  msgDiv.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-bubble">${formattedContent}</div>
  `;
  
  container.appendChild(msgDiv);
  
  // Auto-scroll to the newest message
  container.scrollTop = container.scrollHeight;
}


function showTyping() {
  /**
   * Shows "AI is typing..." animation (three bouncing dots).
   * Returns a unique ID so we can remove it later.
   */
  const container = document.getElementById("chat-messages");
  const id = "typing-" + Date.now();
  
  const typingDiv = document.createElement("div");
  typingDiv.className = "message ai";
  typingDiv.id = id;
  typingDiv.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  
  container.appendChild(typingDiv);
  container.scrollTop = container.scrollHeight;
  
  return id;
}


function removeTyping(typingId) {
  /**
   * Removes the typing indicator by its ID.
   */
  const elem = document.getElementById(typingId);
  if (elem) elem.remove();
}


// ══════════════════════════════════════════════════════
// 📅 CALENDLY — Schedule Interview Modal
// ══════════════════════════════════════════════════════

async function fetchCalendlyUrl() {
  /**
   * Gets the Calendly URL from backend and stores it globally.
   */
  try {
    const response = await fetch(`${BACKEND_URL}/calendly`);
    if (response.ok) {
      const data = await response.json();
      calendlyUrl = data.url;
      document.getElementById("calendly-btn").disabled = false;
    }
  } catch {
    // Backend not running — Calendly button will use fallback
    calendlyUrl = "https://calendly.com"; // Generic fallback
  }
}


function openCalendly() {
  /**
   * Opens the Calendly scheduling modal.
   * Embeds the Calendly widget as an iframe inside the modal.
   */
  const modal = document.getElementById("calendly-modal");
  const container = document.getElementById("calendly-embed-container");
  
  // Build iframe (this is the official Calendly embed method)
  if (calendlyUrl && calendlyUrl !== "https://calendly.com/your-name") {
    container.innerHTML = `
      <iframe 
        src="${calendlyUrl}?embed_domain=localhost&embed_type=Inline"
        width="100%"
        height="100%"
        frameborder="0"
        title="Schedule a meeting"
        style="border:none;"
      ></iframe>
    `;
  } else {
    // No Calendly URL configured yet
    container.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:16px; color:var(--text-secondary);">
        <span style="font-size:48px;">📅</span>
        <p style="font-size:16px; font-weight:600;">Calendly Not Configured</p>
        <p style="font-size:14px; text-align:center; max-width:300px;">
          Set your CALENDLY_URL in the .env file to enable interview scheduling.
        </p>
        <a href="https://calendly.com" target="_blank" 
           style="padding:10px 20px; background:var(--accent); color:white; border-radius:8px; text-decoration:none; font-weight:600;">
          Create Calendly Account →
        </a>
      </div>
    `;
  }
  
  modal.classList.add("open");
}


function closeCalendly(event) {
  /**
   * Closes the Calendly modal.
   * If called from clicking the overlay, only close if clicking outside the modal content.
   */
  const modal = document.getElementById("calendly-modal");
  
  // If clicking the dark overlay itself (not the modal content), close
  if (event && event.target !== modal) return;
  
  modal.classList.remove("open");
  
  // Clear iframe to stop any Calendly scripts running in background
  document.getElementById("calendly-embed-container").innerHTML = "";
}


// ══════════════════════════════════════════════════════
// 🛠️ UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════

function handleKeyDown(event) {
  /**
   * Enter key = send message
   * Shift+Enter = new line (default behavior, we don't interrupt)
   */
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault(); // Prevent actual newline
    sendMessage();
  }
}


function autoResize(textarea) {
  /**
   * Makes the textarea grow taller as the user types more lines.
   * Reset to auto first, then set to scrollHeight so it fits content.
   */
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}


function setLoading(state) {
  /**
   * Enable/disable the send button and update icon.
   */
  isLoading = state;
  const btn = document.getElementById("send-btn");
  const icon = document.getElementById("send-icon");
  
  btn.disabled = state;
  icon.textContent = state ? "⏳" : "➤";
}


function hideSuggestions() {
  /**
   * Hide the suggestion buttons after the first message is sent.
   * They're only meant as a starting point.
   */
  const suggestions = document.getElementById("suggestions");
  if (suggestions) {
    suggestions.style.display = "none";
  }
}


function truncate(text, maxLength) {
  /**
   * Truncate text to maxLength characters, adding "..." if truncated.
   */
  if (!text) return "No description";
  return text.length > maxLength ? text.slice(0, maxLength) + "..." : text;
}


function escapeHtml(text) {
  /**
   * Escape special HTML characters to prevent XSS attacks.
   * e.g. "<script>" becomes "&lt;script&gt;" and won't execute.
   */
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
