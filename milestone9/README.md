# 🤖 TalentTwin — AI Career Digital Twin
### Milestone 9 | AlgoProfessor Internship 2026

An AI agent that represents YOU to recruiters. It reads your GitHub profile,
answers questions about your skills and projects, and lets recruiters schedule
interviews via Calendly — all automatically.

---

## 📁 Project Structure

```
talentwin/
├── backend/
│   ├── main.py          ← FastAPI server (API endpoints)
│   ├── agent.py         ← AI agent (Gemini + RAG logic)
│   ├── github_loader.py ← GitHub API data extraction
│   ├── memory.py        ← Context builder (profile → AI prompt)
│   └── config.py        ← Environment variables
│
├── frontend/
│   ├── index.html       ← Chat UI layout
│   ├── app.js           ← Chat logic (fetch, display messages)
│   └── styles.css       ← Dark theme styling
│
├── data/
│   └── profile.json     ← Your personal knowledge base (auto-generated)
│
├── requirements.txt     ← Python dependencies
├── render.yaml          ← Render deployment config
├── .env.example         ← Template for your .env file
├── .gitignore           ← Files NOT to commit to GitHub
└── README.md            ← This file
```

---

## ⚡ STEP-BY-STEP LOCAL SETUP (VS Code)

### STEP 1 — Download/Clone the Project

If you received this as a ZIP:
- Extract the ZIP to a folder, e.g. `C:\Projects\talentwin`

If you're cloning from GitHub:
```bash
git clone https://github.com/YOUR_USERNAME/talentwin.git
cd talentwin
```

---

### STEP 2 — Open in VS Code

1. Open VS Code
2. Click **File → Open Folder**
3. Select the `talentwin` folder
4. You should see all files in the Explorer panel on the left

---

### STEP 3 — Open the Terminal in VS Code

Press `` Ctrl + ` `` (backtick key, top-left of keyboard)
This opens a terminal at the bottom of VS Code.

---

### STEP 4 — Create a Virtual Environment

A virtual environment keeps this project's packages separate from other projects.

**On Windows:**
```bash
python -m venv venv
```

**On Mac/Linux:**
```bash
python3 -m venv venv
```

You should see a new `venv/` folder appear in the project.

---

### STEP 5 — Activate the Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

✅ SUCCESS: You'll see `(venv)` at the beginning of your terminal prompt.
e.g. `(venv) C:\Projects\talentwin>`

---

### STEP 6 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: FastAPI, Uvicorn, Google Gemini SDK, Requests, Python-dotenv.
Wait for it to finish (~1-2 minutes).

---

### STEP 7 — Create Your .env File

The `.env` file holds your secret API keys. 

1. In VS Code, right-click the `talentwin` folder → **New File**
2. Name it exactly: `.env` (yes, with the dot, no extension)
3. Paste this content and fill in YOUR values:

```
GEMINI_API_KEY=your_key_here
GITHUB_USERNAME=your_github_username
GITHUB_TOKEN=your_github_token_here
CALENDLY_URL=https://calendly.com/your-name/30min
CORS_ORIGINS=*
```

#### How to get your GEMINI API KEY (FREE):
1. Go to: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)
5. Paste it in your `.env` file

#### How to get your GITHUB TOKEN:
1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Give it a name like "TalentTwin"
4. Check only the **"public_repo"** box
5. Click **"Generate token"** at the bottom
6. Copy it immediately (shown only once!)

---

### STEP 8 — Fill in Your Extra Info in profile.json

Open `data/profile.json` and update the `extra_info` section with your real:
- Education
- Work experience
- Skills and frameworks
- Achievements
- LinkedIn URL

This information supplements your GitHub data to give the AI full context.

---

### STEP 9 — Run the Backend Server

```bash
cd backend
python main.py
```

**Expected output:**
```
==================================================
🚀 TalentTwin AI Agent — Starting Up
==================================================

📡 Auto-fetching GitHub data for: your_username
  ✅ Got profile: Your Name
  ✅ Got 15 repositories
  💾 Saved to data/profile.json

🌐 Server running at: http://localhost:8000
📖 API Docs at:       http://localhost:8000/docs
🔍 Health check:      http://localhost:8000/

Press Ctrl+C to stop

INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### STEP 10 — Run the Frontend

Open a **new terminal tab** in VS Code (click the `+` icon in the terminal panel).

```bash
cd frontend
python -m http.server 3000
```

Now open your browser and go to: **http://localhost:3000**

You should see the TalentTwin chat interface!

---

### STEP 11 — Test the System

Try these example questions in the chat:

| Question | Expected Response |
|----------|-------------------|
| "What are your strongest technical skills?" | Lists skills from profile |
| "Tell me about your best GitHub project" | Describes top repo |
| "What is your work experience?" | Shares experience from extra_info |
| "Are you available for full-time work?" | Gives availability info |
| "How can I schedule an interview?" | Provides Calendly link |
| "What programming languages do you know?" | Lists languages from GitHub |

Click the **"📅 Book a Meeting"** button in the sidebar to test Calendly.

---

## 🚀 DEPLOYMENT (FREE — Make it Public)

### BACKEND → Deploy to Render

**Render is free** for web services (spins down after 15 min of inactivity, restarts on request).

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "[Day-65] TalentTwin: AI Career Digital Twin"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/talentwin.git
   git push -u origin main
   ```

2. **Go to Render:** https://render.com

3. **Click:** "New +" → "Web Service"

4. **Connect GitHub:** Click "Connect account" → authorize Render

5. **Select your repository:** `talentwin`

6. **Configure the service:**
   - Name: `talentwin-backend`
   - Region: Oregon (US West)
   - Branch: `main`
   - Root Directory: *(leave empty)*
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Plan: **Free**

7. **Add Environment Variables** (click "Advanced" → "Add Environment Variable"):
   ```
   GEMINI_API_KEY    = AIzaSy...your_key...
   GITHUB_USERNAME   = your_username
   GITHUB_TOKEN      = ghp_your_token
   CALENDLY_URL      = https://calendly.com/your-name/30min
   CORS_ORIGINS      = *
   ```

8. **Click "Create Web Service"**

9. **Wait 3-5 minutes** for deployment. You'll see "Your service is live" ✅

10. **Copy your backend URL:**  
    It looks like: `https://talentwin-backend.onrender.com`

---

### FRONTEND → Deploy to Vercel

1. **Go to Vercel:** https://vercel.com

2. **Click "Add New" → "Project"**

3. **Import your GitHub repository**

4. **Configure:**
   - Framework Preset: **Other**
   - Root Directory: **frontend**
   - Build Command: *(leave empty)*
   - Output Directory: *(leave empty)*

5. **Click "Deploy"**

6. **Wait 1 minute** → You get a URL like: `https://talentwin.vercel.app`

---

### CONNECT FRONTEND TO DEPLOYED BACKEND

After deploying the backend, update one line in `frontend/app.js`:

```javascript
// Line 21 in app.js — CHANGE THIS:
const BACKEND_URL = "http://localhost:8000";

// TO YOUR RENDER URL:
const BACKEND_URL = "https://talentwin-backend.onrender.com";
```

Then push to GitHub (Vercel auto-deploys):
```bash
git add frontend/app.js
git commit -m "Connect frontend to deployed backend"
git push
```

Also update CORS_ORIGINS in your Render environment variables:
```
CORS_ORIGINS = https://talentwin.vercel.app
```

---

### CALENDLY SETUP

1. Go to: https://calendly.com
2. Create a free account
3. Click **"New Event Type"**
4. Choose **"One-on-One"**
5. Set: Name = "Interview Call", Duration = 30 min
6. Copy your link: `https://calendly.com/your-name/30min`
7. Paste it in your `.env` file as `CALENDLY_URL`
8. Redeploy the backend (or update the env var in Render dashboard)

---

## 🐛 COMMON ERRORS & FIXES

### ❌ "GEMINI_API_KEY not found" or authentication error
- Check your `.env` file exists in the `talentwin/` folder (not inside `backend/`)
- Make sure the key starts with `AIza`
- Restart the server after changing `.env`

### ❌ "Profile not found" in the chat
- Run `POST /refresh` to fetch GitHub data:
  ```bash
  curl -X POST http://localhost:8000/refresh
  ```
  Or visit: http://localhost:8000/docs → /refresh → "Try it out" → Execute

### ❌ CORS Error in browser console
- During local development: `CORS_ORIGINS=*` in `.env`
- After deployment: Set `CORS_ORIGINS` to your exact Vercel URL

### ❌ "GitHub API rate limit exceeded"
- Add your `GITHUB_TOKEN` to the `.env` file
- Without token: 60 requests/hour limit
- With token: 5000 requests/hour

### ❌ Render "Application failed to respond"  
- Render free tier sleeps after 15 min of inactivity
- First request takes 30-60 seconds to wake up
- This is normal behavior on free tier

### ❌ Frontend shows "Backend not connected"
- Make sure `python main.py` is running in the backend folder
- Check no other app is using port 8000
- Check BACKEND_URL in `app.js` matches where your backend is running

---

## ✅ FINAL VERIFICATION CHECKLIST

- [ ] Backend running at http://localhost:8000
- [ ] Health check returns `{"status": "ok"}`
- [ ] Profile loads (name shows in sidebar)
- [ ] Chat responds to questions
- [ ] Calendly button opens scheduling modal
- [ ] Backend deployed to Render (public URL works)
- [ ] Frontend deployed to Vercel (public URL works)
- [ ] Both connected (frontend calls deployed backend)
- [ ] GitHub commit submitted by 5:00 PM

---

## 🛠️ Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| AI Model  | Google Gemini 2.5 Flash | FREE (500 req/day) |
| Backend   | FastAPI + Uvicorn | FREE |
| Frontend  | HTML + CSS + JS | FREE |
| GitHub Data | GitHub REST API | FREE |
| Backend Host | Render | FREE |
| Frontend Host | Vercel | FREE |
| Scheduling | Calendly | FREE |

**Total cost: $0** ✅

---

*Built for AlgoProfessor AI R&D Solutions Internship 2026 — Milestone 9: TalentTwin*
