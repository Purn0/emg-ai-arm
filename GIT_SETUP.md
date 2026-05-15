# Git & GitHub setup — beginner walkthrough

This is a one-time setup. After this you'll only ever run three commands to save your work to GitHub: `git add`, `git commit`, `git push`.

---

## 0. Prerequisites

1. Install Git for Windows: https://git-scm.com/download/win — accept all defaults during install.
2. You already have a GitHub account. Good.

Verify Git works. Open **PowerShell** (Win+R → `powershell` → Enter) and run:

```powershell
git --version
```

You should see something like `git version 2.45.2.windows.1`.

---

## 1. Tell Git who you are (one-time)

```powershell
git config --global user.name "Muhtasim Al Ahsan"
git config --global user.email "mahimhassan006@gmail.com"
git config --global init.defaultBranch main
```

Use the same email you use on GitHub so your commits get linked to your profile.

---

## 2. Create the GitHub repo (browser, one-time)

1. Go to https://github.com/new
2. **Repository name:** `emg-ai-arm`
3. **Description:** `Low-cost sEMG-based HMI for multi-DoF robotic arm control. BSc thesis, SUB CSE.`
4. **Public** (recommended — helps scholarship applications). Or Private if you'd rather wait.
5. **Do NOT** check "Add a README", "Add .gitignore", or "Choose a license" — you already have these locally.
6. Click **Create repository**.

GitHub will show a page with setup instructions. Ignore it; use the steps below.

---

## 3. Initialize the local repo (in your project folder)

Open PowerShell, then **cd to your project folder**:

```powershell
cd C:\Users\muhta\Desktop\FinalProject
```

Initialize Git:

```powershell
git init
git branch -M main
```

Add a remote pointing to your GitHub repo (replace `<your-username>` with your GitHub username):

```powershell
git remote add origin https://github.com/<your-username>/emg-ai-arm.git
```

---

## 4. First commit

Stage everything (the `.gitignore` will automatically exclude `.venv`, `.idea`, `__pycache__`):

```powershell
git add .
```

Check what's about to be committed (sanity check):

```powershell
git status
```

You should see `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `Goals.txt`, `emg_ai_arm/` files, etc. **You should NOT see `.venv/`, `.idea/`, or any `__pycache__`.** If you do, stop and tell me — something's wrong with the `.gitignore`.

Commit:

```powershell
git commit -m "Initial commit: software pipeline scaffold + emulator"
```

Push to GitHub:

```powershell
git push -u origin main
```

The first time, a browser window will pop up asking you to authenticate with GitHub. Sign in, click Authorize. After that, Git remembers you.

Refresh your GitHub repo page in the browser — your code should now be live.

---

## 5. Daily workflow (memorize this)

Every time you make changes you want to save:

```powershell
git add .
git commit -m "Short message describing what changed"
git push
```

Examples of good commit messages:

- `Implement bandpass + notch filters`
- `Add ZC and SSC features`
- `GUI: live signal viewer working with emulator`
- `Paper: draft Methods section`

Bad commit messages:

- `update`
- `fix`
- `asdf`

---

## 6. If something goes wrong

**"Author identity unknown"** — you forgot Step 1. Run those `git config` commands.

**"Permission denied (publickey)"** — you're using SSH URL but didn't set up SSH keys. Easier: use the HTTPS URL from Step 3, which is what we set up.

**Accidentally committed `.venv` or large file** — tell me, I'll walk you through `git rm --cached` + amending.

**Want to undo the last commit (before pushing)** — `git reset --soft HEAD~1`

**Want to see what files Git is tracking** — `git ls-files`

---

## 7. Make the repo look professional (do these AFTER first push)

1. Go to your GitHub repo page.
2. Click the gear icon next to "About" (top right).
3. Add a description, website (leave blank if none), and **topics**: `emg`, `prosthetics`, `bci`, `pytorch`, `arduino`, `signal-processing`, `pyqt`.
4. Once you have a working demo, record a 30-second GIF and put it at the top of the README.

That's it. Once your first push is live, send me the repo URL and I'll review it.
