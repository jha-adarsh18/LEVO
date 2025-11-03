Follow these steps to contribute safely without breaking `main`.

---

### 🔹 One-time setup
```bash
# Clone the repo (only once)
git clone git@github.com:jha-adarsh18/PEVSLAM.git
cd PEVSLAM

# Create and switch to your personal branch
# Adarsh:
git checkout -b adarsh
git push -u origin adarsh

# Bhavisha:
git checkout -b bhavisha
git push -u origin bhavisha

🔹 Everyday workflow

cd PEVSLAM

# 2. Switch to your branch
git checkout adarsh     # if you are Adarsh
git checkout bhavisha   # if you are Bhavisha

# 3. Pull the latest updates from GitHub
git pull origin <your_branch>

# 4. Work on your code...
#    (edit files, test, etc.)

# 5. Stage and commit your changes
git add .
git commit -m "update: short description of changes"

# 6. Push your work to GitHub
git push

🔹 Getting updates from the other person’s branch
# Example: Bhavisha wants Adarsh's work
git checkout bhavisha
git pull origin adarsh
