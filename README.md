# 🧠 GitHub Codebase RAG Assistant

A local, privacy-first Desktop Application that allows you to "chat" with any GitHub repository. It downloads the code, indexes it into a vector database, and uses AI (Local Qwen) to explain the architecture, debug logic, and map out the file structure.

---

## 🚀 What is this project?
This tool solves the problem of jumping into a massive, undocumented GitHub repository. Instead of manually reading dozens of files to figure out how a project works, you simply paste the GitHub URL into this app. The app reads the code, maps it out, and gives you a ChatGPT-like interface to ask technical questions about the codebase.

It features **Intent Detection**, meaning it knows the difference between you asking *"List all files"* (which triggers a fast directory tree generation) and *"Explain the authentication logic"* (which triggers a deep AI code analysis).

## ⚙️ How does it work?
1. **Fetch**: Connects to GitHub via their API and downloads all text/code files in the repository (skipping junk like `node_modules` or `.git`).
2. **Chunk**: Cuts the massive codebase into smaller, readable pieces (about 300 tokens each) so they can fit into the AI's brain.
3. **Embed**: Converts these chunks of text into numbers (vectors) using the Jina AI API or a local MiniLM model.
4. **Store**: Saves these numbers into a local database on your hard drive called **ChromaDB**.
5. **Chat**: When you ask a question, it finds the most relevant chunks of code and hands them to the AI to answer your question.

---

## 🤔 What is RAG? (Retrieval-Augmented Generation)
Standard AI models (like ChatGPT) are trained on data from the past, meaning they don't know anything about your private, personal, or brand-new code. 

**RAG** is a technique that fixes this:
* **Retrieval:** Before the AI answers your question, a search engine *retrieves* the exact files and code snippets related to your question from a database.
* **Augmented:** It *augments* (adds to) your prompt by pasting that code invisibly in the background.
* **Generation:** The AI then *generates* an answer by reading both your question and the retrieved code at the same time. 

Because of RAG, the AI doesn't have to guess or hallucinate—it answers by looking directly at your actual source code!

---

## 📂 Project File Manifest

Currently, this entire self-contained project consists of exactly **20 files**. 

### Are all these files necessary?
**Yes, they are absolutely necessary for the AI to work!** 

While 20 files might sound like a lot, you actually only have **6 actual project files** (your python script, your batch launcher, the LLM model, the encrypted key file, and your two markdown readmes). That is incredibly minimal for an entire AI application!

The other **14 files** are strictly database files that ChromaDB creates automatically. Here is why it needs them:

When the AI searches your code, it isn't doing a basic `CTRL+F` word search. It is doing complex vector math. To do this at lightning speed, ChromaDB uses a specialized algorithm called **HNSW** (Hierarchical Navigable Small World).

It is physically impossible for the database to store this mathematical web inside a single text file. Instead, every time you ingest a new repository, it has to generate 4 specialized binary (`.bin`) files:
1. One to hold the raw text (`data_level0.bin`).
2. One to hold the 3D mathematical coordinates of that text.
3. One to hold the "web" of links connecting similar pieces of code together (`link_lists.bin`).
4. One to track the size of the web (`header.bin` and `length.bin`).

If you delete even one of those `.bin` files, the database will corrupt, and the AI will completely lose its memory of that repository.

The good news is that **you never have to look at or manage these files!** The database handles them completely invisibly in the background. If you ever want to delete a repository, just click the "❌ Delete" button in the UI, and the app will automatically clean up those files for you (including the custom `_manifest.json` file so absolutely zero junk is left behind).

### Complete File Breakdown

**Root Folder (The 6 Project Files):**
*   `app.py` — The core application. Contains the UI, GitHub fetcher, AI prompt builder, and Vector search logic.
*   `run.bat` — The Windows batch script you double-click to easily launch the app.
*   `Qwen3.5-4B.q4_k_m.gguf` — The actual brain of the operation. This is your 4-Billion parameter Local AI model compressed into a single file.
*   `.env.enc` — Your securely AES-encrypted API keys (GitHub, Jina) so they aren't stored in plain text.
*   `README.md` — The file you are reading right now.
*   `CODE_BREAKDOWN.md` — An exhaustive, line-by-line annotated explanation of how `app.py` was built.

**Inside `chroma_db/` (The 14 Database Files):**
*   `chroma.sqlite3` — The master database tracking what repositories are indexed.
*   `..._manifest.json` — A custom lightweight text list of files for instant retrieval.
*   `2e2d5fa3.../`, `5109915b.../`, `f713f34b.../` — These three folders represent the three repositories you have indexed, each containing the 4 `.bin` files described above!
