import os
import tkinter as tk
from tkinter import messagebox, scrolledtext

import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

GEMINI_MODEL = "gemini-2.5-flash" 
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# Stores the repositories collected during the latest search. This allows the chatbot to analyse the same data without scraping GitHub again for each question.
current_repo_names = []


def scrape_repository_names(username):
    names = []
    more_pages = True
    page = 1
    headers = {"User-Agent": "Mozilla/5.0 (repo-lister)"}

    while more_pages:
        url = f"https://github.com/{username}?tab=repositories&page={page}"
        response = requests.get(url, headers=headers, timeout=15) # The timeout prevents the application from waiting indefinitely if GitHub is unavailable or the network connection is slow.
        response.raise_for_status()

        organisedResponse = BeautifulSoup(response.text, "html.parser")
        repo_links = organisedResponse.find_all(
            "a", attrs={"itemprop": "name codeRepository"}
        )

        if repo_links:
            for link in repo_links:
                names.append(link.get_text(strip=True))
            page += 1
        else:
            more_pages = False

    return names


def save_names_to_excel(names, username):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Repositories"
    sheet["A1"] = "Repository Name"

    for row_index, name in enumerate(names, start=2):
        sheet.cell(row=row_index, column=1, value=name)

    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_folder, f"{username}_repositories.xlsx")
    workbook.save(file_path)
    return file_path


def get_api_key():
    return os.environ.get("GEMINI_API_KEY", "").strip() or api_key_var.get().strip()

import time

# Function to ask Gemini a question about the repo names
def ask_gemini(question, repo_names, api_key, max_retries=4):
    repo_list = "\n".join(repo_names) if repo_names else "(no repositories loaded yet)"
    prompt = (
        "You are helping vet a data scientist by their GitHub repositories.\n"
        f"Here are the repository names:\n{repo_list}\n\n"
        f"Question: {question}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(max_retries):
        response = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        # 503 = Google overloaded; wait a moment and try again.
        if response.status_code == 503 and attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s...
            continue

        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


def on_generate():
    username = username_var.get().strip()

    if not username:
        messagebox.showwarning("Missing input", "Please enter a GitHub username.")
        return

    try:
        names = scrape_repository_names(username)
    except requests.HTTPError:
        messagebox.showerror(
            "Error", f"Could not find user '{username}' (or page unavailable)."
        )
        return

    if not names:
        messagebox.showinfo("No repositories", f"'{username}' has no public repositories.")
        return

    global current_repo_names
    current_repo_names = names

    file_path = save_names_to_excel(names, username)
    messagebox.showinfo(
        "Done", f"Saved {len(names)} repository names to:\n{file_path}"
    )


def append_chat(speaker, text):
    chat_log.config(state="normal")
    chat_log.insert(tk.END, f"{speaker}: {text}\n\n")
    chat_log.config(state="disabled")
    chat_log.see(tk.END)


def on_send():
    question = question_var.get().strip()
    if not question:
        return

    api_key = get_api_key()
    if not api_key:
        messagebox.showwarning(
            "No API key",
            "Add your Gemini API key via the GEMINI_API_KEY environment "
            "variable or the 'Gemini API key' field.",
        )
        return

    append_chat("You", question)
    question_var.set("")

    try:
        answer = ask_gemini(question, current_repo_names, api_key)
    except requests.HTTPError as exc:
        append_chat("Error", f"Gemini request failed: {exc}")
        return
    except requests.RequestException as exc:
        append_chat("Error", f"Network problem: {exc}")
        return

    append_chat("Gemini", answer)



root = tk.Tk()
root.title("GitHub Repository Lister + AI Chat")

username_var = tk.StringVar()
api_key_var = tk.StringVar()
question_var = tk.StringVar()


tk.Label(root, text="GitHub username:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
tk.Entry(root, textvariable=username_var, width=40).grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Generate Excel", command=on_generate).grid(row=0, column=2, padx=5, pady=5)


tk.Label(root, text="Gemini API key:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
tk.Entry(root, textvariable=api_key_var, width=40, show="*").grid(row=1, column=1, padx=5, pady=5)


chat_log = scrolledtext.ScrolledText(root, width=70, height=15, state="disabled", wrap="word")
chat_log.grid(row=2, column=0, columnspan=3, padx=5, pady=5)


tk.Entry(root, textvariable=question_var, width=55).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
tk.Button(root, text="Send", command=on_send).grid(row=3, column=2, padx=5, pady=5)


root.bind("<Return>", lambda event: on_send())

if __name__ == "__main__":
    root.mainloop()