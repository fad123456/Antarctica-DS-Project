import os
import tkinter as tk
from tkinter import messagebox

import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook



def scrape_repository_names(username):

    names = []
    more_pages = True
    page = 1
    headers = {"User-Agent": "Mozilla/5.0 (repo-lister)"}  # User-Agent identifies the request instead of sending an anonymous default Python request, which GitHub seems to reject.


    while more_pages: # GitHub displays repositories across multiple pages, so each page is requested until no more repository links are found.
        url = f"https://github.com/{username}?tab=repositories&page={page}"
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Stop immediately if GitHub returns an error such as 404 or 500.

        organisedResponse = BeautifulSoup(response.text, "html.parser") # BeautifulSoup converts the HTML into a structure that can be searched.
        repo_links = organisedResponse.find_all("a", attrs={"itemprop": "name codeRepository"}) # GitHub marks repository-name links with this itemprop value

        if repo_links:
            for link in repo_links:
                names.append(link.get_text(strip=True))
            page += 1
        else:
            more_pages = False # An empty result indicates that all repository pages were checked.

    return names


def save_names_to_excel(names, username):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Repositories"
    sheet["A1"] = "Repository Name"

    # Add a heading so the spreadsheet remains understandable on its own.
    for row_index, name in enumerate(names, start=2):
        sheet.cell(row=row_index, column=1, value=name)

    # Saving to Downloads gives the user a predictable place to find the file.
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_folder, f"{username}_repositories.xlsx")
    workbook.save(file_path)
    return file_path


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

    file_path = save_names_to_excel(names, username)
    messagebox.showinfo(
        "Done", f"Saved {len(names)} repository names to:\n{file_path}"
    )


# Create a small graphical interface for GUI so the program can be used without # entering commands or editing the source code.
root = tk.Tk()
root.title("GitHub Repository Lister")

# This variable connects the username entry box to the program logic.
username_var = tk.StringVar()

tk.Label(root, text="GitHub username:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
tk.Entry(root, textvariable=username_var, width=40).grid(row=0, column=1, padx=5, pady=5)
# Clicking the button starts the complete validation, scraping, and export process through the on_generate event handler.
tk.Button(root, text="Generate Excel", command=on_generate).grid(
    row=2, column=1, pady=10
)

# This check prevents the GUI from opening automatically if this file is imported into another Python program.
if __name__ == "__main__":
    root.mainloop()
