from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://chatgpt.com/c/68ba46f3-4408-8328-b6a9-83164e4e005a")
    page.screenshot(path="68ba46f3-4408-8328-b6a9-83164e4e005a.png", full_page=True)
    browser.close()
