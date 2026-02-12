import asyncio
from playwright.async_api import async_playwright
import os
import subprocess
import time

async def verify_button_listener():
    # Start the server
    process = subprocess.Popen(['python3', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto('http://localhost:5000')

            # Inject segments so validation passes
            await page.evaluate("""() => {
                window.currentSegments = [{id: 1, start: 0, end: 5, text: 'Test'}];
            }""")

            # Click the button under the video
            btn = await page.query_selector('#playVideoBtn')
            await btn.click()

            # Check if videoPlaying became true (simulated playback fallback)
            playing = await page.evaluate("window.videoPlaying")
            print(f"videoPlaying state after click: {playing}")

            await browser.close()

            if playing:
                print("Verification passed: button has listener and starts playback.")
            else:
                print("Verification failed: button seems unresponsive.")

    finally:
        process.terminate()

if __name__ == "__main__":
    asyncio.run(verify_button_listener())
