"""
Browser Tools — Playwright actions wrapped as LangGraph-compatible tools.
"""

import json
from typing import Optional
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

# Global browser state — shared across tool calls within one task
_playwright_instance = None
_browser = None
_page: Optional[object] = None


async def get_page():
    """Lazy-init Playwright browser and return the active page."""
    global _playwright_instance, _browser, _page

    # Check if existing page is still usable
    if _page is not None:
        try:
            if not _page.is_closed():
                return _page
        except Exception:
            pass
        _page = None

    from playwright.async_api import async_playwright
    from config import settings

    # Start fresh playwright instance
    _playwright_instance = await async_playwright().start()
    _browser = await _playwright_instance.chromium.launch(
        headless=settings.PLAYWRIGHT_HEADLESS,
        slow_mo=settings.PLAYWRIGHT_SLOW_MO,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = await _browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    _page = await context.new_page()
    logger.info("Browser launched.")
    return _page


async def close_browser():
    """Gracefully close the browser and playwright instance after a task finishes."""
    global _playwright_instance, _browser, _page
    try:
        if _browser:
            await _browser.close()
    except Exception as e:
        logger.warning(f"Error closing browser: {e}")
    try:
        if _playwright_instance:
            await _playwright_instance.stop()
    except Exception as e:
        logger.warning(f"Error stopping playwright: {e}")
    finally:
        _playwright_instance = None
        _browser = None
        _page = None
        logger.info("Browser closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def navigate_to(url: str) -> str:
    """
    Navigate the browser to a given URL.
    Use this as the first step when a URL is provided.

    Args:
        url: Full URL including https:// prefix

    Returns:
        Page title and current URL after navigation
    """
    page = await get_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = await page.title()
        return f"Navigated to: {page.url}\nPage title: {title}"
    except Exception as e:
        return f"Navigation failed: {str(e)}"


@tool
async def click_element(selector: str) -> str:
    """
    Click on an element identified by a CSS selector or text content.
    Try CSS selectors first. For buttons/links with visible text, use:
      text=Submit  or  button:has-text("Login")

    Args:
        selector: CSS selector or Playwright locator string

    Returns:
        Success or error message
    """
    page = await get_page()
    try:
        await page.locator(selector).first.click(timeout=10000)
        await page.wait_for_load_state("domcontentloaded")
        return f"Clicked: {selector}"
    except Exception as e:
        return f"Click failed for '{selector}': {str(e)}"


@tool
async def fill_input(selector: str, value: str) -> str:
    """
    Fill a text input or textarea with the given value.
    Clears existing content before typing.

    Args:
        selector: CSS selector for the input element
        value: Text to type into the input

    Returns:
        Success or error message
    """
    page = await get_page()
    try:
        locator = page.locator(selector).first
        await locator.scroll_into_view_if_needed()
        await locator.clear()
        await locator.fill(value)
        return f"Filled '{selector}' with value (length {len(value)})"
    except Exception as e:
        return f"Fill failed for '{selector}': {str(e)}"


@tool
async def select_dropdown(selector: str, value: str) -> str:
    """
    Select an option from a <select> dropdown element.

    Args:
        selector: CSS selector for the <select> element
        value: The option value or visible label to select

    Returns:
        Success or error message
    """
    page = await get_page()
    try:
        await page.select_option(selector, label=value)
        return f"Selected '{value}' in dropdown '{selector}'"
    except Exception as e:
        try:
            await page.select_option(selector, value=value)
            return f"Selected '{value}' (by value) in '{selector}'"
        except Exception as e2:
            return f"Dropdown selection failed: {str(e2)}"


@tool
async def extract_text(selector: str = "body") -> str:
    """
    Extract visible text from a page element.
    Defaults to entire page body if no selector given.

    Args:
        selector: CSS selector of element to extract text from

    Returns:
        Extracted text (truncated to 3000 chars)
    """
    page = await get_page()
    try:
        text = await page.locator(selector).first.inner_text(timeout=5000)
        return text[:3000] if len(text) > 3000 else text
    except Exception as e:
        return f"Text extraction failed for '{selector}': {str(e)}"


@tool
async def get_page_info() -> str:
    """
    Get current page URL, title, and a summary of visible interactive elements.
    Use this to understand what's on the current page before deciding actions.

    Returns:
        JSON with url, title, and lists of inputs/buttons/links found
    """
    page = await get_page()
    try:
        url = page.url
        title = await page.title()

        inputs = await page.evaluate("""
            () => Array.from(document.querySelectorAll('input, textarea, select'))
              .slice(0, 20)
              .map(el => ({
                tag: el.tagName.toLowerCase(),
                type: el.type || '',
                name: el.name || '',
                id: el.id || '',
                placeholder: el.placeholder || '',
                selector: el.id ? '#' + el.id : (el.name ? `[name="${el.name}"]` : el.tagName.toLowerCase())
              }))
        """)

        buttons = await page.evaluate("""
            () => Array.from(document.querySelectorAll('button, [type="submit"], [type="button"]'))
              .slice(0, 10)
              .map(el => ({text: el.innerText.trim().slice(0, 50), id: el.id || ''}))
        """)

        info = {
            "url": url,
            "title": title,
            "inputs": inputs,
            "buttons": buttons,
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Page info failed: {str(e)}"


@tool
async def take_screenshot() -> str:
    """
    Take a screenshot of the current page for debugging.
    Saves to screenshot.png in the backend directory.

    Returns:
        Confirmation with file path
    """
    page = await get_page()
    try:
        import os
        path = os.path.join(os.path.dirname(__file__), "screenshot.png")
        await page.screenshot(path=path, full_page=False)
        return f"Screenshot saved to {path}"
    except Exception as e:
        return f"Screenshot failed: {str(e)}"


@tool
async def wait_for_element(selector: str, timeout_ms: int = 10000) -> str:
    """
    Wait for an element to appear on the page (useful after navigation or form submission).

    Args:
        selector: CSS selector to wait for
        timeout_ms: Maximum wait time in milliseconds

    Returns:
        Success or timeout message
    """
    page = await get_page()
    try:
        await page.wait_for_selector(selector, timeout=timeout_ms)
        return f"Element '{selector}' is now visible"
    except Exception as e:
        return f"Timeout waiting for '{selector}': {str(e)}"


@tool
async def scroll_page(direction: str = "down", pixels: int = 500) -> str:
    """
    Scroll the page up or down.

    Args:
        direction: 'up' or 'down'
        pixels: Number of pixels to scroll

    Returns:
        Confirmation message
    """
    page = await get_page()
    try:
        scroll_y = pixels if direction == "down" else -pixels
        await page.evaluate(f"window.scrollBy(0, {scroll_y})")
        return f"Scrolled {direction} by {pixels}px"
    except Exception as e:
        return f"Scroll failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    navigate_to,
    click_element,
    fill_input,
    select_dropdown,
    extract_text,
    get_page_info,
    take_screenshot,
    wait_for_element,
    scroll_page,
]
