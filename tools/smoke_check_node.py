#!/usr/bin/env python3
"""Headless smoke-check: ComfyUI must accept PromptLibrary into a workflow.

Drives a headless Chromium against a running ComfyUI, registers the node via
LiteGraph.createNode("PromptLibrary"), and asserts that no JS error fires.
Catches client-side regressions like the bulkBar TDZ bug
(commit 3e829d2) that break onNodeCreated and prevent the node from being
inserted on the Nodes 2.0 frontend.

Requires: playwright + a running ComfyUI on the chosen URL.
    pip install playwright && playwright install chromium

Usage:
    python tools/smoke_check_node.py
    python tools/smoke_check_node.py --url http://127.0.0.1:8188/
"""
import argparse
import asyncio
import sys


PROBE_JS = r"""
async () => {
  const errors = [];
  window.addEventListener("error", (e) => errors.push(String(e.error || e.message)));
  let result;
  try {
    const node = LiteGraph.createNode("PromptLibrary");
    if (!node) return { ok: false, reason: "LiteGraph.createNode returned null — node not registered" };
    window.app.graph.add(node);
    result = {
      ok: true,
      widgetCount: node.widgets.length,
      widgetNames: node.widgets.map(w => w.name),
      promptIdHidden: node.widgets.find(w => w.name === "prompt_id")?.hidden === true,
      size: node.size,
    };
  } catch (e) {
    return { ok: false, error: String(e), stack: e.stack };
  }
  await new Promise(r => setTimeout(r, 1000));
  result.runtimeErrors = errors;
  return result;
}
"""


async def run(url: str) -> int:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await (await browser.new_context(viewport={"width": 1600, "height": 1000})).new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        await page.goto(url, wait_until="networkidle", timeout=60000)
        await page.wait_for_function("() => !!window.app && !!window.app.graph && !!window.LiteGraph", timeout=30000)
        await page.wait_for_timeout(2500)

        result = await page.evaluate(PROBE_JS)
        await browser.close()

    print("result:", result)
    if page_errors:
        print("page-level errors:", page_errors)

    if not result.get("ok"):
        print("FAIL: node could not be added")
        return 1
    if result.get("runtimeErrors"):
        print("FAIL: runtime errors fired after add")
        return 1
    if "prompt_id" not in result.get("widgetNames", []) or "gallery" not in result.get("widgetNames", []):
        print("FAIL: expected prompt_id and gallery widgets")
        return 1
    if not result.get("promptIdHidden"):
        print("FAIL: prompt_id widget should be hidden")
        return 1
    print("OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188/")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.url)))


if __name__ == "__main__":
    main()
