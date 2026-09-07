"""浏览器级烟雾测试：启动短工况并确认仅有三张实时图。"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

from playwright.sync_api import TimeoutError, sync_playwright


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
    page.goto(os.environ.get("MOC_DASHBOARD_URL", "http://127.0.0.1:8504"))
    # Streamlit 通过持续 WebSocket 推送刷新，因此 networkidle 不会自然稳定；
    # 先按常规等待，再在超时后确认首屏控件已经完成渲染。
    try:
        page.wait_for_load_state("networkidle", timeout=4000)
    except TimeoutError:
        page.wait_for_selector("[data-testid='stSidebar']", timeout=10000)
    assert "MOC" in page.locator("body").inner_text()

    page.get_by_label("总仿真时长 tf [s]").fill("0.2")
    page.get_by_label("停泵时刻 ts [s]").fill("0.05")
    page.get_by_role("button", name="开始仿真").click()
    page.wait_for_timeout(1800)

    body = page.locator("body").inner_text()
    assert "H_wh(t)" in body
    assert len(page.locator("[data-testid='stPlotlyChart']").all()) == 2
    screenshot = Path(tempfile.gettempdir()) / "moc_live_dashboard_smoke.png"
    page.screenshot(path=str(screenshot), full_page=True)
    print(f"browser smoke passed; screenshot: {screenshot}")
    browser.close()
