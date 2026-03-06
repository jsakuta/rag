/**
 * Playwright script to take 7 screenshots for answer_support_ui_guide.docx
 *
 * Screenshots:
 *   as_01 - as_04: No API required (UI-only operations)
 *   as_05 - as_07: API required (search execution)
 *
 * Query 1: 積立定期の一部解約の払戻請求書の住所は訂正できない？
 * Query 2: 自己宛小切手口に睡眠対象取扱番号があります。これは伝票をたどって発行者に通知状を発送するべきでしょうか？
 * Business area: 内部事務 (naibujimu)
 */
const { chromium } = require('playwright');
const path = require('path');

const SCREENSHOTS_DIR = path.join(__dirname, 'screenshots');
const BASE_URL = 'http://localhost:8502';
const VIEWPORT = { width: 1280, height: 720 };

const QUERY_1 = '積立定期の一部解約の払戻請求書の住所は訂正できない？';
const QUERY_2 = '自己宛小切手口に睡眠対象取扱番号があります。これは伝票をたどって発行者に通知状を発送するべきでしょうか？';

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitForStreamlit(page) {
  await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 60000 });
  await sleep(3000);
}

async function selectDropdown(page, label, value) {
  const selectBox = page.locator(`[data-testid="stSelectbox"]:has-text("${label}")`);
  await selectBox.locator('[data-baseweb="select"]').click();
  await sleep(500);
  await page.locator(`[role="option"]:has-text("${value}")`).click();
  await sleep(2000);
}

async function fillFormInput(page, text) {
  const input = page.locator('[data-testid="stForm"] input[type="text"]');
  await input.fill(text);
  await sleep(500);
}

async function submitForm(page) {
  const btn = page.locator('[data-testid="stFormSubmitButton"] button');
  await btn.click();
}

async function waitForSearchResults(page, timeoutMs = 180000) {
  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    const cards = await page.locator('.response-card').count();
    if (cards > 0) {
      await sleep(3000);
      return true;
    }
    await sleep(2000);
  }
  return false;
}

async function screenshot(page, name) {
  const filepath = path.join(SCREENSHOTS_DIR, `${name}.png`);
  await page.screenshot({ path: filepath, clip: { x: 0, y: 0, ...VIEWPORT } });
  console.log(`  Saved: ${name}.png`);
}

async function freshLoad(page) {
  await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 120000 });
  await waitForStreamlit(page);
}

(async () => {
  console.log('Starting answer_support screenshot capture...\n');

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  try {
    // ========================================
    // GROUP 1: UI-only screenshots (as_01 - as_04)
    // ========================================
    console.log('=== GROUP 1: UI-only screenshots ===');
    await freshLoad(page);

    // Wait for processor initialization spinner to complete
    console.log('Waiting for search engine initialization...');
    await page.waitForSelector('[data-testid="stForm"]', { timeout: 300000 });
    await sleep(2000);

    // --- as_01: Initial launch ---
    console.log('Taking as_01_launch...');
    await screenshot(page, 'as_01_launch');

    // --- as_02: Business area dropdown ---
    console.log('Taking as_02_business_area...');
    // Click the business area selectbox to open dropdown
    const bizSelect = page.locator('[data-testid="stSelectbox"]:has-text("業務分野")');
    await bizSelect.locator('[data-baseweb="select"]').click();
    await sleep(1000);
    await screenshot(page, 'as_02_business_area');
    // Close dropdown by pressing Escape
    await page.keyboard.press('Escape');
    await sleep(500);

    // --- as_03: Search parameters expander ---
    console.log('Taking as_03_search_params...');
    // Scroll sidebar to show the expander content
    await page.evaluate(() => {
      const sidebar = document.querySelector('[data-testid="stSidebarContent"]');
      if (sidebar) sidebar.scrollTop = 0;
    });
    await sleep(500);
    await screenshot(page, 'as_03_search_params');

    // --- as_04: Query input ---
    console.log('Taking as_04_query_input...');
    await fillFormInput(page, QUERY_1);
    await sleep(500);
    await screenshot(page, 'as_04_query_input');

    // ========================================
    // GROUP 2: Search results (as_05, as_06)
    // ========================================
    console.log('\n=== GROUP 2: Search results ===');
    console.log('Executing search (this may take a while)...');
    await submitForm(page);
    const hasResults = await waitForSearchResults(page);

    if (hasResults) {
      // --- as_05: Results overview ---
      await page.evaluate(() => window.scrollTo(0, 0));
      await sleep(1000);
      console.log('Taking as_05_results...');
      await screenshot(page, 'as_05_results');

      // --- as_06: Result card detail (scroll to first card) ---
      await page.evaluate(() => {
        const card = document.querySelector('.response-card');
        if (card) {
          card.scrollIntoView({ block: 'start', behavior: 'instant' });
        }
      });
      await sleep(1000);
      console.log('Taking as_06_results_detail...');
      await screenshot(page, 'as_06_results_detail');
    } else {
      console.log('WARNING: No search results. as_05/as_06 skipped.');
    }

    // ========================================
    // GROUP 3: Chat history (as_07)
    // ========================================
    console.log('\n=== GROUP 3: Chat history ===');

    // Execute second search
    console.log('Executing second search for chat history...');
    await fillFormInput(page, QUERY_2);
    await sleep(500);
    await submitForm(page);
    const hasResults2 = await waitForSearchResults(page);

    if (hasResults2) {
      // Scroll to top to show chat history
      await page.evaluate(() => window.scrollTo(0, 0));
      await sleep(1000);
      console.log('Taking as_07_chat_history...');
      await screenshot(page, 'as_07_chat_history');
    } else {
      console.log('WARNING: No second search results. as_07 skipped.');
    }

    console.log('\n=== All screenshots completed! ===');

  } catch (err) {
    console.error('Error:', err.message);
    await page.screenshot({ path: path.join(SCREENSHOTS_DIR, 'debug_error.png') });
    console.log('Debug screenshot saved as debug_error.png');
  } finally {
    await browser.close();
  }
})();
