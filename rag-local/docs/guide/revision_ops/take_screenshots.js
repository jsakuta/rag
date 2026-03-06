/**
 * Playwright script to take 6 screenshots for ops_ui_guide.docx
 *
 * Re-take targets: 05, 06, 07, 10, 11, 13
 * Each group uses a fresh page load to avoid chat history contamination.
 */
const { chromium } = require('playwright');
const path = require('path');

const SCREENSHOTS_DIR = path.join(__dirname, 'screenshots');
const BASE_URL = 'http://localhost:8501';
const VIEWPORT = { width: 1280, height: 720 };

const EVAL_QUERY = '健康保険証が廃止され、マイナンバーカードによる資格確認証に変更された場合の手続き';

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitForStreamlit(page) {
  await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 30000 });
  await sleep(2000);
}

async function selectRadioOption(page, groupLabel, optionText) {
  const radioGroup = page.locator(`[data-testid="stRadio"]:has-text("${groupLabel}")`);
  await radioGroup.locator(`label:has-text("${optionText}")`).click();
  await sleep(1500);
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

async function waitForSearchResults(page, timeoutMs = 120000) {
  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    const tabs = await page.locator('[data-testid="stTabs"]').count();
    if (tabs > 0) {
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
  await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 60000 });
  await waitForStreamlit(page);
}

(async () => {
  console.log('Starting Playwright screenshot capture...\n');

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  try {
    // ========================================
    // GROUP 1: Eval mode (05, 06, 07)
    // ========================================
    console.log('=== GROUP 1: Evaluation mode ===');
    await freshLoad(page);

    // Select eval mode + revision ③
    await selectRadioOption(page, 'モード', '評価モード');
    await selectDropdown(page, '改定番号', '③');
    await sleep(1000);

    // --- 05: Query input ---
    await fillFormInput(page, EVAL_QUERY);
    await sleep(500);
    console.log('Taking 05_eval_query_input...');
    await screenshot(page, '05_eval_query_input');

    // --- 06: Search results ---
    console.log('Executing semantic search...');
    await submitForm(page);
    const hasResults = await waitForSearchResults(page);

    if (hasResults) {
      // Scroll to top to show results from the beginning
      await page.evaluate(() => window.scrollTo(0, 0));
      await sleep(1000);
      console.log('Taking 06_eval_results...');
      await screenshot(page, '06_eval_results');

      // --- 07: Result detail (scroll to show card headers) ---
      // Use the actual CSS class "response-card" and scroll the Streamlit main container
      await page.evaluate(() => {
        const card = document.querySelector('.response-card');
        if (card) {
          card.scrollIntoView({ block: 'start', behavior: 'instant' });
        }
      });
      await sleep(1000);
      // Scroll a bit more so the card header is centered
      await page.evaluate(() => {
        const container = document.querySelector('[data-testid="stAppViewContainer"] section[data-testid="stMain"]')
          || document.querySelector('[data-testid="stAppViewContainer"]');
        if (container) container.scrollBy(0, 150);
        else window.scrollBy(0, 150);
      });
      await sleep(1500);
      console.log('Taking 07_eval_results_detail...');
      await screenshot(page, '07_eval_results_detail');
    } else {
      console.log('WARNING: No eval results. 06/07 skipped.');
    }

    // ========================================
    // GROUP 2: Impact mode (10, 11)
    // ========================================
    console.log('\n=== GROUP 2: Impact investigation mode ===');
    await freshLoad(page);

    // Switch to impact mode
    await selectRadioOption(page, 'モード', '影響調査モード');
    await sleep(1000);

    // --- 10: Query input ---
    await fillFormInput(page, EVAL_QUERY);
    await sleep(500);
    console.log('Taking 10_impact_query...');
    await screenshot(page, '10_impact_query');

    // --- 11: Search results ---
    console.log('Executing impact search...');
    await submitForm(page);
    const hasImpact = await waitForSearchResults(page);

    if (hasImpact) {
      await page.evaluate(() => window.scrollTo(0, 0));
      await sleep(1000);
      console.log('Taking 11_impact_results...');
      await screenshot(page, '11_impact_results');
    } else {
      console.log('WARNING: No impact results. 11 skipped.');
    }

    // ========================================
    // GROUP 3: Keyword search (13)
    // ========================================
    console.log('\n=== GROUP 3: Keyword search ===');
    await freshLoad(page);

    // Select eval mode + revision ⑤ + keyword search
    await selectRadioOption(page, 'モード', '評価モード');
    await selectDropdown(page, '改定番号', '⑤');
    await sleep(1000);

    // Scroll sidebar down to show search settings
    await page.evaluate(() => {
      const sidebar = document.querySelector('[data-testid="stSidebarContent"]');
      if (sidebar) sidebar.scrollTop = 300;
    });
    await sleep(500);

    await selectRadioOption(page, '検索タイプ', 'キーワード検索');
    await sleep(1500);

    // Scroll sidebar to show keyword caption clearly
    await page.evaluate(() => {
      const sidebar = document.querySelector('[data-testid="stSidebarContent"]');
      if (sidebar) sidebar.scrollTop = 200;
    });
    await sleep(1000);

    console.log('Taking 13_keyword_search...');
    await screenshot(page, '13_keyword_search');

    console.log('\n=== All screenshots completed! ===');

  } catch (err) {
    console.error('Error:', err.message);
    await page.screenshot({ path: path.join(SCREENSHOTS_DIR, 'debug_error.png') });
    console.log('Debug screenshot saved as debug_error.png');
  } finally {
    await browser.close();
  }
})();
