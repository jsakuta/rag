/**
 * Cosmos DB テストデータ投入スクリプト
 * 実行: npx ts-node scripts/seed-cosmos.ts
 */
import { CosmosClient } from "@azure/cosmos";
import { DefaultAzureCredential } from "@azure/identity";

const ENDPOINT = "https://cosmos-maintenance-poc.documents.azure.com:443/";
const DB_NAME = "maintenance-db";

// --- シナリオ テストデータ ---
const scenarios = [
  {
    id: "scenario-0001",
    dataType: "scenario",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "口座開設手続きフロー",
    content:
      "お客様が口座開設を希望された場合、本人確認書類を2点ご用意いただきます。運転免許証、マイナンバーカード、パスポート等のいずれか2点を確認し、コピーを取得します。未成年の場合は親権者の同意書も必要です。",
    combinedContent:
      "口座開設手続きフロー お客様が口座開設を希望された場合、本人確認書類を2点ご用意いただきます。運転免許証、マイナンバーカード、パスポート等のいずれか2点を確認し、コピーを取得します。未成年の場合は親権者の同意書も必要です。",
    keywords: ["口座開設", "本人確認書類", "運転免許証", "マイナンバーカード"],
    updatedAt: "2026-01-15T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "scenario-0002",
    dataType: "scenario",
    categoryId: "cat-kawase",
    categoryName: "為替",
    title: "海外送金手続きフロー",
    content:
      "海外送金を行う場合、送金依頼書の記入と本人確認書類の提示が必要です。10万円を超える送金の場合はマイナンバーの届出が必要となります。送金手数料は送金先の国・地域により異なります。",
    combinedContent:
      "海外送金手続きフロー 海外送金を行う場合、送金依頼書の記入と本人確認書類の提示が必要です。10万円を超える送金の場合はマイナンバーの届出が必要となります。送金手数料は送金先の国・地域により異なります。",
    keywords: ["海外送金", "本人確認書類", "マイナンバー", "送金手数料"],
    updatedAt: "2026-01-15T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "scenario-0003",
    dataType: "scenario",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "定期預金解約手続きフロー",
    content:
      "定期預金の解約は、預入店舗の窓口で受け付けます。届出印と通帳をご持参ください。中途解約の場合は中途解約利率が適用されます。満期日以降の解約は元利自動継続の設定を確認してください。",
    combinedContent:
      "定期預金解約手続きフロー 定期預金の解約は、預入店舗の窓口で受け付けます。届出印と通帳をご持参ください。中途解約の場合は中途解約利率が適用されます。満期日以降の解約は元利自動継続の設定を確認してください。",
    keywords: ["定期預金", "解約", "届出印", "中途解約利率"],
    updatedAt: "2026-01-20T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "scenario-0004",
    dataType: "scenario",
    categoryId: "cat-yushi",
    categoryName: "融資",
    title: "住宅ローン申込手続きフロー",
    content:
      "住宅ローンの申込には、本人確認書類2点、収入証明書（源泉徴収票または確定申告書）、物件資料が必要です。事前審査は最短3営業日で結果をお知らせします。団体信用生命保険への加入が条件となります。",
    combinedContent:
      "住宅ローン申込手続きフロー 住宅ローンの申込には、本人確認書類2点、収入証明書（源泉徴収票または確定申告書）、物件資料が必要です。事前審査は最短3営業日で結果をお知らせします。団体信用生命保険への加入が条件となります。",
    keywords: ["住宅ローン", "本人確認書類", "収入証明書", "団体信用生命保険"],
    updatedAt: "2026-01-20T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "scenario-0005",
    dataType: "scenario",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "キャッシュカード再発行手続き",
    content:
      "キャッシュカードの紛失・破損時は、届出印と本人確認書類をご持参のうえ窓口で再発行手続きを行います。再発行手数料は1,100円（税込）です。ICキャッシュカードの場合は約2週間で簡易書留にてお届けします。",
    combinedContent:
      "キャッシュカード再発行手続き キャッシュカードの紛失・破損時は、届出印と本人確認書類をご持参のうえ窓口で再発行手続きを行います。再発行手数料は1,100円（税込）です。ICキャッシュカードの場合は約2週間で簡易書留にてお届けします。",
    keywords: ["キャッシュカード", "再発行", "届出印", "本人確認書類"],
    updatedAt: "2026-02-01T10:00:00+09:00",
    isDeleted: false,
  },
];

// --- FAQ テストデータ ---
const faqs = [
  {
    id: "faq-0001",
    dataType: "faq",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "口座開設に必要な書類は何ですか？",
    content:
      "口座開設には本人確認書類が2点必要です。運転免許証、マイナンバーカード、健康保険証、パスポートなどからお選びいただけます。顔写真付きの書類が1点以上含まれていることが望ましいです。",
    combinedContent:
      "口座開設に必要な書類は何ですか？ 口座開設には本人確認書類が2点必要です。運転免許証、マイナンバーカード、健康保険証、パスポートなどからお選びいただけます。顔写真付きの書類が1点以上含まれていることが望ましいです。",
    keywords: ["口座開設", "本人確認書類", "必要書類"],
    updatedAt: "2026-01-10T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0002",
    dataType: "faq",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "通帳を紛失した場合はどうすればよいですか？",
    content:
      "通帳を紛失された場合は、速やかに最寄りの支店窓口へご連絡ください。届出印と本人確認書類をご持参のうえ、再発行手続きを行います。再発行手数料は1,100円（税込）です。",
    combinedContent:
      "通帳を紛失した場合はどうすればよいですか？ 通帳を紛失された場合は、速やかに最寄りの支店窓口へご連絡ください。届出印と本人確認書類をご持参のうえ、再発行手続きを行います。再発行手数料は1,100円（税込）です。",
    keywords: ["通帳", "紛失", "再発行"],
    updatedAt: "2026-01-10T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0003",
    dataType: "faq",
    categoryId: "cat-kawase",
    categoryName: "為替",
    title: "海外送金の手数料はいくらですか？",
    content:
      "海外送金の手数料は送金先の国・地域や金額によって異なります。電信送金の場合、基本手数料は4,000円〜7,500円です。別途、中継銀行手数料（2,500円）や為替手数料がかかる場合があります。",
    combinedContent:
      "海外送金の手数料はいくらですか？ 海外送金の手数料は送金先の国・地域や金額によって異なります。電信送金の場合、基本手数料は4,000円〜7,500円です。別途、中継銀行手数料（2,500円）や為替手数料がかかる場合があります。",
    keywords: ["海外送金", "手数料", "電信送金"],
    updatedAt: "2026-01-10T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0004",
    dataType: "faq",
    categoryId: "cat-kawase",
    categoryName: "為替",
    title: "送金時に必要な身分証明書は何ですか？",
    content:
      "送金時には本人確認書類の提示が必要です。10万円を超える送金の場合は、マイナンバーの届出も必要となります。本人確認書類は運転免許証、パスポート等の顔写真付き書類が必要です。",
    combinedContent:
      "送金時に必要な身分証明書は何ですか？ 送金時には本人確認書類の提示が必要です。10万円を超える送金の場合は、マイナンバーの届出も必要となります。本人確認書類は運転免許証、パスポート等の顔写真付き書類が必要です。",
    keywords: ["送金", "身分証明書", "本人確認書類", "マイナンバー"],
    updatedAt: "2026-01-10T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0005",
    dataType: "faq",
    categoryId: "cat-yushi",
    categoryName: "融資",
    title: "住宅ローンの申込に必要な書類は何ですか？",
    content:
      "住宅ローンの申込には以下の書類が必要です：本人確認書類2点、収入証明書（源泉徴収票2年分）、住民票、印鑑証明書、物件の売買契約書または重要事項説明書のコピーです。",
    combinedContent:
      "住宅ローンの申込に必要な書類は何ですか？ 住宅ローンの申込には以下の書類が必要です：本人確認書類2点、収入証明書（源泉徴収票2年分）、住民票、印鑑証明書、物件の売買契約書または重要事項説明書のコピーです。",
    keywords: ["住宅ローン", "申込", "必要書類", "収入証明書"],
    updatedAt: "2026-01-10T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0006",
    dataType: "faq",
    categoryId: "cat-yushi",
    categoryName: "融資",
    title: "ローン申込に必要な書類は？",
    content:
      "各種ローンの申込には、本人確認書類（運転免許証等）と収入を証明する書類が必要です。カードローンの場合、50万円以下のお借入であれば収入証明書は不要です。",
    combinedContent:
      "ローン申込に必要な書類は？ 各種ローンの申込には、本人確認書類（運転免許証等）と収入を証明する書類が必要です。カードローンの場合、50万円以下のお借入であれば収入証明書は不要です。",
    keywords: ["ローン", "申込", "本人確認書類", "収入証明書"],
    updatedAt: "2026-01-15T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0007",
    dataType: "faq",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "ATMの利用時間と手数料を教えてください",
    content:
      "当行ATMは平日8:45〜18:00は手数料無料でご利用いただけます。時間外（18:00〜翌8:45）および土日祝日は110円の手数料がかかります。コンビニATMは1回あたり110円〜220円の手数料が発生します。",
    combinedContent:
      "ATMの利用時間と手数料を教えてください 当行ATMは平日8:45〜18:00は手数料無料でご利用いただけます。時間外（18:00〜翌8:45）および土日祝日は110円の手数料がかかります。コンビニATMは1回あたり110円〜220円の手数料が発生します。",
    keywords: ["ATM", "利用時間", "手数料"],
    updatedAt: "2026-01-15T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0008",
    dataType: "faq",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "スーパーカード(DC)の申込方法は？",
    content:
      "スーパーカード(DC)のお申込は、最寄りの支店窓口またはインターネットバンキングから可能です。年会費は初年度無料、2年目以降は1,375円（税込）です。キャッシュカードとの一体型もお選びいただけます。",
    combinedContent:
      "スーパーカード(DC)の申込方法は？ スーパーカード(DC)のお申込は、最寄りの支店窓口またはインターネットバンキングから可能です。年会費は初年度無料、2年目以降は1,375円（税込）です。キャッシュカードとの一体型もお選びいただけます。",
    keywords: ["スーパーカード", "DC", "申込", "年会費"],
    updatedAt: "2026-01-20T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0009",
    dataType: "faq",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "AMLフィルターとは何ですか？",
    content:
      "AMLフィルターは、マネー・ロンダリング（資金洗浄）防止のための取引監視システムです。一定金額以上の取引や不審な取引パターンを自動検知し、法令に基づく疑わしい取引の届出を支援します。",
    combinedContent:
      "AMLフィルターとは何ですか？ AMLフィルターは、マネー・ロンダリング（資金洗浄）防止のための取引監視システムです。一定金額以上の取引や不審な取引パターンを自動検知し、法令に基づく疑わしい取引の届出を支援します。",
    keywords: ["AMLフィルター", "マネーロンダリング", "取引監視"],
    updatedAt: "2026-01-25T10:00:00+09:00",
    isDeleted: false,
  },
  {
    id: "faq-0010",
    dataType: "faq",
    categoryId: "cat-yokin",
    categoryName: "預金",
    title: "インターネットバンキングの初期設定方法は？",
    content:
      "インターネットバンキングの初期設定は、仮パスワード通知書に記載されたURLからアクセスし、契約者番号と仮パスワードでログインしてください。初回ログイン時にパスワードの変更が必要です。ワンタイムパスワードの設定も併せてお願いします。",
    combinedContent:
      "インターネットバンキングの初期設定方法は？ インターネットバンキングの初期設定は、仮パスワード通知書に記載されたURLからアクセスし、契約者番号と仮パスワードでログインしてください。初回ログイン時にパスワードの変更が必要です。ワンタイムパスワードの設定も併せてお願いします。",
    keywords: ["インターネットバンキング", "初期設定", "ワンタイムパスワード"],
    updatedAt: "2026-02-01T10:00:00+09:00",
    isDeleted: false,
  },
];

async function main() {
  const credential = new DefaultAzureCredential();
  const client = new CosmosClient({ endpoint: ENDPOINT, aadCredentials: credential });
  const db = client.database(DB_NAME);

  // scenarios コンテナにデータ投入
  const scenarioContainer = db.container("scenarios");
  console.log("=== scenarios コンテナにデータ投入中 ===");
  for (const item of scenarios) {
    const { resource } = await scenarioContainer.items.upsert(item);
    console.log(`  ✓ ${resource?.id}: ${resource?.title}`);
  }

  // faqs コンテナにデータ投入
  const faqContainer = db.container("faqs");
  console.log("\n=== faqs コンテナにデータ投入中 ===");
  for (const item of faqs) {
    const { resource } = await faqContainer.items.upsert(item);
    console.log(`  ✓ ${resource?.id}: ${resource?.title}`);
  }

  console.log(`\n完了: scenarios ${scenarios.length}件, faqs ${faqs.length}件を投入しました`);
}

main().catch((err) => {
  console.error("エラー:", err.message);
  process.exit(1);
});
