# AI Detection System

AI writing analysis and authorship improvement platform MVP.

## Features

- Drag-and-drop document upload and paste-text analysis
- PDF, DOCX, TXT, RTF, ODT, HTML, and EPUB user-facing workflow
- AI-writing signal with confidence, limitations, and uncertainty language
- Passage-level highlights and explanations
- Downloadable detection report
- Free plan: detection report
- Premium plan: EUR 5/month for all features, including Writing Improvement Assistant
- Guided revision workflow with protected-content checks, side-by-side comparison, approval, reanalysis, and revised export
- Static browser fallback for live GitHub Pages demo
- Flask backend API for local model-backed analysis

## Run Locally

```powershell
cd F:\AI-Detector
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Real Premium Payments

The Premium plan uses Stripe Checkout in subscription mode for EUR 5/month.

Local backend checkout:

```powershell
$env:STRIPE_SECRET_KEY="sk_test_..."
python app.py
```

Optional: set `STRIPE_PREMIUM_PRICE_ID` to a recurring EUR 5/month Stripe Price. If it is not set, the backend creates inline recurring price data for the Checkout Session.

Static live site checkout:

- Create a Stripe Payment Link for the Premium EUR 5/month subscription.
- Configure the live page to expose it as `window.STRIPE_PAYMENT_LINK`.

Production note: use Stripe webhooks, especially `checkout.session.completed` and subscription lifecycle events, before unlocking Premium on a real account.

## Live Static Site

The frontend can run as a static site. When the Flask API is unavailable, it uses a conservative browser-only demo analyzer and labels that mode clearly.

GitHub Pages is configured through `.github/workflows/pages.yml`.

## Responsible Use

This app reports statistical writing-pattern signals for human review. It does not prove authorship or academic misconduct. The writing assistant improves clarity and evidence while preserving meaning; it is not designed or marketed to bypass AI detectors.

## Checking User Usage

In this static MVP:

1. Open the website.
2. Go to `Login`.
3. Use the demo admin account:
   - Email: `admin@veritas.local`
   - Password: `admin123`
4. Open `Admin`.
5. Review the `User Usage` table for scans, report downloads, plan, role, and last active time.

This usage data is stored in browser `localStorage`, so it is only for demo/testing on one browser.

The Flask backend now includes a production-style SQLite foundation:

- `users`
- `sessions`
- `usage_records`

The app logs server-side events for:

- `account_created`
- `login`
- `scan_completed`
- `report_downloaded`
- `payment_checkout_started`
- `revised_exported`

Admins can query usage through:

```http
GET /admin/usage
Authorization: Bearer <admin-session-token>
```

For a larger production deployment, move the same schema to PostgreSQL and track:

- `user_id`
- `organization_id`
- `event_type` such as `scan_started`, `report_downloaded`, `premium_checkout_completed`
- `document_id`
- `analysis_id`
- `created_at`
- `metadata`

Admin pages should read usage through protected backend endpoints and require a verified admin role on the server, not only in the browser.

## Checking Website Traffic

In the Flask backend, page views are logged through:

```http
POST /traffic/pageview
```

Admins can view traffic in the Admin page or query:

```http
GET /admin/usage
Authorization: Bearer <admin-session-token>
```

The response includes:

- total page views
- estimated unique visitors
- latest page view time
- usage events by type

On the static live demo, traffic counters use browser-local fallback data. For real public analytics across all visitors, connect a production backend, Cloudflare Web Analytics, Plausible, Google Analytics, or another analytics provider.

## Clean Domain

The current hosted URL is a generated deployment domain. To remove the `s-karthikeyan5401.chatgpt.site` part, attach a custom domain you own, such as:

```text
www.yourdomain.com
```

After you provide the domain name, add it in the Sites deployment and create the required DNS records at your domain registrar.
